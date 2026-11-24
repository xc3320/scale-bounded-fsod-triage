"""论文3 P1-m4：多方法错位率区分度表 + base-small 33% 口径核验。

固定同一批小目标 novel 匹配实例（IoU>=0.3，与 probe_scale_resample 同口径），
对 5+1 个配置（baseline / gate0.0 / dualb3.0 / dualr3.0_0.5 / dualr4.0_1.0 / dualobj3.0，lam=6）
各自在缓存 logits 上施加融合决策，统计：
  - novel small matched: n / bg率(pred==0) / 错位率(pred==0 & IoU>0.5)
  - base  small matched: n / bg率 / 错位率（显式分母，回应审稿 minor#3 的 33% 口径问题）
匹配基于原始 proposals（与融合无关），故实例集在各配置间严格相同。
"""
import argparse
import json
import os

import numpy as np
import torch

from fsod import voc, engine


def iou_xyxy(a, b):
    ix1 = torch.max(a[:, None, 0], b[None, :, 0]); iy1 = torch.max(a[:, None, 1], b[None, :, 1])
    ix2 = torch.min(a[:, None, 2], b[None, :, 2]); iy2 = torch.min(a[:, None, 3], b[None, :, 3])
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter).clamp(min=1e-6)


def apply_mode(mode, lg, s, novel_idx, base_idx, n_all, lam, reg_chunk=None, obj=None):
    """复刻 eval_clip_fusion 的决策公式，就地修改 lg。"""
    if mode == "baseline":
        return lg
    if mode.startswith("gate"):
        thr = float(mode[4:]) if len(mode) > 4 else 0.05
        nsim = s[:, novel_idx].max(1).values
        bsim = s[:, base_idx].max(1).values
        gate = torch.clamp((nsim - bsim - thr) / 0.1, 0, 1)
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
    elif mode.startswith("dualb"):
        bgc = float(mode[5:]) if len(mode) > 5 else lam
        nsim = s[:, novel_idx].max(1).values
        bsim = s[:, base_idx].max(1).values
        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
        lg[:, 0] = lg[:, 0] - bgc * nsim
    elif mode.startswith("dualr"):
        spec = mode[5:]
        bgc, tau = [float(x) for x in spec.split("_")]
        nsim = s[:, novel_idx].max(1).values
        bsim = s[:, base_idx].max(1).values
        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
        rmag = reg_chunk.reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)
        w = 1.0 / (1.0 + rmag / max(tau, 1e-4))
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
        lg[:, 0] = lg[:, 0] - bgc * nsim * w
    elif mode.startswith("dualobj"):
        bgc = float(mode[7:]) if len(mode) > 7 else 3.0
        nsim = s[:, novel_idx].max(1).values
        bsim = s[:, base_idx].max(1).values
        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
        lg[:, 0] = lg[:, 0] - bgc * nsim * obj
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--objectness", default="data/objectness_full.pt")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--lam", type=float, default=6.0)
    ap.add_argument("--modes", default="baseline,gate0.0,dualb3.0,dualr3.0_0.5,dualr4.0_1.0,dualobj3.0")
    ap.add_argument("--out", default="data/probe_misalign_byconfig.json")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_idx = [c - 1 for c in novel_labs]
    base_labs = [c for c in range(1, 21) if c not in novel_labs]
    n_all = 20

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]
    obj_blob = None
    if any(m.startswith("dualobj") for m in args.modes.split(",")):
        obj_blob = torch.load(args.objectness, map_location="cpu", weights_only=False)
        print(f"[obj] keys={list(obj_blob.keys())}")
    cache_offsets = []
    off = 0
    for c in caches:
        cache_offsets.append(off)
        off += len(c["proposals"])

    # ---- 固定实例集：小目标 novel matched + 小目标 base matched（原始 proposals 匹配）----
    inst = []   # dict(img, pi, iou, kind(novel/base), gtcls)
    gi = 0
    for ci, cache in enumerate(caches):
        for bi in range(len(cache["proposals"])):
            img = gi; gi += 1
            props = cache["proposals"][bi].float()
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            props_orig = props * torch.tensor([ow / rw, oh / rh, ow / rw, oh / rh])
            gt = gts[img]
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]
            if len(gb2) == 0:
                continue
            ious = iou_xyxy(props_orig, torch.from_numpy(gb2))
            for j in range(len(gb2)):
                kind = "novel" if gl2[j] in novel_labs else ("base" if gl2[j] in base_labs else None)
                if kind is None:
                    continue
                x1, y1, x2, y2 = gb2[j]
                if max(0, x2 - x1) * max(0, y2 - y1) >= engine.AREA_BINS["small"][1]:
                    continue
                col = ious[:, j]
                if float(col.max()) < 0.3:
                    continue
                pi_ = int(col.argmax().item())
                inst.append({"img": img, "ci": ci, "bi": bi, "pi": pi_,
                             "iou": float(col[pi_]), "kind": kind, "gtcls": int(gl2[j])})
    n_novel = sum(1 for r in inst if r["kind"] == "novel")
    n_base = sum(1 for r in inst if r["kind"] == "base")
    print(f"instances: novel small matched n={n_novel}, base small matched n={n_base}")

    # ---- 每配置判定 ----
    rows = {}
    for mode in args.modes.split(","):
        preds = {}
        for ci, cache in enumerate(caches):
            props_b = cache["proposals"]; s_b = sims[ci]
            logits = cache["logits"]; regs = cache["reg"]
            sizes = [p.shape[0] for p in props_b]
            lg_off, off = 0, 0
            for bi in range(len(props_b)):
                ns = sizes[bi]
                lg = logits[off:off + ns].clone().float()
                s = s_b[bi].float()
                rchunk = regs[off:off + ns].float()
                obj = None
                if mode.startswith("dualobj"):
                    gimg = cache_offsets[ci] + bi
                    obj = obj_blob["objectness"][gimg].float()
                    assert obj.shape[0] == ns
                lg = apply_mode(mode, lg, s, novel_idx,
                                [li for li in range(n_all) if li not in novel_idx],
                                n_all, args.lam, rchunk, obj)
                pred = lg.argmax(1)
                for k in range(ns):
                    preds[(cache_offsets[ci] + bi, k)] = int(pred[k])
                off += ns
        # 汇总
        agg = {"novel": {"n": 0, "bg": 0, "bg_goodloc": 0},
               "base": {"n": 0, "bg": 0, "bg_goodloc": 0}}
        for r in inst:
            pred = preds[(r["img"], r["pi"])]
            a = agg[r["kind"]]
            a["n"] += 1
            if pred == 0:
                a["bg"] += 1
                if r["iou"] > 0.5:
                    a["bg_goodloc"] += 1
        for kind in ["novel", "base"]:
            a = agg[kind]
            a["bg_rate"] = a["bg"] / max(1, a["n"])
            a["mis_rate"] = a["bg_goodloc"] / max(1, a["n"])
        rows[mode] = agg
        nn, nb = agg["novel"], agg["base"]
        print(f"{mode:14s} | novel small: n={nn['n']} bg={nn['bg_rate']*100:5.1f}% mis={nn['mis_rate']*100:5.1f}% "
              f"| base small: n={nb['n']} bg={nb['bg_rate']*100:5.1f}% mis={nb['mis_rate']*100:5.1f}%")

    out = {"lam": args.lam, "n_novel": n_novel, "n_base": n_base, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
