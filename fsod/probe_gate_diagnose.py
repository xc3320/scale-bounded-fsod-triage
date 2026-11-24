"""诊断：被 GT 匹配的 proposal 上，CLIP 的 novel/base 证据分布按尺度 × 类别统计。

目的：回答"为什么 gate(nsim>bsim) 在 small 上失效，哪种判据有区分度"。
不改模型、不训练，只用已落盘的全量缓存 clip_fusion_cache_full.pt。

对每张图：proposals(resized) 反缩放回原图 -> 与 GT 算 IoU -> 每个 GT 取 IoU 最高 proposal，
记录该 proposal 的 CLIP sim：nsim(max novel)、bsim(max base)、bmean、margin=nsim-bsim、
nprob(softmax over 20 类中 novel 概率和, 多温度)。按 small/medium/large × base/novel 汇总。
"""
import argparse
import os
import numpy as np
import torch

from fsod import voc, engine


def area_bin(area):
    # 用 engine.AREA_BINS 一致定义；fallback 到 COCO 近似
    bins = getattr(engine, "AREA_BINS", None)
    if bins:
        s = bins["small"]; m = bins["medium"]
        # 兼容 tuple(area_range) 或 callable
        if isinstance(s, (tuple, list)):
            smax = s[1]
        elif callable(s):
            smax = s
        else:
            smax = 32 ** 2
        if isinstance(m, (tuple, list)):
            mmax = m[1]
        elif callable(m):
            mmax = m
        else:
            mmax = 96 ** 2
    else:
        smax, mmax = 32 ** 2, 96 ** 2
    if area < smax:
        return "small"
    if area < mmax:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--out", default="data/gate_diagnose.json")
    ap.add_argument("--max-imgs", type=int, default=-1)
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data")) if args.data == "0712" \
        else voc.prepare(vr, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_idx = [c - 1 for c in novel_labs]      # sim 列 = 类别-1
    base_labs = [ft_map[c] for c in sp["base"]]
    base_idx = [c - 1 for c in base_labs]

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]

    # 累加器：[scale][kind] -> list of dict
    from collections import defaultdict
    acc = defaultdict(list)

    def iou_xyxy(a, b):
        # a:(N,4) b:(M,4)
        ix1 = torch.max(a[:, None, 0], b[None, :, 0])
        iy1 = torch.max(a[:, None, 1], b[None, :, 1])
        ix2 = torch.min(a[:, None, 2], b[None, :, 2])
        iy2 = torch.min(a[:, None, 3], b[None, :, 3])
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
        area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / union.clamp(min=1e-6)

    n_img = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]          # list per image (resized xyxy)
        s_b = sims[ci]                         # list per image (n,20)
        for bi in range(len(props_b)):
            if args.max_imgs > 0 and n_img >= args.max_imgs:
                break
            props = props_b[bi].float()
            s = s_b[bi].float()                # (n,20)
            oh, ow = cache["orig_sizes"][bi]
            rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[n_img]
            gb = torch.from_numpy(np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4))
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            n_img += 1
            if len(gb) == 0 or props_orig.shape[0] == 0:
                continue
            keep = gd == 0
            gb, gl = gb[keep], gl[keep]
            if len(gb) == 0:
                continue
            ious = iou_xyxy(props_orig, gb)    # (n_prop, n_gt)
            best_i, best_j = ious.max(0)        # values, indices per gt
            best_i = best_i.numpy(); best_j = best_j.numpy()
            for j in range(len(gb)):
                pi = int(best_j[j])
                if float(best_i[j]) < 0.3:
                    continue
                lab = int(gl[j])
                kind = "novel" if lab in novel_labs else ("base" if lab in base_labs else "other")
                if kind == "other":
                    continue
                x1, y1, x2, y2 = gb[j].tolist()
                area = max(0, x2 - x1) * max(0, y2 - y1)
                sc = area_bin(area)
                sv = s[pi]                     # (20,)
                nsim = float(sv[novel_idx].max())
                bsim = float(sv[base_idx].max())
                bmean = float(sv[base_idx].mean())
                nmean = float(sv[novel_idx].mean())
                rec = {"nsim": nsim, "bsim": bsim, "bmean": bmean, "nmean": nmean,
                       "margin": nsim - bsim, "mbm": nsim - bmean,
                       "novel_win": 1.0 if nsim > bsim else 0.0,
                       "iou": float(best_i[j])}
                # softmax 温度下的 novel 概率和
                for T in [0.01, 0.02, 0.05, 0.1]:
                    p = torch.softmax(sv / T, dim=0)
                    rec[f"nprob{T}"] = float(p[novel_idx].sum())
                acc[(sc, kind)].append(rec)
        if ci % 400 == 0:
            print(f"  diag {n_img}/{blob['n_imgs']} imgs", flush=True)

    # 汇总
    print(f"\n{'scale':7s} {'kind':6s} {'n':>5s} | {'nsim':>6s} {'bsim':>6s} {'bmean':>6s} "
          f"| {'margin':>7s} {'mbm':>7s} {'win%':>6s} | {'np.01':>6s} {'np.02':>6s} {'np.05':>6s} {'np.1':>6s}")
    out = {}
    for sc in ["small", "medium", "large"]:
        for kind in ["novel", "base"]:
            rs = acc[(sc, kind)]
            if not rs:
                continue
            keys = ["nsim", "bsim", "bmean", "nmean", "margin", "mbm", "novel_win",
                    "nprob0.01", "nprob0.02", "nprob0.05", "nprob0.1"]
            m = {k: float(np.mean([r[k] for r in rs])) for k in keys}
            out[f"{sc}_{kind}"] = {"n": len(rs), **m}
            print(f"{sc:7s} {kind:6s} {len(rs):5d} | {m['nsim']:6.3f} {m['bsim']:6.3f} {m['bmean']:6.3f} "
                  f"| {m['margin']:+7.3f} {m['mbm']:+7.3f} {m['novel_win']*100:6.1f} "
                  f"| {m['nprob0.01']:6.3f} {m['nprob0.02']:6.3f} {m['nprob0.05']:6.3f} {m['nprob0.1']:6.3f}")

    with open(args.out, "w") as f:
        import json
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
