"""机制验证：FSOD 中"检测分数-定位质量错位"随 novel/小目标加剧，CLIP 语义证据更诚实。

文献支撑：confidence-localization alignment（IoU-aware calibration）是 CVPR/TPAMI 活跃方向，
但没人做 few-shot 下 novel 类特有的错位。本脚本用全量缓存量化：
  - corr(检测前景置信度 pfg, IoU)  —— 检测分数是否反映定位质量
  - corr(CLIP nsim, IoU)           —— 语义证据是否更诚实
按 small/medium/large × novel/base 分组，验证错位随 novel+小目标恶化。
另统计"被判背景但 IoU>0.5"比例 = 错位铁证（定位好却被判背景）。

纯诊断，用缓存，不改模型不训练。
"""
import argparse
import os
import numpy as np
import torch

from fsod import voc, engine


def corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default="data/misalignment.json")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_idx = [c - 1 for c in novel_labs]
    base_idx = [i for i in range(20) if i not in novel_idx]

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]

    from collections import defaultdict
    rec = defaultdict(list)

    def iou_xyxy(a, b):
        ix1 = torch.max(a[:, None, 0], b[None, :, 0]); iy1 = torch.max(a[:, None, 1], b[None, :, 1])
        ix2 = torch.min(a[:, None, 2], b[None, :, 2]); iy2 = torch.min(a[:, None, 3], b[None, :, 3])
        iw = (ix2-ix1).clamp(min=0); ih = (iy2-iy1).clamp(min=0)
        inter = iw*ih
        aa = (a[:, 2]-a[:, 0]).clamp(min=0)*(a[:, 3]-a[:, 1]).clamp(min=0)
        bb = (b[:, 2]-b[:, 0]).clamp(min=0)*(b[:, 3]-b[:, 1]).clamp(min=0)
        return inter/(aa[:, None]+bb[None, :]-inter).clamp(min=1e-6)

    gi = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]; s_b = sims[ci]
        logits = cache["logits"]                       # (N_total,21)
        sizes = [p.shape[0] for p in props_b]
        lg_chunks, off = [], 0
        for ns in sizes:
            lg_chunks.append(logits[off:off+ns]); off += ns
        for bi in range(len(props_b)):
            props = props_b[bi].float(); s = s_b[bi].float(); lg = lg_chunks[bi]
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow/rw, oh/rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[gi]; gi += 1
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            if len(gb) == 0 or props_orig.shape[0] == 0:
                continue
            keep = gd == 0; gb, gl = gb[keep], gl[keep]
            if len(gb) == 0:
                continue
            # 检测分数
            prob = torch.softmax(lg.float(), dim=1)
            pfg = 1 - prob[:, 0]                       # 前景置信度
            pred_bg = lg.argmax(1) == 0                # 是否判背景
            # CLIP 证据
            nsim = s[:, novel_idx].max(1).values
            bsim = s[:, base_idx].max(1).values
            margin = nsim - bsim
            ious = iou_xyxy(props_orig, torch.from_numpy(gb))
            best_i, best_j = ious.max(0)
            best_i = best_i.numpy(); best_j = best_j.numpy()
            for j in range(len(gb)):
                if float(best_i[j]) < args.iou:
                    continue
                pi = int(best_j[j]); lab = int(gl[j])
                kind = "novel" if lab in novel_labs else ("base" if lab not in novel_labs else "other")
                if kind == "other":
                    continue
                x1, y1, x2, y2 = gb[j]
                area = max(0, x2-x1)*max(0, y2-y1)
                sc = "small" if area < engine.AREA_BINS["small"][1] else (
                     "medium" if area < engine.AREA_BINS["medium"][1] else "large")
                rec[(sc, kind)].append({
                    "iou": float(best_i[j]),
                    "pfg": float(pfg[pi]),
                    "pred_bg": int(pred_bg[pi].item()),
                    "nsim": float(nsim[pi]),
                    "margin": float(margin[pi]),
                })

    print(f"\n{'scale':7s} {'kind':6s} {'n':>5s} | {'IoU':>6s} {'pfg':>6s} {'bg%':>6s} "
          f"| {'corr(pfg,IoU)':>13s} {'corr(nsim,IoU)':>14s} {'corr(mgn,IoU)':>13s} | {'bg&IoU>.5':>10s}")
    out = {}
    for sc in ["small", "medium", "large"]:
        for kind in ["novel", "base"]:
            rs = rec[(sc, kind)]
            if len(rs) < 5:
                continue
            ious = [r["iou"] for r in rs]
            pfgs = [r["pfg"] for r in rs]
            nsims = [r["nsim"] for r in rs]
            mgns = [r["margin"] for r in rs]
            bg = [r["pred_bg"] for r in rs]
            bg_good = [r for r in rs if r["pred_bg"] and r["iou"] > 0.5]
            m = {"n": len(rs), "iou": float(np.mean(ious)), "pfg": float(np.mean(pfgs)),
                 "bg_rate": float(np.mean(bg)),
                 "corr_pfg_iou": corr(pfgs, ious),
                 "corr_nsim_iou": corr(nsims, ious),
                 "corr_margin_iou": corr(mgns, ious),
                 "bg_and_goodloc_rate": len(bg_good)/max(1, len(rs))}
            out[f"{sc}_{kind}"] = m
            print(f"{sc:7s} {kind:6s} {len(rs):5d} | {m['iou']:6.3f} {m['pfg']:6.3f} "
                  f"{m['bg_rate']*100:6.1f} | {m['corr_pfg_iou']:13.3f} {m['corr_nsim_iou']:14.3f} "
                  f"{m['corr_margin_iou']:13.3f} | {m['bg_and_goodloc_rate']*100:10.1f}")

    with open(args.out, "w") as f:
        import json
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
