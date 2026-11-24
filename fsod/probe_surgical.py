"""①′ surgical-rescue feasibility: measure the rescue ceiling BEFORE designing.

For every small novel GT-matched proposal (IoU>=0.3), dump (margin, rmag, pred_bg,
goodloc). Then grid-search surgical rules "flip bg->novel iff margin>m0 AND rmag<r0":
  - rescue ceiling: how many bg&goodloc small novel satisfy the rule
  - FP cost: how many UNMATCHED proposals (potential FPs) pass the same rule
Outputs the rescue/FP frontier to choose the method design.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default="data/probe_surgical.json")
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
    n_all = 20

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]

    # collect small novel matched instances + all small unmatched proposals (FP pool)
    matched = []   # dict(margin, rmag, pred_bg, goodloc, iou)
    fp_pool = []   # dict(margin, rmag) for small proposals not matched to any GT at iou
    gi = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]; s_b = sims[ci]
        logits = cache["logits"]; regs = cache["reg"]
        sizes = [p.shape[0] for p in props_b]
        lg_chunks, rg_chunks, off = [], [], 0
        for ns in sizes:
            lg_chunks.append(logits[off:off + ns]); rg_chunks.append(regs[off:off + ns]); off += ns
        for bi in range(len(props_b)):
            props = props_b[bi].float(); s = s_b[bi].float(); lg = lg_chunks[bi]; rg = rg_chunks[bi]
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[gi]; gi += 1
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]

            rmag = rg.float().reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)
            nsim = s[:, novel_idx].max(1).values
            bsim = s[:, base_idx].max(1).values
            margin = nsim - bsim
            pred_bg = lg.float().argmax(1) == 0

            # proposal sizes in original coords
            pw = (props_orig[:, 2] - props_orig[:, 0]).clamp(min=0)
            ph = (props_orig[:, 3] - props_orig[:, 1]).clamp(min=0)
            parea = pw * ph
            small_mask = parea < engine.AREA_BINS["small"][1]

            if len(gb2) > 0 and props_orig.shape[0] > 0:
                ious = iou_xyxy(props_orig, torch.from_numpy(gb2))
                best_i, best_j = ious.max(0)          # per-GT best
                best_i = best_i.numpy()
                best_iou_prop = ious.max(1).values     # per-proposal best IoU
            else:
                best_i = np.zeros(props_orig.shape[0])
                best_iou_prop = torch.zeros(props_orig.shape[0])

            unmatched_small = small_mask & (best_iou_prop < args.iou)

            # GT side: small novel matched
            for j in range(len(gb2)):
                if gl2[j] not in novel_labs:
                    continue
                x1, y1, x2, y2 = gb2[j]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area >= engine.AREA_BINS["small"][1]:
                    continue
                # best matching proposal for this GT
                col = ious[:, j] if len(gb2) > 0 else None
                if col is None or float(col.max()) < args.iou:
                    continue
                pi = int(col.argmax().item())
                matched.append({
                    "margin": float(margin[pi]), "rmag": float(rmag[pi]),
                    "pred_bg": int(pred_bg[pi]), "goodloc": float(col[pi]) > 0.5,
                    "iou": float(col[pi]), "nsim": float(nsim[pi]),
                })

            for pi in torch.nonzero(unmatched_small).flatten().tolist():
                fp_pool.append({"margin": float(margin[pi]), "rmag": float(rmag[pi]),
                                "nsim": float(nsim[pi])})

    bg_good = [r for r in matched if r["pred_bg"] and r["goodloc"]]
    print(f"\nsmall novel matched n={len(matched)}  bg&goodloc n={len(bg_good)}")
    m = np.array([r["margin"] for r in bg_good])
    r_ = np.array([r["rmag"] for r in bg_good])
    print(f"bg&goodloc: margin mean={m.mean():.3f} p25={np.percentile(m,25):.3f} p50={np.percentile(m,50):.3f} p75={np.percentile(m,75):.3f}")
    print(f"bg&goodloc: rmag   mean={r_.mean():.3f} p25={np.percentile(r_,25):.3f} p50={np.percentile(r_,50):.3f} p75={np.percentile(r_,75):.3f}")

    fp_m = np.array([r["margin"] for r in fp_pool])
    fp_r = np.array([r["rmag"] for r in fp_pool])
    print(f"FP pool (small unmatched): n={len(fp_pool)}")

    # grid: rule = margin>m0 AND rmag<r0
    print(f"\n{'rule (m0,r0)':22s} | {'rescue':>7s} {'rescue%':>8s} | {'FP hit':>7s} {'FP rate':>8s}")
    rows = []
    for m0 in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for r0 in [0.8, 1.0, 1.2, 1.5]:
            rescue = int(((m >= m0) & (r_ < r0)).sum())
            fp_hit = int(((fp_m >= m0) & (fp_r < r0)).sum())
            fp_rate = fp_hit / max(1, len(fp_pool))
            rescue_pct = rescue / max(1, len(bg_good))
            rows.append({"m0": m0, "r0": r0, "rescue": rescue, "rescue_pct": rescue_pct,
                         "fp_hit": fp_hit, "fp_rate": fp_rate})
            print(f"margin>{m0:.2f} & rmag<{r0:.1f}   | {rescue:7d} {rescue_pct*100:8.1f} | {fp_hit:7d} {fp_rate*100:8.2f}")

    with open(args.out, "w") as f:
        json.dump({"matched": matched, "fp_pool_n": len(fp_pool), "grid": rows}, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
