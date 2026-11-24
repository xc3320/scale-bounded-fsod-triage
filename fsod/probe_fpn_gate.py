"""①′ Day3 pre-registered feasibility gate: cross-FPN level consistency.

Question: can level-consistency signals separate the 36 bg&goodloc small novel
instances (predicted bg at canonical scale, but well localized) from the
small-unmatched FP pool?

Pre-registered gate (written in 总档 before running):
  the NEW signal must be able to separate >= 5/36 target instances from the
  FP pool at a workable FP cost. If no rule in the evaluated family achieves
  this, ①' is falsified with the (sims, reg, level-consistency) signal set
  and paper 3 restructures per the contingency plan.

Inputs:
  data/clip_fusion_cache_full.pt  (caches, sims, gts) — reference cache
  data/fpn_levels_full.pt         (caches, level_novelprob, level_argmax)

Both iterate the same test loader in the same order; proposal alignment is
verified per image (allclose) before analysis.

Signals per proposal (5 forced levels, box-scaled pooling):
  fgmax   = max over levels of max-fg prob
  nnovel  = # levels whose argmax is a novel class
  nfg     = # levels whose argmax is fg (>0)
  ncanon  = # levels whose argmax equals the proposal's canonical-level argmax
  ngtcls  = # levels whose argmax equals the GT novel class (targets only)

Rules evaluated:
  A: nnovel >= k            (k = 1..5)
  B: fgmax >= t & nnovel>=k (t in {0.3,0.5,0.7,0.9})
  C: ngtcls >= k            (targets only, k = 1..5)
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
    ap.add_argument("--fpn", default="data/fpn_levels_full.pt")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default="data/probe_fpn_gate.json")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_set = set(novel_labs)

    ref = torch.load(args.cache, map_location="cpu", weights_only=False)
    fpn = torch.load(args.fpn, map_location="cpu", weights_only=False)
    caches, sims, gts = ref["caches"], ref["sims"], ref["gts"]
    lv_prob = fpn["level_novelprob"].float()   # (n_imgs, P, 5) max-fg prob per level
    lv_am = fpn["level_argmax"]                # (n_imgs, P, 5)
    n_ims_ref = sum(len(c["proposals"]) for c in caches)
    assert fpn["n_imgs"] == n_ims_ref, f"image count mismatch {fpn['n_imgs']} vs {n_ims_ref}"

    targets = []      # dict(gi, pi, margin, rmag, pred_bg, goodloc, iou, gtcls, sig...)
    fp = []           # per-image arrays appended; concatenated at end
    gi = 0
    n_align_bad = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]; s_b = sims[ci]
        logits = cache["logits"]; regs = cache["reg"]
        sizes = [p.shape[0] for p in props_b]
        lg_chunks, rg_chunks, off = [], [], 0
        for ns in sizes:
            lg_chunks.append(logits[off:off + ns]); rg_chunks.append(regs[off:off + ns]); off += ns
        for bi in range(len(props_b)):
            img = gi
            gi += 1
            props = props_b[bi].float(); s = s_b[bi].float(); lg = lg_chunks[bi]; rg = rg_chunks[bi]
            # alignment check with fpn cache
            fprops = fpn["caches"][ci]["proposals"][bi]
            if not (props.shape == fprops.shape and torch.allclose(props, fprops, atol=1e-3)):
                n_align_bad += 1
                continue
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[img]
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]

            rmag = rg.float().reshape(-1, 21, 4)[:, 1:, :].norm(dim=-1).mean(1)
            nsim = s[:, [c - 1 for c in novel_labs]].max(1).values
            bsim = s[:, [i for i in range(20) if (i + 1) not in novel_set]].max(1).values
            margin = nsim - bsim
            pred_bg = lg.float().argmax(1) == 0

            pw = (props_orig[:, 2] - props_orig[:, 0]).clamp(min=0)
            ph = (props_orig[:, 3] - props_orig[:, 1]).clamp(min=0)
            small_mask = (pw * ph) < engine.AREA_BINS["small"][1]

            # level signals for this image's proposals
            lp = lv_prob[img]                  # (P,5)
            la = lv_am[img]                    # (P,5)
            nnovel = np.stack([np.isin(la[:, L].numpy(), [c - 1 for c in novel_labs])
                               for L in range(5)], axis=1).sum(1)
            nfg = (la.numpy() > 0).sum(1)
            am0 = la[:, 0].numpy()
            ncanon = (la.numpy() == am0[:, None]).sum(1)
            fgmax = lp.max(1).values.numpy()

            if len(gb2) > 0:
                ious = iou_xyxy(props_orig, torch.from_numpy(gb2))
                best_i, best_j = ious.max(0); best_i = best_i.numpy()
                best_iou_prop = ious.max(1).values
            else:
                best_i = np.zeros(props_orig.shape[0]); best_iou_prop = torch.zeros(props_orig.shape[0])

            unmatched_small = small_mask & (best_iou_prop < args.iou)
            upi = torch.nonzero(unmatched_small).flatten().tolist()
            fp.append({
                "margin": margin[upi].numpy(), "nnovel": nnovel[upi], "nfg": nfg[upi],
                "ncanon": ncanon[upi], "fgmax": fgmax[upi],
            })

            for j in range(len(gb2)):
                if gl2[j] not in novel_labs:
                    continue
                x1, y1, x2, y2 = gb2[j]
                if max(0, x2 - x1) * max(0, y2 - y1) >= engine.AREA_BINS["small"][1]:
                    continue
                col = ious[:, j] if len(gb2) > 0 else None
                if col is None or float(col.max()) < args.iou:
                    continue
                pi = int(col.argmax().item())
                gcls = gl2[j] - 1
                targets.append({
                    "img": img, "pi": pi, "margin": float(margin[pi]), "rmag": float(rmag[pi]),
                    "pred_bg": int(pred_bg[pi]), "goodloc": float(col[pi]) > 0.5,
                    "iou": float(col[pi]), "gtcls": int(gcls),
                    "nnovel": int(nnovel[pi]), "nfg": int(nfg[pi]),
                    "ncanon": int(ncanon[pi]), "fgmax": float(fgmax[pi]),
                    "ngtcls": int((la[pi].numpy() == gcls).sum()),
                    "am_seq": la[pi].tolist(),
                })

    print(f"aligned imgs OK; align failures={n_align_bad}")
    fp_all = {k: np.concatenate([d[k] for d in fp]) for k in fp[0]} if fp else {}
    bg_good = [t for t in targets if t["pred_bg"] and t["goodloc"]]
    print(f"targets(matched small novel)={len(targets)}  bg&goodloc={len(bg_good)}  FP pool={len(fp_all['margin'])}")

    # vector rules on FP side
    fp_nnovel, fp_nfg, fp_ncanon, fp_fgmax = fp_all["nnovel"], fp_all["nfg"], fp_all["ncanon"], fp_all["fgmax"]

    rows = []
    print(f"\n{'rule':32s} | {'rescue':>7s} {'/36':>5s} | {'FP hit':>8s} {'FP rate':>8s}")
    for k in range(1, 6):
        res = sum(t["nnovel"] >= k for t in bg_good)
        fph = int((fp_nnovel >= k).sum())
        rows.append({"rule": f"nnovel>={k}", "rescue": res, "fp": fph})
        print(f"nnovel>={k:<22d} | {res:7d} {res/36*100:5.1f} | {fph:8d} {fph/max(1,len(fp_nnovel))*100:8.2f}")
    for t0 in [0.3, 0.5, 0.7, 0.9]:
        for k in [1, 2, 3]:
            res = sum(t["fgmax"] >= t0 and t["nnovel"] >= k for t in bg_good)
            fph = int(((fp_fgmax >= t0) & (fp_nnovel >= k)).sum())
            rows.append({"rule": f"fgmax>={t0}&nnovel>={k}", "rescue": res, "fp": fph})
            print(f"fgmax>={t0:.1f} & nnovel>={k:<7d} | {res:7d} {res/36*100:5.1f} | {fph:8d} {fph/max(1,len(fp_nnovel))*100:8.2f}")
    for k in range(1, 6):
        res = sum(t["ngtcls"] >= k for t in bg_good)
        print(f"ngtcls>={k:<22d} | {res:7d} {res/36*100:5.1f} | (no FP analog)")

    # per-target dump for transparency
    print("\nbg&goodloc targets detail (img,pi,iou,gtcls | am_seq | nnovel,nfg,ncanon,fgmax,ngtcls,margin):")
    for t in bg_good:
        print(f"  ({t['img']:5d},{t['pi']:3d}) iou={t['iou']:.2f} cls={t['gtcls']:2d} | "
              f"am={t['am_seq']} | nn={t['nnovel']} nfg={t['nfg']} nc={t['ncanon']} "
              f"fgmax={t['fgmax']:.3f} ngt={t['ngtcls']} m={t['margin']:.3f}")

    out = {"n_targets": len(targets), "n_bg_good": len(bg_good), "n_fp": int(len(fp_nnovel)),
           "align_failures": n_align_bad, "rules": rows, "targets_detail": bg_good}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
