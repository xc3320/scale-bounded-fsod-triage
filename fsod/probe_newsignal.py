"""④′ new-signal feasibility probe (pre-registered BEFORE running; master archive 2026-11-23 night).

Motivation (upgrade route v0.1): F1/F2/F3 falsified the signal set
{CLIP semantic similarity, regression displacement, cross-FPN consistency}.
This probe tests two signal families OUTSIDE that set, both pure score-level
(no forward pass needed, logits come from the existing cache):

  Experiment L (logit adjustment, FRACAL-style): background-logit suppression.
      Rule family: argmax(z') where z'_0 = z_0 - tau (variant A) or
      z'_c = z_c - tau*log(pi_c) with class-frequency prior (variant B).
      PRE-REGISTERED PASS: exists (variant, tau) with rescue >= 5/36 AND
      fp_hit_new < 2454 (directly comparable to the F2 point 5/36 @ 2454 FP;
      must be strictly better on BOTH axes).

  Experiment E (logits-statistics separability): energy / margin / entropy
      statistics, positive class = the 36 misaligned instances,
      negative class = the small unmatched FP pool.
      PRE-REGISTERED PASS: max AUROC >= 0.7.

Honesty notes:
  - "rescue" = the GT's best-matching proposal (same instance localization as
    probe_surgical.py, IoU>=0.3, area<small, goodloc IoU>0.5, pred_bg) flips
    to a novel-class argmax under the adjusted logits.
  - fp_hit = small unmatched proposals whose adjusted argmax becomes ANY fg
    class (baseline tau=0 fp_hit is reported for reference).
  - True ViM needs RoI features (absent from this cache); this probe measures
    the logits-statistics family only and says so.
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


def auroc(pos, neg):
    """Rank-based AUROC (Mann-Whitney). Higher score = more positive-like."""
    x = np.concatenate([np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ranks for ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default="data/probe_newsignal.json")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_idx = np.array([c - 1 for c in novel_labs])
    base_idx = [i for i in range(20) if (i) not in set(novel_idx.tolist())]
    n_all = 20

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]
    mis_logits = []      # logits vectors of the 36 (bg & goodloc) small novel matched proposals
    fp_logits = []       # logits vectors of small unmatched proposals
    mis_meta, n_matched = [], 0
    mis_aux, fp_aux = [], []   # (clip_margin, rmag) for the same proposals
    gi = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]; s_b = sims[ci]
        logits = cache["logits"]; regs = cache["reg"]
        sizes = [p.shape[0] for p in props_b]
        lg_chunks, rg_chunks, s_chunks, off = [], [], [], 0
        for ns in sizes:
            lg_chunks.append(logits[off:off + ns]); rg_chunks.append(regs[off:off + ns]); off += ns
        for bi in range(len(props_b)):
            props = props_b[bi].float(); lg = lg_chunks[bi]; rg = rg_chunks[bi]; s = s_b[bi].float()
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[gi]; gi += 1
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]

            pw = (props_orig[:, 2] - props_orig[:, 0]).clamp(min=0)
            ph = (props_orig[:, 3] - props_orig[:, 1]).clamp(min=0)
            small_mask = (pw * ph) < engine.AREA_BINS["small"][1]

            rmag = rg.float().reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)
            nsim = s[:, novel_idx].max(1).values
            bsim = s[:, base_idx].max(1).values
            clip_margin = nsim - bsim

            if len(gb2) > 0 and props_orig.shape[0] > 0:
                ious = iou_xyxy(props_orig, torch.from_numpy(gb2))
                best_iou_prop = ious.max(1).values
            else:
                ious = None; best_iou_prop = torch.zeros(props_orig.shape[0])

            unmatched_small = small_mask & (best_iou_prop < args.iou)
            fp_logits.append(lg[unmatched_small].numpy())
            fp_aux.append(torch.stack([clip_margin[unmatched_small], rmag[unmatched_small]], dim=1).numpy())

            if ious is not None:
                for j in range(len(gb2)):
                    if gl2[j] not in novel_labs:
                        continue
                    x1, y1, x2, y2 = gb2[j]
                    if max(0, x2 - x1) * max(0, y2 - y1) >= engine.AREA_BINS["small"][1]:
                        continue
                    col = ious[:, j]
                    if float(col.max()) < args.iou:
                        continue
                    pi = int(col.argmax().item())
                    n_matched += 1
                    if lg[pi].argmax().item() == 0 and float(col[pi]) > 0.5:
                        mis_logits.append(lg[pi].numpy())
                        mis_aux.append([float(clip_margin[pi]), float(rmag[pi])])
                        mis_meta.append({"iou": float(col[pi])})

    mis = np.stack(mis_logits)  # (36, 21)
    fp = np.concatenate(fp_logits)  # (~193k, 21)
    mis_aux = np.array(mis_aux)     # (36, 2) clip_margin, rmag
    fp_aux = np.concatenate(fp_aux) # (~193k, 2)
    print(f"small novel matched n={n_matched}; misaligned (bg&goodloc) n={len(mis)}; FP pool n={len(fp)}")

    # ---------- Experiment L: background-logit suppression ----------
    fg = mis[:, 1:]; bg_m = mis[:, 0]
    fg_fp = fp[:, 1:]; bg_fp = fp[:, 0]
    novel_mat = np.zeros(20, dtype=bool); novel_mat[novel_idx] = True

    # per-instance logit diagnostics
    lm_margin = fg.max(1) - bg_m
    print(f"mis logit margin (max fg - bg): mean={lm_margin.mean():.3f} p25={np.percentile(lm_margin,25):.3f} "
          f"p50={np.percentile(lm_margin,50):.3f} p75={np.percentile(lm_margin,75):.3f}")

    rows = []
    tau_grid = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    # variant A: z'_bg = z_bg - tau
    for tau in tau_grid:
        arg_m = fg.argmax(1) + 1
        flip = (fg.max(1) - (bg_m - tau)) > 0
        rescue = int((flip & novel_mat[arg_m]).sum())
        arg_fp = fg_fp.argmax(1) + 1
        flip_fp = (fg_fp.max(1) - (bg_fp - tau)) > 0
        fp_hit = int(flip_fp.sum())
        rows.append({"variant": "A_bg_suppress", "tau": tau, "rescue": rescue, "fp_hit": fp_hit,
                     "fp_hit_new": fp_hit, "fp_rate": fp_hit / len(fp)})
        print(f"A tau={tau:4d} | rescue {rescue:2d}/36 | fp_hit {fp_hit:6d} ({fp_hit/len(fp)*100:.2f}%)")
    # variant B: z'_c = z_c - tau*log(pi_c); pi_bg=0.99, pi_fg=0.01/20
    log_pi = np.log(np.concatenate([[0.99], np.full(20, 0.01 / 20)]))
    for tau in [0, 1, 2, 4, 8]:
        adj = tau * log_pi
        # per-class log-prior adjustment, then argmax over 21 classes
        z_m = mis - adj[None, :]
        am = z_m.argmax(1)
        rescue = int(((am != 0) & novel_mat[am - 1]).sum())
        z_fp = fp - adj[None, :]
        afp = z_fp.argmax(1)
        fp_hit = int((afp != 0).sum())
        rows.append({"variant": "B_logprior", "tau": tau, "rescue": rescue, "fp_hit": fp_hit,
                     "fp_hit_new": fp_hit, "fp_rate": fp_hit / len(fp)})
        print(f"B tau={tau:4d} | rescue {rescue:2d}/36 | fp_hit {fp_hit:6d} ({fp_hit/len(fp)*100:.2f}%)")

    best = max((r for r in rows if r["tau"] > 0), key=lambda r: (r["rescue"], -r["fp_hit"]))
    expL_pass = any(r["rescue"] >= 5 and r["fp_hit"] < 2454 for r in rows if r["tau"] > 0)
    print(f"\n[Exp L] PASS={expL_pass}  (pre-registered: rescue>=5/36 AND fp_hit<2454)")

    # ---------- Experiment E: logits-statistics separability ----------
    def safe_logsumexp(x, axis=1):
        m = x.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    # compute each stat for mis and fp properly
    def stat_values(name, z):
        fgb = z[:, 1:]; bgc = z[:, 0]
        if name == "energy_fg":
            return -safe_logsumexp(fgb)
        if name == "energy_all":
            return -safe_logsumexp(z)
        if name == "logit_margin_novel":
            return fgb[:, novel_idx].max(1) - bgc
        if name == "max_fg_logit":
            return fgb.max(1)
        if name == "entropy":
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            return -(p * np.log(p + 1e-12)).sum(1)
        if name == "max_fg_prob":
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            return p[:, 1:].max(1)
        raise KeyError(name)

    aucs = {}
    for name in ["energy_fg", "energy_all", "logit_margin_novel", "max_fg_logit", "entropy", "max_fg_prob"]:
        s_pos = stat_values(name, mis); s_neg = stat_values(name, fp)
        a = auroc(s_pos, s_neg); aucs[name] = a
        print(f"[Exp E] AUROC {name:22s} = {a:.4f}")
    expE_pass = max(aucs.values()) >= 0.7
    print(f"[Exp E] PASS={expE_pass}  (pre-registered: max AUROC >= 0.7)")

    # ---------- Conditional-domain control: FP pool restricted to pred_bg proposals ----------
    # The 36 misaligned positives are all argmax=bg; the operational task is
    # "among proposals ALREADY judged background, separate true targets from FPs".
    fp_bg = fp[fp.argmax(1) == 0]
    aucs_bg = {}
    for name in ["energy_fg", "energy_all", "logit_margin_novel", "max_fg_logit", "entropy", "max_fg_prob"]:
        s_pos = stat_values(name, mis); s_neg = stat_values(name, fp_bg)
        aucs_bg[name] = auroc(s_pos, s_neg)
        print(f"[Exp E|bg-domain] AUROC {name:22s} = {aucs_bg[name]:.4f}  (fp_bg n={len(fp_bg)})")

    # ---------- Rescue/FP frontier for the best statistic (same shape as F2 grid) ----------
    # rule: logit_margin_novel > m0 -> flip to novel
    s_pos_m = stat_values("logit_margin_novel", mis)
    s_fp_m = stat_values("logit_margin_novel", fp)
    frontier = []
    for m0 in [-12, -11, -10, -9, -8, -7, -6, -5, -4]:
        rescue = int((s_pos_m > m0).sum())
        fp_hit = int((s_fp_m > m0).sum())
        frontier.append({"m0": m0, "rescue": rescue, "fp_hit": fp_hit, "fp_rate": fp_hit / len(fp)})
        print(f"[frontier] margin>{m0:5.1f} | rescue {rescue:2d}/36 | fp_hit {fp_hit:6d} ({fp_hit/len(fp)*100:.2f}%)")
    frontierL_pass = any(r["rescue"] >= 5 and r["fp_hit"] < 2454 for r in frontier)
    print(f"[frontier] PASS={frontierL_pass}  (rescue>=5/36 AND fp_hit<2454)")

    # ---------- EXPLORATORY (NOT pre-registered): three-signal joint rule ----------
    # rule: logit_margin > m0 AND clip_margin > m1 AND rmag < r0
    # baseline for comparison: F2 two-signal rule best point = 5/36 @ 2454 FP
    lm_pos = stat_values("logit_margin_novel", mis); lm_fp = stat_values("logit_margin_novel", fp)
    cm_pos, rg_pos = mis_aux[:, 0], mis_aux[:, 1]
    cm_fp, rg_fp = fp_aux[:, 0], fp_aux[:, 1]
    joint = []
    for m0 in [-6, -5.5, -5, -4.5, -4, -3.5, -3]:
        for m1 in [0.0, 0.02]:
            for r0 in [1.2, 1.5, 2.0, 3.0]:
                rescue = int(((lm_pos > m0) & (cm_pos > m1) & (rg_pos < r0)).sum())
                fp_hit = int(((lm_fp > m0) & (cm_fp > m1) & (rg_fp < r0)).sum())
                joint.append({"m0": m0, "m1": m1, "r0": r0, "rescue": rescue, "fp_hit": fp_hit,
                              "fp_rate": fp_hit / len(fp)})
    top = sorted(joint, key=lambda r: (r["rescue"], -r["fp_hit"]), reverse=True)[:12]
    print("\n[JOINT exploratory] top rules by (rescue desc, fp asc):")
    for r in top:
        print(f"  lm>{r['m0']:5.1f} & cm>{r['m1']:.2f} & rmag<{r['r0']:.1f} | rescue {r['rescue']:2d}/36 | fp {r['fp_hit']:6d}")
    joint_pass = any(r["rescue"] >= 5 and r["fp_hit"] < 2454 for r in joint)
    print(f"[JOINT exploratory] beats F2 point (>=5/36 @ <2454 FP): {joint_pass}")

    out = {
        "pre_registered": {
            "expL": "rescue>=5/36 AND fp_hit<2454 (vs F2 point 5/36@2454)",
            "expE": "max AUROC(pos=36 misaligned, neg=FP pool) >= 0.7",
            "frozen": "2026-11-23 night, before execution (master archive entry)",
        },
        "n_matched": n_matched, "n_misaligned": len(mis), "fp_pool_n": len(fp),
        "mis_logit_margin": {"mean": float(lm_margin.mean()), "p50": float(np.percentile(lm_margin, 50)),
                             "p25": float(np.percentile(lm_margin, 25)), "p75": float(np.percentile(lm_margin, 75))},
        "grid": rows, "expL_pass": bool(expL_pass),
        "aucs": aucs, "expE_pass": bool(expE_pass),
        "aucs_bg_domain": aucs_bg, "fp_bg_n": int((fp.argmax(1) == 0).sum()),
        "margin_frontier": frontier, "margin_frontier_pass": bool(frontierL_pass),
        "joint_exploratory_top": top, "joint_exploratory_pass": bool(joint_pass),
        "mis_meta_first5": mis_meta[:5],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
