"""①′ prototype probe: size-stratified effect of dual-evidence gating.

Question: does adding Q_loc (reg-delta reliability weighting) to the semantic
gate rescue well-localized small novel objects that the baseline judges as
background (the 72% injustice), without breaking base?

For each mode (baseline / gate0.0 / rel{inv,sig}{tau} / loc), recompute the
modified logits from the cached forward, then for every GT object matched at
IoU>=0.3 record whether its best-matching proposal is judged background.
Report per (scale, kind): bg_rate, bg&IoU>0.5 rate, and corr(nsim,IoU).

CPU-only, uses cache. Mirrors eval_clip_fusion mode math exactly.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from fsod import voc, engine


def corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def iou_xyxy(a, b):
    ix1 = torch.max(a[:, None, 0], b[None, :, 0]); iy1 = torch.max(a[:, None, 1], b[None, :, 1])
    ix2 = torch.min(a[:, None, 2], b[None, :, 2]); iy2 = torch.min(a[:, None, 3], b[None, :, 3])
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter).clamp(min=1e-6)


def apply_mode(lg, s, reg_n, n_all, novel_idx, mode, lam):
    """Replicate/extend eval_clip_fusion gating math. lg modified in place.

    Modes:
      none                    baseline
      gate{thr}               semantic-only boost (paper-2 form)
      rel{inv|sig}{tau}       semantic boost * loc reliability weight
      bg                      plain bg suppression: bg -= lam * nsim
      bgr{tau}                bg suppression gated by loc reliability: bg -= lam*nsim*w
      dual{tau}               gate0.0 novel boost + bgr bg suppression
      dualb{tau}              gate0.0 novel boost + plain bg suppression
    """
    base_idx = [i for i in range(n_all) if i not in novel_idx]
    if mode == "none" or lam == 0:
        return lg
    nsim = s[:, novel_idx].max(1).values
    bsim = s[:, base_idx].max(1).values
    sem_gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
    rmag = reg_n

    def w_inv(tau):
        return 1.0 / (1.0 + rmag / max(tau, 1e-4))

    if mode.startswith("gate"):
        thr = float(mode[4:]) if len(mode) > 4 else 0.05
        gate = torch.clamp((nsim - bsim - thr) / 0.1, 0, 1)
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
    elif mode.startswith("rel"):
        spec = mode[3:]
        kind, rest = ("inv", spec[3:]) if spec.startswith("inv") else ("sig", spec[3:])
        tau = float(rest)
        w = w_inv(tau) if kind == "inv" else torch.sigmoid((tau - rmag) / 0.3)
        gate = sem_gate * w
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
    elif mode.startswith("loc"):
        parts = mode[3:].split("_")
        thr = float(parts[0])
        loc_gate = torch.clamp((thr - rmag) / 1.0 + 0.5, 0, 1)
        gate = sem_gate * loc_gate
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
    elif mode == "bg":
        lg[:, 0] = lg[:, 0] - lam * nsim
    elif mode.startswith("bgr"):
        tau = float(mode[3:])
        lg[:, 0] = lg[:, 0] - lam * nsim * w_inv(tau)
    elif mode.startswith("dualr"):
        spec = mode[5:]
        bgc, tau = [float(x) for x in spec.split("_")]
        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
        w = 1.0 / (1.0 + rmag / max(tau, 1e-4))
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
        lg[:, 0] = lg[:, 0] - bgc * nsim * w
    elif mode.startswith("dualb"):
        bgc = float(mode[5:]) if len(mode) > 5 else lam
        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
        lg[:, 0] = lg[:, 0] - bgc * nsim
    elif mode.startswith("dual"):
        tau = float(mode[4:])
        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
        for li in novel_idx:
            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
        lg[:, 0] = lg[:, 0] - lam * nsim * w_inv(tau)
    else:
        raise ValueError(mode)
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--modes", default="none,gate0.0,bg,bgr0.3,bgr0.5,bgr1.0,dual0.5,dual1.0,dualb")
    ap.add_argument("--out", default="data/probe_dualgate_size.json")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_idx = [c - 1 for c in novel_labs]
    n_all = 20

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]

    modes = args.modes.split(",")
    rec = {m: defaultdict(list) for m in modes}

    gi = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]; s_b = sims[ci]
        logits = cache["logits"]; regs = cache["reg"]
        sizes = [p.shape[0] for p in props_b]
        lg_chunks, rg_chunks, off = [], [], 0
        for ns in sizes:
            lg_chunks.append(logits[off:off + ns]); rg_chunks.append(regs[off:off + ns]); off += ns
        for bi in range(len(props_b)):
            props = props_b[bi].float(); s = s_b[bi].float(); lg0 = lg_chunks[bi]; rg = rg_chunks[bi]
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
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
            reg_n = rg.float().reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)
            ious = iou_xyxy(props_orig, torch.from_numpy(gb))
            best_i, best_j = ious.max(0)
            best_i = best_i.numpy(); best_j = best_j.numpy()
            for j in range(len(gb)):
                if float(best_i[j]) < args.iou:
                    continue
                pi = int(best_j[j]); lab = int(gl[j])
                kind = "novel" if lab in novel_labs else "base"
                x1, y1, x2, y2 = gb[j]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                sc = "small" if area < engine.AREA_BINS["small"][1] else (
                     "medium" if area < engine.AREA_BINS["medium"][1] else "large")
                for m in modes:
                    lg = lg0.clone()
                    apply_mode(lg, s, reg_n, n_all, novel_idx, m, args.lam)
                    am = int(lg[pi].argmax().item())
                    r = {
                        "iou": float(best_i[j]),
                        "nsim": float(s[pi, novel_idx].max().item()),
                        "rmag": float(reg_n[pi].item()),
                        "goodloc": float(best_i[j]) > 0.5,
                        "pred_correct": int(am == lab),
                        "pred_bg": int(am == 0),
                        "pred_novel": int(am in [x + 1 for x in novel_idx]),
                    }
                    rec[m][(sc, kind)].append(r)

    out = {}
    print(f"\nlam={args.lam:g}  IoU>={args.iou}  (novel: 判对% = argmax命中本类；base: 误判novel% = 门控副作用)\n")
    hdr = f"{'mode':14s} {'scale':7s} {'kind':6s} {'n':>5s} | {'判对%':>6s} {'bg%':>6s} {'误判novel%':>10s} {'框准判对%':>9s}"
    print(hdr); print("-" * len(hdr))
    for m in modes:
        for sc in ["small", "medium", "large"]:
            for kind in ["novel", "base"]:
                rs = rec[m][(sc, kind)]
                if len(rs) < 5:
                    continue
                ok = [r["pred_correct"] for r in rs]
                bg = [r["pred_bg"] for r in rs]
                nv = [r["pred_novel"] for r in rs]
                good_rs = [r for r in rs if r["goodloc"]]
                ok_good = sum(r["pred_correct"] for r in good_rs) / max(1, len(good_rs))
                bg_good = sum(r["pred_bg"] for r in good_rs) / max(1, len(good_rs))
                ious = [r["iou"] for r in rs]; nsims = [r["nsim"] for r in rs]
                rmags = [r["rmag"] for r in rs]
                key = f"{m}|{sc}_{kind}"
                out[key] = {"n": len(rs), "correct_rate": float(np.mean(ok)),
                            "bg_rate": float(np.mean(bg)),
                            "novel_mis_rate": float(np.mean(nv)),
                            "correct_goodloc_rate": ok_good,
                            "bg_goodloc_rate": bg_good,
                            "corr_nsim_iou": corr(nsims, ious),
                            "corr_rmag_iou": corr(rmags, ious),
                            "rmag_mean": float(np.mean(rmags))}
                print(f"{m:14s} {sc:7s} {kind:6s} {len(rs):5d} | {np.mean(ok)*100:6.1f} "
                      f"{np.mean(bg)*100:6.1f} {np.mean(nv)*100:10.1f} {bg_good*100:9.1f} "
                      f"corr(rmag,IoU)={out[key]['corr_rmag_iou']:+.3f} rmag={np.mean(rmags):.2f}")

    with open(args.out, "w") as f:
        json.dump({"lam": args.lam, "iou_thr": args.iou, "results": out}, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
