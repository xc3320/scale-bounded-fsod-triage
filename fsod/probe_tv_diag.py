"""TV(p_hat, uniform) diagnostic on detection query sets -> M2 statistic design input.

Uses cached detector logits (fsod.eval_clip_fusion forward cache, VOC split1, 4952 imgs).
Per image: p_hat = class distribution of proposal argmax labels (20 fg classes).

Statistics (candidates for M2):
  tv_unif     TV(p_hat, uniform over 20)
  tv_ds       TV(p_hat, dataset marginal p_bar)
  novel_mass  p_hat mass on novel classes (idx 15..19)
  max_share   max class share

Discrimination targets (AUROC, rank-based):
  T1 gt_novel>=1 vs =0   (quota-appropriate? novel present)
  T2 gt_novel>=2 vs <2
  T3 unknown-injected f=0.1 vs clean  (confound: TV rises for unrelated reasons)

Unknown injection: fraction f of proposals reassigned to a 21st virtual class,
TV over 21 categories (tv_unif21). Reports mean shift + AUROC(clean vs injected).

CPU-only. Output: data/tv_diag.json
"""
import json
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "data", "clip_fusion_cache_full.pt")
OUT = os.path.join(HERE, "data", "tv_diag.json")
N_NOVEL = 5          # novel classes occupy logits cols 16..20 (1-based 16..20? use 0-based 15..19)
N_FG = 20
rng = np.random.default_rng(0)


def auroc(pos, neg):
    """Rank-based AUROC. pos/neg: 1-D arrays."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].mean()
    return float((r_pos - (len(pos) + 1) / 2.0) / len(neg))


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def main():
    blob = torch.load(CACHE, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]
    n_img = blob.get("n_imgs", len(caches))
    print(f"[load] {n_img} images")

    rows = []
    all_counts = np.zeros(N_FG)
    # pass 1: collect per-image predicted class counts
    for c in caches:
        lg = c["logits"].numpy()          # (N,21)
        pred = lg[:, 1:].argmax(1)        # (N,) 0..19
        counts = np.bincount(pred, minlength=N_FG).astype(np.float64)
        all_counts += counts
        rows.append(counts)
    p_bar = all_counts / all_counts.sum()
    uni = np.full(N_FG, 1.0 / N_FG)

    stats = {"tv_unif": [], "tv_ds": [], "novel_mass": [], "max_share": [],
             "gt_novel": [], "n_prop": []}
    for counts, g in zip(rows, gts):
        n = counts.sum()
        if n == 0:
            continue
        p = counts / n
        stats["tv_unif"].append(tv(p, uni))
        stats["tv_ds"].append(tv(p, p_bar))
        stats["novel_mass"].append(p[N_NOVEL:].sum())
        stats["max_share"].append(p.max())
        stats["n_prop"].append(int(n))
        stats["gt_novel"].append(int((np.asarray(g["labels"]) > N_NOVEL).sum()))

    tv_u = np.array(stats["tv_unif"])
    tv_d = np.array(stats["tv_ds"])
    nm = np.array(stats["novel_mass"])
    ms = np.array(stats["max_share"])
    gn = np.array(stats["gt_novel"])

    res = {
        "n_images": len(tv_u),
        "dataset_marginal_top5": {str(i): float(p_bar[i]) for i in np.argsort(-p_bar)[:5]},
        "tv_unif_vs_uniform": {"mean": float(tv_u.mean()), "std": float(tv_u.std()),
                                "q10": float(np.quantile(tv_u, .1)),
                                "q50": float(np.quantile(tv_u, .5)),
                                "q90": float(np.quantile(tv_u, .9))},
        "tv_ds_vs_marginal": {"mean": float(tv_d.mean()), "std": float(tv_d.std()),
                               "q50": float(np.quantile(tv_d, .5)),
                               "q90": float(np.quantile(tv_d, .9))},
        "gt_novel_dist": {str(k): int((gn == k).sum()) for k in range(0, int(gn.max()) + 1)},
        "AUROC": {
            "T1_novel>=1_tv_unif": auroc(tv_u[gn >= 1], tv_u[gn == 0]),
            "T1_novel>=1_tv_ds": auroc(tv_d[gn >= 1], tv_d[gn == 0]),
            "T1_novel>=1_novel_mass": auroc(nm[gn >= 1], nm[gn == 0]),
            "T1_novel>=1_max_share_neg": auroc(-ms[gn >= 1], -ms[gn == 0]),
            "T2_novel>=2_tv_unif": auroc(tv_u[gn >= 2], tv_u[gn < 2]),
            "T2_novel>=2_novel_mass": auroc(nm[gn >= 2], nm[gn < 2]),
        },
    }

    # unknown injection scenario: f=0.05/0.10/0.20 -> TV over 21 cats
    inj = {}
    for f in (0.05, 0.10, 0.20):
        tv21_clean, tv21_inj = [], []
        for counts, g in zip(rows, gts):
            n = int(counts.sum())
            if n == 0:
                continue
            k = int(round(f * n))
            base = np.concatenate([counts, [0.0]]).astype(np.float64)
            inj_c = base.copy()
            inj_c[:N_FG] -= k / N_FG * np.ones(N_FG) * (k > 0)  # subtract proportionally later
            # simpler: move k proposals from random classes to unknown
            src = rng.choice(N_FG, size=k, p=counts / n)
            inj_c = base.copy()
            for s in src:
                inj_c[s] -= 1
            inj_c[N_FG] += k
            p21 = np.clip(inj_c, 0, None) / n
            uni21 = np.full(21, 1.0 / 21)
            tv21_inj.append(tv(p21, uni21))
            tv21_clean.append(tv(base / n, uni21))
        tv21_clean = np.array(tv21_clean); tv21_inj = np.array(tv21_inj)
        inj[f"f={f}"] = {"clean_mean": float(tv21_clean.mean()),
                          "injected_mean": float(tv21_inj.mean()),
                          "AUROC_clean_vs_injected": auroc(tv21_inj, tv21_clean)}
    res["unknown_injection_TV21"] = inj

    with open(OUT, "w", encoding="utf-8") as fjson:
        json.dump(res, fjson, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1)[:2000])
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
