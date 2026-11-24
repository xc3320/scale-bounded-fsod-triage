"""论文4 Phase 3 第一步：COCO 跨数据集机制探针（纯缓存 CPU，秒级）。

【预注册判据 — 执行前冻结，见论文4立项书 §5 Phase 3】
  可比性：COCO = 77 图 + 80类全监督检测器，非 few-shot、无 base/novel。
          只验证机制方向是否跨数据集复现，不比精确数字，不测 novel 专属量。

  命题 A（尺度普遍效应 F(s)）：bg_rate 与 bg_and_goodloc_rate 均随尺度减小单调上升
          (small>medium>large)。PASS = 两者都单调。
  命题 B1（可分性）：正例=matched(IoU>=0.3)&pred_bg&goodloc(IoU>0.5) 的 GT proposal，
          负例=unmatched small FP。6 类 logit 统计量 max AUROC >= 0.7。
  命题 B2（可修复性）：背景 logit 抑制 z'_bg=z_bg-tau，是否存在 tau>0 使
          rescue>=5 且 fp_hit<2454。若不存在 => "可分!=可修复" 在 COCO 复现。

  判定：A、B1、B2(不复现) 方向一致 => 机制跨数据集复现成立；否则如实收窄。

口径与 probe_coco_misalignment.py / probe_newsignal.py 一致：
  - proposal logits 映射到 GT：每个 GT 取 IoU 最大的 proposal；
  - COCO 尺度：small<32^2 / medium<96^2 / large；
  - logits 91 类，0=background，1..90 为 COCO category id。
"""
import argparse
import json

import numpy as np
import torch


def iou_xyxy(a, b):
    ix1 = torch.max(a[:, None, 0], b[None, :, 0]); iy1 = torch.max(a[:, None, 1], b[None, :, 1])
    ix2 = torch.min(a[:, None, 2], b[None, :, 2]); iy2 = torch.min(a[:, None, 3], b[None, :, 3])
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter).clamp(min=1e-6)


def scale_of(area):
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def auroc(pos, neg):
    """Rank-based AUROC (Mann-Whitney). Higher score = more positive-like."""
    x = np.concatenate([np.asarray(pos, np.float64), np.asarray(neg, np.float64)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/coco_cache_full.pt")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default="data/phase3_coco.json")
    args = ap.parse_args()

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]

    # scale-level counters for 命题 A
    scale_cnt = {s: {"n": 0, "bg": 0, "bg_good": 0} for s in ("small", "medium", "large")}
    pos_logits = []   # matched & pred_bg & goodloc, any scale -> for B1/B2 positive
    neg_logits = []   # unmatched small FP -> for B1/B2 negative
    gi = 0
    for cache in caches:
        props = cache["proposals"][0].float()
        logits = cache["logits"]                       # (N, 91)
        oh, ow = cache["orig_sizes"][0]
        rh, rw = cache["image_sizes"][0]
        sx, sy = ow / rw, oh / rh
        props_orig = props * torch.tensor([sx, sy, sx, sy])
        gt = gts[gi]; gi += 1
        gb = np.asarray(gt["boxes"], np.float32).reshape(-1, 4)
        gl = np.asarray(gt["labels"], np.int64)
        if len(gb) == 0 or props_orig.shape[0] == 0:
            continue
        pred = logits.argmax(1)
        pw = (props_orig[:, 2] - props_orig[:, 0]).clamp(min=0)
        ph = (props_orig[:, 3] - props_orig[:, 1]).clamp(min=0)
        small_mask = (pw * ph) < 32 ** 2

        ious = iou_xyxy(props_orig, torch.from_numpy(gb))   # (n_prop, n_gt)
        best_i, best_j = ious.max(0)                        # per GT best proposal
        best_prop_iou = ious.max(1).values                  # per proposal best IoU (for FP)

        # ---- FP pool: unmatched small proposals ----
        unmatched_small = small_mask & (best_prop_iou < args.iou)
        neg_logits.append(logits[unmatched_small].numpy())

        # ---- matched GT -> per-scale stats + positive set ----
        for j in range(len(gb)):
            if float(best_i[j]) < args.iou:
                continue
            pi = int(best_j[j].item())
            x1, y1, x2, y2 = gb[j]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            sc = scale_of(area)
            is_bg = int(pred[pi].item() == 0)
            c = scale_cnt[sc]
            c["n"] += 1
            c["bg"] += is_bg
            goodloc = float(best_i[j]) > 0.5
            if is_bg and goodloc:
                c["bg_good"] += 1
                pos_logits.append(logits[pi].numpy())

    scale_stats = {}
    for s in ("small", "medium", "large"):
        c = scale_cnt[s]
        scale_stats[s] = {
            "n": c["n"],
            "bg_rate": round(c["bg"] / c["n"], 4) if c["n"] else None,
            "bg_goodloc_rate": round(c["bg_good"] / c["n"], 4) if c["n"] else None,
        }

    pos = np.stack(pos_logits) if pos_logits else np.zeros((0, 91))
    neg = np.concatenate(neg_logits) if neg_logits else np.zeros((0, 91))
    print(f"positive (bg&goodloc matched) n={len(pos)}; negative (unmatched small FP) n={len(neg)}")

    # ---------- 命题 A ----------
    bg_rates = [scale_stats[s]["bg_rate"] for s in ("small", "medium", "large")]
    bg_good_rates = [scale_stats[s]["bg_goodloc_rate"] for s in ("small", "medium", "large")]
    A_bg_mono = bg_rates[0] > bg_rates[1] > bg_rates[2]
    A_bggood_mono = bg_good_rates[0] > bg_good_rates[1] > bg_good_rates[2]
    A_pass = bool(A_bg_mono and A_bggood_mono)
    print("\n[命题 A 尺度普遍效应]")
    for s in ("small", "medium", "large"):
        print(f"  {s:7s} n={scale_stats[s]['n']:5d}  bg={scale_stats[s]['bg_rate']*100:5.1f}%  "
              f"bg&goodloc={scale_stats[s]['bg_goodloc_rate']*100:5.1f}%")
    print(f"  bg 单调 small>med>large: {A_bg_mono}; bg&goodloc 单调: {A_bggood_mono}; => A {'PASS' if A_pass else 'FAIL'}")

    # ---------- 命题 B1 可分性 ----------
    def safe_logsumexp(x):
        m = x.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).squeeze(1)

    def stat_values(name, z):
        fgb = z[:, 1:]          # COCO: class 1..90 foreground
        bgc = z[:, 0]
        if name == "energy_fg":
            return -safe_logsumexp(fgb)
        if name == "energy_all":
            return -safe_logsumexp(z)
        if name == "logit_margin":
            return fgb.max(1) - bgc
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
    print("\n[命题 B1 可分性]")
    for name in ["energy_fg", "energy_all", "logit_margin", "max_fg_logit", "entropy", "max_fg_prob"]:
        a = auroc(stat_values(name, pos), stat_values(name, neg))
        aucs[name] = round(a, 4)
        print(f"  AUROC {name:16s} = {a:.4f}")
    b1_max = max(aucs.values())
    B1_pass = bool(b1_max >= 0.7)
    # conditional bg domain: negatives already pred_bg (operational subset)
    fp_bg = neg[neg.argmax(1) == 0] if len(neg) else neg
    aucs_bg = {}
    for name in aucs:
        aucs_bg[name] = round(auroc(stat_values(name, pos), stat_values(name, fp_bg)), 4)
    print(f"  max AUROC = {b1_max:.4f} => B1 {'PASS' if B1_pass else 'FAIL'} (pre-reg >=0.7)")
    print(f"  [bg-domain control] fp_bg n={len(fp_bg)}; max AUROC={max(aucs_bg.values()):.4f}")

    # ---------- 命题 B2 可修复性 ----------
    print("\n[命题 B2 可修复性] bg-logit suppression tau grid")
    fg_pos = pos[:, 1:]; bg_pos = pos[:, 0]
    fg_neg = neg[:, 1:]; bg_neg = neg[:, 0]
    grid = []
    for tau in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
        flip_pos = (fg_pos.max(1) - (bg_pos - tau)) > 0
        rescue = int(flip_pos.sum())
        flip_neg = (fg_neg.max(1) - (bg_neg - tau)) > 0
        fp_hit = int(flip_neg.sum())
        grid.append({"tau": tau, "rescue": rescue, "fp_hit": fp_hit,
                     "fp_rate": round(fp_hit / max(1, len(neg)), 5)})
        print(f"  tau={tau:4d} | rescue {rescue:3d}/{len(pos)} | fp_hit {fp_hit:6d} ({fp_hit/max(1,len(neg))*100:.2f}%)")
    repairable = any(r["tau"] > 0 and r["rescue"] >= 5 and r["fp_hit"] < 2454 for r in grid)
    B2_reproduce = not repairable   # "可分!=可修复"复现 = 修不回来
    print(f"  exists tau>0 with rescue>=5 AND fp_hit<2454: {repairable}")
    print(f"  => 'separability!=repairability' reproduced on COCO: {B2_reproduce}")

    all_consistent = bool(A_pass and B1_pass and B2_reproduce)
    print(f"\n[整体] A={A_pass} B1={B1_pass} B2_reproduce={B2_reproduce} => "
          f"机制跨数据集复现 {'成立' if all_consistent else '不成立（如实收窄）'}")

    out = {
        "pre_registered": {
            "A": "bg_rate and bg_goodloc_rate both monotone small>medium>large",
            "B1": "max AUROC (6 logit stats) >= 0.7",
            "B2": "NO tau>0 with rescue>=5 AND fp_hit<2454 => separability!=repairability reproduced",
            "frozen": "2026-11-23 before execution (立项书 §5 Phase 3)",
        },
        "dataset": "COCO 77-val, 80-class fully-supervised detector, non-fewshot (no base/novel)",
        "scale_stats": scale_stats,
        "A": {"bg_mono": A_bg_mono, "bg_goodloc_mono": A_bggood_mono, "pass": A_pass},
        "n_positive": len(pos), "n_negative": len(neg),
        "B1": {"aucs": aucs, "aucs_bg_domain": aucs_bg, "fp_bg_n": len(fp_bg),
               "max_auc": b1_max, "pass": B1_pass},
        "B2": {"grid": grid, "repairable": repairable, "reproduce_inseparable": B2_reproduce},
        "all_consistent": all_consistent,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
