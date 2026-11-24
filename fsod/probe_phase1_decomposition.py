"""论文4 Phase 1：判背景率两层分解模型的"可预测性"检验（秒级缓存计算）。

【预注册判据（执行前冻结，见论文4立项书 §5 Phase 1）】
  分解模型 MAE < 8pp  且  优于朴素基线（G=0，即 novel_pred = base_rate）。
  通过 → "可预测"成立（现象升级为定律），推进 Phase 2。
  失败 → 模型降级为"描述性分解"，创新点改用分诊闭环/双瓶颈天花板。

【科学依据】
  错位率分解：P(bg | scale s, group g) = F(s) ⊕ G(g)
    F(s) = 尺度普遍效应（base/novel 共有，随 s 减小单调上升）——用 base 判背景率估计
    G(s) = novel 相对 base 的额外惩罚（少样本偏置）——从源 split 估计
  可预测性 = 用源 split 估的 G(s)，叠加目标 split 自身的 F(s)，
             能否预测目标 split（未见）的 novel 各尺度判背景率。
  留一 split 交叉验证（LOSO）：三个 split 轮流当源，测其余两个，报告 MAE 均值±std。

【判决口径】
  缓存 logits 是 proposal 的（600/图），非 GT 框。用 IoU 匹配把 proposal 判决映射到 GT：
  对每个 GT 框取最佳匹配 proposal（IoU 最大），若 IoU≥阈值，
  判背景 = (argmax(logits)==0)。GT 按原始面积分箱 small<1024/medium<9216/large。
  主口径 IoU≥0.5（论文3 "定位好却判背景"错位定义）；IoU≥0.3 作稳健对照。

【档位】决定性测量（LOSO 交叉验证，规则无调参；阈值 0.5/0.3 双口径为预设稳健性检查）。
"""
import argparse
import gc
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


SCALES = ("small", "medium", "large")
THRS = (0.5, 0.3)


def collect_split(cache_path, ann, tr, te, split):
    """返回 {thr: {group: {scale: {'n','bg','correct'}}}}"""
    sp = voc.build_split(ann, tr, te, split)
    ft_map = sp["ft_map"]
    novel_labs = set(ft_map[c] for c in sp["novel"])

    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]

    cnt = {thr: {g: {s: {"n": 0, "bg": 0, "correct": 0} for s in SCALES}
                 for g in ("base", "novel")} for thr in THRS}
    gi = 0
    for cache in caches:
        props_b = cache["proposals"]
        logits = cache["logits"]
        sizes = [p.shape[0] for p in props_b]
        lg_chunks, off = [], 0
        for ns in sizes:
            lg_chunks.append(logits[off:off + ns]); off += ns
        for bi in range(len(props_b)):
            props = props_b[bi].float(); lg = lg_chunks[bi]
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[gi]; gi += 1
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]
            if len(gb2) == 0 or props_orig.shape[0] == 0:
                continue
            ious = iou_xyxy(props_orig, torch.from_numpy(gb2))  # (n_prop, n_gt)
            pred = lg.float().argmax(1)
            for j in range(len(gb2)):
                x1, y1, x2, y2 = gb2[j]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                scale = next((s for s in SCALES if engine.AREA_BINS[s][0] <= area < engine.AREA_BINS[s][1]), "large")
                group = "novel" if int(gl2[j]) in novel_labs else "base"
                col = ious[:, j]
                best_iou = float(col.max())
                pi = int(col.argmax())
                p = int(pred[pi])
                for thr in THRS:
                    if best_iou >= thr:
                        d = cnt[thr][group][scale]
                        d["n"] += 1
                        d["bg"] += int(p == 0)
                        d["correct"] += int(p == int(gl2[j]))
    del blob, caches, gts
    gc.collect()
    # 转成比率
    out = {}
    for thr in THRS:
        out[thr] = {}
        for g in ("base", "novel"):
            out[thr][g] = {}
            for s in SCALES:
                d = cnt[thr][g][s]
                out[thr][g][s] = {
                    "n": d["n"],
                    "bg_rate": round(d["bg"] / d["n"], 4) if d["n"] else None,
                    "top1": round(d["correct"] / d["n"], 4) if d["n"] else None,
                }
    return out


def rate_table(raw, thr):
    """提取 {split: {group: [r_small,r_med,r_large]}}"""
    return {sp: {g: [raw[sp][thr][g][s]["bg_rate"] for s in SCALES]
                 for g in ("base", "novel")} for sp in raw}


def loso_eval(tbl):
    """留一 split 外推。返回逐 (源,目标) 对的 MAE 与朴素基线，及均值。"""
    splits = sorted(tbl.keys())
    rows = []
    for src in splits:
        G = [tbl[src]["novel"][i] - tbl[src]["base"][i] for i in range(3)]  # 源 split 的偏置 G(s)
        for tgt in splits:
            if tgt == src:
                continue
            actual = tbl[tgt]["novel"]
            base = tbl[tgt]["base"]
            pred = [float(np.clip(base[i] + G[i], 0, 1)) for i in range(3)]   # 分解模型
            naive = base[:]                                                   # 朴素基线 G=0
            mae = float(np.mean([abs(pred[i] - actual[i]) for i in range(3)]))
            mae_naive = float(np.mean([abs(naive[i] - actual[i]) for i in range(3)]))
            rows.append({"src": src, "tgt": tgt, "G": [round(g, 4) for g in G],
                         "actual": [round(x, 4) for x in actual],
                         "base": [round(x, 4) for x in base],
                         "pred": [round(x, 4) for x in pred],
                         "mae_pp": round(mae * 100, 2),
                         "mae_naive_pp": round(mae_naive * 100, 2)})
    maes = [r["mae_pp"] for r in rows]
    naives = [r["mae_naive_pp"] for r in rows]
    return {"rows": rows,
            "mae_mean_pp": round(float(np.mean(maes)), 2),
            "mae_std_pp": round(float(np.std(maes)), 2),
            "mae_naive_mean_pp": round(float(np.mean(naives)), 2),
            "mae_naive_std_pp": round(float(np.std(naives)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/phase1_decomposition.json")
    ap.add_argument("--max-imgs", type=int, default=0, help="0=全量；>0 截断每 split 图数（调试用）")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))

    cache_map = {
        1: "data/clip_fusion_cache_full.pt",
        2: "data/clip_fusion_cache_full_s2.pt",
        3: "data/clip_fusion_cache_full_s3.pt",
    }
    raw = {}
    for sp, path in cache_map.items():
        print(f"[collect] split {sp} <- {path}", flush=True)
        raw[sp] = collect_split(path, ann, tr, te, sp)

    print("\n========== 判背景率表（主口径 IoU≥0.5）==========")
    for sp in sorted(raw):
        for g in ("base", "novel"):
            cells = raw[sp][THRS[0]][g]
            print(f"  split{sp} {g:5s}: " + "  ".join(
                f"{s}={cells[s]['bg_rate']:.3f}(n={cells[s]['n']})" for s in SCALES), flush=True)

    results = {"raw": raw, "thresholds": list(THRS), "scales": list(SCALES)}
    verdict = {}
    for thr in THRS:
        tbl = rate_table(raw, thr)
        ev = loso_eval(tbl)
        results[f"loso_iou{thr}"] = ev
        print(f"\n========== LOSO 外推（IoU≥{thr}）==========")
        for r in ev["rows"]:
            print(f"  src{r['src']}->tgt{r['tgt']}: G={r['G']}  pred={r['pred']}  "
                  f"actual={r['actual']}  MAE={r['mae_pp']:.2f}pp (naive {r['mae_naive_pp']:.2f}pp)", flush=True)
        print(f"  >> 均值: 分解 {ev['mae_mean_pp']:.2f}±{ev['mae_std_pp']:.2f}pp | "
              f"朴素 {ev['mae_naive_mean_pp']:.2f}±{ev['mae_naive_std_pp']:.2f}pp", flush=True)
        passed = (ev["mae_mean_pp"] < 8.0) and (ev["mae_mean_pp"] < ev["mae_naive_mean_pp"])
        verdict[f"iou{thr}"] = {"pass": bool(passed),
                                 "criterion": "MAE<8pp AND better than naive(G=0)"}
        print(f"  >> 判定: {'PASS' if passed else 'FAIL'} (预注册: MAE<8pp 且优于朴素基线)", flush=True)

    results["verdict"] = verdict
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
