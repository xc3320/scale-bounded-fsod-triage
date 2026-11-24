"""论文3 缺口清单#5：语义盲区（F1, margin≈0）是否跨 CLIP 塔稳健（B/32 → B/16）。

关键认识：错位现象（框准却判背景）由检测器决定（logits/proposals 与 CLIP 塔无关），
换塔真正改变的是语义证据。因此只需对 50 个匹配实例的 proposal 裁剪块
（B/32 已定位）用 B/16 重新编码语义分数，比较 margin 分布：
  - 若 B/16 下 margin 仍 ≈0 → 盲区是"跨 CLIP 塔稳健"的结构性现象（强结论）
  - 若 B/16 下 margin 显著可分 → 盲区是 B/32 特有（措辞降级为塔相关）
支持集原型用与 eval_clip_fusion 相同的流水线（5-shot, seed=0）。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from fsod import voc
from fsod.eval_clip_fusion import crop_tensor, encode_crops, build_fewshot_set
from fsod.data import VOCDetDataset  # noqa: F401


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
    ap.add_argument("--clip16", default="torch_home/clip/ViT-B-16.pt")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--out", default="data/probe_b16_margin.json")
    args = ap.parse_args()

    here = os.getcwd()
    dev = torch.device("cuda")
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    nidx = [c - 1 for c in novel_labs]
    bidx = [i for i in range(20) if i not in nidx]
    inv_ft = {v: k for k, v in ft_map.items()}

    # ---- 1) 从 B/32 缓存重建匹配实例（与 probe_misalign_fig 同逻辑）----
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]
    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    matched = []
    gi = 0
    for ci, cache in enumerate(caches):
        for bi in range(len(cache["proposals"])):
            img = gi; gi += 1
            props = cache["proposals"][bi].float()
            lg = cache["logits"]
            off = sum(p.shape[0] for p in cache["proposals"][:bi])
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[img]
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]
            if len(gb2) == 0:
                continue
            ious = iou_xyxy(props_orig, torch.from_numpy(gb2))
            best_iou_prop = ious.max(1).values
            from fsod import engine
            pw = (props_orig[:, 2] - props_orig[:, 0]).clamp(min=0)
            ph = (props_orig[:, 3] - props_orig[:, 1]).clamp(min=0)
            small_mask = (pw * ph) < engine.AREA_BINS["small"][1]
            s32 = sims[ci][bi].float()
            margin32_all = s32[:, nidx].max(1).values - s32[:, bidx].max(1).values
            pred_bg_all = lg[off:off + props.shape[0]].float().argmax(1) == 0
            for j in range(len(gb2)):
                if gl2[j] not in novel_labs:
                    continue
                x1, y1, x2, y2 = gb2[j]
                if max(0, x2 - x1) * max(0, y2 - y1) >= engine.AREA_BINS["small"][1]:
                    continue
                col = ious[:, j]
                if float(col.max()) < 0.3:
                    continue
                pi = int(col.argmax().item())
                if float(col[pi]) <= 0.3:
                    continue
                matched.append({
                    "img": img, "pi": pi, "file": te_samples[img]["file"],
                    "prop_box": [float(v) for v in props_orig[pi]],
                    "gtcls": int(gl2[j]), "cls": inv_ft[int(gl2[j])],
                    "iou": float(col[pi]), "margin_b32": float(margin32_all[pi]),
                    "pred_bg": bool(pred_bg_all[pi]), "goodloc": float(col[pi]) > 0.5,
                })
    n = len(matched)
    bg_good = [m for m in matched if m["pred_bg"] and m["goodloc"]]
    print(f"matched={n}  bg&goodloc={len(bg_good)} (B/32 基准=36)")

    # ---- 2) B/16 原型（与主评测同流水线：5-shot seed=0 支持集）----
    import clip as clip_lib
    clip16_path = os.path.join(here, args.clip16) if not os.path.isabs(args.clip16) else args.clip16
    clip16, _ = clip_lib.load(clip16_path, device=dev)
    clip16.eval()
    ft_samples, _ = build_fewshot_set(ann, tr, ft_map, sp["base"], sp["novel"], args.shot, seed=0)
    sup_crops = [crop_tensor(s["file"], bx) for s in ft_samples for bx in s["boxes"]]
    sup_labs = [lb for s in ft_samples for lb in s["labels"]]
    sup_feat = encode_crops(clip16, sup_crops, dev)
    protos = []
    for lb in range(1, 21):
        m = torch.tensor([l == lb for l in sup_labs])
        pf = sup_feat[m].mean(0) if m.sum() else torch.zeros(512)
        protos.append(F.normalize(pf.unsqueeze(0), dim=-1).squeeze(0))
    proto16 = torch.stack(protos).to(dev)

    # ---- 3) 对匹配实例的 proposal 裁剪块编码 B/16 语义 ----
    crops = [crop_tensor(m["file"], m["prop_box"]) for m in matched]
    feat16 = encode_crops(clip16, crops, dev)          # (n, 512)
    sim16 = (feat16 @ proto16.T.cpu()).numpy()
    for k, m in enumerate(matched):
        s = sim16[k]
        m["margin_b16"] = float(s[nidx].max() - s[bidx].max())

    def stats(recs, key):
        v = np.array([r[key] for r in recs])
        return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                "p75": float(np.percentile(v, 75)), "max": float(v.max()),
                "n_pos": int((v > 0).sum()), "n_pos05": int((v > 0.05).sum())}

    out = {
        "matched": {k: stats(matched, k) for k in ["margin_b32", "margin_b16"]},
        "bg_good": {k: stats(bg_good, k) for k in ["margin_b32", "margin_b16"]},
        "corr_b32_b16_matched": float(np.corrcoef(
            [m["margin_b32"] for m in matched], [m["margin_b16"] for m in matched])[0, 1]),
        "records": matched,
    }
    print("\nmatched (n=%d): margin_b32 mean %.3f / median %.3f | margin_b16 mean %.3f / median %.3f" % (
        n, out["matched"]["margin_b32"]["mean"], out["matched"]["margin_b32"]["median"],
        out["matched"]["margin_b16"]["mean"], out["matched"]["margin_b16"]["median"]))
    print("bg_good  (n=%d): margin_b32 mean %.3f / median %.3f | margin_b16 mean %.3f / median %.3f" % (
        len(bg_good), out["bg_good"]["margin_b32"]["mean"], out["bg_good"]["margin_b32"]["median"],
        out["bg_good"]["margin_b16"]["mean"], out["bg_good"]["margin_b16"]["median"]))
    print(f"corr(margin_b32, margin_b16) = {out['corr_b32_b16_matched']:.3f}")
    print(f"bg_good 中 margin_b16>0.05 的实例数: {out['bg_good']['margin_b16']['n_pos05']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
