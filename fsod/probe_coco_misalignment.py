"""COCO 跨数据集机制验证：语义-定位反相关是否在 COCO+全监督检测器上复现。

VOC 发现（probe_misalignment.py）：
  small novel corr(CLIP相似度, IoU) = -0.204（语义证据与定位质量反相关）
  72% small novel proposal "IoU>0.5 却判背景"
本脚本在 COCO（完全不同的数据集 + 80类全监督检测器，无 few-shot）上重复：
  按 GT 框尺度分组，统计 corr(nsim, IoU)、corr(pfg, IoU)、"定位好却判背景"率。
  若 small 上同样出现 corr(nsim,IoU)<0 且显著低于中大型 => 机制跨数据集普适。

不训练、不改模型，用 coco_forward_cache.py 的缓存。
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch


def corr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 8 or a.std() < 1e-9 or b.std() < 1e-9:
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


# COCO 尺度定义（COCO 标准）
def scale_of(area):
    if area < 32 ** 2:
        return "small"
    if area < 96 ** 2:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/coco_cache.pt")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--out", default="data/coco_misalignment.json")
    args = ap.parse_args()

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]
    rec = defaultdict(list)

    for ci, cache in enumerate(caches):
        props = cache["proposals"][0].float()
        s = sims[ci][0].float()               # (N, C_proto)
        logits = cache["logits"]              # (N, 91)
        oh, ow = cache["orig_sizes"][0]
        rh, rw = cache["image_sizes"][0]
        sx, sy = ow / rw, oh / rh
        props_orig = props * torch.tensor([sx, sy, sx, sy])
        gt = gts[ci]
        gb = np.asarray(gt["boxes"], np.float32).reshape(-1, 4)
        gl = np.asarray(gt["labels"], np.int64)
        if len(gb) == 0 or props_orig.shape[0] == 0:
            continue
        prob = torch.softmax(logits.float(), dim=1)
        pfg = 1 - prob[:, 0]
        pred_bg = logits.argmax(1) == 0
        # CLIP 相似度：用全部原型的最大（语义证据强度）
        nsim = s.max(1).values
        ious = iou_xyxy(props_orig, torch.from_numpy(gb))
        best_i, best_j = ious.max(0)
        best_i = best_i.numpy(); best_j = best_j.numpy()
        for j in range(len(gb)):
            if float(best_i[j]) < args.iou:
                continue
            pi = int(best_j[j])
            x1, y1, x2, y2 = gb[j]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            sc = scale_of(area)
            # 该 GT 类在 CLIP 原型里的相似度（若该类有原型）
            lab = int(gl[j])
            proto_labs = blob["proto_labs"]
            if lab in proto_labs:
                li = proto_labs.index(lab)
                cls_sim = float(s[pi, li])
            else:
                cls_sim = float("nan")
            rec[sc].append({
                "iou": float(best_i[j]), "pfg": float(pfg[pi]),
                "pred_bg": int(pred_bg[pi].item()),
                "nsim": float(nsim[pi]), "cls_sim": cls_sim,
            })

    print(f"\n{'scale':8s} {'n':>6s} | {'IoU':>6s} {'pfg':>6s} {'bg%':>6s} "
          f"| {'corr(pfg,IoU)':>13s} {'corr(nsim,IoU)':>14s} {'corr(cls,IoU)':>13s} | {'bg&IoU>.5':>10s}")
    out = {}
    for sc in ["small", "medium", "large"]:
        rs = rec[sc]
        if len(rs) < 8:
            continue
        ious = [r["iou"] for r in rs]
        pfgs = [r["pfg"] for r in rs]
        nsims = [r["nsim"] for r in rs]
        clss = [r["cls_sim"] for r in rs if not np.isnan(r["cls_sim"])]
        cls_iou = [r["iou"] for r in rs if not np.isnan(r["cls_sim"])]
        bg = [r["pred_bg"] for r in rs]
        bg_good = [r for r in rs if r["pred_bg"] and r["iou"] > 0.5]
        m = {"n": len(rs), "iou": float(np.mean(ious)), "pfg": float(np.mean(pfgs)),
             "bg_rate": float(np.mean(bg)),
             "corr_pfg_iou": corr(pfgs, ious),
             "corr_nsim_iou": corr(nsims, ious),
             "corr_cls_iou": corr(clss, cls_iou),
             "bg_and_goodloc_rate": len(bg_good) / max(1, len(rs))}
        out[sc] = m
        print(f"{sc:8s} {len(rs):6d} | {m['iou']:6.3f} {m['pfg']:6.3f} {m['bg_rate']*100:6.1f} "
              f"| {m['corr_pfg_iou']:13.3f} {m['corr_nsim_iou']:14.3f} {m['corr_cls_iou']:13.3f} "
              f"| {m['bg_and_goodloc_rate']*100:10.1f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
