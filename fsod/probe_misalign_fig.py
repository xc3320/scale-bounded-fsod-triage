"""论文3 缺口清单#2：语义-定位错位实例可视化（纯CPU）。

从 clip_fusion_cache_full.pt 复现 bg&goodloc small novel 实例的匹配，
选出跨类别代表性样本，在原图上画 GT 框(绿) + 匹配proposal框(红)，
放大裁剪并标注（类别/IoU/模型判背景），输出单图 + 拼图。
写作用途：第3章现象证据图。
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fsod import voc, engine
from fsod.data import VOCDetDataset  # noqa: F401  (保持与缓存生成同一 import 路径)

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# ft_map 把类名重映射为连续索引，gts 里的 label 是 ft_map 索引；
# 类名必须用逆映射还原，不能用 canonical VOC 顺序查表。
def make_cls_name(sp):
    inv = {v: k for k, v in sp["ft_map"].items()}
    return lambda label: inv[int(label)]


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
    ap.add_argument("--gate", default="data/probe_fpn_gate.json")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--out-dir", default="figures/misalign_examples")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_set = set(novel_labs)

    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    cls_name = make_cls_name(sp)
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]
    gate = json.load(open(args.gate, encoding="utf-8"))

    # 复现匹配，收集 (img, pi, gt_j, gt_box, cls_name, iou)
    found = []
    gi = 0
    for ci, cache in enumerate(caches):
        for bi in range(len(cache["proposals"])):
            img = gi; gi += 1
            props = cache["proposals"][bi].float()
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
                if float(col[pi]) <= 0.5:
                    continue
                # 预测=背景（与 gate json 一致的条件：argmax==0）
                # gate json 的 targets_detail 是权威来源，这里只取坐标；
                # 校验：img/pi 是否在其中
                match = any(t["img"] == img and t["pi"] == pi for t in gate["targets_detail"])
                if not match:
                    continue
                found.append({
                    "img": img, "pi": pi, "gt_j": j, "gt_box": [float(v) for v in gb2[j]],
                    "prop_box": [float(v) for v in props_orig[pi]],
                    "cls": cls_name(gl2[j]), "iou": float(col[pi]),
                    "file": te_samples[img]["file"],
                })
    print(f"verified bg&goodloc instances with coords: {len(found)} (gate n={gate['n_bg_good']})")

    # 选代表：每类最多2个，按 IoU 降序，跨图片
    by_cls = {}
    for f in sorted(found, key=lambda d: -d["iou"]):
        by_cls.setdefault(f["cls"], []).append(f)
    picks = []
    for cls, lst in by_cls.items():
        picks.extend(lst[:2])
    picks = sorted(picks, key=lambda d: -d["iou"])[:args.n_examples]
    print(f"selected {len(picks)} examples: {[(p['cls'], p['iou']) for p in picks]}")

    os.makedirs(args.out_dir, exist_ok=True)
    jpg_dir = os.path.join(here, "datasets", "VOCdevkit", "VOC2007", "JPEGImages")
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    crops = []
    for k, p in enumerate(picks):
        im = Image.open(os.path.join(jpg_dir, p["file"])).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle(p["gt_box"], outline=(0, 200, 0), width=3)
        d.rectangle(p["prop_box"], outline=(230, 30, 30), width=3)
        x1, y1, x2, y2 = p["gt_box"]
        # 放大裁剪：以 GT 为中心，边长 = 4*max(h,w)
        bw, bh = x2 - x1, y2 - y1
        half = max(bw, bh) * 2.0
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box = [max(0, cx - half), max(0, cy - half),
               min(im.width, cx + half), min(im.height, cy + half)]
        crop = im.crop([int(v) for v in box])
        # 统一裁剪尺寸便于拼图；文字画在缩放后的固定位置（带底衬，不再裁边）
        crop = crop.resize((440, 440))
        d2 = ImageDraw.Draw(crop)
        try:
            font2 = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font2 = ImageFont.load_default()
        lines = [f"{p['cls']}  IoU={p['iou']:.2f}", "pred: background"]
        # 绿框/红框映射到裁剪坐标系，只重画细线提升可见度
        ox, oy = box[0], box[1]
        sc = 440.0 / (box[2] - box[0])
        gx1, gy1 = (p["gt_box"][0] - ox) * sc, (p["gt_box"][1] - oy) * sc
        gx2, gy2 = (p["gt_box"][2] - ox) * sc, (p["gt_box"][3] - oy) * sc
        px1, py1 = (p["prop_box"][0] - ox) * sc, (p["prop_box"][1] - oy) * sc
        px2, py2 = (p["prop_box"][2] - ox) * sc, (p["prop_box"][3] - oy) * sc
        d2.rectangle([gx1, gy1, gx2, gy2], outline=(0, 220, 0), width=2)
        d2.rectangle([px1, py1, px2, py2], outline=(255, 60, 60), width=2)
        for li, txt in enumerate(lines):
            ty = 6 + li * 20
            tw = d2.textlength(txt, font=font2)
            d2.rectangle([4, ty - 2, 4 + tw + 8, ty + 18], fill=(0, 0, 0))
            d2.text((8, ty), txt, fill=(255, 255, 100), font=font2)
        out_p = os.path.join(args.out_dir, f"misalign_{k:02d}_{p['cls']}_iou{p['iou']:.2f}.png")
        crop.save(out_p)
        crops.append(crop)
        print(f"saved {out_p}")

    # 2x3 拼图
    if crops:
        cols, rows = 3, (len(crops) + 2) // 3
        montage = Image.new("RGB", (cols * 440 + (cols + 1) * 8, rows * 440 + (rows + 1) * 8), (255, 255, 255))
        for i, c in enumerate(crops):
            r, col = divmod(i, cols)
            montage.paste(c, (8 + col * (440 + 8), 8 + r * (440 + 8)))
        out_m = os.path.join(args.out_dir, "misalign_montage.png")
        montage.save(out_m)
        print(f"saved montage {out_m}")


if __name__ == "__main__":
    main()
