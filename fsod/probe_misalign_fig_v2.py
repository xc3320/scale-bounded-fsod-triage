"""论文3 m5：语义-定位错位图扩样 + 正确检出负对照行。

在 v1（probe_misalign_fig.py）基础上：
1) --n-misalign 控制错位实例（bg&goodloc）数量（v1 为 6）。
2) 新增正确检出负对照：small novel 匹配实例中预测类==GT类 且 IoU>0.5 者，默认 3 例。
3) 拼图把对照排在最后一整行并加标题，单图注记区分 background / correct。
输出到新目录 figures/misalign_examples_v2，不覆盖 v1。

判据沿用 v1：ft_map 逆映射取类名；gate targets_detail 是错位实例权威来源。
对照实例不依赖 gate，改用 logits argmax==GT label 判定（正确分类且定位良好）。
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from fsod import voc, engine
from fsod.data import VOCDetDataset  # noqa: F401  保持与缓存生成同一 import 路径


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
    ap.add_argument("--n-misalign", type=int, default=6)
    ap.add_argument("--n-control", type=int, default=3)
    ap.add_argument("--out-dir", default="figures/misalign_examples_v2")
    args = ap.parse_args()

    here = os.getcwd()
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]

    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    cls_name = make_cls_name(sp)
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]
    gate = json.load(open(args.gate, encoding="utf-8"))
    gate_pairs = {(t["img"], t["pi"]) for t in gate["targets_detail"]}

    misalign, controls = [], []
    gi = 0
    for ci, cache in enumerate(caches):
        props_b = cache["proposals"]; logits = cache["logits"]
        sizes = [p.shape[0] for p in props_b]
        lg_off = 0
        for bi in range(len(props_b)):
            props = props_b[bi].float()
            lg = logits[lg_off:lg_off + sizes[bi]].float()
            lg_off += sizes[bi]
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            img = gi; gi += 1
            gt = gts[img]
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            keep = gd == 0; gb2, gl2 = gb[keep], gl[keep]
            if len(gb2) == 0:
                continue
            ious = iou_xyxy(props_orig, torch.from_numpy(gb2))
            pred = lg.argmax(1)
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
                rec = {"img": img, "pi": pi, "gt_j": j,
                       "gt_box": [float(v) for v in gb2[j]],
                       "prop_box": [float(v) for v in props_orig[pi]],
                       "cls": cls_name(gl2[j]), "iou": float(col[pi]),
                       "file": te_samples[img]["file"]}
                if (img, pi) in gate_pairs and int(pred[pi]) == 0:
                    misalign.append(rec)
                elif int(pred[pi]) == int(gl2[j]):
                    controls.append(rec)
    print(f"verified misalign(bg&goodloc)={len(misalign)} (gate n={gate['n_bg_good']})  "
          f"correct-detection controls={len(controls)}")

    def pick(lst, n, cap_per_cls):
        by_cls = {}
        for f in sorted(lst, key=lambda d: -d["iou"]):
            by_cls.setdefault(f["cls"], []).append(f)
        out = []
        for cls, sub in by_cls.items():
            out.extend(sub[:cap_per_cls])
        return sorted(out, key=lambda d: -d["iou"])[:n]

    pick_m = pick(misalign, args.n_misalign, 3)
    pick_c = pick(controls, args.n_control, 3)
    print(f"picked misalign={[(p['cls'], p['iou']) for p in pick_m]}")
    print(f"picked controls={[(p['cls'], p['iou']) for p in pick_c]}")

    os.makedirs(args.out_dir, exist_ok=True)
    jpg_dir = os.path.join(here, "datasets", "VOCdevkit", "VOC2007", "JPEGImages")
    try:
        font2 = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font2 = ImageFont.load_default()

    def render(p, is_control, k):
        im = Image.open(os.path.join(jpg_dir, p["file"])).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle(p["gt_box"], outline=(0, 200, 0), width=3)
        d.rectangle(p["prop_box"], outline=(230, 30, 30), width=3)
        x1, y1, x2, y2 = p["gt_box"]
        bw, bh = x2 - x1, y2 - y1
        half = max(bw, bh) * 2.0
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box = [max(0, cx - half), max(0, cy - half),
               min(im.width, cx + half), min(im.height, cy + half)]
        crop = im.crop([int(v) for v in box]).resize((440, 440))
        d2 = ImageDraw.Draw(crop)
        ox, oy = box[0], box[1]
        sc = 440.0 / (box[2] - box[0])
        gx1, gy1 = (p["gt_box"][0] - ox) * sc, (p["gt_box"][1] - oy) * sc
        gx2, gy2 = (p["gt_box"][2] - ox) * sc, (p["gt_box"][3] - oy) * sc
        px1, py1 = (p["prop_box"][0] - ox) * sc, (p["prop_box"][1] - oy) * sc
        px2, py2 = (p["prop_box"][2] - ox) * sc, (p["prop_box"][3] - oy) * sc
        d2.rectangle([gx1, gy1, gx2, gy2], outline=(0, 220, 0), width=2)
        d2.rectangle([px1, py1, px2, py2], outline=(255, 60, 60), width=2)
        lines = [f"{p['cls']}  IoU={p['iou']:.2f}",
                 "pred: background" if not is_control else f"pred: {p['cls']} (correct)"]
        for li, txt in enumerate(lines):
            ty = 6 + li * 20
            tw = d2.textlength(txt, font=font2)
            d2.rectangle([4, ty - 2, 4 + tw + 8, ty + 18], fill=(0, 0, 0))
            d2.text((8, ty), txt, fill=(255, 255, 100), font=font2)
        tag = "ctrl" if is_control else "misalign"
        out_p = os.path.join(args.out_dir, f"{tag}_{k:02d}_{p['cls']}_iou{p['iou']:.2f}.png")
        crop.save(out_p)
        return crop

    crops_m = [render(p, False, k) for k, p in enumerate(pick_m)]
    crops_c = [render(p, True, k) for k, p in enumerate(pick_c)]

    cols = 3
    rows_m = (len(crops_m) + cols - 1) // cols
    rows = rows_m + (1 if crops_c else 0)
    cell, label_h = 448, 28
    montage = Image.new("RGB", (cols * cell + 8, rows * (cell + label_h) + 8), (255, 255, 255))
    md = ImageDraw.Draw(montage)
    for i, c in enumerate(crops_m):
        r, col = divmod(i, cols)
        montage.paste(c, (8 + col * cell, 8 + r * (cell + label_h)))
    if crops_c:
        r0 = rows_m
        md.text((8, 8 + r0 * (cell + label_h)),
                "Negative controls: correctly detected small novel instances",
                fill=(0, 0, 0), font=font2)
        for i, c in enumerate(crops_c):
            col = i % cols
            montage.paste(c, (8 + col * cell, 8 + r0 * (cell + label_h) + label_h))
    out_m = os.path.join(args.out_dir, "misalign_montage_v2.png")
    montage.save(out_m)
    print(f"saved montage {out_m}  ({montage.width}x{montage.height})")


if __name__ == "__main__":
    main()
