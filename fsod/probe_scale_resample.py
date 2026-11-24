"""论文3 P1-M2②：真实尺度变化的直接检验（图像重采样 + 重跑完整检测管线）。

背景：F3 的强制池化检验缩放的是"框在金字塔上的读取位置"，非目标真实外观。
本实验做严格版本：把图像整体重采样 2×/4×（同步放大 transform 的 min/max size），
重跑 transform→backbone→RPN(topk300)→box head 完整管线，
在重采样坐标系下对同一批 small novel GT 重新匹配，统计错位率随尺度的变化。

判读：若 2×/4× 下错位率显著下降（错位实例被检出）→ 尺度伪影成立，F3 结论需降级；
      若错位率保持（仍判背景）→ 尺度不变盲区在真实尺度变化下成立（F3 解释被严格证实）。
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

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
    ap.add_argument("--ckpt", default="runs/ft2_ctrl_base/ft.pth")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--out", default="data/probe_scale_resample.json")
    args = ap.parse_args()

    here = os.getcwd()
    dev = torch.device("cuda")
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    novel_labs = set(ft_map[c] for c in sp["novel"])

    from fsod.model import build_model
    from fsod.data import VOCDetDataset, detection_collate  # noqa: F401
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(21, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    base_min = model.transform.min_size[0] if isinstance(model.transform.min_size, (list, tuple)) else model.transform.min_size
    base_max = model.transform.max_size[0] if isinstance(model.transform.max_size, (list, tuple)) else model.transform.max_size
    print(f"transform base min/max = {base_min}/{base_max}")

    # ---- 重建匹配实例（与 probe_misalign_fig 同逻辑，B/32 缓存定位 f=1 的 GT 群体）----
    blob = torch.load("data/clip_fusion_cache_full.pt", map_location="cpu", weights_only=False)
    caches, sims, gts = blob["caches"], blob["sims"], blob["gts"]

    def iou_match(props, gt):
        return iou_xyxy(props, gt[None, :])[0] if len(props) else torch.zeros(0)

    targets = []   # (img, gt_box(4,), gtcls)
    gi = 0
    for ci, cache in enumerate(caches):
        for bi in range(len(cache["proposals"])):
            img = gi; gi += 1
            props = cache["proposals"][bi].float()
            oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
            props_orig = props * torch.tensor([ow / rw, oh / rh, ow / rw, oh / rh])
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
                targets.append({"img": img, "gt_box": [float(v) for v in gb2[j]],
                                "gtcls": int(gl2[j])})
    print(f"candidate GT instances (small novel matched): {len(targets)}")

    # ---- 每张涉及图像，多尺度前向 ----
    jpg_dir = os.path.join(here, "datasets", "VOCdevkit", "VOC2007", "JPEGImages")
    by_img = {}
    for t in targets:
        by_img.setdefault(t["img"], []).append(t)
    scales = [float(s) for s in args.scales.split(",")]
    results = {f: {"n": 0, "matched": 0, "unmatched": 0, "bg": 0, "bg_goodloc": 0} for f in scales}
    per_inst = {}
    n_engaged = len(by_img)
    print(f"images to re-detect: {n_engaged}, scales {scales}")

    with torch.no_grad():
        for img, tl in by_img.items():
            ci = img // 2; bi = img % 2
            smp = te_samples[img]
            smp_file = smp["file"]
            pil = Image.open(os.path.join(jpg_dir, smp_file)).convert("RGB")
            for f in scales:
                if f != 1:
                    w, h = pil.size
                    pil_f = pil.resize((int(round(w * f)), int(round(h * f))), Image.BICUBIC)
                else:
                    pil_f = pil
                import torchvision.transforms.functional as TF
                img_t = TF.to_tensor(pil_f)
                img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img_t = img_t.to(dev)
                model.transform.min_size = (base_min * f,)
                model.transform.max_size = base_max * f
                imgs, _ = model.transform([img_t], None)
                feats = model.backbone(imgs.tensors)
                props, _ = model.rpn(imgs, feats, imgs.image_sizes)
                props = [p[:300] for p in props]
                roi_pool = model.roi_heads.box_roi_pool
                box_head = model.roi_heads.box_head
                box_pred = model.roi_heads.box_predictor
                pooled = roi_pool(feats, props, imgs.image_sizes)
                feat = box_head(pooled)
                lg, _ = box_pred(feat)
                lg = lg.float()
                oh, ow = imgs.image_sizes[0]
                # proposal 在 resized 坐标；GT 放大到 f 倍原始坐标后转 resized 坐标
                # 更稳：把 proposal 反缩放回 f 倍原始坐标再匹配（GT×f）
                props_up = props[0].float()  # resized 坐标（f 倍原图空间）
                # resized→f×orig: 乘 (orig_w*f)/rw 等
                ih, iw = imgs.image_sizes[0]
                # 用原图尺寸×f 作参照
                w0, h0 = pil.size  # f≠1 时 pil 未缩放，pil_f 才是
                wf, hf = (pil_f.size if f != 1 else pil.size)
                sx = wf / iw; sy = hf / ih
                props_orig_f = props_up * torch.tensor([sx, sy, sx, sy], device=dev)
                argmax_fg = lg.argmax(1)
                for t in tl:
                    results[f]["n"] += 1
                    gb_f = torch.tensor(t["gt_box"], device=dev) * f
                    rec = {"img": t["img"], "gt": t["gt_box"], "gtcls": t["gtcls"]}
                    if len(props_orig_f) == 0:
                        results[f]["unmatched"] += 1
                        rec.update({"status": "unmatched", "iou": 0.0, "pred": -1})
                        per_inst.setdefault(f"{t['img']}_{t['gtcls']}_{t['gt_box']}", {})[f] = rec
                        continue
                    ious = iou_xyxy(props_orig_f, gb_f[None, :])
                    col = ious[:, 0]
                    if float(col.max()) < 0.3:
                        results[f]["unmatched"] += 1
                        rec.update({"status": "unmatched", "iou": 0.0, "pred": -1})
                        per_inst.setdefault(f"{t['img']}_{t['gtcls']}_{t['gt_box']}", {})[f] = rec
                        continue
                    pi = int(col.argmax().item())
                    results[f]["matched"] += 1
                    pred = int(argmax_fg[pi])
                    rec.update({"status": "matched", "iou": float(col[pi]), "pred": pred})
                    per_inst.setdefault(f"{t['img']}_{t['gtcls']}_{t['gt_box']}", {})[f] = rec
                    if pred == 0:
                        results[f]["bg"] += 1
                        if float(col[pi]) > 0.5:
                            results[f]["bg_goodloc"] += 1
    # f=1 基准（B/32 缓存口径）
    base = {"n": len(targets)}
    # 统计 f=1（重跑管线自检）
    out = {"scales": {str(k): v for k, v in results.items()},
           "n_candidate_gt": len(targets), "n_images": n_engaged,
           "per_instance": per_inst}
    print(json.dumps({str(k): v for k, v in results.items()}, indent=2))
    with open(args.out, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
