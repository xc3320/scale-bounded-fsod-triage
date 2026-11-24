"""最小验证：CLIP 冻结图像特征（同模态原型匹配）能否捞回被检测器判背景的 novel 小目标。

与已证伪的"CLIP 文本锚"区别：
  - 文本锚：text-image 对齐，跨模态、有 gap；
  - 这里：novel 支持框裁剪 -> CLIP 图像塔 -> 类原型；test GT 框裁剪 -> CLIP -> 余弦匹配。
    纯 image-image 同模态，对应 DeFRCN-PCB / FSOD-VFM(ICLR26) / no-time-to-train 路线。

判定（不训练，纯推理探针）：
  对 test 集 novel GT 框，同时算
    (a) 检测器分类头判决 det_pred（已知小目标 ~95% 判背景）
    (b) CLIP 与 20 类 few-shot 图像原型的余弦相似度 -> clip_pred
  分尺度统计。核心看：det 判背景的 novel 小目标里，clip 能捞回正确 novel 类的比例。

用法：
  python -m fsod.probe_clip_rescue --ckpt runs/ft2_ctrl_base/ft.pth --data 0712 \
      --shot 5 --max-imgs 300 --out data/clip_rescue.json
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from fsod import voc, engine
from fsod.model import build_model
from fsod.data import VOCDetDataset, detection_collate
from fsod.finetune import build_fewshot_set

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def crop_tensor(path, box, size=224):
    """按框裁剪原图 -> CLIP 输入张量 (3,size,size)。"""
    im = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im.width, x2), min(im.height, y2)
    if x2 <= x1 or y2 <= y1:
        x2, y2 = min(im.width, x1 + 1), min(im.height, y1 + 1)
    crop = im.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
    a = np.asarray(crop, dtype=np.float32) / 255.0
    a = (a - np.array(CLIP_MEAN)) / np.array(CLIP_STD)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


@torch.no_grad()
def encode_crops(clip_model, crops, device, bs=64):
    feats = []
    for i in range(0, len(crops), bs):
        batch = torch.stack(crops[i:i + bs]).to(device)
        with torch.amp.autocast("cuda", enabled=True):
            f = clip_model.encode_image(batch)
        f = f.float()
        f = F.normalize(f, dim=-1)
        feats.append(f.cpu())
    if not feats:
        return torch.zeros(0, 512)
    return torch.cat(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--max-imgs", type=int, default=300)
    ap.add_argument("--clip-path", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    voc12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    if args.data == "0712":
        ann, tr, te = voc.prepare_0712(voc_root, voc12, os.path.join(here, "data"))
    else:
        ann, tr, te = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]           # 20 类 -> label 1..20
    idx2name = {v: k for k, v in ft_map.items()}
    n_all = len(ft_map)
    novel = sp["novel"]
    novel_labs = [ft_map[c] for c in novel]
    novel_set = set(novel_labs)
    base_labs = [ft_map[c] for c in sp["base"]]
    print(f"[data] novel={novel} novel_labels={novel_labs}", flush=True)

    # ---- 1. 支持集 few-shot 框 -> 20 类 CLIP 图像原型 ----
    ft_samples, picks = build_fewshot_set(ann, tr, ft_map, sp["base"], novel, args.shot, seed=0)
    sup_crops, sup_labs = [], []
    for s in ft_samples:
        for box, lb in zip(s["boxes"], s["labels"]):
            sup_crops.append(crop_tensor(s["file"], box))
            sup_labs.append(lb)
    print(f"[support] imgs={len(ft_samples)} boxes={len(sup_crops)}", flush=True)

    # ---- 2. 加载 CLIP 图像塔 ----
    import clip as clip_lib
    clip_path = args.clip_path or os.path.join(here, "torch_home", "clip", "ViT-B-32.pt")
    dev = torch.device("cuda")
    clip_model, _ = clip_lib.load(clip_path, device=dev)
    clip_model.eval()
    sup_feat = encode_crops(clip_model, sup_crops, dev)   # (N,512) L2
    protos = {}
    for lb in range(1, n_all + 1):
        m = torch.tensor([l == lb for l in sup_labs])
        if m.sum() == 0:
            continue
        pf = sup_feat[m].mean(0)
        protos[lb] = F.normalize(pf.unsqueeze(0), dim=-1).squeeze(0)
    proto_mat = torch.stack([protos[l] for l in range(1, n_all + 1)], 0).to(dev)  # (20,512)
    proto_labs = list(range(1, n_all + 1))
    novel_proto_idx = [i for i, l in enumerate(proto_labs) if l in novel_set]
    print(f"[clip] prototypes={proto_mat.shape} novel_proto_idx={novel_proto_idx}", flush=True)

    # ---- 3. 加载微调检测器（与 head_diagnostic 同口径）----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(n_all + 1, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2,
                           shuffle=False, num_workers=2, collate_fn=detection_collate)

    # ---- 4. 逐图：GT 框 -> 检测器 logits + CLIP 特征 ----
    bins = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
    rec = {b: {"n": 0, "det_correct": 0, "det_bg": 0,
               "clip20_correct": 0, "clip5_correct": 0,
               "bg_recovered20": 0, "bg_recovered5": 0, "bg_n": 0}
           for b in bins}
    clip_crops, meta = [], []   # meta: (bin, y, det_pred)
    n_img = 0
    for images, targets in te_loader:
        if n_img >= args.max_imgs:
            break
        images = [im.to(dev) for im in images]
        orig = [tuple(im.shape[-2:]) for im in images]
        # --- 检测器分类头判决 ---
        with torch.amp.autocast("cuda", enabled=True):
            imgs, _ = model.transform(images, None)
            feats = model.backbone(imgs.tensors)
            props = []
            for i, t in enumerate(targets):
                sy = imgs.image_sizes[i][0] / orig[i][0]
                sx = imgs.image_sizes[i][1] / orig[i][1]
                props.append(t["boxes"].to(dev) * torch.tensor([sx, sy, sx, sy], device=dev))
            pooled = model.roi_heads.box_roi_pool(feats, props, imgs.image_sizes)
            feat = model.roi_heads.box_head(pooled)
            logits, _ = model.roi_heads.box_predictor(feat)
            logits = logits.float().cpu()
        det_pred = logits.argmax(1).tolist()
        # --- CLIP 裁剪 ---
        off = 0
        for i, t in enumerate(targets):
            boxes = t["boxes"].cpu().numpy()
            labels = t["labels"].cpu().numpy()
            smp = te_samples[n_img + i] if n_img + i < len(te_samples) else None
            for k in range(len(labels)):
                y = int(labels[k])
                if y not in novel_set:
                    continue
                # 用原图坐标裁剪（targets boxes 已是原图坐标）
                box = boxes[k]
                bx = [smp["boxes"][k][0], smp["boxes"][k][1],
                      smp["boxes"][k][2], smp["boxes"][k][3]] if smp else box
                area = float((bx[2] - bx[0]) * (bx[3] - bx[1]))
                bn = next((b for b, (lo, hi) in bins.items() if lo <= area < hi), "large")
                dp = int(det_pred[off + k])
                is_bg = (dp == 0)
                rec[bn]["n"] += 1
                if dp == y:
                    rec[bn]["det_correct"] += 1
                if is_bg:
                    rec[bn]["det_bg"] += 1
                    rec[bn]["bg_n"] += 1
                path = smp["file"] if smp else ""
                if path:
                    clip_crops.append(crop_tensor(path, bx))
                    meta.append((bn, y, dp, is_bg))
            off += len(labels)
        n_img += len(images)

    print(f"[test] imgs={n_img} novel_gt_crops={len(clip_crops)}", flush=True)
    test_feat = encode_crops(clip_model, clip_crops, dev).to(dev)   # (M,512)
    sim20 = test_feat @ proto_mat.T                          # (M,20)
    clip20_pred = np.array([proto_labs[i] for i in sim20.argmax(1).cpu().tolist()])
    # 只在 5 个 novel 原型里取最近（类似去背景 top1）
    sim5 = test_feat @ proto_mat[novel_proto_idx, :].T
    clip5_pred = np.array([novel_labs[i] for i in sim5.argmax(1).cpu().tolist()])

    for j, (bn, y, dp, is_bg) in enumerate(meta):
        d = rec[bn]
        if clip20_pred[j] == y:
            d["clip20_correct"] += 1
        if clip5_pred[j] == y:
            d["clip5_correct"] += 1
        if is_bg:
            if clip20_pred[j] == y:
                d["bg_recovered20"] += 1
            if clip5_pred[j] == y:
                d["bg_recovered5"] += 1

    # ---- 5. 汇总 ----
    out = {"ckpt": args.ckpt, "data": args.data, "shot": args.shot,
           "clip": "ViT-B-32 image tower, crop->L2->cos to 20 class prototypes", "by_scale": {}}
    print("\n=== 结果：novel GT 框，检测器分类头 vs CLIP 同模态原型 ===", flush=True)
    print(f"{'scale':8s} {'n':>5s} {'det_top1':>9s} {'det_bg':>7s} "
          f"{'clip20':>7s} {'clip5':>7s} {'bg捞回20':>9s} {'bg捞回5':>8s}", flush=True)
    for bn in ("small", "medium", "large"):
        d = rec[bn]
        n = max(d["n"], 1)
        bgn = max(d["bg_n"], 1)
        row = {
            "n": d["n"],
            "det_top1": round(d["det_correct"] / n, 4),
            "det_bg": round(d["det_bg"] / n, 4),
            "clip20_top1": round(d["clip20_correct"] / n, 4),
            "clip5_top1": round(d["clip5_correct"] / n, 4),
            "bg_n": d["bg_n"],
            "bg_recovered20": round(d["bg_recovered20"] / bgn, 4),
            "bg_recovered5": round(d["bg_recovered5"] / bgn, 4),
        }
        out["by_scale"][bn] = row
        print(f"{bn:8s} {d['n']:5d} {row['det_top1']:9.3f} {row['det_bg']:7.3f} "
              f"{row['clip20_top1']:7.3f} {row['clip5_top1']:7.3f} "
              f"{row['bg_recovered20']:9.3f} {row['bg_recovered5']:8.3f}", flush=True)

    # 判定
    s = out["by_scale"]["small"]
    verdict = "PASS" if (s["clip5_top1"] >= 0.5 and s["bg_recovered5"] >= 0.4) else "FAIL"
    out["verdict"] = verdict
    print(f"\n[VERDICT] {verdict}  (判定线: small clip5_top1>=0.50 且 bg捞回5>=0.40)", flush=True)
    print("随机基线: clip20=1/20=0.05, clip5=1/5=0.20", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
