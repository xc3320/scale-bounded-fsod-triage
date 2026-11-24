"""COCO 跨数据集机制验证：缓存 COCO 预训练检测器的 logits/reg/proposals + CLIP 相似度。

用 torchvision COCO_V1 预训练 Faster R-CNN（91类），不改模型不训练。
CLIP 原型从 val 图标注框 crop 得到（每类最多 --proto-per-cls 个样本）。
缓存落盘后，probe_misalignment 的 COCO 版可秒级复用。

用法:
  python -m fsod.coco_forward_cache --max-imgs 600 --out data/coco_cache.pt
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)


def set_torch_home():
    here = os.getcwd()
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))


class CocoValDS(Dataset):
    def __init__(self, root, ann, ids):
        self.root = root
        self.ann = ann
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        iid = self.ids[k]
        info = self.ann["id2img"][iid]
        # 本地保存为纯数字 ID 名（服务器 URL 名），非 file_name
        path = os.path.join(self.root, "%012d.jpg" % iid)
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            # 坏图返回空，主循环跳过
            return {"image": None, "boxes": np.zeros((0, 4), np.float32),
                    "labels": np.zeros(0, np.int64), "size": (1, 1),
                    "file": "", "image_id": iid}
        w, h = im.size
        arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)
        # GT
        gts = [a for a in self.ann["id2ann"][iid] if not a.get("iscrowd", 0)]
        boxes = np.array([[a["bbox"][0], a["bbox"][1],
                           a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in gts], dtype=np.float32).reshape(-1, 4) if gts else np.zeros((0, 4), np.float32)
        labels = np.array([a["category_id"] for a in gts], dtype=np.int64)
        return {"image": arr, "boxes": boxes, "labels": labels,
                "size": (h, w), "file": "%012d.jpg" % iid, "image_id": iid}


def collate(batch):
    return batch


@torch.no_grad()
def build_clip_protos(clip_model, pre, samples, dev, per_cls=8):
    """samples: list of (path, box) -> 每类 crop 的 CLIP 特征均值原型。"""
    from collections import defaultdict
    feats = defaultdict(list)
    imgs = []
    labs = []
    for lab, path, box in samples:
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(im.width, x2), min(im.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = im.crop((x1, y1, x2, y2)).resize((224, 224))
        t = pre(crop).unsqueeze(0)
        imgs.append(t)
        labs.append(lab)
        if len(imgs) >= 16:
            flush(imgs, labs, feats, clip_model, dev)
            imgs, labs = [], []
    if imgs:
        flush(imgs, labs, feats, clip_model, dev)
    protos = {}
    for lab, fl in feats.items():
        f = torch.stack(fl).mean(0)
        protos[lab] = torch.nn.functional.normalize(f, dim=0)
    return protos


def flush(imgs, labs, feats, model, dev):
    import torch.nn.functional as F
    x = torch.cat(imgs, 0).to(dev)
    with torch.amp.autocast("cuda", enabled=True):
        f = model.encode_image(x)
    f = F.normalize(f.float(), dim=1).cpu()
    for lab, vec in zip(labs, f):
        feats[lab].append(vec)


@torch.no_grad()
def forward_cache(model, image, dev, topk=300):
    """手动走前向，缓存 logits/reg/proposals（resized 坐标）。"""
    imgs = [image.to(dev)]
    imgs_t, _ = model.transform(imgs, None)
    feats = model.backbone(imgs_t.tensors)
    proposals, _ = model.rpn(imgs_t, feats, imgs_t.image_sizes)
    # 只取前 topk proposals 做 ROI pooling（省显存；RPN 已按 score 排序）
    proposal_single = [p[:topk] for p in proposals]
    # 测试模式：RoIHeads.forward 内部用 rpn 全部 proposals，经 nms 后取前 k
    box_features = model.roi_heads.box_roi_pool(feats, proposal_single, imgs_t.image_sizes)
    box_features = model.roi_heads.box_head(box_features)
    logits = model.roi_heads.box_predictor.cls_score(box_features)
    reg = model.roi_heads.box_predictor.bbox_pred(box_features)
    return {"logits": logits.cpu(), "reg": reg.cpu(),
            "proposals": [p.cpu() for p in proposal_single],
            "image_sizes": imgs_t.image_sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="datasets/coco/coco/annotations/val.json")
    ap.add_argument("--imgdir", default="datasets/coco/val2017")
    ap.add_argument("--max-imgs", type=int, default=600)
    ap.add_argument("--proto-per-cls", type=int, default=8)
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--out", default="data/coco_cache.pt")
    args = ap.parse_args()

    set_torch_home()
    import clip
    dev = torch.device("cuda")
    ann = json.load(open(args.ann, encoding="utf-8"))
    id2img = {im["id"]: im for im in ann["images"]}
    id2ann = {}
    for a in ann["annotations"]:
        id2ann.setdefault(a["image_id"], []).append(a)
    ann["id2img"], ann["id2ann"] = id2img, id2ann

    # 只用已下载的图
    avail = set(os.listdir(args.imgdir))
    ids = []
    for im in ann["images"]:
        p = os.path.join(args.imgdir, "%012d.jpg" % im["id"])
        if ("%012d.jpg" % im["id"]) not in avail:
            continue
        try:
            with open(p, "rb") as fh:
                if fh.read(3) != b"\xff\xd8\xff":
                    continue
            Image.open(p).verify()
        except Exception:
            continue
        ids.append(im["id"])
        if len(ids) >= args.max_imgs:
            break
    print(f"[coco] using {len(ids)} verified images", flush=True)

    # ===== 阶段1：检测器前向，只保留 CPU 结果，随后彻底释放模型 =====
    w = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=w).to(dev).eval()
    cats = w.meta["categories"]  # 0=bg, 1..80

    ds = CocoValDS(args.imgdir, ann, ids)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    caches, gts = [], []
    t0 = time.time()
    for k, batch in enumerate(dl):
        smp = batch[0]
        if smp["image"] is None:
            continue
        cache = forward_cache(model, smp["image"], dev, topk=args.topk)
        oh, ow = smp["size"]
        props = cache["proposals"][0][:args.topk]
        logits = cache["logits"][:args.topk]
        reg = cache["reg"][:args.topk]
        caches.append({"logits": logits.cpu(), "reg": reg.cpu(),
                       "proposals": [props.cpu()], "image_sizes": cache["image_sizes"],
                       "orig_sizes": [(oh, ow)], "file": smp["file"]})
        gts.append({"boxes": smp["boxes"], "labels": smp["labels"]})
        del cache, props, logits, reg
        if (k + 1) % 10 == 0:
            print(f"[det] {k+1}/{len(ids)} in {(time.time()-t0)/60:.1f}min", flush=True)
    del model, dl, ds
    torch.cuda.empty_cache()
    print(f"[det] done {len(caches)} imgs in {(time.time()-t0)/60:.1f}min "
          f"mem={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)

    # ===== 阶段2：CLIP 原型 + crop 编码，检测器已不在显存 =====
    import torch.nn.functional as F
    clip_path = os.path.join(os.getcwd(), "torch_home", "clip", "ViT-B-32.pt")
    clip_model, pre = clip.load(clip_path, device=dev)
    clip_model.eval()

    proto_samples = []
    per_cls = {}
    for iid in ids:
        for a in id2ann[iid]:
            if a.get("iscrowd", 0):
                continue
            lab = a["category_id"]
            if per_cls.get(lab, 0) >= args.proto_per_cls:
                continue
            x, y, bw, bh = a["bbox"]
            path = os.path.join(args.imgdir, "%012d.jpg" % iid)
            proto_samples.append((lab, path, (x, y, x + bw, y + bh)))
            per_cls[lab] = per_cls.get(lab, 0) + 1
    protos = build_clip_protos(clip_model, pre, proto_samples, dev, args.proto_per_cls)
    proto_labs = sorted(protos.keys())
    proto_mat = torch.stack([protos[l] for l in proto_labs]).to(dev)  # (C,512)
    print(f"[clip] built {len(protos)} prototypes mem={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)

    sims = []
    with torch.no_grad():
      for ci, cache in enumerate(caches):
        props = cache["proposals"][0].float()
        oh, ow = cache["orig_sizes"][0]
        rh, rw = cache["image_sizes"][0]
        sx, sy = ow / rw, oh / rh
        im_full = Image.open(os.path.join(args.imgdir, cache["file"])).convert("RGB")
        crops = []
        for p in props:
            bx = [p[0].item() * sx, p[1].item() * sy, p[2].item() * sx, p[3].item() * sy]
            x1, y1, x2, y2 = [int(round(v)) for v in bx]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(ow, x2), min(oh, y2)
            if x2 > x1 and y2 > y1:
                crop = im_full.crop((x1, y1, x2, y2)).resize((224, 224))
            else:
                crop = Image.new("RGB", (224, 224), (0, 0, 0))
            crops.append(pre(crop).unsqueeze(0))
        if crops:
            sim_chunks = []
            for ci2 in range(0, len(crops), 16):
                x = torch.cat(crops[ci2:ci2 + 16], 0).to(dev)
                with torch.amp.autocast("cuda", enabled=True):
                    pf = clip_model.encode_image(x)
                sim_chunks.append((F.normalize(pf.float(), dim=1) @ proto_mat.T).cpu())
                del x, pf
            sim = torch.cat(sim_chunks, 0)
            del sim_chunks
        else:
            sim = torch.zeros(0, len(proto_labs))
        sims.append([sim])
        del crops, sim
        if (ci + 1) % 10 == 0:
            print(f"[clip] {ci+1}/{len(caches)} in {(time.time()-t0)/60:.1f}min "
                  f"mem={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)

    torch.save({"caches": caches, "sims": sims, "gts": gts,
                "proto_labs": proto_labs, "cats": cats, "n_imgs": len(caches)}, args.out)
    print(f"[coco] SAVED {args.out} ({len(caches)} imgs) in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
