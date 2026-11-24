"""①′ Day3: cross-FPN level consistency forward.

For each proposal, force pooling at ALL 5 FPN levels (scale boxes to fool the
canonical-level assignment), run box_head+predictor per level, and store
per-level novel probabilities + argmax. Proposals/logits/reg saved in the same
format as clip_fusion_cache_full.pt so the two caches zip 1:1 by index.

Feasibility question (pre-registered gate): can level-consistency separate the
36 bg&goodloc small-novel instances from the unmatched-FP pool?
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from fsod import voc
from fsod.data import VOCDetDataset, detection_collate
from fsod.model import build_model


@torch.no_grad()
def forward_levels(model, images, dev, topk):
    imgs, _ = model.transform(images, None)
    feats = model.backbone(imgs.tensors)          # list of 5 level maps
    proposals, _ = model.rpn(imgs, feats, imgs.image_sizes)
    if topk > 0:
        proposals = [p[:topk] for p in proposals]
    roi_pool = model.roi_heads.box_roi_pool
    box_head = model.roi_heads.box_head
    box_pred = model.roi_heads.box_predictor

    # canonical level per proposal (torchvision formula, resized coords)
    lv_list = []
    for p in proposals:
        wh = (p[:, 2] - p[:, 0]).clamp(min=1) * (p[:, 3] - p[:, 1]).clamp(min=1)
        lv = torch.floor(4 + torch.log2(wh.sqrt() / 224 + 1e-8)).clamp(0, 4)
        lv_list.append(lv)

    level_logits = []                             # per level: list of (N_i,21)
    n_per = [p.shape[0] for p in proposals]
    for L in range(5):
        scaled = []
        for p, lv in zip(proposals, lv_list):
            f = torch.pow(torch.tensor(2.0, device=p.device), (L - lv).float()).unsqueeze(1)
            c = (p[:, :2] + p[:, 2:]) / 2
            hw = (p[:, 2:] - p[:, :2]) / 2 * f
            scaled.append(torch.cat([c - hw, c + hw], dim=1))
        pooled = roi_pool(feats, scaled, imgs.image_sizes)
        feat = box_head(pooled)
        lg, _ = box_pred(feat)
        level_logits.append(lg.cpu().split(n_per, dim=0))

    # canonical (unmodified) forward for reference logits/reg
    pooled0 = roi_pool(feats, proposals, imgs.image_sizes)
    feat0 = box_head(pooled0)
    logits0, reg0 = box_pred(feat0)

    out = {
        "logits": logits0.cpu().float(), "reg": reg0.cpu().float(),
        "proposals": [p.cpu() for p in proposals],
        "image_sizes": imgs.image_sizes,
        "orig_sizes": [tuple(im.shape[-2:]) for im in images],
    }
    # per-image compact per-level novel prob + argmax
    for i, n_i in enumerate([p.shape[0] for p in proposals]):
        lv_np = np.stack([level_logits[L][i].float().softmax(1)[:, 1:].numpy().max(1)
                          for L in range(5)], axis=1)          # (N_i,5) novel prob per level
        lv_am = np.stack([level_logits[L][i].argmax(1).numpy() for L in range(5)], axis=1)
        out.setdefault("level_novelprob_chunks", []).append(torch.from_numpy(lv_np).half())
        out.setdefault("level_argmax_chunks", []).append(torch.from_numpy(lv_am))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ft2_ctrl_base/ft.pth")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--max-imgs", type=int, default=4952)
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--out", default="data/fpn_levels_full.pt")
    args = ap.parse_args()

    here = os.getcwd()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    n_all = 20

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

    caches, np_chunks, am_chunks = [], [], []
    ptr, t0 = 0, __import__("time").time()
    for images, targets in te_loader:
        if ptr >= args.max_imgs:
            break
        images = [im.to(dev) for im in images]
        out = forward_levels(model, images, dev, args.topk)
        caches.append({"logits": out["logits"].half(), "reg": out["reg"].half(),
                       "proposals": out["proposals"], "image_sizes": out["image_sizes"],
                       "orig_sizes": out["orig_sizes"]})
        np_chunks.extend(out["level_novelprob_chunks"])
        am_chunks.extend(out["level_argmax_chunks"])
        ptr += len(images)
        if ptr % 200 == 0:
            el = (__import__("time").time() - t0) / 60
            print(f"  {ptr}/{args.max_imgs} imgs {el:.1f}min eta~{el/ptr*(args.max_imgs-ptr):.0f}min",
                  flush=True)

    blob = {"caches": caches,
            "level_novelprob": torch.stack(np_chunks) if np_chunks else torch.empty(0),
            "level_argmax": torch.stack(am_chunks) if am_chunks else torch.empty(0),
            "n_imgs": ptr}
    torch.save(blob, args.out)
    print(f"[saved] {args.out}  imgs={ptr}", flush=True)


if __name__ == "__main__":
    main()
