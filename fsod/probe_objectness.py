"""论文3 缺口清单#1a：转储每个 proposal 的 RPN objectness（objectness 对照消融的数据基础）。

monkey-patch model.rpn.filter_proposals 捕获 scores（与返回 proposals 1:1），
图片顺序与 clip_fusion_cache_full.pt 一致（同一 DataLoader 顺序），
proposal 对齐在分析时校验。
输出 data/objectness_full.pt: {objectness: list[N_i tensor], proposals: 同 cache}
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from fsod import voc
from fsod.data import VOCDetDataset, detection_collate
from fsod.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ft2_ctrl_base/ft.pth")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--max-imgs", type=int, default=4952)
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--out", default="data/objectness_full.pt")
    args = ap.parse_args()

    here = os.getcwd()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    te_samples = voc.samples_for(ann, te, sp["ft_map"], keep_difficult=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2,
                           shuffle=False, num_workers=2, collate_fn=detection_collate)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(21, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    orig_fp = model.rpn.filter_proposals
    captured = {}

    def patched(proposals, objectness, image_sizes, num_anchors_per_level):
        boxes, scores = orig_fp(proposals, objectness, image_sizes, num_anchors_per_level)
        captured["scores"] = scores
        return boxes, scores

    model.rpn.filter_proposals = patched

    obj_all, props_all, ptr, t0 = [], [], 0, __import__("time").time()
    with torch.no_grad():
        for images, targets in te_loader:
            if ptr >= args.max_imgs:
                break
            images = [im.to(dev) for im in images]
            imgs, _ = model.transform(images)
            props, _ = model.rpn(imgs, model.backbone(imgs.tensors), imgs.image_sizes)
            if args.topk > 0:
                props = [p[:args.topk] for p in props]
                captured["scores"] = [s[:args.topk] for s in captured["scores"]]
            obj_all.extend([s.cpu().float() for s in captured["scores"]])
            props_all.extend([p.cpu() for p in props])
            ptr += len(images)
            if ptr % 500 == 0:
                el = (__import__("time").time() - t0) / 60
                print(f"  {ptr}/{args.max_imgs} {el:.1f}min eta~{el/ptr*(args.max_imgs-ptr):.0f}min", flush=True)

    torch.save({"objectness": obj_all, "proposals": props_all, "n_imgs": ptr}, args.out)
    print(f"[saved] {args.out} imgs={ptr} n_prop={sum(len(o) for o in obj_all)}", flush=True)


if __name__ == "__main__":
    main()
