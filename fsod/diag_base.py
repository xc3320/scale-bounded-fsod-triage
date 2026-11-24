"""纯 base 模型（仅 base 类）在 GT 框上的分尺度分类判决诊断。

用法：
  python -m fsod.diag_base --ckpt runs/base_s1_0712/ep10.pth --data 0712 --tag baseline
输出：base 类在 small/medium/large 上的 top1 与判背景率。
用于最小闭环判定：SCC 短训后 base 小目标判背景是否下降。
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from fsod import voc, engine
from fsod.model import build_model
from fsod.data import VOCDetDataset, detection_collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-imgs", type=int, default=400)
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
    sp = voc.build_split(ann, tr, te, 1)
    base_map = sp["base_map"]
    n_base = len(base_map)
    te_samples = voc.samples_for(ann, te, base_map, keep_difficult=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2,
                           shuffle=False, num_workers=2, collate_fn=detection_collate)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    dev = torch.device("cuda")
    model = build_model(n_base + 1, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])

    # base 类当"novel_set"喂给诊断，分尺度统计即 base 类的判决
    base_labels_int = list(range(1, n_base + 1))
    d = engine.head_diagnostic(model, te_loader, te_samples, dev,
                               novel_labels=base_labels_int, base_labels=[],
                               max_imgs=args.max_imgs)
    out = {"tag": args.tag, "ckpt": args.ckpt,
           "base_top1": d["novel_top1"], "base_to_bg": d["novel_to_bg"],
           "by_scale": {b: {"n": d[f"n_novel_{b}"],
                            "top1": d[f"novel_top1_{b}"],
                            "tobg": d[f"novel_tobg_{b}"],
                            "top1_nobg": d[f"novel_top1nobg_{b}"]}
                        for b in ("small", "medium", "large")}}
    print(f"[{args.tag}] base top1={out['base_top1']:.3f} to_bg={out['base_to_bg']:.3f}")
    for b, v in out["by_scale"].items():
        print(f"    {b:7s} n={v['n']:4d} top1={v['top1']:.3f} tobg={v['tobg']:.3f} "
              f"top1_nobg={v['top1_nobg']:.3f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
