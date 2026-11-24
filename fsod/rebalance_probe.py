"""推理时分类头再平衡：按范数比例放大 novel 类 logit / 压低背景 logit。

动机（实测）：微调后 cls_score 行范数 = 背景 2.461 / base 1.648 / novel 1.075，
背景 bias=+1.045 而 novel≈-0.02。novel 行系统性最弱，导致小目标 95.5% 判背景。

本脚本不改训练，只在评测时对 box_predictor 的 logit 做后处理，扫缩放系数。
"""
import argparse
import json
import os
import torch
import torch.nn as nn

from fsod import voc, engine
from fsod.data import VOCDetDataset, detection_collate
from fsod.model import build_model, expand_classifier
from torch.utils.data import DataLoader


class LogitRebalance(nn.Module):
    def __init__(self, novel_idx, novel_scale, bg_scale):
        super().__init__()
        self.novel_idx = novel_idx
        self.novel_scale = novel_scale
        self.bg_scale = bg_scale

    def install(self, model):
        pred = model.roi_heads.box_predictor
        orig = pred.forward

        def forward(x):
            cls, box = orig(x)
            cls = cls.clone()
            cls[:, 0] = cls[:, 0] * self.bg_scale
            cls[:, self.novel_idx] = cls[:, self.novel_idx] * self.novel_scale
            return cls, box

        pred.forward = forward
        return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True)
    ap.add_argument("--base-ckpt", default="runs/base_s1_0712/ep10.pth")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--out", default="data/rebalance_sweep.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    device = torch.device("cuda")

    r07 = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    r12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    if args.data == "0712":
        ann, tr, te = voc.prepare_0712(r07, r12, os.path.join(here, "data"))
    else:
        ann, tr, te = voc.prepare(r07, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    n_base, n_all = len(sp["base"]), len(ft_map)
    novel_idx = [ft_map[c] for c in sp["novel"]]

    ck = torch.load(os.path.join(here, args.base_ckpt), map_location="cpu", weights_only=False)
    fck = torch.load(os.path.join(here, args.ft), map_location="cpu", weights_only=False)

    def fresh():
        m = build_model(n_base + 1, "resnet50", 600, 1000, pretrained=False)
        m.load_state_dict(ck["model"])
        m = expand_classifier(m, n_all + 1, n_base + 1)
        m.load_state_dict(fck["model"])
        return m

    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2, shuffle=False,
                        num_workers=2, collate_fn=detection_collate, pin_memory=True)
    novel_names = {ft_map[c]: c for c in sp["novel"]}

    results = {}
    for ns in [1.0, 1.5, 2.0]:
        for bs in [1.0, 0.7]:
            tag = f"novel{ns}_bg{bs}"
            model = fresh()
            m = LogitRebalance(novel_idx, ns, bs).install(model)
            m = m.to(device).eval()
            p, g = engine.evaluate(m, loader, te_samples, device)
            _, mn = engine.report_ap(p, g, novel_names, tag="")
            base_names = {ft_map[c]: c for c in sp["base"]}
            _, mb = engine.report_ap(p, g, base_names, tag="")
            results[tag] = {"novel": mn, "base": mb}
            print(f"{tag:14s} novel={mn*100:6.2f} base={mb*100:6.2f}", flush=True)

    with open(os.path.join(here, args.out), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
