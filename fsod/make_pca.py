"""用 base 检测器在 base 类 GT 框上拟合 pooled RoI 特征的 PCA 子空间（旁路初始化）。"""
import argparse
import os
import sys

import torch

from fsod import voc
from fsod.model import build_model
from fsod.probe_bypass import extract, pca_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--quota", type=int, default=400)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--out", default="data/pca_basis_s1.pt")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    device = torch.device("cuda")

    ann_all, train_ids, test_ids = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann_all, train_ids, test_ids, args.split)
    base_map = sp["base_map"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ck.get("args", {})
    model = build_model(len(base_map) + 1, ca.get("backbone", "resnet50"),
                        ca.get("min_size", 600), ca.get("max_size", 1000),
                        pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    print(f"[ckpt] epoch={ck.get('epoch')}", flush=True)

    samples = voc.samples_for(ann_all, train_ids, base_map)
    Pb, Bb, yb, ab = extract(model, samples, device, args.quota)
    print(f"[pooled] {tuple(Pb.shape)}", flush=True)
    mean, V = pca_basis(Pb, args.k, device="cuda")
    print(f"[pca] V {tuple(V.shape)}", flush=True)
    p = os.path.join(here, args.out)
    torch.save({"mean": mean, "V": V, "quota": args.quota, "epoch": ck.get("epoch")}, p)
    print("saved", p, flush=True)


if __name__ == "__main__":
    main()
