"""判定实验：绕开微调训练动态，直接测"特征 + 5-shot 原型"在 21 类上的上限。

回答一个问题：novel 类在 box_head 特征空间里到底能不能跟 15 个 base 类 + 背景区分开。
若这里能到 40%+，说明微调循环有问题；若也接近 0，说明 21 类设定下 5-shot 原型本身就不可分。
"""
import argparse
import json
import os

import numpy as np
import torch

from fsod import voc
from fsod.model import build_model
from fsod.probe_bypass import extract, l2n


def centroids(X, y, labels):
    P = []
    for c in labels:
        m = y == c
        P.append(X[m].mean(0) if m.sum() else torch.zeros(X.shape[1]))
    return l2n(torch.stack(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base_s1/ep10.pth")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--base-quota", type=int, default=200)
    ap.add_argument("--test-imgs", type=int, default=600)
    ap.add_argument("--out", default="data/feat_21way.json")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    device = torch.device("cuda")

    ann_all, train_ids, test_ids = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann_all, train_ids, test_ids, args.split)
    ft_map, base_map = sp["ft_map"], sp["base_map"]

    ck = torch.load(os.path.join(here, args.ckpt), map_location="cpu", weights_only=False)
    ca = ck.get("args", {})
    model = build_model(len(base_map) + 1, ca.get("backbone", "resnet50"),
                        ca.get("min_size", 600), ca.get("max_size", 1000),
                        pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    # base 原型：train split 的 base 类 GT 框
    base_samples = voc.samples_for(ann_all, train_ids, base_map)
    Pb, Bb, yb, _ = extract(model, base_samples, device, args.base_quota)

    # novel 原型：只取 shot 个实例/类
    novel_map = {c: i + 1 for i, c in enumerate(sp["novel"])}
    novel_samples = voc.samples_for(ann_all, train_ids, novel_map)
    Pn, Bn, yn, _ = extract(model, novel_samples, device, args.shot)

    # 测试集
    te_samples = voc.samples_for(ann_all, test_ids, ft_map)
    import random
    rng = random.Random(0)
    te_sub = rng.sample(te_samples, min(args.test_imgs, len(te_samples)))
    Pt, Bt, yt, _ = extract(model, te_sub, device, 9999)

    res = {"shot": args.shot, "split": args.split,
           "n_support_base": int(len(yb)), "n_support_novel": int(len(yn)),
           "n_test": int(len(yt))}
    labels = list(range(1, 21))
    for name, sup_b, sup_n, test in (("boxhead", Bb, Bn, Bt), ("pooled", Pb, Pn, Pt)):
        Proto_b = centroids(l2n(sup_b), yb, labels[:15])
        Proto_n = centroids(l2n(sup_n), yn + 15, list(range(16, 21)))
        P = l2n(torch.cat([Proto_b, Proto_n], 0))
        Q = l2n(test)
        pred = (Q @ P.t()).argmax(1).numpy() + 1
        out = {}
        for tag, ys_sel in (("novel", [ft_map[c] for c in sp["novel"]]),
                            ("base", [ft_map[c] for c in sp["base"]])):
            m = np.isin(yt, ys_sel)
            if m.sum() == 0:
                continue
            out[tag + "_n"] = int(m.sum())
            out[tag + "_top1"] = float((pred[m] == yt[m]).mean())
            if tag == "novel":
                out["novel_to_base"] = float(np.isin(pred[m],
                                             [ft_map[c] for c in sp["base"]]).mean())
        res[name] = out
        print(f"[{name}] novel_top1={out.get('novel_top1',0):.3f} "
              f"novel_to_base={out.get('novel_to_base',0):.3f} base_top1={out.get('base_top1',0):.3f}",
              flush=True)

    with open(os.path.join(here, args.out), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
