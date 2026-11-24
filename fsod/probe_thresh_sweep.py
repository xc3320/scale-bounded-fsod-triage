"""论文3 缺口清单#1b：朴素降阈值对照（"72%框准被判背景，降阈值不就行了？"的必答洞）。

在缓存 logits 上（无门控，lam=0）扫 roi_heads.score_thresh，
实测降低检测分数阈值的代价：FP 洪泛（每图检测框数）、novel/base mAP、
small novel / small base AP。
预登记判读：若降阈值能让 small novel 检出达标而 base 不塌，则对照成立；
预期（由错位现象决定）base small AP 先崩。
"""
import argparse
import json
import os

import torch

from fsod import engine, voc
from fsod.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ft2_ctrl_base/ft.pth")
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--thresholds", default="0.05,0.02,0.01,0.005,0.001")
    ap.add_argument("--out", default="data/probe_thresh_sweep.json")
    args = ap.parse_args()

    here = os.getcwd()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_names = {ft_map[c]: c for c in sp["novel"]}
    base_names = {ft_map[c]: c for c in sp["base"]}

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(21, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    default_thresh = model.roi_heads.score_thresh
    print(f"roi_heads default score_thresh = {default_thresh}", flush=True)

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]
    all_gts = gts

    from fsod.eval_clip_fusion import postprocess

    rows = []
    for t in [float(x) for x in args.thresholds.split(",")]:
        model.roi_heads.score_thresh = t
        all_preds = []
        for cache in caches:
            all_preds.extend(postprocess(model, cache, cache["logits"], dev))
        n_det = sum(len(p["scores"]) for p in all_preds) / max(1, len(all_preds))
        _, nm = engine.report_ap(all_preds, all_gts, novel_names, tag=f"novel thr={t}")
        _, bm = engine.report_ap(all_preds, all_gts, base_names, tag=f"base thr={t}")
        _, nsm = engine.report_ap(all_preds, all_gts, novel_names,
                                  tag=f"novel-small thr={t}", area_range=engine.AREA_BINS["small"])
        _, bsm = engine.report_ap(all_preds, all_gts, base_names,
                                  tag=f"base-small thr={t}", area_range=engine.AREA_BINS["small"])
        rows.append({"thresh": t, "novel_mAP": nm, "base_mAP": bm,
                     "novel_small": nsm, "base_small": bsm, "det_per_img": n_det})
        print(f"== thr={t}: novel {nm*100:.2f} base {bm*100:.2f} | small novel {nsm*100:.2f} "
              f"small base {bsm*100:.2f} | det/img {n_det:.1f}", flush=True)

    model.roi_heads.score_thresh = default_thresh
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "split": args.split}, f, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
