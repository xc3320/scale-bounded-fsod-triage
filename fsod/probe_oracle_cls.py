"""决定性诊断：假设分类头完美，small AP 的天花板是多少？

用已落盘缓存，把每个 proposal 的 logits 强制为它匹配的 GT 类（IoU>=thr），
未匹配设为背景，然后跑 postprocess + AP。定位(reg)不变。

若 oracle 分类下 small AP 依然低 -> 瓶颈在 RPN/定位，分类侧方法救不了 small，停止攻 small。
若 oracle 分类下 small AP 很高 -> 瓶颈在分类，值得继续攻（CLIP 只是还不够强）。
"""
import argparse
import os
import numpy as np
import torch

from fsod import voc, engine
from fsod.model import build_model
from fsod.ap import compute_ap_per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--ckpt", default="runs/ft2_ctrl_base/ft.pth")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default="data/oracle_cls_probe.json")
    args = ap.parse_args()

    here = os.getcwd()
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel = sp["novel"]
    novel_labs = [ft_map[c] for c in novel]

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    dev = torch.device("cuda")
    n_all = len(ft_map)
    model = build_model(n_all + 1, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    def iou_xyxy(a, b):
        ix1 = torch.max(a[:, None, 0], b[None, :, 0]); iy1 = torch.max(a[:, None, 1], b[None, :, 1])
        ix2 = torch.min(a[:, None, 2], b[None, :, 2]); iy2 = torch.min(a[:, None, 3], b[None, :, 3])
        iw = (ix2-ix1).clamp(min=0); ih = (iy2-iy1).clamp(min=0)
        inter = iw*ih
        aa = (a[:, 2]-a[:, 0]).clamp(min=0)*(a[:, 3]-a[:, 1]).clamp(min=0)
        bb = (b[:, 2]-b[:, 0]).clamp(min=0)*(b[:, 3]-b[:, 1]).clamp(min=0)
        return inter/(aa[:, None]+bb[None, :]-inter).clamp(min=1e-6)

    import contextlib, io
    def ap_of(preds, labs, area=None):
        with contextlib.redirect_stdout(io.StringIO()):
            aps = compute_ap_per_class(preds, gts, labs, area_range=area)
        return float(np.mean(list(aps.values()))) if aps else 0.0

    # baseline preds（用缓存 logits 原样）
    def collect(mode):
        all_preds = []
        for ci, cache in enumerate(caches):
            logits = cache["logits"].clone()
            if mode == "oracle":
                for bi in range(len(cache["proposals"])):
                    props = cache["proposals"][bi].float()
                    oh, ow = cache["orig_sizes"][bi]; rh, rw = cache["image_sizes"][bi]
                    sx, sy = ow/rw, oh/rh
                    props_orig = props * torch.tensor([sx, sy, sx, sy])
                    gi = sum(len(c["proposals"]) for c in caches[:ci]) + bi
                    gt = gts[gi]
                    gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
                    gl = np.asarray(gt["labels"], dtype=np.int64).tolist()
                    gd = np.asarray(gt["difficult"], dtype=np.int64).tolist() if "difficult" in gt else [0]*len(gl)
                    valid = [k for k in range(len(gb)) if gd[k] == 0]
                    off = sum(p.shape[0] for p in cache["proposals"][:bi])
                    np_ = props.shape[0]
                    lg = logits[off:off+np_]
                    lg[:] = -30.0
                    if valid and props_orig.shape[0] > 0:
                        gbt = torch.from_numpy(gb[valid])
                        ious = iou_xyxy(props_orig, gbt)          # (np, nvalid)
                        best_i, best_j = ious.max(1)
                        for p in range(np_):
                            if float(best_i[p]) >= args.iou:
                                lg[p, gl[valid[int(best_j[p])]]] = 30.0
                            else:
                                lg[p, 0] = 30.0
                    else:
                        lg[:, 0] = 30.0
            # postprocess 整批
            ab, asc, al = model.roi_heads.postprocess_detections(
                logits.to(dev), cache["reg"].to(dev),
                [p.to(dev) for p in cache["proposals"]], cache["image_sizes"])
            for i, (b, s, l) in enumerate(zip(ab, asc, al)):
                oh, ow = cache["orig_sizes"][i]; rh, rw = cache["image_sizes"][i]
                b = b * torch.tensor([ow/rw, oh/rh, ow/rw, oh/rh], device=b.device)
                all_preds.append({"boxes": b.cpu().numpy(), "scores": s.cpu().numpy(),
                                  "labels": l.cpu().numpy()})
        return all_preds

    preds_base = collect("base")
    preds_orc = collect("oracle")
    novel_names = [ft_map[c] for c in novel]
    print("\n=== 分类天花板诊断（定位不变）===")
    print(f"{'scale':8s} {'baseline':>9s} {'oracle_cls':>10s} {'gap':>8s}")
    for sc in ["small", "medium", "large"]:
        a = ap_of(preds_base, novel_labs, engine.AREA_BINS[sc])
        b = ap_of(preds_orc, novel_labs, engine.AREA_BINS[sc])
        print(f"{sc:8s} {a*100:9.2f} {b*100:10.2f} {(b-a)*100:+8.2f}")
    a = ap_of(preds_base, novel_labs); b = ap_of(preds_orc, novel_labs)
    print(f"{'overall':8s} {a*100:9.2f} {b*100:10.2f} {(b-a)*100:+8.2f}")

    res = {"iou_thr": args.iou,
           "small": {"base": ap_of(preds_base, novel_labs, engine.AREA_BINS["small"]),
                     "oracle": ap_of(preds_orc, novel_labs, engine.AREA_BINS["small"])}}
    for sc in ["medium", "large"]:
        res[sc] = {"base": ap_of(preds_base, novel_labs, engine.AREA_BINS[sc]),
                   "oracle": ap_of(preds_orc, novel_labs, engine.AREA_BINS[sc])}
    with open(args.out, "w") as f:
        import json
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
