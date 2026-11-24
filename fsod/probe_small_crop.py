"""诊断：small 目标上，CLIP crop 用 GT 框 vs 用 RPN proposal 框的判别力差距。

假说：probe_clip_rescue 用 GT 框裁剪 small clip5_top1=71.7%，
但 eval_clip_fusion 用 proposal 框裁剪，RPN 小目标 proposal 偏移导致 crop 背景污染，
这才是 small 无增益的根因（而非门控判据）。

对每个 novel GT 框：找 IoU 最高 proposal，分别用 GT 框和 proposal 框 crop -> CLIP，
算在 5 个 novel 原型 + 20 类里的 top1。只重编码 small GT 对应的框（秒级）。
"""
import argparse
import os
import numpy as np
import torch

from fsod import voc, engine, eval_clip_fusion as ecf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/clip_fusion_cache_full.pt")
    ap.add_argument("--ckpt", default="runs/ft2_ctrl_base/ft.pth")
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--out", default="data/small_crop_probe.json")
    args = ap.parse_args()

    here = os.getcwd()
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    vr = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    v12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    ann, tr, te = voc.prepare_0712(vr, v12, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    novel_labs = [ft_map[c] for c in sp["novel"]]
    novel_idx = [c - 1 for c in novel_labs]

    from fsod.finetune import build_fewshot_set
    ft_samples, _ = build_fewshot_set(ann, tr, ft_map, sp["base"], sp["novel"], args.shot, seed=0)
    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)

    import clip
    dev = torch.device("cuda")
    clip_model, _ = clip.load(os.path.join(here, "torch_home", "clip", "ViT-B-32.pt"), device=dev)
    clip_model.eval()

    # 支持集 -> 20 类原型（与 probe_clip_rescue 一致）
    sup_crops, sup_labs = [], []
    for s in ft_samples:
        lab = max(s["labels"])
        im = __import__("PIL.Image", fromlist=["Image"]).open(s["file"]).convert("RGB")
        x1, y1, x2, y2 = s["boxes"][0]
        crop = im.crop((int(x1), int(y1), int(x2), int(y2))).resize((224, 224))
        a = np.asarray(crop, dtype=np.float32) / 255.0
        a = (a - np.array(ecf.CLIP_MEAN)) / np.array(ecf.CLIP_STD)
        sup_crops.append(torch.from_numpy(a.transpose(2, 0, 1).copy()))
        sup_labs.append(int(lab))
    proto_mat = []
    for lb in range(1, 21):
        idxs = [i for i, l in enumerate(sup_labs) if l == lb]
        if not idxs:
            proto_mat.append(torch.zeros(512)); continue
        f = ecf.encode_crops(clip_model, [sup_crops[i] for i in idxs], dev).mean(0)
        proto_mat.append(f)
    proto_mat = torch.stack(proto_mat).to(dev)  # (20,512)

    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, gts = blob["caches"], blob["gts"]

    def iou_xyxy(a, b):
        ix1 = torch.max(a[:, None, 0], b[None, :, 0]); iy1 = torch.max(a[:, None, 1], b[None, :, 1])
        ix2 = torch.min(a[:, None, 2], b[None, :, 2]); iy2 = torch.min(a[:, None, 3], b[None, :, 3])
        iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
        inter = iw * ih
        aa = (a[:, 2]-a[:, 0]).clamp(min=0)*(a[:, 3]-a[:, 1]).clamp(min=0)
        bb = (b[:, 2]-b[:, 0]).clamp(min=0)*(b[:, 3]-b[:, 1]).clamp(min=0)
        return inter / (aa[:, None]+bb[None, :]-inter).clamp(min=1e-6)

    gt_crops, prop_crops, gt_labs = [], [], []
    gi = 0
    for ci, cache in enumerate(caches):
        for bi in range(len(cache["proposals"])):
            props = cache["proposals"][bi].float()
            oh, ow = cache["orig_sizes"][bi]
            rh, rw = cache["image_sizes"][bi]
            sx, sy = ow / rw, oh / rh
            props_orig = props * torch.tensor([sx, sy, sx, sy])
            gt = gts[gi]; gi += 1
            gb = np.asarray(gt["boxes"], dtype=np.float32).reshape(-1, 4)
            gl = np.asarray(gt["labels"], dtype=np.int64)
            gd = np.asarray(gt["difficult"], dtype=np.int64) if "difficult" in gt else np.zeros(len(gl))
            smp = te_samples[gi - 1] if gi - 1 < len(te_samples) else None
            path = smp["file"] if smp else ""
            for j in range(len(gb)):
                if gl[j] not in novel_labs or gd[j] != 0:
                    continue
                x1, y1, x2, y2 = gb[j]
                area = max(0, x2-x1)*max(0, y2-y1)
                if area >= engine.AREA_BINS["small"][1]:
                    continue
                if not path:
                    continue
                # GT 框 crop
                gt_crops.append(ecf.crop_tensor(path, [x1, y1, x2, y2]))
                gt_labs.append(int(gl[j]))
                # proposal 框 crop（IoU 最高）
                gbt = torch.from_numpy(gb[j:j+1])
                if props_orig.shape[0] > 0:
                    ious = iou_xyxy(props_orig, gbt).squeeze(1)
                    pi = int(ious.argmax())
                    px1, py1, px2, py2 = props_orig[pi].tolist()
                    prop_crops.append(ecf.crop_tensor(path, [px1, py1, px2, py2]))
                else:
                    prop_crops.append(gt_crops[-1])
    print(f"[probe] small novel GT n={len(gt_crops)}", flush=True)

    fg = ecf.encode_crops(clip_model, gt_crops, dev).to(dev)     # (n,512)
    fp = ecf.encode_crops(clip_model, prop_crops, dev).to(dev)   # (n,512)

    def top1(feat):
        sim = feat @ proto_mat.T
        return {
            "clip20": float((sim.argmax(1).cpu().numpy() + 1 == np.array(gt_labs)).mean()),
            "clip5": float((sim[:, novel_idx].argmax(1).cpu().numpy() == np.array([novel_idx.index(l-1) for l in gt_labs])).mean()),
            "novel_win": float((sim[:, novel_idx].max(1).values > sim[:, [i for i in range(20) if i not in novel_idx]].max(1).values).float().mean()),
            "nsim": float(sim[:, novel_idx].max(1).values.mean()),
            "margin": float((sim[:, novel_idx].max(1).values - sim[:, [i for i in range(20) if i not in novel_idx]].max(1).values).mean()),
        }

    res = {"n": len(gt_crops), "gt_box": top1(fg), "proposal_box": top1(fp)}
    print("\n=== small novel: GT框 vs proposal框 crop ===")
    for k in ["gt_box", "proposal_box"]:
        r = res[k]
        print(f"{k:14s} clip20={r['clip20']:.3f} clip5={r['clip5']:.3f} "
              f"novel_win={r['novel_win']:.3f} nsim={r['nsim']:.3f} margin={r['margin']:+.3f}")
    with open(args.out, "w") as f:
        import json
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
