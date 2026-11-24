"""Phase 5：把分诊做成真算法 + label-free λ 选择（论文4 投稿前突破）。

复用 eval_clip_fusion 的 forward+CLIP 缓存，纯评测、不训练。
四种模式对照：
  - gate0   : 基线，所有 proposal 语义门控（现状，λ 人工最优）
  - tri_hard: small proposal 关语义门控(权重0)，medium/large 开(权重1)
  - tri_soft: 门控权重 = min(1, area/96^2) 连续递增
  - lambda_auto: 用 base 类验证选 λ（label-free on novel），再用于 novel

判据（立项书 §5 Phase 5 冻结，2026-11-24）：
  D1 分诊有效性: tri_hard/tri_soft 的 novel mAP >= gate0 同 λ (不损失), 
                 且 small novel AP 不高于 gate0; 任一行 < -1pp 判 FAIL。
  D2 λ自动化: lambda_auto 的 novel mAP 相对人工最优落差 <= 2pp 且 base 不掉。

用法:
  PYTHONPATH=. env_fsl/Scripts/python.exe -m fsod.eval_phase5_triage \
      --ckpt runs/ft_s1_ctrl_base/ft.pth --data 0712 --split 1 --shot 5 \
      --cache data/clip_fusion_cache_full.pt --out data/phase5_triage_s1.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from fsod import voc, engine
from fsod.model import build_model
from fsod.ap import compute_ap_per_class


def proposal_scale_weight(cache, gi, mode):
    """返回该图每个 proposal 的尺度门控权重 (n_i,)。
    proposals 是 resized 坐标 → 乘 orig/image 缩放得原图框 → 面积 → 分箱权重。"""
    props = cache["proposals"][gi]                       # (n_i,4) resized
    oh, ow = cache["orig_sizes"][gi]
    rh, rw = cache["image_sizes"][gi]
    sx, sy = ow / rw, oh / rh
    bw = (props[:, 2] - props[:, 0]) * sx
    bh = (props[:, 3] - props[:, 1]) * sy
    area = torch.clamp(bw, min=0) * torch.clamp(bh, min=0)
    if mode == "gate0":
        return torch.ones(props.shape[0])
    if mode == "tri_hard":
        # small(<32^2) 弃权; medium/large 放行
        return (area >= 32.0 ** 2).float()
    if mode == "tri_soft":
        # 连续递增: 小→弱、大→强; 下限0.1 避免完全掐断
        return torch.clamp(area / (96.0 ** 2), min=0.0, max=1.0) * 0.9 + 0.1
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--cache", required=True)
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
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    n_all = len(ft_map)
    novel = sp["novel"]
    novel_labs = [ft_map[c] for c in novel]
    novel_idx = [l - 1 for l in novel_labs]
    base = sp["base"]
    base_labs = [ft_map[c] for c in base]
    novel_names = {ft_map[c]: c for c in novel}
    base_names = {ft_map[c]: c for c in base}
    dev = torch.device("cuda")

    # 加载缓存
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    caches, sims, all_gts = blob["caches"], blob["sims"], blob["gts"]
    ptr = blob["n_imgs"]
    # 重建 orig_sizes（兼容旧缓存）
    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    gi = 0
    for c in caches:
        if "orig_sizes" not in c:
            sizes = []
            for _ in c["proposals"]:
                smp = te_samples[gi]
                w, h = ann[smp["image_id"]]["size"]
                sizes.append((h, w)); gi += 1
            c["orig_sizes"] = sizes
        else:
            gi += len(c["proposals"])
    print(f"[cache] {ptr} imgs, novel={novel}", flush=True)

    # 重建 detector 用于 postprocess
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(n_all + 1, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    from fsod.eval_clip_fusion import postprocess
    import contextlib, io

    def run(lam, mode, classes_for_ap):
        """对给定 λ/模式跑一遍，返回指定类集合的 mAP 与 novel 分尺度 AP。"""
        all_preds = []
        for ci, cache in enumerate(caches):
            logits = cache["logits"].clone()
            sc = sims[ci]
            sizes = [p.shape[0] for p in cache["proposals"]]
            lchunks, off = [], 0
            for ns in sizes:
                lchunks.append(logits[off:off + ns]); off += ns
            for gimg, s in enumerate(sc):
                lg = lchunks[gimg]
                if lam > 0:
                    base_idx = [li for li in range(n_all) if li not in novel_idx]
                    nsim = s[:, novel_idx].max(1).values
                    bsim = s[:, base_idx].max(1).values
                    gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
                    sw = proposal_scale_weight(cache, gimg, mode)   # per-proposal 尺度权重
                    eff = gate * sw
                    for li in novel_idx:
                        lg[:, li + 1] = lg[:, li + 1] + lam * eff * s[:, li]
            all_preds.extend(postprocess(model, cache, logits, dev))
        with contextlib.redirect_stdout(io.StringIO()):
            named, mm = engine.report_ap(all_preds, all_gts, classes_for_ap)
        return mm, all_preds

    def ap_of(preds, labs, area=None):
        aps = compute_ap_per_class(preds, all_gts, labs, area_range=area)
        return float(np.mean(list(aps.values()))) if aps else 0.0

    # ---- baseline lam=0 ----
    mm0, preds0 = run(0, "gate0", novel_names)
    with contextlib.redirect_stdout(io.StringIO()):
        _, base0 = engine.report_ap(preds0, all_gts, base_names)
    b_small0 = ap_of(preds0, novel_labs, engine.AREA_BINS["small"])
    print(f"[baseline] novel={mm0*100:.2f} base={base0*100:.2f} small_novel={b_small0*100:.2f}", flush=True)

    results = {"split": args.split, "baseline": {
        "novel_mAP": mm0, "base_mAP": base0, "small_novel_AP": b_small0}}

    # ---- λ 自动化：用 base 类验证选 λ（label-free on novel）----
    lam_grid = [3, 6, 10, 15]
    base_by_lam = {}
    for lam in lam_grid:
        _, preds_b = run(lam, "gate0", base_names)
        with contextlib.redirect_stdout(io.StringIO()):
            _, bm = engine.report_ap(preds_b, all_gts, base_names)
        base_by_lam[lam] = bm
    # 选 base mAP 不降且 base 增益最大的 λ
    valid = [(lam, bm - base0) for lam, bm in base_by_lam.items() if bm >= base0 - 1e-6]
    lam_auto = max(valid, key=lambda x: x[1])[0] if valid else 0
    results["lambda_auto"] = {"base_by_lam": base_by_lam, "chosen_lam": lam_auto,
                              "base_gain": base_by_lam.get(lam_auto, base0) - base0}

    # ---- 人工最优 λ（对照上界）：从现有 sweep 结果读 ----
    # 各 split 人工最优: s1=10, s2=6, s3=10（冻结的既有事实）
    lam_manual = {1: 10, 2: 6, 3: 10}[args.split]

    # ---- 三模式对照：用人工最优 λ（隔离分诊效果）----
    for mode in ["gate0", "tri_hard", "tri_soft"]:
        mm, preds = run(lam_manual, mode, novel_names)
        with contextlib.redirect_stdout(io.StringIO()):
            _, bm = engine.report_ap(preds, all_gts, base_names)
        sm = ap_of(preds, novel_labs, engine.AREA_BINS["small"])
        results[mode] = {"lam": lam_manual, "novel_mAP": mm, "base_mAP": bm,
                         "small_novel_AP": sm, "delta_novel_pp": (mm - mm0) * 100,
                         "delta_small_pp": (sm - b_small0) * 100}
        print(f"[{mode:9s} lam{lam_manual}] novel={mm*100:.2f} ({100*(mm-mm0):+5.2f}) "
              f"base={bm*100:.2f} small={sm*100:.2f} ({100*(sm-b_small0):+5.2f})", flush=True)

    # ---- lambda_auto 的 novel 效果（用自动 λ + 最佳分诊模式）----
    if lam_auto > 0:
        mm_a, preds_a = run(lam_auto, "tri_soft", novel_names)
        with contextlib.redirect_stdout(io.StringIO()):
            _, bm_a = engine.report_ap(preds_a, all_gts, base_names)
        results["auto_final"] = {"lam": lam_auto, "mode": "tri_soft",
                                 "novel_mAP": mm_a, "base_mAP": bm_a,
                                 "delta_novel_pp": (mm_a - mm0) * 100}
        print(f"[auto_final lam{lam_auto} tri_soft] novel={mm_a*100:.2f} ({100*(mm_a-mm0):+5.2f}) "
              f"base={bm_a*100:.2f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
