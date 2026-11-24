"""Stage-2：K-shot 平衡微调（TFA 协议）+ novel mAP 评测。

用法：
  python -m fsod.finetune --split 1 --shot 5 --ckpt runs/base_s1/last.pth --out runs/ft_s1_5shot
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fsod import voc, engine
from fsod.data import VOCDetDataset, detection_collate
from fsod.model import expand_classifier, freeze_backbone_and_rpn, count_params


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_fewshot_set(ann_all, train_ids, ft_map, base, novel, shot, seed=0):
    """每类取 shot 张含该类目标的图；返回样本列表（图内保留全部类别标注）。"""
    rng = random.Random(seed)
    all_samples = voc.samples_for(ann_all, train_ids, ft_map, keep_difficult=True)
    # 每类 -> 含该类的样本下标
    by_class = {}
    for idx, s in enumerate(all_samples):
        for lb in set(s["labels"]):
            by_class.setdefault(lb, []).append(idx)
    chosen = set()
    picks = {}
    for cname in base + novel:
        lb = ft_map[cname]
        pool = sorted(by_class.get(lb, []))
        if len(pool) == 0:
            raise RuntimeError(f"class {cname} has no image")
        k = min(shot, len(pool))
        sel = rng.sample(pool, k)
        picks[cname] = len(sel)
        chosen.update(sel)
    out = [all_samples[i] for i in sorted(chosen)]
    return out, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--data", type=str, default="07", choices=["07", "0712"])
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--head", type=str, default="tfa", choices=["tfa", "bypass"])
    ap.add_argument("--basis", type=str, default="", help="PCA 基 .pt（bypass 用）")
    ap.add_argument("--bypass-mode", type=str, default="pca", choices=["pca", "random", "frozen"])
    ap.add_argument("--bypass-mid", type=int, default=1024)
    ap.add_argument("--train-proj", type=int, default=0)
    ap.add_argument("--eval-gt", type=int, default=0, help="1=额外评测 oracle-proposal nAP")
    ap.add_argument("--prop-recall", type=int, default=0, help="1=额外统计 RPN 召回")
    ap.add_argument("--diag-head", type=int, default=0, help="1=在 GT RoI 上直接诊断分类头")
    ap.add_argument("--skip-eval", type=int, default=0, help="1=跳过全量 AP 评测（调参用）")
    ap.add_argument("--train-rpn", type=int, default=0, help="1=微调时解冻 RPN")
    ap.add_argument("--freeze-boxhead", type=int, default=0, help="1=冻结 box_head 只训分类头")
    ap.add_argument("--diag-train", type=int, default=0, help="1=在微调集自身上做分类头诊断（过拟合测试）")
    ap.add_argument("--fg-margin", type=float, default=0.0,
                    help=">0 启用训练时前景边距损失（SAFM）；0=原始 TFA")
    ap.add_argument("--margin-lam", type=float, default=1.0, help="边距辅助损失权重")
    ap.add_argument("--margin-val", type=float, default=1.0, help="边距 margin")
    ap.add_argument("--margin-thr", type=int, default=32, help="小目标面积阈值（原图像素边长）")
    ap.add_argument("--margin-novel-only", type=int, default=1, help="1=只对 novel 正样本加边距")
    ap.add_argument("--margin-scale-cond", type=int, default=1, help="1=只对小 RoI 加边距")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")

    voc12_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    if getattr(args, "data", "07") == "0712":
        ann_all, train_ids, test_ids = voc.prepare_0712(voc_root, voc12_root, os.path.join(here, "data"))
    else:
        ann_all, train_ids, test_ids = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann_all, train_ids, test_ids, args.split)
    ft_map = sp["ft_map"]
    n_base = len(sp["base"])
    n_all = len(ft_map)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    from fsod.model import build_model
    model = build_model(n_base + 1, ck_args.get("backbone", "resnet50"),
                        ck_args.get("min_size", 600), ck_args.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False)
    model.load_state_dict(ck["model"])
    if args.head == "tfa":
        model = expand_classifier(model, n_all + 1, n_base + 1)
        model = freeze_backbone_and_rpn(model, freeze_rpn=not bool(args.train_rpn))
        if args.freeze_boxhead:
            for p in model.roi_heads.box_head.parameters():
                p.requires_grad = False
    else:
        from fsod.bypass import install_bypass, freeze_bypass_tuning
        basis = None
        if args.bypass_mode == "pca":
            assert args.basis, "--basis required for pca mode"
            obj = torch.load(args.basis, map_location="cpu", weights_only=False)
            basis = obj["V"]
        in_dim = 256 * 7 * 7
        model = install_bypass(model, in_dim, n_all + 1, n_base + 1,
                               mid=args.bypass_mid, mode=args.bypass_mode, init_basis=basis)
        model = freeze_bypass_tuning(model, train_proj=bool(args.train_proj))
    model.to(device)
    if args.fg_margin > 0:
        from fsod.margin_loss import LogitCapture
        LogitCapture(model)
        print(f"[SAFM] margin={args.fg_margin} lam={args.margin_lam} thr={args.margin_thr} "
              f"novel_only={args.margin_novel_only} scale_cond={args.margin_scale_cond}", flush=True)
    print(f"[model] head={args.head} classes={n_all+1} trainable={count_params(model)/1e6:.2f}M", flush=True)

    ft_samples, picks = build_fewshot_set(ann_all, train_ids, ft_map, sp["base"], sp["novel"],
                                          args.shot, seed=args.seed)
    te_samples = voc.samples_for(ann_all, test_ids, ft_map, keep_difficult=True)
    print(f"[data] shot={args.shot} ft_imgs={len(ft_samples)} per_class={picks} "
          f"test_imgs={len(te_samples)}", flush=True)

    ds = VOCDetDataset(ft_samples, train=True, hflip=0.5)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0,
                        collate_fn=detection_collate, drop_last=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2, shuffle=False,
                           num_workers=max(2, args.workers // 2), collate_fn=detection_collate,
                           pin_memory=True, persistent_workers=args.workers > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))

    # backbone 冻结 -> 必须锁在 eval，否则 BN running stats 会被少样本batch污染
    model.train()
    model.backbone.eval()

    t0 = time.time()
    it_done = 0
    aux_hits = 0
    aux_n_total = 0
    ep = 0
    while it_done < args.iters:
        ep += 1
        before = it_done
        for images, targets in loader:
            if it_done >= args.iters:
                break
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            aux_info = None
            if args.fg_margin > 0:
                model._cap.clear()  # 防止跨迭代持有计算图张量
            with torch.amp.autocast("cuda", enabled=bool(args.amp)):
                loss_dict = model(images, targets)
                # 冻结 RPN 时不回传 rpn loss；解冻时必须回传，否则 RPN 永远不更新
                if args.train_rpn:
                    loss = sum(loss_dict.values())
                else:
                    loss = sum(v for k, v in loss_dict.items() if "rpn" not in k)
                aux_info = None
                if args.fg_margin > 0:
                    from fsod.margin_loss import margin_aux
                    out = margin_aux(model, images, n_base + 1,
                                     margin=args.margin_val,
                                     small_thr=args.margin_thr ** 2,
                                     novel_only=bool(args.margin_novel_only),
                                     scale_cond=bool(args.margin_scale_cond))
                    if out is not None:
                        aux, aux_n = out
                        loss = loss + args.margin_lam * aux
                        aux_info = (float(aux.detach()), aux_n)
                        aux_hits += 1
                        aux_n_total += aux_n
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            it_done += 1
            if it_done % 50 == 0:
                print(f"  ft it{it_done}/{args.iters} loss={loss.item():.4f} "
                      f"cls={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.4f} "
                      f"box={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f} "
                      f"aux={aux_info[0]:.4f}(n={aux_info[1]})" if aux_info else
                      f"  ft it{it_done}/{args.iters} loss={loss.item():.4f} "
                      f"cls={loss_dict.get('loss_classifier', torch.tensor(0.)).item():.4f} "
                      f"box={loss_dict.get('loss_box_reg', torch.tensor(0.)).item():.4f}", flush=True)
        if it_done == before:
            break
    print(f"[ft] {args.iters} iters in {(time.time()-t0)/60:.1f} min", flush=True)
    if args.fg_margin > 0:
        print(f"[SAFM] applied on {aux_hits}/{args.iters} iters, {aux_n_total} RoIs total", flush=True)
    torch.save({"model": model.state_dict(), "ft_map": ft_map, "split": args.split,
                "shot": args.shot, "args": vars(args)}, os.path.join(args.out, "ft.pth"))

    novel_names = {ft_map[c]: c for c in sp["novel"]}
    base_names = {ft_map[c]: c for c in sp["base"]}
    res = {"split": args.split, "shot": args.shot, "tag": args.tag,
           "iters": args.iters, "lr": args.lr}

    if args.diag_train:
        ft_loader = DataLoader(VOCDetDataset(ft_samples, train=False), batch_size=2,
                               shuffle=False, num_workers=0, collate_fn=detection_collate)
        dt = engine.head_diagnostic(model, ft_loader, ft_samples, device,
                                    list(novel_names.keys()), list(base_names.keys()), max_imgs=10**6)
        res["head_diag_train"] = dt
        print("[head-diag-train] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in dt.items()), flush=True)
    if args.diag_head:
        d = engine.head_diagnostic(model, te_loader, te_samples, device,
                                   list(novel_names.keys()), list(base_names.keys()))
        res["head_diag"] = d
        print("[head-diag] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in d.items()), flush=True)
    if args.skip_eval:
        with open(os.path.join(args.out, "result.json"), "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(json.dumps(res.get("head_diag", {}), ensure_ascii=False), flush=True)
        return

    p, g = engine.evaluate(model, te_loader, te_samples, device, amp=bool(args.amp))
    named_n, m_n = engine.report_ap(p, g, novel_names, tag=f"novel {args.tag}")
    named_b, m_b = engine.report_ap(p, g, base_names, tag=f"base  {args.tag}")
    scale_n = engine.report_ap_by_scale(p, g, novel_names, tag=f"novel-scale {args.tag}")
    res.update({"novel_mAP50": m_n, "base_mAP50": m_b,
                "novel_ap": named_n, "base_ap": named_b,
                "novel_scale": {k: v["mAP50"] for k, v in scale_n.items()},
                "novel_scale_detail": scale_n})
    if args.eval_gt:
        pg, gg = engine.evaluate_gt_proposals(model, te_loader, te_samples, device, amp=bool(args.amp))
        _, m_g = engine.report_ap(pg, gg, novel_names, tag=f"novel-GTprop {args.tag}")
        res["novel_mAP50_gtprop"] = m_g
        _, m_gb = engine.report_ap(pg, gg, base_names, tag=f"base-GTprop {args.tag}")
        res["base_mAP50_gtprop"] = m_gb
    if args.prop_recall:
        rec, tot = engine.proposal_recall(model, te_loader, te_samples, device)
        print(f"[prop-recall {args.tag}] {rec} tot={tot}", flush=True)
        res["prop_recall"] = rec
        res["prop_recall_total"] = tot
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: (round(v * 100, 2) if isinstance(v, float) else v)
                      for k, v in res.items() if k in ("novel_mAP50", "base_mAP50")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
