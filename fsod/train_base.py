"""Stage-1：base 类训练（VOC2007 trainval，仅 base 15 类）。

用法：
  python -m fsod.train_base --split 1 --epochs 12 --bs 4 --lr 0.02 --out runs/base_s1
"""
import argparse
import json
import os
import random
import sys
import time
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from fsod import voc
from fsod.data import VOCDetDataset, detection_collate
from fsod.model import build_model, build_optimizer, count_params
from fsod import engine


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def multistep_lr(opt, base_lrs, iters_per_epoch, total_epochs, warmup_iters,
                 milestones=(0.7, 0.9), gamma=0.1):
    """warmup + multistep 的按迭代调度器。"""
    tot = iters_per_epoch * total_epochs

    def fn(it):
        if it < warmup_iters:
            a = (it + 1) / max(warmup_iters, 1)
            return a
        prog = it / max(tot, 1)
        m = 1.0
        for ms in milestones:
            if prog > ms:
                m *= gamma
        return m

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--min-size", type=int, default=600)
    ap.add_argument("--max-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help=">0 则每 N epoch 在 test 上评估 base AP")
    ap.add_argument("--max-iters", type=int, default=0, help="调试用：每 epoch 限制迭代数")
    ap.add_argument("--resume", type=int, default=0, help="1=从 out 目录的 last.pth 续训")
    ap.add_argument("--data", type=str, default="07", choices=["07", "0712"],
                    help="07=VOC2007 trainval；0712=VOC2012 trainval（文献标准协议），评测均为 VOC2007 test")
    ap.add_argument("--scc", type=int, default=0,
                    help="1=启用 base 阶段尺度-上下文一致性训练（SCC）")
    ap.add_argument("--scc-expand", type=float, default=2.0, help="SCC 扩张框倍数")
    ap.add_argument("--scc-lam", type=float, default=1.0, help="SCC 辅助损失权重")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    voc12_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    os.makedirs(args.out, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "0712":
        ann_all, train_ids, test_ids = voc.prepare_0712(voc_root, voc12_root, os.path.join(here, "data"))
    else:
        ann_all, train_ids, test_ids = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann_all, train_ids, test_ids, args.split)
    base_map = sp["base_map"]
    tr_samples = voc.samples_for(ann_all, train_ids, base_map, keep_difficult=True)
    te_samples = voc.samples_for(ann_all, test_ids, base_map, keep_difficult=True)
    print(f"[data] split{args.split} base={len(base_map)} train_imgs={len(tr_samples)} "
          f"test_imgs={len(te_samples)}", flush=True)

    tr_ds = VOCDetDataset(tr_samples, train=True)
    te_ds = VOCDetDataset(te_samples, train=False)
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                           collate_fn=detection_collate, drop_last=True, pin_memory=True,
                           persistent_workers=args.workers > 0)
    te_loader = DataLoader(te_ds, batch_size=2, shuffle=False, num_workers=max(2, args.workers // 2),
                           collate_fn=detection_collate, pin_memory=True,
                           persistent_workers=args.workers > 0)

    num_classes = len(base_map) + 1
    model = build_model(num_classes, args.backbone, args.min_size, args.max_size,
                        trainable_backbone_layers=5, pretrained=True)
    model.to(device)
    print(f"[model] {args.backbone} num_classes={num_classes} "
          f"trainable={count_params(model)/1e6:.1f}M", flush=True)

    aux_fn = None
    if args.scc:
        from fsod.scc import ScaleContextConsistency
        scc = ScaleContextConsistency(model, expand=args.scc_expand, lam=args.scc_lam)
        aux_fn = scc.aux
        print(f"[SCC] expand={args.scc_expand} lam={args.scc_lam}", flush=True)

    opt = build_optimizer(model, args.lr, backbone_lr_mult=1.0, momentum=0.9, wd=1e-4)
    iters_per_epoch = len(tr_loader) if args.max_iters <= 0 else args.max_iters
    sched = multistep_lr(opt, None, iters_per_epoch, args.epochs, warmup_iters=min(800, iters_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))

    start_ep = 0
    ck_path = os.path.join(args.out, "last.pth")
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        start_ep = int(ck.get("epoch", 0))
        for _ in range(start_ep * iters_per_epoch):
            sched.step()
        print(f"[resume] from epoch {start_ep}, lr now {opt.param_groups[0]['lr']:.5f}", flush=True)

    label_names = {v: k for k, v in base_map.items()}
    log = {"args": vars(args), "epochs": []}
    t_start = time.time()

    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        print(f"=== epoch {ep+1}/{args.epochs} ===", flush=True)
        engine.train_one_epoch(model, opt, tr_loader, device, ep + 1, print_freq=50,
                               amp=bool(args.amp), scaler=scaler, lr_scheduler=sched,
                               iter_limit=args.max_iters if args.max_iters > 0 else None,
                               aux_fn=aux_fn)
        dt = time.time() - t0
        print(f"=== epoch {ep+1} done in {dt/60:.1f} min, total {(time.time()-t_start)/3600:.2f} h ===", flush=True)

        ckpt = {"model": model.state_dict(), "base_map": base_map, "split": args.split,
                "args": vars(args), "epoch": ep + 1}
        torch.save(ckpt, os.path.join(args.out, "last.pth"))
        if (ep + 1) % 4 == 0 or ep + 1 == args.epochs:
            torch.save(ckpt, os.path.join(args.out, f"ep{ep+1:02d}.pth"))

        rec = {"epoch": ep + 1, "minutes": round(dt / 60, 2)}
        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs):
            p, g = engine.evaluate(model, te_loader, te_samples, device, amp=bool(args.amp))
            named, m = engine.report_ap(p, g, label_names, tag=f"base ep{ep+1}")
            rec["base_mAP50"] = m
        log["epochs"].append(rec)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print("DONE base training", flush=True)


if __name__ == "__main__":
    main()
