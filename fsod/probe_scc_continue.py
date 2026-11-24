"""最小闭环：从同一收敛 base checkpoint 出发，短训 N iter，唯一变量=是否加 SCC。

用法：
  python -m fsod.probe_scc_continue --ckpt runs/base_s1_0712/ep10.pth \
      --data 0712 --out runs/scc_ctrl --scc 0 --iters 1200 --lr 3e-4
  python -m fsod.probe_scc_continue --ckpt runs/base_s1_0712/ep10.pth \
      --data 0712 --out runs/scc_on --scc 1 --iters 1200 --lr 3e-4
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from fsod import voc, engine
from fsod.model import build_model
from fsod.data import VOCDetDataset, detection_collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scc", type=int, default=0)
    ap.add_argument("--scc-expand", type=float, default=2.0)
    ap.add_argument("--scc-lam", type=float, default=1.0)
    ap.add_argument("--scc-mode", type=str, default="sup", choices=["sup", "distill"])
    ap.add_argument("--scc-margin", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    voc12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    os.makedirs(args.out, exist_ok=True)
    if args.data == "0712":
        ann, tr, te = voc.prepare_0712(voc_root, voc12, os.path.join(here, "data"))
    else:
        ann, tr, te = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, 1)
    tr_samples = voc.samples_for(ann, tr, sp["base_map"], keep_difficult=True)

    import random
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(16, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])

    aux_fn = None
    if args.scc:
        from fsod.scc import ScaleContextConsistency
        scc = ScaleContextConsistency(model, expand=args.scc_expand, lam=args.scc_lam,
                                      mode=args.scc_mode, margin=args.scc_margin)
        aux_fn = scc.aux
        print(f"[SCC] expand={args.scc_expand} lam={args.scc_lam} "
              f"mode={args.scc_mode} margin={args.scc_margin}", flush=True)

    ds = VOCDetDataset(tr_samples, train=True)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0,
                    collate_fn=detection_collate, drop_last=True)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    t0 = time.time()
    it = 0
    model.train()
    while it < args.iters:
        for imgs, tgts in dl:
            if it >= args.iters:
                break
            imgs = [x.to(dev, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            if hasattr(model, "_cap"):
                model._cap.clear()
            with torch.amp.autocast("cuda", enabled=True):
                ld = model(imgs, tgts)
                loss = sum(ld.values())
                if aux_fn is not None:
                    out = aux_fn()
                    if out is not None:
                        loss = loss + out[0]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
            scaler.step(opt); scaler.update()
            it += 1
            if it % 200 == 0:
                print(f"  it{it}/{args.iters} loss={float(loss):.4f} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
    torch.save({"model": model.state_dict(), "args": cka, "probe_args": vars(args)},
               os.path.join(args.out, "last.pth"))
    print(f"[done] scc={args.scc} {args.iters}it in {(time.time()-t0)/60:.1f}min", flush=True)

    # 进程内直接诊断 base 类分尺度判背景率（免去二次加载）
    te_samples = voc.samples_for(ann, te, sp["base_map"], keep_difficult=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2,
                           shuffle=False, num_workers=2, collate_fn=detection_collate)
    d = engine.head_diagnostic(model, te_loader, te_samples, dev,
                               novel_labels=list(range(1, 16)), base_labels=[], max_imgs=400)
    print(f"[DIAG scc={args.scc}] base_small tobg={d['novel_tobg_small']:.3f} "
          f"top1={d['novel_top1_small']:.3f} n={d['n_novel_small']} | "
          f"med tobg={d['novel_tobg_medium']:.3f} | large tobg={d['novel_tobg_large']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
