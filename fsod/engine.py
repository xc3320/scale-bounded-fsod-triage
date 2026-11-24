"""训练 / 评测循环。"""
import math
import sys
import time
import numpy as np
import torch
from .ap import compute_ap_per_class


def train_one_epoch(model, optimizer, loader, device, epoch, print_freq=50,
                    amp=True, scaler=None, grad_clip=10.0, lr_scheduler=None,
                    iter_limit=None, aux_fn=None):
    model.train()
    # backbone 冻结时必须锁 BN 统计量，否则会污染预训练特征
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()
    metric_logger = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "loss_rpn": 0.0,
                     "aux": 0.0, "aux_n": 0, "n": 0}
    t0 = time.time()
    for it, (images, targets) in enumerate(loader):
        if iter_limit is not None and it >= iter_limit:
            break
        images = [im.to(device, non_blocking=True) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 辅助损失 hook（如 SCC）持有的张量含计算图，每迭代开头清空，防止跨迭代持有
        if hasattr(model, "_cap"):
            model._cap.clear()

        if amp:
            with torch.amp.autocast("cuda", enabled=True):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
                aux_val, aux_n = 0.0, 0
                if aux_fn is not None:
                    out = aux_fn()
                    if out is not None:
                        aux_loss, aux_n = out
                        losses = losses + aux_loss
                        aux_val = float(aux_loss.detach())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            aux_val, aux_n = 0.0, 0
            if aux_fn is not None:
                out = aux_fn()
                if out is not None:
                    aux_loss, aux_n = out
                    losses = losses + aux_loss
                    aux_val = float(aux_loss.detach())
            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in loss_dict.items():
            if "loss_classifier" in k:
                metric_logger["loss_cls"] += v.item()
            elif "loss_box_reg" in k:
                metric_logger["loss_box"] += v.item()
            elif "rpn" in k:
                metric_logger["loss_rpn"] += v.item()
        metric_logger["loss"] += losses.item()
        metric_logger["aux"] += aux_val
        metric_logger["aux_n"] += aux_n
        metric_logger["n"] += 1

        if (it + 1) % print_freq == 0:
            n = metric_logger["n"]
            el = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  ep{epoch} it{it+1}/{len(loader)} loss={metric_logger['loss']/n:.4f} "
                  f"cls={metric_logger['loss_cls']/n:.4f} box={metric_logger['loss_box']/n:.4f} "
                  f"rpn={metric_logger['loss_rpn']/n:.4f} aux={metric_logger['aux']/n:.4f}"
                  f"(n={metric_logger['aux_n']}) lr={lr:.5f} "
                  f"{el/(it+1):.2f}s/it", flush=True)
            metric_logger = {"loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0, "loss_rpn": 0.0,
                             "aux": 0.0, "aux_n": 0, "n": 0}
    return metric_logger


@torch.no_grad()
def evaluate_gt_proposals(model, loader, samples, device, amp=True, score_thr=0.001):
    """用 GT 框替换 RPN proposal，隔离"定位/召回"与"分类"两段损失。

    得到的是 oracle-proposal nAP：若它远高于真实 nAP，说明瓶颈在 RPN 的
    base 类偏置（novel 物体根本进不了 RoI），而不是分类头。
    """
    model.eval()
    all_preds, all_gts = [], []
    ptr = 0
    for images, targets in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        orig_sizes = [tuple(im.shape[-2:]) for im in images]
        with torch.amp.autocast("cuda", enabled=amp):
            imgs, _ = model.transform(images, None)
            feats = model.backbone(imgs.tensors)
            # GT 框需从原图坐标系缩放到 resize 后的坐标系
            props = []
            for i, t in enumerate(targets):
                sy = imgs.image_sizes[i][0] / orig_sizes[i][0]
                sx = imgs.image_sizes[i][1] / orig_sizes[i][1]
                b = t["boxes"].to(device) * torch.tensor([sx, sy, sx, sy], device=device)
                props.append(b)
            out = model.roi_heads(feats, props, imgs.image_sizes)
            dets = out[0] if isinstance(out, tuple) else out
            dets = model.transform.postprocess(dets, imgs.image_sizes, orig_sizes)
        for d in dets:
            b = d["boxes"].float().cpu().numpy()
            s = d["scores"].float().cpu().numpy()
            l = d["labels"].cpu().numpy()
            m = s >= score_thr
            all_preds.append({"boxes": b[m], "scores": s[m], "labels": l[m]})
        for _ in range(len(dets)):
            smp = samples[ptr]
            all_gts.append({
                "boxes": np.array(smp["boxes"], dtype=np.float32).reshape(-1, 4),
                "labels": np.array(smp["labels"], dtype=np.int64),
                "difficult": np.array(smp["difficult"], dtype=np.int64),
            })
            ptr += 1
    return all_preds, all_gts


@torch.no_grad()
def proposal_recall(model, loader, samples, device, iou_thr=0.5, max_props=1000):
    """RPN 对 novel 类目标的召回率（按面积分箱），衡量 base 偏置有多严重。"""
    model.eval()
    bins = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
    hit = {k: 0 for k in bins}
    tot = {k: 0 for k in bins}
    ptr = 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        orig = [tuple(im.shape[-2:]) for im in images]
        with torch.amp.autocast("cuda", enabled=True):
            imgs, _ = model.transform(images, None)
            feats = model.backbone(imgs.tensors)
            props, _ = model.rpn(imgs, feats, imgs.image_sizes)
        for i, (p, _t) in enumerate(zip(props, targets)):
            pb = p[:max_props].cpu().numpy()
            smp = samples[ptr]
            gb = np.array(smp["boxes"], dtype=np.float32).reshape(-1, 4)
            # GT 需缩放到 resize 后的坐标系（面积仍按原图尺度分箱）
            sy = imgs.image_sizes[i][0] / orig[i][0]
            sx = imgs.image_sizes[i][1] / orig[i][1]
            gb_t = gb * np.array([sx, sy, sx, sy], dtype=np.float32)
            ptr += 1
            if len(gb) == 0 or len(pb) == 0:
                continue
            ov = _iou(pb, gb_t)
            best = ov.max(0)
            area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            for k, (lo, hi) in bins.items():
                m = (area >= lo) & (area < hi)
                tot[k] += int(m.sum())
                hit[k] += int((best[m] >= iou_thr).sum())
    return {k: (hit[k] / tot[k] if tot[k] else float("nan")) for k in bins}, tot


@torch.no_grad()
def head_diagnostic(model, loader, samples, device, novel_labels, base_labels,
                    amp=True, max_imgs=300):
    """在 GT RoI 上直接看分类头的判决，定位失败模式（novel→base / novel→bg）。

    跳过 RPN 与 NMS，因此与 nAP 无关，只回答一个题：给对了框，分类头认得出来吗。
    """
    model.eval()
    st = {"novel_total": 0, "novel_correct": 0, "novel_to_base": 0, "novel_to_bg": 0,
          "base_total": 0, "base_correct": 0, "base_to_novel": 0}
    bins = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
    by_scale = {b: {"n": 0, "correct": 0, "to_bg": 0, "to_base": 0} for b in bins}
    novel_set = set(novel_labels)
    n = 0
    for images, targets in loader:
        if n >= max_imgs:
            break
        images = [im.to(device) for im in images]
        orig = [tuple(im.shape[-2:]) for im in images]
        with torch.amp.autocast("cuda", enabled=amp):
            imgs, _ = model.transform(images, None)
            feats = model.backbone(imgs.tensors)
            props, gts = [], []
            for i, t in enumerate(targets):
                sy = imgs.image_sizes[i][0] / orig[i][0]
                sx = imgs.image_sizes[i][1] / orig[i][1]
                props.append(t["boxes"].to(device) * torch.tensor([sx, sy, sx, sy], device=device))
                gts.append(t["labels"].to(device))
            pooled = model.roi_heads.box_roi_pool(feats, props, imgs.image_sizes)
            feat = model.roi_heads.box_head(pooled)
            logits, _ = model.roi_heads.box_predictor(feat)
            logits = logits.float()
            logits_nobg = logits.clone()
            logits_nobg[:, 0] = -1e9
        off = 0
        for i, gl in enumerate(gts):
            m = gl.shape[0]
            lg = logits[off:off + m]
            pred = lg.argmax(1)
            pred_nb = logits_nobg[off:off + m].argmax(1)
            gb_orig = targets[i]["boxes"]
            for k in range(m):
                y = int(gl[k].item())
                p = int(pred[k].item())
                bb = gb_orig[k]
                ar = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
                bin_name = next((bn for bn, (lo, hi) in bins.items() if lo <= ar < hi), "large")
                pn = int(pred_nb[k].item())
                if y in novel_set:
                    by_scale[bin_name]["n"] += 1
                    by_scale[bin_name].setdefault("nb_correct", 0)
                    if pn == y:
                        by_scale[bin_name]["nb_correct"] += 1
                    if p == y:
                        by_scale[bin_name]["correct"] += 1
                    elif p == 0:
                        by_scale[bin_name]["to_bg"] += 1
                    elif p in base_labels:
                        by_scale[bin_name]["to_base"] += 1
                if y in novel_set:
                    st["novel_total"] += 1
                    if p == y:
                        st["novel_correct"] += 1
                    elif p == 0:
                        st["novel_to_bg"] += 1
                    elif p in base_labels:
                        st["novel_to_base"] += 1
                elif y in base_labels:
                    st["base_total"] += 1
                    if p == y:
                        st["base_correct"] += 1
                    elif p in novel_set:
                        st["base_to_novel"] += 1
            off += m
        n += 1
    out = {}
    nt = max(st["novel_total"], 1)
    bt = max(st["base_total"], 1)
    out["novel_top1"] = st["novel_correct"] / nt
    out["novel_to_base"] = st["novel_to_base"] / nt
    out["novel_to_bg"] = st["novel_to_bg"] / nt
    out["base_top1"] = st["base_correct"] / bt
    out["base_to_novel"] = st["base_to_novel"] / bt
    out["n_novel_gt"] = st["novel_total"]
    out["n_base_gt"] = st["base_total"]
    for bn, d in by_scale.items():
        nn = max(d["n"], 1)
        out[f"novel_top1_{bn}"] = d["correct"] / nn
        out[f"novel_top1nobg_{bn}"] = d.get("nb_correct", 0) / nn
        out[f"novel_tobg_{bn}"] = d["to_bg"] / nn
        out[f"n_novel_{bn}"] = d["n"]
    return out


@torch.no_grad()
def proposal_recall_curve(model, loader, samples, device, iou_thr=0.5, ks=(100, 300, 1000, 3000)):
    """RPN 召回随 proposal 预算的变化，按面积分箱。用于区分"召回不足"与"预算被大目标挤占"。"""
    model.eval()
    bins = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}
    hit = {k: {b: 0 for b in bins} for k in ks}
    tot = {b: 0 for b in bins}
    old_top = model.rpn._post_nms_top_n["testing"]
    old_pre = model.rpn._pre_nms_top_n["testing"]
    model.rpn._post_nms_top_n["testing"] = max(ks)
    model.rpn._pre_nms_top_n["testing"] = max(ks) * 6
    ptr = 0
    try:
        for images, targets in loader:
            images = [im.to(device) for im in images]
            orig = [tuple(im.shape[-2:]) for im in images]
            with torch.amp.autocast("cuda", enabled=True):
                imgs, _ = model.transform(images, None)
                feats = model.backbone(imgs.tensors)
                props, _ = model.rpn(imgs, feats, imgs.image_sizes)
            for i, p in enumerate(props):
                smp = samples[ptr]
                gb = np.array(smp["boxes"], dtype=np.float32).reshape(-1, 4)
                sy = imgs.image_sizes[i][0] / orig[i][0]
                sx = imgs.image_sizes[i][1] / orig[i][1]
                gb_t = gb * np.array([sx, sy, sx, sy], dtype=np.float32)
                ptr += 1
                if len(gb) == 0:
                    continue
                area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
                pb = p.cpu().numpy()
                ov = _iou(pb, gb_t)
                for b, (lo, hi) in bins.items():
                    m = (area >= lo) & (area < hi)
                    tot[b] += int(m.sum())
                    if m.sum() == 0:
                        continue
                    for k in ks:
                        best = ov[:k].max(0)
                        hit[k][b] += int((best[m] >= iou_thr).sum())
    finally:
        model.rpn._post_nms_top_n["testing"] = old_top
        model.rpn._pre_nms_top_n["testing"] = old_pre
    return {k: {b: (hit[k][b] / tot[b] if tot[b] else float("nan")) for b in bins} for k in ks}, tot


def _iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


@torch.no_grad()
def evaluate(model, loader, samples, device, amp=True, score_thr=0.001):
    model.eval()
    all_preds, all_gts = [], []
    ptr = 0
    t0 = time.time()
    for images, targets in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        with torch.amp.autocast("cuda", enabled=amp):
            outs = model(images)
        for out in outs:
            b = out["boxes"].cpu().numpy()
            s = out["scores"].cpu().numpy()
            l = out["labels"].cpu().numpy()
            m = s >= score_thr
            all_preds.append({"boxes": b[m], "scores": s[m], "labels": l[m]})
        for _ in range(len(outs)):
            smp = samples[ptr]
            all_gts.append({
                "boxes": np.array(smp["boxes"], dtype=np.float32).reshape(-1, 4),
                "labels": np.array(smp["labels"], dtype=np.int64),
                "difficult": np.array(smp["difficult"], dtype=np.int64),
            })
            ptr += 1
    print(f"  inference {time.time()-t0:.1f}s for {len(all_preds)} imgs", flush=True)
    return all_preds, all_gts


def report_ap(all_preds, all_gts, class_names, tag="", area_range=None):
    """class_names: {label_int: name}"""
    aps = compute_ap_per_class(all_preds, all_gts, list(class_names.keys()), area_range=area_range)
    named = {class_names[k]: v for k, v in aps.items()}
    mean = float(np.mean(list(named.values()))) if named else 0.0
    if tag:
        print(f"[{tag}] mAP50={mean*100:.2f}", flush=True)
    for k, v in named.items():
        print(f"    {k:14s} {v*100:6.2f}", flush=True)
    return named, mean


# COCO 面积区间（像素面积）
AREA_BINS = {
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e10),
}


def report_ap_by_scale(all_preds, all_gts, class_names, tag=""):
    out = {}
    for name, rng in AREA_BINS.items():
        named, m = report_ap(all_preds, all_gts, class_names, tag=f"{tag}/{name}", area_range=rng)
        out[name] = {"mAP50": m, "per_class": named}
    return out
