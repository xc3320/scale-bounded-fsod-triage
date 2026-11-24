"""Base 阶段尺度-上下文一致性训练（SCC, Scale-Context Consistency）。

动机（全部来自本机实测，见 knowledge/SAFM训练时边距_机制撬动但AP负增益.md）：
  - 小目标判背景是两层叠加：尺度普遍效应（base 小目标也 61.5% 判背景）
    + novel 少样本偏置（叠加后到 100%、判对归零）。
  - base 靠大量样本"隐式"学会了抗上下文（小目标仍有 34.1% 判对），novel 学不会。
  - 一切推理时事后校正（logit 缩放/背景校准/上下文扣除）与一刀切训练时 margin
    (SAFM) 均失败：前者碰不到机制，后者以过拟合换判对率、小目标类崩溃。

核心假说：
  "抗上下文"是一种**类别无关的表征能力**——从含大量上下文的 RoI 里提取目标
  特征。它应当能在 base 训练阶段被显式注入，并迁移到只有 K 张图的 novel 类。

做法（只在 base 训练期，推理路径完全不变）：
  对采样到的每个 RoI，再构造一个"扩张框视图"（框按中心放大 expand 倍，纳入
  更多上下文），两个视图共用 box_head / box_predictor。对正样本约束：
      L = mean relu( margin_tight.detach() - margin_expand )
  其中 margin = logit[GT类] - logit[背景]。即：当扩张框（上下文更多）导致
  前景-背景边缘下降时惩罚模型，逼它在含上下文的 RoI 里仍保持目标证据。
  tight 视图 detach，梯度只经 expand 分支回传，但二者共享参数，故整体朝
  "对上下文不敏感"更新。

实现用轻量 hook：包装 select_training_samples / box_roi_pool.forward /
box_predictor.forward 三个入口，不改 torchvision 源码。训练循环每迭代需
调用 model._cap.clear() 并在 loss 里加 scc.aux()。
"""
import torch
import torch.nn.functional as F


class ScaleContextConsistency:
    def __init__(self, model, expand=2.0, lam=1.0, mode="sup", margin=1.0):
        rh = model.roi_heads
        self.rh = rh
        self.expand = float(expand)
        self.lam = float(lam)
        # mode="sup": 用 GT 标签直接监督扩张视图的前景边缘（v2，正确）
        # mode="distill": tight 视图自蒸馏（v1，已证伪——tight 小目标本身就判背景，会教错）
        self.mode = mode
        self.margin = float(margin)
        cap = {}
        model._cap = cap
        self.cap = cap

        orig_select = rh.select_training_samples
        orig_pool_fwd = rh.box_roi_pool.forward
        orig_pred_fwd = rh.box_predictor.forward

        def select_wrapped(proposals, targets):
            # torchvision 0.26: (proposals, matched_idxs, labels, regression_targets)
            out = orig_select(proposals, targets)
            cap["labels"] = out[2]
            return out

        def pool_wrapped(features, proposals, image_shapes):
            pooled = orig_pool_fwd(features, proposals, image_shapes)
            if rh.training:
                expand_props = []
                for p, shp in zip(proposals, image_shapes):
                    boxes = p.detach()
                    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1)
                    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1)
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2
                    nw = w * self.expand / 2
                    nh = h * self.expand / 2
                    ex = torch.stack([cx - nw, cy - nh, cx + nw, cy + nh], 1)
                    ex[:, 0::2] = ex[:, 0::2].clamp(0, shp[1])
                    ex[:, 1::2] = ex[:, 1::2].clamp(0, shp[0])
                    expand_props.append(ex)
                cap["pooled_expand"] = orig_pool_fwd(features, expand_props, image_shapes)
            return pooled

        def pred_wrapped(features):
            logits, bbox = orig_pred_fwd(features)
            if rh.training and "pooled_expand" in cap:
                feat_e = rh.box_head(cap["pooled_expand"])
                logits_e, _ = orig_pred_fwd(feat_e)
                cap["logits_expand"] = logits_e
                cap["logits_tight"] = logits
            return logits, bbox

        rh.select_training_samples = select_wrapped
        rh.box_roi_pool.forward = pool_wrapped
        rh.box_predictor.forward = pred_wrapped
        self._orig = (orig_select, orig_pool_fwd, orig_pred_fwd)

    def aux(self):
        """返回 (loss, n_pos)；无有效样本返回 None（不破坏梯度图）。"""
        cap = self.cap
        if "logits_expand" not in cap:
            return None
        lt = cap["logits_tight"].float()
        le = cap["logits_expand"].float()
        labels = torch.cat(cap["labels"], 0)
        if labels.numel() != lt.shape[0] or le.shape[0] != lt.shape[0]:
            return None
        pos = labels > 0
        idx = torch.nonzero(pos).flatten()
        if idx.numel() == 0:
            return None
        y = labels[idx]
        fg_e = le[idx, y]
        bg_e = le[idx, 0]
        if self.mode == "sup":
            # v2: 用 GT 标签直接要求扩张视图（含更多上下文）前景-背景边缘 >= margin。
            # 不依赖 tight 视图当教师（那本身判背景），避免自蒸馏教错。
            loss = self.lam * F.softplus(bg_e - fg_e + self.margin).mean()
        else:
            # v1 (证伪): tight 自蒸馏
            fg_t = lt[idx, y]
            bg_t = lt[idx, 0]
            margin_t = (fg_t - bg_t).detach()
            margin_e = fg_e - bg_e
            loss = self.lam * F.relu(margin_t - margin_e).mean()
        return loss, int(idx.numel())
