"""训练时前景边距（Scale-Aware Foreground Margin, SAFM）。

动机（来自实测）：
  - 小 novel 目标在 GT 框上 95.5% 被判为背景；去掉背景行后判对率从 4.5% 跳到 59.1%。
  - 一切推理时的 logit 事后缩放（压背景 / 放大 novel / 按尺度校准）全部负增益。
  => 背景压制是训练动态造成的，必须在训练时干预，而不是推理时改分数。

做法：
  在 K-shot 微调的前向里捕获 RoI 分类头的 logits / 采样标签 / proposal 坐标，
  对"小面积的正样本 RoI"加一项边距损失：
      L_aux = mean softplus( bg_logit - fg_logit + margin )
  它只在训练时作用，推理路径完全不变，因此不会引入事后校准那类副作用。

实现用轻量 hook：只在训练期替换 select_training_samples / box_roi_pool.forward /
box_predictor.forward 三个入口来取张量，不改动 torchvision 源码。
"""
import torch
import torch.nn.functional as F


class LogitCapture:
    """挂到 model 上，训练前向结束后从 model._cap 读取所需张量。"""

    def __init__(self, model):
        self.model = model
        cap = {}
        model._cap = cap

        rh = model.roi_heads
        orig_select = rh.select_training_samples
        orig_pool_fwd = rh.box_roi_pool.forward
        orig_pred_fwd = rh.box_predictor.forward

        def select_wrapped(proposals, targets):
            # torchvision 0.26 返回 (proposals, matched_idxs, labels, regression_targets)
            out = orig_select(proposals, targets)
            cap["proposals"], cap["labels"] = out[0], out[2]
            return out

        def pool_wrapped(features, proposals, image_shapes):
            cap["image_shapes"] = image_shapes
            return orig_pool_fwd(features, proposals, image_shapes)

        def pred_wrapped(features):
            logits, bbox = orig_pred_fwd(features)
            cap["logits"] = logits
            return logits, bbox

        rh.select_training_samples = select_wrapped
        rh.box_roi_pool.forward = pool_wrapped
        rh.box_predictor.forward = pred_wrapped
        self._orig = (orig_select, orig_pool_fwd, orig_pred_fwd)


def margin_aux(model, images, novel_start, margin=1.0, small_thr=32 ** 2,
               novel_only=True, scale_cond=True):
    """计算前景边距辅助损失；无符合条件样本时返回 0（不破坏梯度图）。

    novel_start: novel 类在扩展分类头里的起始行号（base 类数 + 1）。
    """
    cap = getattr(model, "_cap", {})
    if not all(k in cap for k in ("logits", "labels", "proposals", "image_shapes")):
        return None
    logits = cap["logits"].float()
    labels = torch.cat(cap["labels"], dim=0)
    n = labels.numel()
    if n == 0 or logits.shape[0] != n:
        return None

    if scale_cond:
        areas = []
        for p, shp, im in zip(cap["proposals"], cap["image_shapes"], images):
            sx = float(shp[1]) / float(im.shape[-1])
            sy = float(shp[0]) / float(im.shape[-2])
            a = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]) / max(sx * sy, 1e-6)
            areas.append(a)
        area = torch.cat(areas)
        sel = (labels > 0) & (area < small_thr)
    else:
        sel = labels > 0
    if novel_only:
        sel = sel & (labels >= novel_start)
    if int(sel.sum()) == 0:
        return None

    idx = torch.nonzero(sel).flatten()
    fg = logits[idx, labels[idx]]
    bg = logits[idx, 0]
    return F.softplus(bg - fg + margin).mean(), int(idx.numel())
