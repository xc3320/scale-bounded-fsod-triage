"""尺度条件化的前景/背景分数校准（SCBC）。

依据（实测）：
  - 微调后 novel 类错误几乎全是 "novel -> 背景"：小目标 95.5%、大目标 58.1%
  - 去掉背景行后，小目标 novel 的 top-1 从 4.5% 升到 77.3%
  说明前景证据存在，只是被背景分数淹没，且淹没程度强烈依赖目标尺度。

做法：
  在 softmax 之前，按 RoI 面积所属尺度箱，从背景 logit 上减去 delta(bin)。
  delta 由 base 类在同一尺度箱上的前景/背景边缘分布估计（只用 base 数据，符合 FSOD 约束）。
"""
import torch
import torch.nn as nn


class ScaleBgCalibrator:
    """包装 box_roi_pool（记录 RoI 面积）与 box_predictor（调整背景 logit）。"""

    def __init__(self, deltas, bins=((0, 32 ** 2), (32 ** 2, 96 ** 2), (96 ** 2, 1e12))):
        self.deltas = deltas            # list, 与 bins 等长
        self.bins = bins
        self.areas = None
        self.enabled = True

    def install(self, model):
        cal = self
        roi_heads = model.roi_heads
        orig_pool = roi_heads.box_roi_pool
        orig_pred = roi_heads.box_predictor

        class _PoolWrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, features, proposals, image_shapes):
                if cal.enabled:
                    with torch.no_grad():
                        bs = [p.detach() for p in proposals]
                        cal.areas = torch.cat([(b[:, 2] - b[:, 0]).clamp(min=0) *
                                               (b[:, 3] - b[:, 1]).clamp(min=0)
                                               for b in bs], 0) if bs else None
                return self.inner(features, proposals, image_shapes)

        class _PredWrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                cls_logits, bbox = self.inner(x)
                if cal.enabled and cal.areas is not None \
                        and cls_logits.shape[0] == cal.areas.shape[0] \
                        and abs(sum(cal.deltas)) > 0:
                    adj = torch.zeros(cls_logits.shape[0], 1,
                                      device=cls_logits.device, dtype=cls_logits.dtype)
                    a = cal.areas.to(cls_logits.device)
                    for d, (lo, hi) in zip(cal.deltas, cal.bins):
                        if d == 0:
                            continue
                        m = (a >= lo) & (a < hi)
                        if m.any():
                            adj[m, 0] = float(d)
                    cls_logits = cls_logits.clone()
                    cls_logits[:, 0:1] = cls_logits[:, 0:1] - adj
                return cls_logits, bbox

        roi_heads.box_roi_pool = _PoolWrap(orig_pool)
        roi_heads.box_predictor = _PredWrap(orig_pred)
        return model


class ContextSubtract:
    """用"扩张框"的池化特征做上下文扣除：pooled - lam * pooled_ctx。

    动机：小 RoI 的 256x7x7 特征里，像素大部分落在目标之外，特征被周边背景主导，
    背景行因此稳定胜出（实测小目标 95.5% 判为背景）。扣掉同尺度上下文分量后，
    剩余分量更接近目标本身。
    """

    def __init__(self, lam=0.5, expand=1.6):
        self.lam = lam
        self.expand = expand

    def install(self, model):
        cs = self
        orig_pool = model.roi_heads.box_roi_pool

        class _Wrap(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, features, proposals, image_shapes):
                pooled = self.inner(features, proposals, image_shapes)
                if cs.lam <= 0 or not proposals:
                    return pooled
                ctx = []
                for b in proposals:
                    b = b.detach()
                    w = (b[:, 2] - b[:, 0]).clamp(min=1)
                    h = (b[:, 3] - b[:, 1]).clamp(min=1)
                    cx = (b[:, 0] + b[:, 2]) / 2
                    cy = (b[:, 1] + b[:, 3]) / 2
                    nw = w * cs.expand / 2
                    nh = h * cs.expand / 2
                    ctx.append(torch.stack([cx - nw, cy - nh, cx + nw, cy + nh], 1))
                pooled_ctx = self.inner(features, ctx, image_shapes)
                return pooled - cs.lam * pooled_ctx

        model.roi_heads.box_roi_pool = _Wrap(orig_pool)
        return model
