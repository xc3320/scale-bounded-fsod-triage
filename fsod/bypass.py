"""少样本微调用的旁路分类头（Bypass-CAL）。

动机（来自本项目的实测证据）：
  - pooled RoI 特征（256x7x7=12544）的 novel 类可分性远高于 box_head 输出（1024）
  - 说明 base 训练把表征压缩成了"只对 base 类判别"的低维流形
  - 微调时若继续复用这条被 base 占满的 1024 维通路，novel 判别信息无处可去

做法：
  h = fc7(fc6(x))                    # 冻结，保留 base 判别几何
  b = LayerNorm(proj(x))             # 旁路：proj 由 base 数据 PCA 初始化后冻结
  feat = [h ; gamma * b]             # 2048 维送入新分类头

初始化采用"单侧零初始化"：分类头的旁路列全零，而非 gamma=0。
这样 t=0 时输出与 TFA 基线逐比特一致，同时旁路列的梯度 = gamma * <dL/dlogits, b> ≠ 0，
不会出现双重零初始化导致的梯度死锁。
"""
import torch
import torch.nn as nn


class BoxHeadWithBypass(nn.Module):
    def __init__(self, box_head, in_dim, mid=1024, mode="pca", init_basis=None):
        super().__init__()
        self.box_head = box_head          # 原 TwoMLPHead（fc6/fc7），冻结
        self.d0 = 1024
        self.mid = mid
        self.mode = mode
        self.proj = nn.Linear(in_dim, mid, bias=False)
        self.norm = nn.LayerNorm(mid)
        if init_basis is not None:
            with torch.no_grad():
                self.proj.weight.copy_(init_basis[:, :mid].t().contiguous())
        else:
            nn.init.normal_(self.proj.weight, std=0.02)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h = self.box_head(x)                      # TwoMLPHead 内部自己 flatten
        xf = x.flatten(start_dim=1)               # (N, 256*7*7)
        b = self.norm(self.proj(xf))
        return torch.cat([h, self.gamma * b], dim=1)


class BypassPredictor(nn.Module):
    """cls 走 (1024+mid) 维；bbox 回归仍只用 box_head 的 1024 维。"""

    def __init__(self, in_dim, num_classes, d0=1024):
        super().__init__()
        self.d0 = d0
        self.cls_score = nn.Linear(in_dim, num_classes)
        self.bbox_pred = nn.Linear(d0, num_classes * 4)

    def forward(self, x):
        return self.cls_score(x), self.bbox_pred(x[:, : self.d0])


def install_bypass(model, in_dim, new_num_classes, n_old, mid=1024, mode="pca",
                   init_basis=None, novel_mean_init=True):
    """把 roi_heads 换成旁路版。

    new_num_classes: 扩展后的类别数（含背景）
    n_old:           旧分类头的类别数（含背景），用于继承 base 阶段学到的权重
    """
    old_head = model.roi_heads.box_head
    old_pred = model.roi_heads.box_predictor
    dev = old_pred.cls_score.weight.device

    bh = BoxHeadWithBypass(old_head, in_dim, mid, mode, init_basis).to(dev)
    bp = BypassPredictor(1024 + mid, new_num_classes, d0=1024).to(dev)

    with torch.no_grad():
        bp.cls_score.weight.zero_()          # 旁路列零初始化 -> 起点等价基线
        bp.cls_score.bias.zero_()
        bp.cls_score.weight[:n_old, :1024] = old_pred.cls_score.weight[:n_old]
        bp.cls_score.bias[:n_old] = old_pred.cls_score.bias[:n_old]

        bp.bbox_pred.weight.zero_()
        bp.bbox_pred.bias.zero_()
        bp.bbox_pred.weight[:n_old] = old_pred.bbox_pred.weight[:n_old]
        bp.bbox_pred.bias[:n_old] = old_pred.bbox_pred.bias[:n_old]

        if novel_mean_init and new_num_classes > n_old:
            fg_w = old_pred.cls_score.weight[1:n_old].mean(0)
            fg_b = old_pred.cls_score.bias[1:n_old].mean(0)
            bw = old_pred.bbox_pred.weight[1:n_old].mean(0)
            bb = old_pred.bbox_pred.bias[1:n_old].mean(0)
            for c in range(n_old, new_num_classes):
                bp.cls_score.weight[c, :1024] = fg_w + 0.01 * torch.randn_like(fg_w)
                bp.cls_score.bias[c] = fg_b
                bp.bbox_pred.weight[c] = bw
                bp.bbox_pred.bias[c] = bb

    model.roi_heads.box_head = bh
    model.roi_heads.box_predictor = bp
    return model


def freeze_bypass_tuning(model, train_proj=False):
    """只训 box_predictor + gamma（+ 可选 proj）。"""
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.rpn.parameters():
        p.requires_grad = False
    bh = model.roi_heads.box_head
    for p in bh.box_head.parameters():
        p.requires_grad = False
    for p in model.roi_heads.box_predictor.parameters():
        p.requires_grad = True
    bh.gamma.requires_grad = True
    bh.norm.requires_grad = True
    for p in bh.proj.parameters():
        p.requires_grad = train_proj
    model.backbone.eval()
    return model
