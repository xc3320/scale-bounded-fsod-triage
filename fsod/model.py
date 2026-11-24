"""Faster R-CNN 构建 / 分类头扩展 / 冻结策略（TFA 协议）。"""
import copy
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet50, ResNet50_Weights

BACKBONES = {
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
}


def build_model(num_classes, backbone="resnet50", min_size=600, max_size=1000,
                trainable_backbone_layers=5, pretrained=True):
    fn, wts, _ = BACKBONES[backbone]
    bb = fn(weights=wts if pretrained else None)
    backbone_fpn = resnet_fpn_backbone(
        backbone_name=backbone,
        weights=wts if pretrained else None,
        trainable_layers=trainable_backbone_layers,
        norm_layer=nn.BatchNorm2d,
    )
    model = FasterRCNN(
        backbone_fpn,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=None,
        box_detections_per_img=100,
        box_score_thresh=0.01,
        box_nms_thresh=0.5,
    )
    # 默认初始化
    return model


def expand_classifier(model, new_num_classes, n_old):
    """把 cls_score / bbox_pred 扩展到 new_num_classes，旧权重保留，新类用 base 均值初始化（TFA）。"""
    in_ch = model.roi_heads.box_predictor.cls_score.in_features
    old_score = model.roi_heads.box_predictor.cls_score
    old_bbox = model.roi_heads.box_predictor.bbox_pred
    new_pred = FastRCNNPredictor(in_ch, new_num_classes)
    with torch.no_grad():
        new_pred.cls_score.weight[:n_old] = old_score.weight[:n_old]
        new_pred.cls_score.bias[:n_old] = old_score.bias[:n_old]
        new_pred.bbox_pred.weight[:n_old] = old_bbox.weight[:n_old]
        new_pred.bbox_pred.bias[:n_old] = old_bbox.bias[:n_old]
        # 新类：用前景类（1..n_old-1）均值 + 小扰动
        fg_w = old_score.weight[1:n_old].mean(0)
        fg_b = old_score.bias[1:n_old].mean(0)
        for c in range(n_old, new_num_classes):
            new_pred.cls_score.weight[c] = fg_w + 0.01 * torch.randn_like(fg_w)
            new_pred.cls_score.bias[c] = fg_b
            new_pred.bbox_pred.weight[c] = old_bbox.weight[1:n_old].mean(0)
            new_pred.bbox_pred.bias[c] = old_bbox.bias[1:n_old].mean(0)
    model.roi_heads.box_predictor = new_pred
    return model


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def freeze_backbone_and_rpn(model, freeze_rpn=True):
    """TFA 微调：冻结 backbone(含 FPN)，可选冻结 RPN，训 roi_heads。"""
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.rpn.parameters():
        p.requires_grad = not freeze_rpn
    for p in model.roi_heads.parameters():
        p.requires_grad = True
    model.backbone.eval()
    set_bn_eval(model.backbone)
    return model


def count_params(model, only_trainable=True):
    ps = [p for p in model.parameters() if (p.requires_grad or not only_trainable)]
    return sum(p.numel() for p in ps)


def build_optimizer(model, base_lr, backbone_lr_mult=0.1, momentum=0.9, wd=1e-4):
    params = []
    bb, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone"):
            bb.append((n, p))
        else:
            rest.append((n, p))
    groups = [{"params": [p for _, p in rest], "lr": base_lr, "weight_decay": wd}]
    if bb:
        groups.append({"params": [p for _, p in bb],
                       "lr": base_lr * backbone_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.SGD(groups, momentum=momentum)
    return opt
