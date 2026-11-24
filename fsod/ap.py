"""VOC 风格 AP（all-point 插值，IoU=0.5），支持 difficult 忽略。"""
import numpy as np


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def _area_filter(boxes, lo, hi):
    a = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep = (a >= lo) & (a < hi)
    return keep


def compute_ap_per_class(preds, gts, classes, iou_thr=0.5, score_thr=0.001,
                         area_range=None):
    """preds/gts: 长度一致的两个 list，逐图。
    preds[i] = {"boxes":(N,4),"scores":(N,),"labels":(N,)}
    gts[i]   = {"boxes":(N,4),"labels":(N,),"difficult":(N,)}
    area_range: (lo, hi) 时按 COCO 语义做尺度分层——GT 与 DT 都只保留该面积区间的框，
                区间外的检测自动成为 FP。
    返回 {class_id: ap}
    """
    aps = {}
    n_img = len(preds)
    for c in classes:
        tp, fp, sc = [], [], []
        npos = 0
        for i in range(n_img):
            g = gts[i]
            gmask = g["labels"] == c
            gb = g["boxes"][gmask]
            gd = g["difficult"][gmask]
            if area_range is not None and len(gb):
                k = _area_filter(gb, area_range[0], area_range[1])
                gb, gd = gb[k], gd[k]
            npos += int((gd == 0).sum())
            p = preds[i]
            pmask = p["labels"] == c
            pb = p["boxes"][pmask]
            ps = p["scores"][pmask]
            if area_range is not None and len(pb):
                k = _area_filter(pb, area_range[0], area_range[1])
                pb, ps = pb[k], ps[k]
            if len(pb) == 0:
                continue
            ov = _iou_matrix(pb, gb)
            order = np.argsort(-ps)
            ov = ov[order]
            ps_sorted = ps[order]
            used = np.zeros(len(gb), dtype=bool)
            for k in range(len(pb)):
                best_iou, best_j = 0.0, -1
                for j in range(len(gb)):
                    if used[j] or gd[j] == 1:
                        continue
                    if ov[k, j] > best_iou:
                        best_iou = ov[k, j]
                        best_j = j
                if best_iou >= iou_thr and best_j >= 0:
                    used[best_j] = True
                    tp.append(1.0); fp.append(0.0)
                else:
                    tp.append(0.0); fp.append(1.0)
                sc.append(ps_sorted[k])
        if npos == 0:
            continue
        if len(sc) == 0:
            aps[c] = 0.0
            continue
        order = np.argsort(-np.array(sc))
        tp = np.array(tp)[order]
        fp = np.array(fp)[order]
        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        rec = ctp / max(npos, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        aps[c] = _voc_ap(rec, prec)
    return aps
