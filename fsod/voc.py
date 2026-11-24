"""VOC2007 解析与 FSOD（base/novel）划分。

协议参照 TFA / FSCE 的 VOC 设定：
  - 数据：VOC2007 trainval(5011) 训练，VOC2007 test(4952) 评测
  - 20 类切成 15 base + 5 novel，共 3 组 split
  - 标注中 difficult=1 的框在评测时忽略
"""
import os
import json
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# TFA (Wang et al., CVPR2020) / FSCE 使用的 VOC novel split
NOVEL_SPLITS = {
    1: ["aeroplane", "bottle", "cow", "pottedplant", "sofa"],
    2: ["bicycle", "bus", "horse", "motorbike", "tvmonitor"],
    3: ["boat", "cat", "chair", "sheep", "train"],
}


def parse_voc_dir(voc_root):
    """解析 Annotations/*.xml -> {image_id: {"size":(w,h), "objects":[{name,box,difficult}]}}"""
    ann_dir = os.path.join(voc_root, "Annotations")
    img_dir = os.path.join(voc_root, "JPEGImages")
    out = {}
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith(".xml"):
            continue
        img_id = fn[:-4]
        tree = ET.parse(os.path.join(ann_dir, fn))
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        objs = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            if name not in VOC_CLASSES:
                continue
            bb = obj.find("bndbox")
            x1 = float(bb.find("xmin").text) - 1.0
            y1 = float(bb.find("ymin").text) - 1.0
            x2 = float(bb.find("xmax").text) - 1.0
            y2 = float(bb.find("ymax").text) - 1.0
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w - 1), x2), min(float(h - 1), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            diff = int(obj.find("difficult").text)
            objs.append({"name": name, "box": [x1, y1, x2, y2], "difficult": diff})
        out[img_id] = {
            "size": (w, h),
            "objects": objs,
            "file": os.path.join(img_dir, img_id + ".jpg"),
        }
    return out


def read_ids(voc_root, split):
    p = os.path.join(voc_root, "ImageSets", "Main", split + ".txt")
    ids = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def build_split(ann_all, train_ids, test_ids, split_id):
    """返回 label 映射与 train/test 样本列表。

    stage='base' 时标签空间为 1..15；stage='ft' 时为 1..20（novel 追加在末尾）。
    """
    novel = NOVEL_SPLITS[split_id]
    base = [c for c in VOC_CLASSES if c not in novel]
    # 统一标签空间：base 在前，novel 在后
    ft_classes = base + novel
    base_map = {c: i + 1 for i, c in enumerate(base)}
    ft_map = {c: i + 1 for i, c in enumerate(ft_classes)}
    return {
        "split_id": split_id,
        "novel": novel,
        "base": base,
        "ft_classes": ft_classes,
        "base_map": base_map,
        "ft_map": ft_map,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def samples_for(ann_all, ids, class_map, keep_difficult=True):
    """把 image_id 列表转成样本列表，标签用 class_map 重映射；丢弃无有效目标图。"""
    out = []
    for i in ids:
        a = ann_all[i]
        boxes, labels, diff = [], [], []
        for o in a["objects"]:
            if o["name"] not in class_map:
                continue
            boxes.append(o["box"])
            labels.append(class_map[o["name"]])
            diff.append(o["difficult"])
        if not boxes:
            continue
        if not keep_difficult:
            idx = [k for k, d in enumerate(diff) if d == 0]
            boxes = [boxes[k] for k in idx]
            labels = [labels[k] for k in idx]
            diff = [diff[k] for k in idx]
            if not boxes:
                continue
        out.append({
            "image_id": i,
            "file": a["file"],
            "boxes": boxes,
            "labels": labels,
            "difficult": diff,
        })
    return out


def prepare_0712(voc07_root, voc12_root, out_dir, cache_name="voc_ann_0712.json"):
    """VOC07+12 协议：VOC2012 trainval 训练，VOC2007 test 评测。

    image_id 不冲突：VOC2007 是 6 位数字（如 000027），VOC2012 是 2007_000027 / 2012_000001，
    因此合并后一个 id 唯一对应一张图与一套标注。
    注意：VOC2012 沿用 2007 图像时重命名并重标注，此时用 2012 版本（标注质量更高），
    而 2007 test 的评测仍用 2007 原始标注，保证与已有基线同口径。
    """
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, cache_name)
    if os.path.exists(cache):
        with open(cache, "r") as f:
            d = json.load(f)
        return d["ann"], d["train_ids"], d["test_ids"]

    ann07 = parse_voc_dir(voc07_root)
    ann12 = parse_voc_dir(voc12_root)
    ann = dict(ann07)
    ann.update(ann12)
    train_ids = read_ids(voc12_root, "trainval") + read_ids(voc07_root, "trainval")
    test_ids = read_ids(voc07_root, "test")
    with open(cache, "w") as f:
        json.dump({"ann": ann, "train_ids": train_ids, "test_ids": test_ids}, f)
    return ann, train_ids, test_ids


def prepare(voc_root, out_dir, cache_name="voc_ann.json"):
    """解析并缓存标注（解析一次约 10-20s）。"""
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, cache_name)
    if os.path.exists(cache):
        with open(cache, "r") as f:
            ann_all = json.load(f)
    else:
        ann_all = parse_voc_dir(voc_root)
        with open(cache, "w") as f:
            json.dump(ann_all, f)
    train_ids = read_ids(voc_root, "trainval")
    test_ids = read_ids(voc_root, "test")
    return ann_all, train_ids, test_ids


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "VOCdevkit", "VOC2007")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ann_all, tr, te = prepare(root, out)
    print("images", len(ann_all), "train", len(tr), "test", len(te))
    for sp in (1, 2, 3):
        s = build_split(ann_all, tr, te, sp)
        n_tr = samples_for(ann_all, tr, s["base_map"])
        n_te = samples_for(ann_all, te, s["base_map"])
        print(f"split{sp} novel={s['novel']} base_train_imgs={len(n_tr)} test_imgs={len(n_te)}")
