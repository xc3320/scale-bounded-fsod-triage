"""探针：定位 novel 判别信息在哪一层丢失。

对比四条通路在真实训练过的 base 检测器上的 few-shot 可分性：
  A. box_head 输出 (1024d)            —— 现有 TFA / Faster R-CNN 分类头的实际输入
  B. pooled RoI 特征 (256x7x7=12544)  —— box_head 的输入，未经压缩
  C. PCA-1024(pooled)                 —— 用 base 数据拟合的主子空间旁路（本方法候选）
  D. PCA-k(pooled) for k in 256/512   —— 旁路维度敏感性

若 A << C ≈ B，说明瓶颈在 TwoMLPHead 的 12:1 压缩，而不是在主干特征本身。
"""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

from fsod import voc
from fsod.model import build_model


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def extract(model, samples, device, quota_per_class, seed=0, max_images=None):
    """按类配额提取 GT 框的 pooled 特征与 box_head 特征。"""
    model.eval()
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    cnt = defaultdict(int)
    pooled_list, box_list, lab_list, area_list = [], [], [], []
    n_img = 0
    for idx in order:
        if max_images and n_img >= max_images:
            break
        s = samples[idx]
        keep = []
        for b, l in zip(s["boxes"], s["labels"]):
            if cnt[l] < quota_per_class:
                keep.append((b, l))
                cnt[l] += 1
        if not keep:
            continue
        boxes = torch.tensor([k[0] for k in keep], dtype=torch.float32, device=device)
        labels = [k[1] for k in keep]
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(s["file"]).convert("RGB")
        t = TF.pil_to_tensor(img).float().div(255.0)
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
        imgs, _ = model.transform([t], None)
        feats = model.backbone(imgs.tensors)
        pooled = model.roi_heads.box_roi_pool(feats, [boxes], imgs.image_sizes)  # (N,256,7,7)
        flat = pooled.flatten(1)
        bf = model.roi_heads.box_head(flat)
        pooled_list.append(flat.float().cpu())
        box_list.append(bf.float().cpu())
        lab_list.extend(labels)
        area_list.extend([(b[2] - b[0]) * (b[3] - b[1]) for b in [k[0] for k in keep]])
        n_img += 1
        if n_img % 200 == 0:
            print(f"    extracted {n_img} imgs, {sum(cnt.values())} boxes", flush=True)
    return (torch.cat(pooled_list), torch.cat(box_list),
            np.array(lab_list), np.array(area_list))


def pca_basis(X, k, device="cuda"):
    """在 n<d 情形下用 Gram 矩阵做 PCA，返回 (mean, V) ，V: (d,k)。"""
    X = X.to(device)
    mean = X.mean(0, keepdim=True)
    Xc = X - mean
    n = Xc.shape[0]
    if n <= k:
        k = max(1, n - 1)
    G = Xc @ Xc.t()
    evals, U = torch.linalg.eigh(G)
    evals = evals.flip(0)[:k]
    U = U.flip(1)[:, :k]
    V = Xc.t() @ U
    V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return mean.cpu(), V.float().cpu()


def l2n(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)


def episode_acc(X, y, classes, K, n_ep, rng):
    """最近类均值分类（cosine），多 episode 平均。"""
    accs = []
    idx_by_c = {c: np.where(y == c)[0] for c in classes}
    for _ in range(n_ep):
        sup, qry = [], []
        ok = True
        for c in classes:
            ids = idx_by_c[c]
            if len(ids) < K + 5:
                ok = False
                break
            perm = rng.permutation(len(ids))
            sup.extend(ids[perm[:K]])
            qry.extend(ids[perm[K:K + 200]])
        if not ok:
            continue
        sup = np.array(sup); qry = np.array(qry)
        protos = []
        for c in classes:
            m = y[sup] == c
            protos.append(X[sup[m]].mean(0))
        P = l2n(torch.stack(protos))
        Q = l2n(X[qry])
        pred = (Q @ P.t()).argmax(1).numpy()
        gt = np.array([classes.index(c) for c in y[qry]])
        accs.append(float((pred == gt).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--quota", type=int, default=400)
    ap.add_argument("--out", type=str, default="data/bypass_probe.json")
    ap.add_argument("--reuse", type=int, default=0, help="1=复用已缓存的特征")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    set_seed(0)
    device = torch.device("cuda")

    ann_all, train_ids, test_ids = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann_all, train_ids, test_ids, args.split)
    ft_map = sp["ft_map"]
    base_map = sp["base_map"]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ca = ck.get("args", {})
    model = build_model(len(base_map) + 1, ca.get("backbone", "resnet50"),
                        ca.get("min_size", 600), ca.get("max_size", 1000),
                        pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    print(f"[ckpt] epoch={ck.get('epoch')} classes={len(base_map)}", flush=True)

    # base 类：拟合 PCA 的支撑集
    cache = os.path.join(here, "data", "bypass_feats.pt")
    if args.reuse and os.path.exists(cache):
        obj = torch.load(cache, map_location="cpu", weights_only=False)
        Pb, Bb, yb, Pn, Bn, yn = obj["Pb"], obj["Bb"], obj["yb"], obj["Pn"], obj["Bn"], obj["yn"]
        print("[reuse] loaded cached features", flush=True)
    else:
        base_samples = voc.samples_for(ann_all, train_ids, base_map)
        print(f"[extract] base pooled, {len(base_samples)} imgs", flush=True)
        Pb, Bb, yb, ab = extract(model, base_samples, device, args.quota)

        # novel 类：探针目标（标签换到 base 空间外，用 ft_map 名到序号）
        novel_samples = voc.samples_for(ann_all, train_ids, {c: i + 1 for i, c in enumerate(sp["novel"])})
        print(f"[extract] novel pooled, {len(novel_samples)} imgs", flush=True)
        Pn, Bn, yn, an = extract(model, novel_samples, device, args.quota)
        torch.save({"Pb": Pb[:3000], "Pn": Pn, "Bb": Bb[:3000], "Bn": Bn, "yb": yb, "yn": yn}, cache)

    print(f"[shapes] base pooled {tuple(Pb.shape)} novel pooled {tuple(Pn.shape)}", flush=True)
    novel_classes = list(range(1, len(sp["novel"]) + 1))

    res = {"ckpt_epoch": ck.get("epoch"), "split": args.split,
           "n_base": int(len(yb)), "n_novel": int(len(yn)), "novel": sp["novel"], "results": {}}
    rng2 = np.random.RandomState(0)

    # A. box_head 输出
    XA = l2n(Bn)
    # B. 原始 pooled
    XB = l2n(Pn)
    # C. PCA 旁路（用 base 拟合）
    mean, V = pca_basis(Pb, 1024)
    print(f"[pca] basis {tuple(V.shape)}", flush=True)
    for k in (256, 512, 1024):
        Vk = V[:, :k]
        XC = l2n((Pn - mean) @ Vk)
        for K in (1, 5, 10):
            res["results"][f"pca{k}_K{K}"] = episode_acc(XC, yn, novel_classes, K, 20, rng2)
    # 对照：对 box_head 特征也做一次 PCA（维度不变，检验"是不是只是去相关"）
    meanb, Vb = pca_basis(Bb, 1024)
    XAb = l2n((Bn - meanb) @ Vb)
    for K in (1, 5, 10):
        res["results"][f"boxhead_pca_K{K}"] = episode_acc(XAb, yn, novel_classes, K, 20, rng2)
        res["results"][f"boxhead_raw_K{K}"] = episode_acc(XA, yn, novel_classes, K, 20, rng2)
        res["results"][f"pooled_raw_K{K}"] = episode_acc(XB, yn, novel_classes, K, 20, rng2)

    for k, v in sorted(res["results"].items()):
        print(f"  {k:20s} {v*100:6.2f}", flush=True)
    os.makedirs(os.path.dirname(os.path.join(here, args.out)), exist_ok=True)
    with open(os.path.join(here, args.out), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    torch.save({"Pb": Pb[:2000], "Pn": Pn, "Bb": Bb[:2000], "Bn": Bn, "yb": yb, "yn": yn},
               os.path.join(here, "data", "bypass_feats.pt"))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
