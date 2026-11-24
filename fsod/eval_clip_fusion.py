"""推理时 CLIP 语义注入：在 NMS 前的 proposal class_logits 上叠加 CLIP 同模态相似度。

动机（见 knowledge/CLIP同模态原型捞回_最小验证PASS.md）：
  被检测头判背景的 novel 目标不产生 detection box，事后重打分无法捞回；
  必须在 box_predictor 输出、NMS 之前注入。CLIP 相似度逐 proposal、由语义证据驱动，
  满足 SAFM 教训"不能一刀切放大 novel"。

高效实现：手动走一次 RoIHeads 前向，缓存 logits/proposals/reg；
CLIP 相似度算一次；所有 (lam, mode) 组合共享缓存，只重跑 postprocess + AP。

用法：
  python -m fsod.eval_clip_fusion --ckpt runs/ft2_ctrl_base/ft.pth --data 0712 \
      --shot 5 --max-imgs 500 --topk 300 --out data/clip_fusion_sweep.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from fsod import voc, engine
from fsod.model import build_model
from fsod.data import VOCDetDataset, detection_collate
from fsod.finetune import build_fewshot_set

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def crop_tensor(path, box, size=224):
    im = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(im.width, x2), min(im.height, y2)
    if x2 <= x1 or y2 <= y1:
        x2, y2 = min(im.width, x1 + 1), min(im.height, y1 + 1)
    crop = im.crop((x1, y1, x2, y2)).resize((size, size), Image.BILINEAR)
    a = np.asarray(crop, dtype=np.float32) / 255.0
    a = (a - np.array(CLIP_MEAN)) / np.array(CLIP_STD)
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


@torch.no_grad()
def encode_crops(clip_model, crops, dev, bs=128):
    out = []
    for i in range(0, len(crops), bs):
        batch = torch.stack(crops[i:i + bs]).to(dev)
        with torch.amp.autocast("cuda", enabled=True):
            f = clip_model.encode_image(batch)
        out.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(out, 0) if out else torch.zeros(0, 512)


@torch.no_grad()
def forward_cache(model, images, dev, topk):
    """手动走 backbone+rpn+roi_head 到 logits，缓存（不后处理）。返回每图 resume 信息。"""
    imgs, _ = model.transform(images, None)
    feats = model.backbone(imgs.tensors)
    proposals, _ = model.rpn(imgs, feats, imgs.image_sizes)
    # proposals: list[Tensor[N_i,4]]，已在 resized 坐标且按 score 降序
    if topk > 0:
        proposals = [p[:topk] for p in proposals]
    pooled = model.roi_heads.box_roi_pool(feats, proposals, imgs.image_sizes)
    feat = model.roi_heads.box_head(pooled)
    logits, reg = model.roi_heads.box_predictor(feat)
    return {
        "logits": logits.cpu(), "reg": reg.cpu(),
        "proposals": [p.cpu() for p in proposals],
        "image_sizes": imgs.image_sizes,
        "orig_sizes": [tuple(im.shape[-2:]) for im in images],  # (h,w)
    }


@torch.no_grad()
def postprocess(model, cache, logits, dev):
    ab, asc, al = model.roi_heads.postprocess_detections(
        logits.to(dev), cache["reg"].to(dev), [p.to(dev) for p in cache["proposals"]],
        cache["image_sizes"])
    out = []
    for i, (b, s, l) in enumerate(zip(ab, asc, al)):
        # postprocess_detections 输出的是 resized 图坐标；FasterRCNN.forward 外层会反缩放回原图，
        # 手动调用必须补上这一步，否则 boxes 落在 resized 尺度导致 AP 崩塌。
        oh, ow = cache["orig_sizes"][i]
        rh, rw = cache["image_sizes"][i]
        b = b * torch.tensor([ow / rw, oh / rh, ow / rw, oh / rh], device=b.device)
        out.append({"boxes": b.cpu().numpy(), "scores": s.cpu().numpy(),
                    "labels": l.cpu().numpy()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--max-imgs", type=int, default=500)
    ap.add_argument("--topk", type=int, default=300, help="每图保留 top-K proposals 算 CLIP")
    ap.add_argument("--clip-path", default="")
    ap.add_argument("--lams", default="0,3,6,10,15")
    ap.add_argument("--modes", default="all,novel,bg")
    ap.add_argument("--cache", default="", help="加载已落盘的 forward+CLIP 缓存，跳过最慢步骤")
    ap.add_argument("--save-cache", default="", help="把 forward+CLIP 缓存落盘")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    voc12 = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    if args.data == "0712":
        ann, tr, te = voc.prepare_0712(voc_root, voc12, os.path.join(here, "data"))
    else:
        ann, tr, te = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann, tr, te, args.split)
    ft_map = sp["ft_map"]
    n_all = len(ft_map)
    novel = sp["novel"]
    novel_labs = [ft_map[c] for c in novel]
    novel_idx = [l - 1 for l in novel_labs]   # logits 列索引（0-based）
    dev = torch.device("cuda")
    print(f"[data] novel={novel} novel_labs={novel_labs} topk={args.topk}", flush=True)

    # ---- CLIP 图像原型（支持集）----
    ft_samples, _ = build_fewshot_set(ann, tr, ft_map, sp["base"], novel, args.shot, seed=0)
    sup_crops, sup_labs = [], []
    for s in ft_samples:
        for box, lb in zip(s["boxes"], s["labels"]):
            sup_crops.append(crop_tensor(s["file"], box))
            sup_labs.append(lb)
    import clip as clip_lib
    clip_path = args.clip_path or os.path.join(here, "torch_home", "clip", "ViT-B-32.pt")
    clip_model, _ = clip_lib.load(clip_path, device=dev)
    clip_model.eval()
    sup_feat = encode_crops(clip_model, sup_crops, dev)
    protos = []
    for lb in range(1, n_all + 1):
        m = torch.tensor([l == lb for l in sup_labs])
        pf = sup_feat[m].mean(0) if m.sum() else torch.zeros(512)
        protos.append(F.normalize(pf.unsqueeze(0), dim=-1).squeeze(0))
    proto_mat = torch.stack(protos).to(dev)   # (21,512)，第0行=背景占位(零)
    print(f"[clip] prototypes={proto_mat.shape}", flush=True)

    # ---- 检测器 ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cka = ck.get("args", {})
    model = build_model(n_all + 1, cka.get("backbone", "resnet50"),
                        cka.get("min_size", 600), cka.get("max_size", 1000),
                        trainable_backbone_layers=5, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    te_samples = voc.samples_for(ann, te, ft_map, keep_difficult=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2,
                           shuffle=False, num_workers=2, collate_fn=detection_collate)

    # ---- 一次前向，缓存 logits + proposals，并算 CLIP sim ----
    caches, sims, all_gts = [], [], []
    ptr = 0
    t0 = __import__("time").time()
    if args.cache and os.path.exists(args.cache):
        blob = torch.load(args.cache, map_location="cpu", weights_only=False)
        caches, sims, all_gts = blob["caches"], blob["sims"], blob["gts"]
        ptr = blob["n_imgs"]
        # 旧缓存未存 orig_sizes：从 te_samples 的顺序重建（每 cache 2 张图）
        gi = 0
        for c in caches:
            if "orig_sizes" not in c:
                sizes = []
                for _ in c["proposals"]:
                    smp = te_samples[gi]
                    w, h = ann[smp["image_id"]]["size"]
                    sizes.append((h, w))
                    gi += 1
                c["orig_sizes"] = sizes
            else:
                gi += len(c["proposals"])
        print(f"[cache] loaded {ptr} imgs from {args.cache}", flush=True)
    else:
        for images, targets in te_loader:
            if ptr >= args.max_imgs:
                break
            images = [im.to(dev) for im in images]
            cache = forward_cache(model, images, dev, args.topk)
            # proposals(resized) -> 原图坐标 -> crop -> CLIP
            crops = []
            for i, props in enumerate(cache["proposals"]):
                oh, ow = cache["orig_sizes"][i]
                rh, rw = cache["image_sizes"][i]
                sx, sy = ow / rw, oh / rh
                smp = te_samples[ptr + i] if ptr + i < len(te_samples) else None
                path = smp["file"] if smp else ""
                for p in props:
                    bx = [p[0].item() * sx, p[1].item() * sy, p[2].item() * sx, p[3].item() * sy]
                    crops.append(crop_tensor(path, bx) if path else torch.zeros(3, 224, 224))
            pfeat = encode_crops(clip_model, crops, dev)   # (N,512) cpu
            sim = pfeat @ proto_mat.T.cpu()                 # (N,21) cpu, cos
            # 按图切分 sim（与 proposals 顺序一致）
            sim_chunks, off = [], 0
            for props in cache["proposals"]:
                n = props.shape[0]
                sim_chunks.append(sim[off:off + n])
                off += n
            caches.append({"logits": cache["logits"], "reg": cache["reg"],
                           "proposals": cache["proposals"], "image_sizes": cache["image_sizes"],
                           "orig_sizes": cache["orig_sizes"]})
            sims.append(sim_chunks)
            for _ in range(len(images)):
                smp = te_samples[ptr]
                all_gts.append({"boxes": np.array(smp["boxes"], dtype=np.float32).reshape(-1, 4),
                                "labels": np.array(smp["labels"], dtype=np.int64),
                                "difficult": np.array(smp["difficult"], dtype=np.int64)})
                ptr += 1
            if ptr % 200 == 0:
                print(f"  forward+clip {ptr}/{args.max_imgs} imgs {(__import__('time').time()-t0)/60:.1f}min", flush=True)
        print(f"[forward] {ptr} imgs in {(__import__('time').time()-t0)/60:.1f}min", flush=True)
        if args.save_cache:
            torch.save({"caches": caches, "sims": sims, "gts": all_gts, "n_imgs": ptr},
                       args.save_cache)
            print(f"[cache] saved {ptr} imgs -> {args.save_cache}", flush=True)

    # ---- baseline（lam=0 直接用缓存 logits 后处理）----
    def run_config(lam, mode):
        all_preds = []
        for ci, cache in enumerate(caches):
            logits = cache["logits"].clone()
            sc = sims[ci]   # per-图 list，每个 (n_i, 21)
            if lam > 0:
                # 按图切分 logits，保证与 sc 的图顺序、proposal 顺序一致
                sizes = [p.shape[0] for p in cache["proposals"]]
                logit_chunks, reg_chunks, off = [], [], 0
                for ns in sizes:
                    logit_chunks.append(logits[off:off + ns])
                    reg_chunks.append(cache["reg"][off:off + ns])
                    off += ns
                for gi, s in enumerate(sc):
                    lg = logit_chunks[gi]
                    if mode == "all":
                        # s 的 20 列对应类别 1..20；logits[:,1:] 是类别 1..20，逐列对齐
                        lg[:, 1:] = lg[:, 1:] + lam * s
                    elif mode == "novel":
                        # novel_idx 是 0-based 类索引(15..19)；sim 列 = 类索引，logits 列 = 类索引+1
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * s[:, li]
                    elif mode.startswith("gate"):
                        # 门控：只有 CLIP 认为 novel 证据显著强于 base 时才放大 novel，
                        # 避免把 base 框误判成 novel（解决 novel 模式 base 掉点）。
                        # gate 阈值编码在 mode 名里，如 gate0.05 / gate0.10
                        thr = float(mode[4:]) if len(mode) > 4 else 0.05
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bsim = s[:, base_idx].max(1).values
                        gate = torch.clamp((nsim - bsim - thr) / 0.1, 0, 1)  # 0..1 连续门控
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                    elif mode.startswith("mbm"):
                        # 判据2：nsim - base 均值（而非 base 最大值）。
                        # 诊断显示 small 上 margin(nsim-bsim) 仅 +0.005 而 mbm +0.112，
                        # base 均值不受个别虚高 base 类干扰，对小目标更稳。
                        thr = float(mode[3:]) if len(mode) > 3 else 0.0
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bmean = s[:, base_idx].mean(1)
                        gate = torch.clamp((nsim - bmean - thr) / 0.1, 0, 1)
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                    elif mode.startswith("loc"):
                        # 判据4：定位×语义联合门控（一区机制对症）。
                        # 定位质量 proxy = reg delta 幅度（位移小=框准）；只在定位可靠时信任语义。
                        # 格式 loc{thr}_{lambda_scale}，如 loc2.0_1.0
                        parts = mode[3:].split("_")
                        thr = float(parts[0]); 
                        # cache["reg"] 已在 cache 里；此处需按图切分
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bsim = s[:, base_idx].max(1).values
                        sem_gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
                        # reg 位移幅度（前景类平均 L2），位移越小越可靠
                        rchunk = reg_chunks[gi]                      # (n_i, 84)
                        rmag = rchunk.reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)  # 前景类平均位移
                        loc_gate = torch.clamp((thr - rmag) / 1.0 + 0.5, 0, 1)  # rmag<thr -> 高
                        gate = sem_gate * loc_gate
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                    elif mode.startswith("rel"):
                        # 可靠性加权证据融合（一区方法）：corr(nsim,IoU) 在 small novel 为负，
                        # 裸 nsim 加权会误放行定位差的 proposal。用 reg delta 范数作定位代理，
                        # 让语义证据的注入强度随"定位可靠性"连续衰减，而非简单线性门控。
                        # 格式 rel{kind}{tau}，如 relinv1.0 / relsig0.5；强度由 --lams 控制
                        spec = mode[3:]
                        if spec.startswith("inv"):
                            kind, rest = "inv", spec[3:]
                        elif spec.startswith("sig"):
                            kind, rest = "sig", spec[3:]
                        else:
                            kind, rest = "inv", spec
                        tau = float(rest)
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bsim = s[:, base_idx].max(1).values
                        sem_gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
                        rchunk = reg_chunks[gi]                       # (n_i, 84)
                        rmag = rchunk.reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)
                        if kind == "inv":
                            w = 1.0 / (1.0 + rmag / max(tau, 1e-4))   # 位移越大权重越低
                        else:
                            w = torch.sigmoid((tau - rmag) / 0.3)      # 尖锐门控
                        gate = sem_gate * w
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                    elif mode.startswith("np"):
                        # 判据3：低温 softmax 的 novel 概率和。诊断显示 small novel=0.48
                        # vs base=0.047（10 倍），不依赖 margin 绝对值。
                        # 格式 np{T}_{thr}，如 np0.01_0.2
                        parts = mode[2:].split("_")
                        T = float(parts[0]); thr = float(parts[1]) if len(parts) > 1 else 0.1
                        prob = torch.softmax(s / T, dim=1)
                        nprob = prob[:, novel_idx].sum(1)
                        gate = torch.clamp((nprob - thr) / (1 - thr + 1e-6), 0, 1)
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                    elif mode == "bg":
                        # 背景 logit 减去 lam * 最强 novel 证据（完全贴合病根）
                        bgv = s[:, novel_idx].max(1).values
                        lg[:, 0] = lg[:, 0] - lam * bgv
                    elif mode.startswith("dualb"):
                        # dualb{c}: gate0.0 增强(lam) + 无加权 bg 抑制(c)。
                        bgc = float(mode[5:]) if len(mode) > 5 else lam
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bsim = s[:, base_idx].max(1).values
                        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                        lg[:, 0] = lg[:, 0] - bgc * nsim
                    elif mode.startswith("dualr"):
                        # ①′ dualr{c}_{tau}: gate0.0 增强(lam) + 可靠性加权 bg 抑制(c*nsim*w)。
                        # corr(rmag,IoU)≈-0.41~-0.56（代理有效）；w 乘在抑制侧：
                        # 只对定位可靠的 proposal 压 bg（捞回框准 novel、放过不可靠 FP）。
                        spec = mode[5:]
                        bgc, tau = [float(x) for x in spec.split("_")]
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bsim = s[:, base_idx].max(1).values
                        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
                        rchunk = reg_chunks[gi]
                        rmag = rchunk.reshape(-1, n_all + 1, 4)[:, 1:, :].norm(dim=-1).mean(1)
                        w = 1.0 / (1.0 + rmag / max(tau, 1e-4))
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                        lg[:, 0] = lg[:, 0] - bgc * nsim * w
                    elif mode.startswith("dualobj"):
                        # 论文3 缺口#1a：RPN objectness 作 Q_loc 对照（v1.3 防御3）。
                        # 与 dualr 同构：w=objectness（高分=可靠），乘在 bg 抑制侧。
                        # dualobj{c}: gate0.0 增强(lam) + c*nsim*obj 抑制背景。
                        bgc = float(mode[7:]) if len(mode) > 7 else 3.0
                        gimg = cache_offsets[ci] + gi
                        ow = obj_blob["objectness"][gimg].to(lg.dtype)
                        assert ow.shape[0] == lg.shape[0], f"obj/proposal misalign img {gimg}"
                        base_idx = [li for li in range(n_all) if li not in novel_idx]
                        nsim = s[:, novel_idx].max(1).values
                        bsim = s[:, base_idx].max(1).values
                        gate = torch.clamp((nsim - bsim) / 0.1, 0, 1)
                        for li in novel_idx:
                            lg[:, li + 1] = lg[:, li + 1] + lam * gate * s[:, li]
                        lg[:, 0] = lg[:, 0] - bgc * nsim * ow
            all_preds.extend(postprocess(model, cache, logits, dev))
        return all_preds

    novel_names = {ft_map[c]: c for c in novel}
    base_names = {ft_map[c]: c for c in sp["base"]}
    lams = [float(x) for x in args.lams.split(",")]
    modes = args.modes.split(",")

    # ---- objectness 对照数据（论文3 缺口#1a，预加载）----
    obj_blob = None
    cache_offsets = []
    if any(m.startswith("dualobj") for m in modes):
        obj_blob = torch.load("data/objectness_full.pt", map_location="cpu", weights_only=False)
        _o = 0
        for c in caches:
            cache_offsets.append(_o); _o += len(c["proposals"])
        assert len(obj_blob["objectness"]) == sum(len(c["proposals"]) for c in caches), \
            f"obj imgs {len(obj_blob['objectness'])} vs cache imgs {sum(len(c['proposals']) for c in caches)}"
    results = {}
    # baseline
    print("\n=== baseline (lam=0) ===", flush=True)
    preds0 = run_config(0, "all")
    import contextlib as _cl, io as _io
    with _cl.redirect_stdout(_io.StringIO()):
        nm0, mm0 = engine.report_ap(preds0, all_gts, novel_names, tag="novel base")
        bm0, bbase0 = engine.report_ap(preds0, all_gts, base_names, tag="base base")
    results["baseline"] = {"novel_mAP": mm0, "base_mAP": bbase0}
    from fsod.ap import compute_ap_per_class as _capc
    _bs = {sc: (lambda aps: float(np.mean(list(aps.values()))) if aps else 0.0)(
        _capc(preds0, all_gts, novel_labs, area_range=engine.AREA_BINS[sc]))
        for sc in ["small", "medium", "large"]}
    results["baseline"]["novel_scale"] = _bs
    print(f"  baseline novel={mm0*100:.2f} base={bbase0*100:.2f}  "
          f"s/m/l={_bs['small']*100:5.2f}/{_bs['medium']*100:5.2f}/{_bs['large']*100:5.2f}", flush=True)

    from fsod.ap import compute_ap_per_class
    def scale_ap(preds, labs, area):
        aps = compute_ap_per_class(preds, all_gts, labs, area_range=area)
        return float(np.mean(list(aps.values()))) if aps else 0.0

    import contextlib, io
    for mode in modes:
        for lam in lams:
            if lam == 0:
                continue
            preds = run_config(lam, mode)
            with contextlib.redirect_stdout(io.StringIO()):
                named, mm = engine.report_ap(preds, all_gts, novel_names)
                bnamed, bmm = engine.report_ap(preds, all_gts, base_names)
            scales = {sc: scale_ap(preds, novel_labs, engine.AREA_BINS[sc])
                      for sc in ["small", "medium", "large"]}
            key = f"{mode}_lam{lam:g}"
            results[key] = {"novel_mAP": mm, "base_mAP": bmm,
                            "novel_scale": {k: v for k, v in scales.items()}}
            print(f"  {key:18s} novel={mm*100:6.2f} ({(mm-mm0)*100:+5.2f})  "
                  f"base={bmm*100:6.2f} ({(bmm-bbase0)*100:+5.2f})  "
                  f"s/m/l={scales['small']*100:5.2f}/{scales['medium']*100:5.2f}/{scales['large']*100:5.2f}",
                  flush=True)

    out = {"ckpt": args.ckpt, "data": args.data, "shot": args.shot, "topk": args.topk,
           "max_imgs": ptr, "clip": "ViT-B-32 image tower, proposal crop -> cos to 20 proto",
           "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
