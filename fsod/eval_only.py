"""只做评测：加载已微调的 ft.pth，跑完整 nAP / oracle-proposal / RPN 召回 / 尺度分层。"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from fsod import voc, engine
from fsod.data import VOCDetDataset, detection_collate
from fsod.model import build_model, expand_classifier, freeze_backbone_and_rpn
from fsod.bypass import install_bypass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True, help="ft.pth 路径")
    ap.add_argument("--base-ckpt", default="", help="留空则按 --data 自动选择")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--data", type=str, default="07", choices=["07", "0712"])
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--head", default="tfa", choices=["tfa", "bypass"])
    ap.add_argument("--basis", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval-gt", type=int, default=1)
    ap.add_argument("--prop-recall", type=int, default=1)
    ap.add_argument("--ctx-sub", type=float, default=0.0, help="上下文扣除系数 lam")
    ap.add_argument("--diag-head", type=int, default=0, help="1=GT 框上直接诊断分类头（含按尺度/去背景行）")
    ap.add_argument("--bg-delta", default="0,0,0", help="按 small,medium,large 从背景 logit 减去的值")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("TORCH_HOME", os.path.join(here, "torch_home"))
    voc_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2007")
    device = torch.device("cuda")

    voc12_root = os.path.join(here, "datasets", "VOCdevkit", "VOC2012")
    if getattr(args, "data", "07") == "0712":
        ann_all, train_ids, test_ids = voc.prepare_0712(voc_root, voc12_root, os.path.join(here, "data"))
    else:
        ann_all, train_ids, test_ids = voc.prepare(voc_root, os.path.join(here, "data"))
    sp = voc.build_split(ann_all, train_ids, test_ids, args.split)
    ft_map = sp["ft_map"]
    n_base, n_all = len(sp["base"]), len(ft_map)

    if not args.base_ckpt:
        args.base_ckpt = "runs/base_s1_0712/ep10.pth" if args.data == "0712" else "runs/base_s1/ep10.pth"
    ck = torch.load(os.path.join(here, args.base_ckpt), map_location="cpu", weights_only=False)
    ca = ck.get("args", {})
    model = build_model(n_base + 1, ca.get("backbone", "resnet50"),
                        ca.get("min_size", 600), ca.get("max_size", 1000),
                        pretrained=False)
    model.load_state_dict(ck["model"])
    if args.head == "tfa":
        model = expand_classifier(model, n_all + 1, n_base + 1)
    else:
        basis = torch.load(os.path.join(here, args.basis), map_location="cpu",
                           weights_only=False)["V"]
        model = install_bypass(model, 256 * 7 * 7, n_all + 1, n_base + 1,
                               mid=1024, mode="pca", init_basis=basis)
    sd = torch.load(os.path.join(here, args.ft), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd)
    deltas = [float(x) for x in args.bg_delta.split(",")]
    if args.ctx_sub > 0:
        from fsod.calib import ContextSubtract
        model = ContextSubtract(args.ctx_sub).install(model)
    if any(abs(d) > 0 for d in deltas):
        from fsod.calib import ScaleBgCalibrator
        model = ScaleBgCalibrator(deltas).install(model)
    model.to(device)
    model.eval()

    te_samples = voc.samples_for(ann_all, test_ids, ft_map, keep_difficult=True)
    te_loader = DataLoader(VOCDetDataset(te_samples, train=False), batch_size=2,
                           shuffle=False, num_workers=2, collate_fn=detection_collate,
                           pin_memory=True)
    novel_names = {ft_map[c]: c for c in sp["novel"]}
    base_names = {ft_map[c]: c for c in sp["base"]}

    p, g = engine.evaluate(model, te_loader, te_samples, device)
    _, m_n = engine.report_ap(p, g, novel_names, tag="novel")
    _, m_b = engine.report_ap(p, g, base_names, tag="base")
    sn = engine.report_ap_by_scale(p, g, novel_names, tag="novel-scale")
    sb = engine.report_ap_by_scale(p, g, base_names, tag="base-scale")
    res = {"novel_mAP50": m_n, "base_mAP50": m_b,
           "novel_scale": {k: v["mAP50"] for k, v in sn.items()},
           "base_scale": {k: v["mAP50"] for k, v in sb.items()}}
    if args.eval_gt:
        pg, gg = engine.evaluate_gt_proposals(model, te_loader, te_samples, device)
        _, m_g = engine.report_ap(pg, gg, novel_names, tag="novel-GTprop")
        _, m_gb = engine.report_ap(pg, gg, base_names, tag="base-GTprop")
        res["novel_mAP50_gtprop"] = m_g
        res["base_mAP50_gtprop"] = m_gb
        sng = engine.report_ap_by_scale(pg, gg, novel_names, tag="novel-GTprop-scale")
        res["novel_gtprop_scale"] = {k: v["mAP50"] for k, v in sng.items()}
    if args.diag_head:
        d = engine.head_diagnostic(model, te_loader, te_samples, device,
                                   list(novel_names.keys()), list(base_names.keys()))
        res["head_diag"] = d
        print("[head-diag] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in d.items()), flush=True)
    if args.prop_recall:
        rec, tot = engine.proposal_recall(model, te_loader, te_samples, device)
        print(f"[prop-recall] {rec} tot={tot}", flush=True)
        res["prop_recall"] = rec
        res["prop_recall_total"] = tot
    out = args.out or (os.path.dirname(args.ft) + "/eval.json")
    with open(os.path.join(here, out) if not os.path.isabs(out) else out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
