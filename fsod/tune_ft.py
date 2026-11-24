"""微调超参粗筛：批量跑 finetune(跳过全量评测) + eval_only(只测 nAP)，汇总成表。

用法（base 训练完成后）：
  python -m fsod.tune_ft --ckpt runs/base_s1_0712/ep10.pth --data 0712 --shot 5

设计依据：
  - 之前 lr=0.02/2000it 在微调集上 novel 记住 98.9%，但测试 nAP 仅 24.29 —— 明显过拟合。
  - TFA 原文只训 box_predictor（冻结 box_head），lr=0.001，约 1000 it。
  - 因此扫 freeze_boxhead × lr × iters，先固定 K=5 找方向，再在最优配置上跑 K=1/10。
"""
import argparse
import itertools
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, "env_fsl", "Scripts", "python.exe")


def run(cmd, log_path):
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    print(f"    rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--shot", type=int, default=5)
    ap.add_argument("--freeze", default="0,1", help="freeze_boxhead 取值列表")
    ap.add_argument("--lr", default="0.005,0.01,0.02")
    ap.add_argument("--iters", default="1000,2000")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="data/tune_ft.json")
    args = ap.parse_args()

    freezes = [int(x) for x in args.freeze.split(",") if x]
    lrs = [float(x) for x in args.lr.split(",") if x]
    iters = [int(x) for x in args.iters.split(",") if x]

    res_path = os.path.join(HERE, args.out_json)
    results = {}
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)

    combos = list(itertools.product(freezes, lrs, iters))
    print(f"[tune] {len(combos)} configs, shot={args.shot}, data={args.data}", flush=True)

    for fb, lr, it in combos:
        tag = f"tune_fb{fb}_lr{lr:g}_it{it}_s{args.split}_k{args.shot}"
        if tag in results and "novel_mAP50" in results[tag]:
            print(f"[skip] {tag}", flush=True)
            continue
        out = os.path.join(HERE, "runs", tag)
        log = os.path.join(HERE, "runs", f"{tag}.log")
        os.makedirs(out, exist_ok=True)
        print(f"[run] {tag}", flush=True)

        cmd_ft = [PY, "-u", "-m", "fsod.finetune",
                  "--split", str(args.split), "--shot", str(args.shot),
                  "--ckpt", args.ckpt, "--out", out,
                  "--iters", str(it), "--lr", str(lr), "--bs", str(args.bs),
                  "--head", "tfa", "--tag", tag, "--data", args.data,
                  "--seed", str(args.seed), "--workers", "4",
                  "--skip-eval", "1", "--freeze-boxhead", str(fb)]
        if run(cmd_ft, log) != 0:
            results[tag] = {"error": "finetune failed"}
            continue

        eval_json = f"runs/{tag}/eval_quick.json"
        cmd_ev = [PY, "-u", "-m", "fsod.eval_only",
                  "--ft", os.path.join(out, "ft.pth"),
                  "--base-ckpt", args.ckpt,
                  "--split", str(args.split), "--shot", str(args.shot),
                  "--data", args.data, "--head", "tfa",
                  "--eval-gt", "0", "--prop-recall", "0", "--diag-head", "0",
                  "--out", eval_json]
        if run(cmd_ev, log) != 0:
            results[tag] = {"error": "eval failed"}
            continue

        with open(os.path.join(HERE, eval_json)) as f:
            ev = json.load(f)
        ev.update({"freeze_boxhead": fb, "lr": lr, "iters": it})
        results[tag] = ev
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"    novel={ev.get('novel_mAP50',0)*100:.2f} base={ev.get('base_mAP50',0)*100:.2f}",
              flush=True)

    print("\n===== SUMMARY (sorted by novel) =====", flush=True)
    valid = [(t, r) for t, r in results.items() if "novel_mAP50" in r]
    for tag, r in sorted(valid, key=lambda x: -x[1]["novel_mAP50"]):
        sc = r.get("novel_scale", {})
        print(f"{tag:38s} novel={r['novel_mAP50']*100:6.2f}  base={r['base_mAP50']*100:6.2f}"
              f"  sm/med/lg={sc.get('small',0)*100:5.2f}/{sc.get('medium',0)*100:5.2f}/"
              f"{sc.get('large',0)*100:5.2f}", flush=True)
    for tag, r in results.items():
        if "novel_mAP50" not in r:
            print(f"{tag:38s} {r}", flush=True)


if __name__ == "__main__":
    main()
