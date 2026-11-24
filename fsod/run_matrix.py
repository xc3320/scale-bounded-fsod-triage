"""批量跑微调矩阵：{shot} x {head}，汇总 nAP。"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, "env_fsl", "Scripts", "python.exe")


def run(cmd, log_path):
    print("[run]", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    print(f"[done] rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--basis", default="")
    ap.add_argument("--shots", default="1,5,10")
    ap.add_argument("--heads", default="tfa,bypass")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extra", default="", help="额外参数，如 '--bypass-mode random'")
    ap.add_argument("--tag-prefix", default="")
    ap.add_argument("--data", default="07", choices=["07", "0712"])
    ap.add_argument("--out-json", default="data/matrix_results.json")
    args = ap.parse_args()

    shots = [int(x) for x in args.shots.split(",") if x]
    heads = [x for x in args.heads.split(",") if x]
    res_path = os.path.join(HERE, args.out_json)
    results = {}
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)

    for shot in shots:
        for head in heads:
            tag = f"{args.tag_prefix}{head}_s{args.split}_k{shot}"
            out = os.path.join(HERE, "runs", tag)
            log = os.path.join(HERE, "runs", f"{tag}.log")
            cmd = [PY, "-u", "-m", "fsod.finetune",
                   "--split", str(args.split), "--shot", str(shot),
                   "--ckpt", args.ckpt, "--out", out,
                   "--iters", str(args.iters), "--bs", str(args.bs),
                   "--lr", str(args.lr), "--tag", tag, "--head", head,
                   "--data", args.data]
            if head == "bypass":
                cmd += ["--basis", args.basis]
            if args.extra:
                cmd += args.extra.split()
            rc = run(cmd, log)
            rj = os.path.join(out, "result.json")
            if rc == 0 and os.path.exists(rj):
                with open(rj) as f:
                    results[tag] = json.load(f)
            else:
                results[tag] = {"error": f"rc={rc}"}
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====", flush=True)
    for tag, r in sorted(results.items()):
        if "novel_mAP50" in r:
            print(f"{tag:32s} novel={r['novel_mAP50']*100:6.2f}  base={r['base_mAP50']*100:6.2f}", flush=True)
        else:
            print(f"{tag:32s} {r}", flush=True)


if __name__ == "__main__":
    main()
