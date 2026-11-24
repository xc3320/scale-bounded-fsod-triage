"""一键续训：检测已完成 epoch，未到目标就自动从最新检查点续训。

用法：python -m fsod.auto_resume_out --out runs/base_s1_0712 --epochs 10
中断后重跑本脚本即可，无需手动判断该从哪里接。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, "env_fsl", "Scripts", "python.exe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--data", default="0712", choices=["07", "0712"])
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()

    out_abs = os.path.join(HERE, args.out)
    log_path = os.path.join(out_abs, "log.json")
    ck_path = os.path.join(out_abs, "last.pth")

    done = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            rec = json.load(f)
        done = max([e.get("epoch", 0) for e in rec.get("epochs", [])] or [0])

    if done >= args.epochs:
        print(f"[auto-resume] already done: {done}/{args.epochs} epochs")
        return

    if not os.path.exists(ck_path):
        print(f"[auto-resume] no checkpoint at {ck_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[auto-resume] {done}/{args.epochs} done, resuming from checkpoint epoch {done}")
    cmd = [PY, "-u", "-m", "fsod.train_base",
           "--data", args.data, "--split", str(args.split),
           "--epochs", str(args.epochs), "--bs", str(args.bs), "--lr", str(args.lr),
           "--out", args.out, "--workers", str(args.workers),
           "--eval-every", str(args.eval_every), "--resume", "1"]
    logf = os.path.join(HERE, "runs", f"{os.path.basename(args.out)}_auto.log")
    with open(logf, "a", encoding="utf-8") as f:
        f.write("\n[auto-resume] " + " ".join(cmd) + "\n")
        rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[auto-resume] exit={rc}, log={logf}")


if __name__ == "__main__":
    main()
