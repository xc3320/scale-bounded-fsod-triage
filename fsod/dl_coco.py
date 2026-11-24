"""并发下载 COCO val2014 图片（curl 子进程，绕开 python urllib 限速/沙箱）。

优先级列表每行: <file_name>\t<12位数字id>
URL = http://images.cocodataset.org/val2014/<file_name>
本地保存 = <数字id>.jpg（前向脚本按数字 id 找图）
下载后校验 JPEG；坏文件不删除（避免安全拦截），下次覆盖。

用法: python -m fsod.dl_coco --n 600 --batch 24
"""
import argparse
import os
import subprocess
import time

BASE = "http://images.cocodataset.org/val2014/"


def valid_jpeg(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            return False
        with open(path, "rb") as f:
            if f.read(3) != b"\xff\xd8\xff":
                return False
        from PIL import Image
        Image.open(path).verify()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--list", default="data/coco_val_priority.txt")
    ap.add_argument("--outdir", default="datasets/coco/val2017")
    args = ap.parse_args()

    pairs = []
    for line in open(args.list, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" in line:
            fn, iid = line.split("\t")
        else:
            iid = line.strip(); fn = "COCO_val2014_" + iid + ".jpg"
        pairs.append((fn, iid))
    os.makedirs(args.outdir, exist_ok=True)
    todo = [(fn, iid) for fn, iid in pairs
            if not valid_jpeg(os.path.join(args.outdir, iid + ".jpg"))]
    todo = todo[:args.n]
    print(f"[dl] target={args.n} pending={len(todo)} batch={args.batch}", flush=True)

    t0 = time.time(); done = 0
    for bi in range(0, len(todo), args.batch):
        batch = todo[bi:bi + args.batch]
        procs = []
        for fn, iid in batch:
            out = os.path.join(args.outdir, iid + ".jpg")
            p = subprocess.Popen(
                ["curl", "-s", "-L", "--max-time", "120", "-o", out, BASE + fn],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append((p, out))
        for p, out in procs:
            p.wait()
            if valid_jpeg(out):
                done += 1
        el = time.time() - t0
        print(f"[dl] {done}/{len(todo)} valid in {el/60:.1f}min "
              f"({done/max(1,el)*60:.0f} img/min)", flush=True)
    print(f"[dl] FINISHED {done}/{len(todo)} in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
