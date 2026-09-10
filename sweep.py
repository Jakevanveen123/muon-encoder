import argparse
import glob
import json
import math
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--size", default="17m")
parser.add_argument("--optimizer", required=True, choices=["muon", "adamw"])
parser.add_argument("--phase", required=True, choices=["sweep", "seeds", "budget"])
parser.add_argument("--seeds", type=int, nargs="+", default=None)
parser.add_argument("--out_dir", default="runs")
args, extra = parser.parse_known_args()

LRS = dict(
    muon=[0.005, 0.01, 0.02, 0.04, 0.08],
    adamw=[3e-4, 6e-4, 1e-3, 2e-3, 4e-3],
)
lrs = LRS[args.optimizer] if args.size in ("17m", "32m") else LRS[args.optimizer][1:4]


def run(**kwargs):
    cmd = ["python", "train.py", "--size", args.size, "--out_dir", args.out_dir]
    cmd += [f"--{k}={v}" for k, v in kwargs.items()] + extra
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def best_lr():
    while True:
        paths = glob.glob(f"{args.out_dir}/{args.size}-{args.optimizer}-*-tpp20-s0/results.json")
        results = [json.load(open(p)) for p in paths]
        if len(results) >= len(lrs):
            finite = [r for r in results if math.isfinite(r["final_val_loss"])]
            return min(finite, key=lambda r: r["final_val_loss"])["args"]["lr"]
        print(f"waiting for sweep: {len(results)}/{len(lrs)} finished", flush=True)
        time.sleep(60)


if args.phase == "sweep":
    for lr in lrs:
        run(treatment=args.optimizer, lr=lr, seed=0)
elif args.phase == "seeds":
    for seed in args.seeds or (1, 2):
        run(treatment=args.optimizer, lr=best_lr(), seed=seed)
elif args.phase == "budget":
    for tokens_per_param in (30, 40):
        for seed in args.seeds or (0, 1, 2):
            run(treatment="adamw", lr=best_lr(), seed=seed, tokens_per_param=tokens_per_param)
