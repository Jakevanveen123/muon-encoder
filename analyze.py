import argparse
import glob
import json
import math
import random
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", default="runs")
parser.add_argument("--size", default="17m")
parser.add_argument("--bootstrap", type=int, default=20000)
args = parser.parse_args()

results = [json.load(open(p)) for p in glob.glob(f"{args.runs_dir}/{args.size}-*/results.json")]
results = [r for r in results if math.isfinite(r["final_val_loss"])]


def select(treatment, lr=None, tpp=20):
    out = []
    for r in results:
        a = r["args"]
        if a["treatment"] == treatment and a["tokens_per_param"] == tpp:
            if lr is None or a["lr"] == lr:
                out.append(r)
    return out


def best_lr(treatment):
    by_lr = {}
    for r in select(treatment):
        by_lr.setdefault(r["args"]["lr"], []).append(r["final_val_loss"])
    means = {lr: statistics.mean(v) for lr, v in by_lr.items()}
    best = min(means, key=means.get)
    grid = sorted(means)
    i = grid.index(best)
    bracketed = 0 < i < len(grid) - 1
    return best, means, bracketed


def fit_line(points):
    xs = [math.log(n) for n, _ in points]
    ys = [l for _, l in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    return slope, my - slope * mx


def tokens_to_reach(points, target):
    slope, intercept = fit_line(points)
    return math.exp((target - intercept) / slope)


muon_lr, muon_means, muon_ok = best_lr("muon")
adamw_lr, adamw_means, adamw_ok = best_lr("adamw")

print(f"=== {args.size} learning-rate sweeps (tpp20) ===")
for name, means, best, ok in [("muon", muon_means, muon_lr, muon_ok), ("adamw", adamw_means, adamw_lr, adamw_ok)]:
    line = "  ".join(f"{lr:g}:{means[lr]:.4f}" + ("*" if lr == best else "") for lr in sorted(means))
    print(f"{name:6s} {line}")
    print(f"       best {best:g}, {'bracketed' if ok else 'AT GRID EDGE — baseline may be under-tuned'}")

muon_losses = [r["final_val_loss"] for r in select("muon", muon_lr)]
budgets = sorted({r["args"]["tokens_per_param"] for r in results if r["args"]["treatment"] == "adamw" and r["args"]["lr"] == adamw_lr})
adamw_by_budget = {}
for tpp in budgets:
    rs = select("adamw", adamw_lr, tpp)
    adamw_by_budget[tpp] = (rs[0]["tokens"], [r["final_val_loss"] for r in rs])

print(f"\n=== final losses (best lr, {len(muon_losses)} seeds) ===")
print(f"muon  tpp20  {statistics.mean(muon_losses):.4f}  sd {statistics.stdev(muon_losses):.4f}")
for tpp, (n, ls) in adamw_by_budget.items():
    print(f"adamw tpp{tpp:<4g} {statistics.mean(ls):.4f}  sd {statistics.stdev(ls):.4f}   ({n / 1e6:.0f}M tokens)")

muon_n = select("muon", muon_lr)[0]["tokens"]
target = statistics.mean(muon_losses)
curve = [(n, statistics.mean(ls)) for n, ls in adamw_by_budget.values()]
point = tokens_to_reach(curve, target) / muon_n

boots = []
for _ in range(args.bootstrap):
    t = statistics.mean(random.choices(muon_losses, k=len(muon_losses)))
    c = [(n, statistics.mean(random.choices(ls, k=len(ls)))) for n, ls in adamw_by_budget.values()]
    boots.append(tokens_to_reach(c, t) / muon_n)
boots.sort()
lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

gap = statistics.mean(adamw_by_budget[20][1]) - target
noise = statistics.stdev(adamw_by_budget[20][1])
print(f"\n=== speedup ===")
print(f"loss gap at equal tokens  {gap:+.4f}  ({gap / noise:.1f}x adamw seed sd)")
print(f"speedup {point:.3f}x   95% CI [{lo:.3f}, {hi:.3f}]")
print("DIFFERENCE WITHIN SEED NOISE" if lo < 1.0 < hi else "significant at the 5% level")
