import argparse
import glob
import json
import math
import statistics
NS_COEFFS = (3.4445, -4.7750, 2.0315)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def ns_threshold(target=0.1, steps=5):
    a, b, c = NS_COEFFS
    lo, hi = 1e-8, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        x = mid
        for _ in range(steps):
            x = a * x + b * x**3 + c * x**5
        lo, hi = (mid, hi) if x < target else (lo, mid)
    return hi


NS_THRESHOLD = ns_threshold()

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", default="runs")
parser.add_argument("--size", default="17m")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--stable_from", type=float, default=0.5)
parser.add_argument("--paper_threshold", type=float, default=0.003)
args = parser.parse_args()

paths = sorted(glob.glob(f"{args.runs_dir}/{args.size}-muon-*/spectra.jsonl"))
if args.lr is not None:
    paths = [p for p in paths if f"-lr{args.lr:g}-" in p]
rows = []
for path in paths:
    run = path.split("/")[-2]
    for line in open(path):
        r = json.loads(line)
        r["run"] = run
        rows.append(r)

last = max(r["step"] for r in rows)
early = [r for r in rows if r["step"] <= 0.05 * last]
stable = [r for r in rows if r["step"] >= args.stable_from * last]


def frac_below(row, value):
    points = sorted(((row[f"sv_q{q}"], q) for q in QUANTILES), reverse=True)
    if value >= points[0][0]:
        return 1 - points[0][1]
    if value <= points[-1][0]:
        return 1 - points[-1][1]
    for (s1, q1), (s2, q2) in zip(points, points[1:]):
        if s2 <= value <= s1:
            w = (math.log(s1) - math.log(value)) / (math.log(s1) - math.log(s2))
            return 1 - (q1 + w * (q2 - q1))
    return 1 - points[-1][1]


def group(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


print(f"runs: {len(paths)}   steps 0..{last}   NS threshold {NS_THRESHOLD:.2e}")
for name, phase in [("first 5%", early), (f"last {100 * (1 - args.stable_from):.0f}%", stable)]:
    per = [statistics.mean([r["frac_below_threshold"] for r in v]) for v in group(phase, lambda r: (r["run"], r["layer"])).values()]
    print(f"  {name:10s} frac below NS threshold: mean {statistics.mean(per):.3f}  max {max(per):.3f}")

print(f"\nstable-phase medians by layer (pooled over {len(paths)} runs)")
print(f"{'layer':16s} {'sv_q0.1':>9s} {'sv_q0.5':>9s} {'sv_q0.9':>9s} {'<NS':>7s} {'<' + str(args.paper_threshold):>8s}")
for layer, rs in sorted(group(stable, lambda r: r["layer"]).items(), key=lambda kv: (int(kv[0].split(".")[0]), kv[0])):
    med = {q: statistics.median([r[f"sv_q{q}"] for r in rs]) for q in QUANTILES}
    ns = statistics.median([r["frac_below_threshold"] for r in rs])
    paper = statistics.median([frac_below(r, args.paper_threshold) for r in rs])
    chk = statistics.median([frac_below(r, NS_THRESHOLD) for r in rs])
    print(f"{layer:16s} {med[0.1]:9.2e} {med[0.5]:9.2e} {med[0.9]:9.2e} {ns:7.3f} {paper:8.3f} {chk:7.3f}")

print("\nby matrix type (stable phase)")
for kind, rs in sorted(group(stable, lambda r: r["layer"].split(".", 1)[1]).items()):
    ns = statistics.median([r["frac_below_threshold"] for r in rs])
    paper = statistics.median([frac_below(r, args.paper_threshold) for r in rs])
    print(f"  {kind:10s} <NS {ns:.3f}   <{args.paper_threshold} {paper:.3f}")
