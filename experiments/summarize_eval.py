"""Summarize MAPF Evaluator results.csv files (one per map) into one table.

    python experiments/summarize_eval.py "~/Projects/MAPF/MAPF Evaluator/results/tlg-*/results.csv"

Per (map, N) and solver: mean soc/lb over instances every solver solved, solved/total.
"""
import csv, glob, sys, os
from collections import defaultdict

files = [f for pat in sys.argv[1:] for f in glob.glob(os.path.expanduser(pat))]
rows = [r for f in files for r in csv.DictReader(open(f))]
solvers = []
for r in rows:
    if r["solver_id"] not in solvers:
        solvers.append(r["solver_id"])
short = {s: s.replace("lg_lacam_fork:", "C++ ").replace("traffic_load_lacam:", "py ") for s in solvers}
keys = sorted({(r["map"], int(r["agents"])) for r in rows})
print(f"{'map':<18}{'N':>4} | " + " | ".join(f"{short[s]:>16}" for s in solvers))
for m, N in keys:
    sel = [r for r in rows if r["map"] == m and int(r["agents"]) == N]
    inst = {(r["scen"], r["seed"]) for r in sel}
    solved = defaultdict(dict)
    for r in sel:
        if r["status"] == "solved":
            solved[r["solver_id"]][(r["scen"], r["seed"])] = float(r["soc"]) / float(r["soc_lb"])
    common = [k for k in inst if all(k in solved[s] for s in solvers)]
    cells = []
    for s in solvers:
        mean = sum(solved[s][k] for k in common) / len(common) if common else float("nan")
        cells.append(f"{mean:6.3f} {len(solved[s]):2d}/{len(inst):<2d}")
    print(f"{m:<18}{N:>4} | " + " | ".join(f"{c:>16}" for c in cells) + f"   (common {len(common)})")
