"""Summarize the PP-guided evaluator runs (pp_guided_tl manifest).

    python experiments/summarize_pp.py "~/Projects/MAPF/MAPF Evaluator/results/pptl-*/results.csv"

For every (map, N) and for both orders (shortest, traffic-last) the change from
plain SIPP to SIPP with the traffic load tie-break (--pp_tl):
  plan     mean change of the plan's flowtime (pp_plan_soc), complete plans only
  exec     mean change of the executed SOC, instances solved by both
  complete number of instances with a complete plan (pp_failed == 0), of total
  rounds   mean demotion rounds (plain -> tl)
  ms       mean planning time (plain -> tl)
"""
import csv, glob, json, os, sys
from collections import defaultdict

files = [f for pat in sys.argv[1:] for f in glob.glob(os.path.expanduser(pat))]
rows = [r for f in sorted(files) for r in csv.DictReader(open(f))]
for r in rows:
    st = json.loads(r["solver_stats"]) if r["solver_stats"] else {}
    r["_plan"] = int(st.get("pp_plan_soc", -1))
    r["_failed"] = int(st.get("pp_failed", -1))
    r["_rounds"] = int(st.get("pp_demote_rounds", -1))
    r["_ms"] = float(st.get("pp_plan_ms", "nan"))
    r["_var"] = r["solver_id"].split(":")[1]

pairs = [("shortest", "shortest_tl"), ("traffic", "traffic_tl")]
keys = sorted({(r["map"], int(r["agents"])) for r in rows})
print(f"{'map':<24}{'N':>4} | " + " | ".join(f"{a + ' -> +tl':^44}" for a, _ in pairs))
print(f"{'':<28} | " + " | ".join(f"{'plan':>7} {'exec':>7} {'complete':>11} {'rounds':>8} {'ms':>8}" for _ in pairs))
for m, N in keys:
    sel = [r for r in rows if r["map"] == m and int(r["agents"]) == N]
    by = defaultdict(dict)
    for r in sel:
        by[r["_var"]][(r["scen"], r["seed"])] = r
    cells = []
    for a, b in pairs:
        A, B = by[a], by[b]
        inst = sorted(set(A) | set(B))
        both_complete = [k for k in inst if k in A and k in B and A[k]["_failed"] == 0 and B[k]["_failed"] == 0]
        both_solved = [k for k in inst if k in A and k in B and A[k]["status"] == "solved" and B[k]["status"] == "solved"]
        plan = sum(B[k]["_plan"] / A[k]["_plan"] - 1 for k in both_complete) / len(both_complete) * 100 if both_complete else float("nan")
        ex = sum(float(B[k]["soc"]) / float(A[k]["soc"]) - 1 for k in both_solved) / len(both_solved) * 100 if both_solved else float("nan")
        ca = sum(1 for k in A if A[k]["_failed"] == 0)
        cb = sum(1 for k in B if B[k]["_failed"] == 0)
        ra = sum(A[k]["_rounds"] for k in A) / len(A) if A else float("nan")
        rb = sum(B[k]["_rounds"] for k in B) / len(B) if B else float("nan")
        ma = sum(A[k]["_ms"] for k in A) / len(A) if A else float("nan")
        mb = sum(B[k]["_ms"] for k in B) / len(B) if B else float("nan")
        cells.append(f"{plan:+6.1f}% {ex:+6.1f}% {ca:3d}->{cb:<3d}/{len(inst):<2d} {ra:3.1f}->{rb:<3.1f} {ma:3.0f}->{mb:<4.0f}")
    print(f"{m:<24}{N:>4} | " + " | ".join(cells))
