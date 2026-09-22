"""Sweep traffic-load variants over MovingAI instances, in process, in parallel.

    .venv/bin/python experiments/sweep.py run  --out results/sweep.csv [--maps random32-20 room-32-4] [--agents 50 100] [--seeds 0 1 2] [--scens 5] [--timeout 20] [--jobs 12]
    .venv/bin/python experiments/sweep.py summarize results/sweep.csv

Every row is one (map, scen, N, seed, variant) run of LaCAM without refinement
(flg_star=False), so `soc` is the cost of the first solution PIBT+LaCAM found.
`lb` is the sum of BFS distances start->goal (the trivial lower bound), so
soc/lb is comparable across instances.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BENCH = Path(os.environ.get("MAPF_BENCH", Path.home() / "Projects/MAPF/MyMAPFBenchmarks/BenchmarkV1"))

MAPS = {  # scen folder -> map file
    "random32-20": "random-32-32-20.map",
    "room-32-4": "room-32-32-4.map",
    "maze-32-4": "maze-32-32-4.map",
    "empty32": "empty-32-32.map",
    "warehouse-10-20-2-1": "warehouse-10-20-10-2-1.map",
    "den312d": "den312d.map",
    "Paris": "Paris_1_256.map",
}

# name -> TrafficLoadConfig kwargs
VARIANTS = {
    "vanilla": dict(mode="off"),
    "tb_step_w10": dict(mode="tiebreak", update="step", window=10),
    "tb_agent_w10": dict(mode="tiebreak", update="agent", window=10),
    "tb_start_w10": dict(mode="tiebreak", update="start", window=10),
    "tb_step_wall": dict(mode="tiebreak", update="step", window=None),
    "pen0.1_step_w10": dict(mode="penalty", lam=0.1, update="step", window=10),
}


@dataclass
class Row:
    map: str
    scen: str
    agents: int
    seed: int
    variant: str
    solved: bool
    soc: int
    makespan: int
    lb: int
    time_ms: float
    forward_calls: int
    rebuilds: int


def scen_files(folder: str, k: int) -> list[Path]:
    files = list((BENCH / "Scens" / folder).glob("*.scen"))

    def num(p: Path) -> int:
        m = re.search(r"-(\d+)\.scen$", p.name)
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=num)[:k]


def run_one(job: tuple) -> Row:
    folder, scen, N, seed, variant, timeout_ms = job
    from pycam import LaCAM, TrafficLoadConfig, get_grid, get_scenario, get_sum_of_loss
    from pycam.dist_table import DistTable

    grid = get_grid(BENCH / "Maps" / MAPS[folder])
    starts, goals = get_scenario(scen, N)
    cfg = TrafficLoadConfig(**VARIANTS[variant])
    planner = LaCAM()
    t0 = time.perf_counter()
    sol = planner.solve(
        grid=grid, starts=starts, goals=goals, time_limit_ms=timeout_ms,
        flg_star=False, seed=seed, verbose=0, load_cfg=cfg,
    )
    dt = (time.perf_counter() - t0) * 1000
    lb = sum(DistTable(grid, g).get(s) for s, g in zip(starts, goals))
    stats = planner.traffic_load.stats if planner.traffic_load else {}
    solved = len(sol) > 0
    return Row(
        map=folder, scen=Path(scen).stem, agents=N, seed=seed, variant=variant,
        solved=solved, soc=get_sum_of_loss(sol) if solved else -1,
        makespan=len(sol) - 1 if solved else -1, lb=lb, time_ms=round(dt, 1),
        forward_calls=stats.get("forward_calls", 0), rebuilds=stats.get("rebuilds", 0),
    )


def cmd_run(a: argparse.Namespace) -> None:
    jobs = [
        (folder, str(scen), N, seed, variant, int(a.timeout * 1000))
        for folder in a.maps
        for scen in scen_files(folder, a.scens)
        for N in a.agents
        for seed in a.seeds
        for variant in a.variants
    ]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} runs -> {out}", flush=True)
    t0 = time.time()
    with out.open("w", newline="") as f, Pool(a.jobs) as pool:
        writer = None
        for k, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(asdict(row)))
                writer.writeheader()
            writer.writerow(asdict(row))
            f.flush()
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)


def cmd_summarize(a: argparse.Namespace) -> None:
    rows = list(csv.DictReader(open(a.csv)))
    for r in rows:
        r["solved"] = r["solved"] == "True"
        for k in ("agents", "seed", "soc", "lb", "makespan"):
            r[k] = int(r[k])
        r["time_ms"] = float(r["time_ms"])
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    keys = sorted({(r["map"], r["agents"]) for r in rows}, key=lambda k: (k[0], k[1]))
    print(f"{'map':<22}{'N':>5} | " + " | ".join(f"{v:>16}" for v in variants))
    print("soc/lb on instances solved by every variant (mean), solved/total, mean time s")
    for m, N in keys:
        sel = [r for r in rows if r["map"] == m and r["agents"] == N]
        inst = {(r["scen"], r["seed"]) for r in sel}
        common = {
            key for key in inst
            if all(any(r["variant"] == v and (r["scen"], r["seed"]) == key and r["solved"] for r in sel) for v in variants)
        }
        cells = []
        for v in variants:
            rv = [r for r in sel if r["variant"] == v]
            rc = [r for r in rv if (r["scen"], r["seed"]) in common]
            ratio = sum(r["soc"] / r["lb"] for r in rc) / len(rc) if rc else float("nan")
            solved = sum(r["solved"] for r in rv)
            t = sum(r["time_ms"] for r in rv) / len(rv) / 1000 if rv else float("nan")
            cells.append(f"{ratio:6.3f} {solved:2d}/{len(rv):<2d} {t:4.1f}")
        print(f"{m:<22}{N:>5} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="results/sweep.csv")
    r.add_argument("--maps", nargs="+", default=["random32-20", "room-32-4", "maze-32-4", "empty32", "warehouse-10-20-2-1"])
    r.add_argument("--agents", nargs="+", type=int, default=[50, 100, 150, 200])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--scens", type=int, default=5)
    r.add_argument("--variants", nargs="+", default=list(VARIANTS))
    r.add_argument("--timeout", type=float, default=20.0, help="seconds per run")
    r.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summarize")
    s.add_argument("csv")
    s.set_defaults(fn=cmd_summarize)
    a = p.parse_args()
    a.fn(a)
