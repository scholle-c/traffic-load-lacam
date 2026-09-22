# py-lacam-pibt

The main branch provides the simplest implementation of LaCAM(\*) with __random action selection__.
Instead, this branch provides a simple implementation of LaCAM* with a vanilla __[PIBT](https://kei18.github.io/pibt2/)__, like the [AAAI-23](https://kei18.github.io/lacam) paper.
Now it is scalable.
See the main branch for more details.

Note that this implementation still omits many useful techniques as seen in the [IJCAI-23](https://kei18.github.io/lacam2) or [AAMAS-24](https://kei18.github.io/lacam3/) papers.

## Setup

This repository uses [uv](https://github.com/astral-sh/uv) for Python package management.
After cloning this repo, run:

```sh
uv sync --all-extras
```

## Example

```sh
uv run python app.py -m assets/random-32-32-10.map -i assets/random-32-32-10-random-1.scen -N 400 --no-flg_star
```

![](./assets/random-32-32-10.gif)

_540ms on my Mac Book Pro (M2 Max)_

## Traffic load guidance (this fork)

Idea (Carsten, 2026-09-22): before PIBT picks an agent's next cell it looks at the
*traffic load* of the candidate cells: the expected number of other agents whose
shortest path passes through that cell, assuming every agent picks uniformly among
its shortest paths. Cells that many others will need are avoided when the goal
distance leaves a choice. See `src/pycam/traffic_load.py` for the definition.

```sh
uv run python app.py -m assets/random-32-32-10.map -i assets/random-32-32-10-random-1.scen -N 200 --no-flg_star \
    --load-mode tiebreak --load-window 10 --load-update step
```

| flag | values | meaning |
|---|---|---|
| `--load-mode` | `off` (default), `tiebreak`, `penalty` | `tiebreak` sorts candidates by `(dist, load)`, `penalty` by `dist + lambda*load` |
| `--load-lambda` | float, default 1.0 | weight in `penalty` mode; values >= 0.5 make agents wander, try 0.1 |
| `--load-window` | int, default 10, `-1` = whole DAG | how many future steps of the other agents' shortest paths count |
| `--load-update` | `start`, `step`, `agent` (default) | when the map is (re)built: once from the starts, per configuration, or per configuration plus after every agent's decision within the step |

`--load-mode off` is bit-identical to upstream py-lacam (tested).

Sweep over the MovingAI benchmark (`MAPF_BENCH` points at `MyMAPFBenchmarks/BenchmarkV1`):

```sh
uv run python experiments/sweep.py run --out results/sweep.csv
uv run python experiments/sweep.py summarize results/sweep.csv
```

For the MAPF Evaluator copy `evaluator/traffic_load_lacam.yaml` into its `solvers/` folder.
