"""Traffic load: expected number of agents whose shortest path passes a cell.

Definition (Carsten's sketch, 2026-09-22). Agent i stands at v with goal g. Every
shortest v->g path is equally likely, so the probability that i visits cell u is

    p_i(u) = N(v->u) * N(u->g) / N(v->g)

where N(a->b) counts shortest a->b paths. The traffic load of a cell is the sum of
p_i(u) over agents. Because every shortest path reaches u after exactly
tau = dist(v,g) - dist(u,g) steps, the load is naturally time indexed: agent i is
expected at u at time offset tau with probability p_i(u).

Implementation notes
- Per agent, once: a full BFS distance table from the goal and the path counts
  N(u->g) for every cell, both level by level with numpy. Counts live in log space
  because they grow like binomial coefficients on open maps.
- Per query: the forward probabilities from the current position are propagated
  level by level along the shortest-path DAG (only edges that decrease the goal
  distance by one). A window W stops after W levels, so a query costs O(W) small
  numpy calls instead of O(cells). window=None walks the whole DAG (Carsten's
  static, time-free load).
- The map keeps one dense array `total` (all agents, summed over the window) plus
  each agent's own contribution, so "load seen by agent i" = total - own.
- Cells use a padded flat index (obstacle border of width 1), so the four
  neighbour offsets never leave the array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .mapf_utils import Config, Coord, Grid

LoadMode = Literal["off", "tiebreak", "penalty"]
LoadUpdate = Literal["start", "step", "agent"]


@dataclass(frozen=True)
class TrafficLoadConfig:
    """How the traffic load enters PIBT's candidate ordering.

    mode
        off       vanilla PIBT (sort by goal distance only)
        tiebreak  sort by (dist, load): load only decides between equally good cells
        penalty   sort by dist + lam * load: load may buy a detour
    lam
        weight of the load in penalty mode
    window
        number of future steps the load looks at (None = whole shortest path DAG,
        i.e. the time-free load of the sketch)
    update
        start  load computed once from the start configuration
        step   recomputed for every configuration PIBT expands
        agent  like step, and refreshed after every agent that commits a move
               inside the step (the "update map" arrow in the sketch)
    """

    mode: LoadMode = "off"
    lam: float = 1.0
    window: int | None = 10
    update: LoadUpdate = "agent"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


class AgentPathCounts:
    """Goal distance and log path counts N(u->g) of one agent on the padded grid."""

    INF = np.iinfo(np.int32).max

    def __init__(self, geometry: _Geometry, goal: Coord) -> None:
        self.geo = geometry
        self.goal = geometry.flat(goal)
        self.dist, self.levels = self._bfs()
        self.log_paths = self._count_paths()

    def _bfs(self) -> tuple[np.ndarray, list[np.ndarray]]:
        geo = self.geo
        dist = np.full(geo.size, self.INF, dtype=np.int32)
        dist[self.goal] = 0
        levels = [np.array([self.goal], dtype=np.int64)]
        frontier = levels[0]
        d = 0
        while len(frontier) > 0:
            d += 1
            nb = np.concatenate([frontier + off for off in geo.offsets])
            nb = nb[geo.free[nb] & (dist[nb] == self.INF)]
            if len(nb) == 0:
                break
            nb = np.unique(nb)
            dist[nb] = d
            levels.append(nb)
            frontier = nb
        return dist, levels

    def _count_paths(self) -> np.ndarray:
        geo = self.geo
        log_n = np.full(geo.size, -np.inf)
        log_n[self.goal] = 0.0
        for d, cells in enumerate(self.levels[1:], start=1):
            parts = []
            for off in geo.offsets:
                nb = cells + off
                parts.append(np.where(self.dist[nb] == d - 1, log_n[nb], -np.inf))
            log_n[cells] = np.logaddexp.reduce(parts, axis=0)
        return log_n

    def forward(self, v: int, window: int | None) -> tuple[np.ndarray, np.ndarray]:
        """Visit probabilities from v for up to `window` levels, (cells, probs).

        Level 0 is v itself with probability 1. Probabilities on each later level
        sum to 1 as well. Returns unique cells with summed probability."""
        d0 = int(self.dist[v])
        cells = [np.array([v], dtype=np.int64)]
        probs = [np.array([1.0])]
        if d0 == self.INF:
            return cells[0], probs[0]
        frontier, pf = cells[0], probs[0]
        limit = d0 if window is None else min(d0, window)
        for tau in range(1, limit + 1):
            target = d0 - tau
            nbs, contribs = [], []
            for off in self.geo.offsets:
                nb = frontier + off
                ok = self.dist[nb] == target
                if not ok.any():
                    continue
                nb_ok, fr_ok = nb[ok], frontier[ok]
                nbs.append(nb_ok)
                contribs.append(
                    pf[ok] * np.exp(self.log_paths[nb_ok] - self.log_paths[fr_ok])
                )
            frontier, inv = np.unique(np.concatenate(nbs), return_inverse=True)
            pf = np.bincount(inv, weights=np.concatenate(contribs))
            cells.append(frontier)
            probs.append(pf)
        if len(cells) == 1:
            return cells[0], probs[0]
        uniq, inv = np.unique(np.concatenate(cells), return_inverse=True)
        return uniq, np.bincount(inv, weights=np.concatenate(probs))


class _Geometry:
    """Padded flat indexing of the grid."""

    def __init__(self, grid: Grid) -> None:
        self.H, self.W = grid.shape
        self.Wp = self.W + 2
        self.size = (self.H + 2) * self.Wp
        free = np.zeros((self.H + 2, self.Wp), dtype=bool)
        free[1:-1, 1:-1] = grid
        self.free = free.ravel()
        self.offsets = (1, -1, self.Wp, -self.Wp)

    def flat(self, coord: Coord) -> int:
        y, x = coord
        return (y + 1) * self.Wp + (x + 1)


class TrafficLoad:
    """Traffic load map over all agents, queryable per agent (excluding itself)."""

    def __init__(self, grid: Grid, goals: Config, cfg: TrafficLoadConfig) -> None:
        self.cfg = cfg
        self.geo = _Geometry(grid)
        self.agents = [AgentPathCounts(self.geo, g) for g in goals]
        self.N = len(self.agents)
        self.total = np.zeros(self.geo.size)
        self.own: list[dict[int, float]] = [{} for _ in range(self.N)]
        self._base: np.ndarray | None = None
        self._base_own: list[dict[int, float]] | None = None
        self._cache: dict[tuple[Coord, ...], tuple[np.ndarray, list[dict[int, float]]]] = {}
        self.stats = {"forward_calls": 0, "rebuilds": 0, "cache_hits": 0}

    # -- building -----------------------------------------------------------

    def rebuild(self, Q: Config) -> None:
        """Recompute the whole map for configuration Q (memoised, small LRU)."""
        key = tuple(Q.positions)
        hit = self._cache.get(key)
        if hit is not None:
            self.stats["cache_hits"] += 1
            self.total, self.own = hit[0].copy(), [d.copy() for d in hit[1]]
        else:
            self.stats["rebuilds"] += 1
            self.total = np.zeros(self.geo.size)
            self.own = [{} for _ in range(self.N)]
            for i, v in enumerate(Q.positions):
                self._add(i, self.geo.flat(v))
            if len(self._cache) >= 32:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (self.total.copy(), [d.copy() for d in self.own])
        self._base = self.total.copy()
        self._base_own = [d.copy() for d in self.own]

    def restore(self) -> None:
        """Undo in-step agent updates (update='agent') after PIBT finished a step."""
        if self._base is not None and self.cfg.update == "agent":
            self.total = self._base.copy()
            self.own = [d.copy() for d in self._base_own]

    def commit(self, i: int, u: Coord) -> None:
        """Agent i has chosen u as its next cell: move its expected load there."""
        if self.cfg.update != "agent":
            return
        self._remove(i)
        self._add(i, self.geo.flat(u))

    def _add(self, i: int, v: int) -> None:
        self.stats["forward_calls"] += 1
        cells, probs = self.agents[i].forward(v, self.cfg.window)
        self.total[cells] += probs
        self.own[i] = dict(zip(cells.tolist(), probs.tolist()))

    def _remove(self, i: int) -> None:
        own = self.own[i]
        if own:
            cells = np.fromiter(own.keys(), dtype=np.int64, count=len(own))
            self.total[cells] -= np.fromiter(own.values(), dtype=float, count=len(own))
            self.own[i] = {}

    # -- querying -----------------------------------------------------------

    def load_for(self, i: int, u: Coord) -> float:
        """Load at u from every agent except i."""
        f = self.geo.flat(u)
        return max(float(self.total[f]) - self.own[i].get(f, 0.0), 0.0)

    def as_grid(self, exclude: int | None = None) -> np.ndarray:
        """Dense (H, W) load map, e.g. for plotting."""
        total = self.total.copy()
        if exclude is not None:
            for f, p in self.own[exclude].items():
                total[f] -= p
        return total.reshape(self.geo.H + 2, self.geo.Wp)[1:-1, 1:-1]
