from pathlib import Path

import numpy as np
import pytest

from pycam.dist_table import DistTable
from pycam.lacam import LaCAM
from pycam.mapf_utils import (
    Config,
    get_grid,
    get_scenario,
    get_sum_of_loss,
    is_valid_mapf_solution,
)
from pycam.traffic_load import AgentPathCounts, TrafficLoad, TrafficLoadConfig, _Geometry

DIR = Path(__file__).parent
ASSETS = DIR.parent / "assets"


def test_two_shortest_paths_on_open_2x2():
    grid = np.full((2, 2), True)
    geo = _Geometry(grid)
    apc = AgentPathCounts(geo, goal=(1, 1))
    cells, probs = apc.forward(geo.flat((0, 0)), window=None)
    load = dict(zip(cells.tolist(), probs.tolist()))
    assert load[geo.flat((0, 0))] == pytest.approx(1.0)
    assert load[geo.flat((0, 1))] == pytest.approx(0.5)
    assert load[geo.flat((1, 0))] == pytest.approx(0.5)
    assert load[geo.flat((1, 1))] == pytest.approx(1.0)
    # two shortest paths from the corner
    assert np.exp(apc.log_paths[geo.flat((0, 0))]) == pytest.approx(2.0)


def test_single_corridor_is_deterministic():
    grid = get_grid(ASSETS / "tunnel.map")
    geo = _Geometry(grid)
    starts, goals = get_scenario(ASSETS / "tunnel.scen", 1)
    apc = AgentPathCounts(geo, goals[0])
    cells, probs = apc.forward(geo.flat(starts[0]), window=None)
    d = DistTable(grid, goals[0]).get(starts[0])
    assert len(cells) == d + 1
    assert np.allclose(probs, 1.0)


def test_bfs_matches_dist_table():
    grid = get_grid(ASSETS / "random-32-32-10.map")
    starts, goals = get_scenario(ASSETS / "random-32-32-10-random-1.scen", 5)
    geo = _Geometry(grid)
    for s, g in zip(starts, goals):
        apc = AgentPathCounts(geo, g)
        assert apc.dist[geo.flat(s)] == DistTable(grid, g).get(s)


def test_window_limits_levels_and_mass():
    grid = get_grid(ASSETS / "random-32-32-10.map")
    starts, goals = get_scenario(ASSETS / "random-32-32-10-random-1.scen", 1)
    geo = _Geometry(grid)
    apc = AgentPathCounts(geo, goals[0])
    v = geo.flat(starts[0])
    d = int(apc.dist[v])
    assert d > 5
    cells, probs = apc.forward(v, window=5)
    # one unit of probability per level, levels 0..5
    assert probs.sum() == pytest.approx(6.0)
    assert (apc.dist[cells] >= d - 5).all()
    cells, probs = apc.forward(v, window=None)
    assert probs.sum() == pytest.approx(d + 1)


def test_load_for_excludes_self_and_rebuild_is_idempotent():
    grid = get_grid(ASSETS / "random-32-32-10.map")
    starts, goals = get_scenario(ASSETS / "random-32-32-10-random-1.scen", 20)
    tl = TrafficLoad(grid, goals, TrafficLoadConfig(mode="penalty", window=8))
    tl.rebuild(starts)
    first = tl.total.copy()
    # own start cell carries probability 1 for the agent itself
    assert tl.load_for(0, starts[0]) == pytest.approx(float(first[tl.geo.flat(starts[0])]) - 1.0)
    assert tl.as_grid().sum() == pytest.approx(first.sum())
    # commit + restore leaves the base map untouched
    tl.commit(0, starts[0])
    tl.restore()
    assert np.allclose(tl.total, first)
    tl.rebuild(starts)
    assert np.allclose(tl.total, first)
    assert tl.stats["cache_hits"] == 1


def _solve(load_cfg, N=30, seed=0):
    grid = get_grid(ASSETS / "random-32-32-10.map")
    starts, goals = get_scenario(ASSETS / "random-32-32-10-random-1.scen", N)
    sol = LaCAM().solve(
        grid=grid, starts=starts, goals=goals, flg_star=False, seed=seed,
        verbose=0, load_cfg=load_cfg, time_limit_ms=10000,
    )
    assert is_valid_mapf_solution(grid, starts, goals, sol)
    return sol


def test_mode_off_is_bit_identical_to_vanilla():
    a = _solve(None)
    b = _solve(TrafficLoadConfig(mode="off"))
    assert [c.positions for c in a] == [c.positions for c in b]


@pytest.mark.parametrize("mode", ["tiebreak", "penalty"])
@pytest.mark.parametrize("update", ["start", "step", "agent"])
def test_guided_lacam_solves(mode, update):
    sol = _solve(TrafficLoadConfig(mode=mode, update=update, window=6))
    assert get_sum_of_loss(sol) > 0
