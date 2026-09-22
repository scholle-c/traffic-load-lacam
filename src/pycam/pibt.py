"""
a toy PIBT implementation taken from
https://github.com/Kei18/pypibt
"""

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Coord, get_neighbors
from .traffic_load import TrafficLoad


class PIBT:
    def __init__(
        self,
        dist_tables: list[DistTable],
        seed: int = 0,
        traffic_load: TrafficLoad | None = None,
    ) -> None:
        self.N = len(dist_tables)
        assert self.N > 0
        self.dist_tables = dist_tables
        self.grid = self.dist_tables[0].grid

        # traffic load guidance (None -> vanilla PIBT)
        self.load = traffic_load if (traffic_load and traffic_load.cfg.enabled) else None

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(self.grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(self.grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=self.sort_key(i))

        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue

            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i
            if self.load is not None:
                self.load.commit(i, v)

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        if self.load is not None:
            self.load.commit(i, Q_from[i])
        return False

    def sort_key(self, i: int):
        dist = self.dist_tables[i].get
        if self.load is None:
            return dist
        load = self.load
        if load.cfg.mode == "tiebreak":
            return lambda u: (dist(u), load.load_for(i, u))
        lam = load.cfg.lam
        return lambda u: dist(u) + lam * load.load_for(i, u)

    def step(
        self,
        Q_from: Config,
        Q_to: Config,
        order: list[int],
    ) -> bool:
        flg_success = True

        # traffic load for this configuration
        if self.load is not None and self.load.cfg.update != "start":
            self.load.rebuild(Q_from)

        # setup
        for i, (v_i_from, v_i_to) in enumerate(zip(Q_from, Q_to)):
            self.occupied_now[v_i_from] = i
            if v_i_to != self.NIL_COORD:
                #  check vertex collision
                if self.occupied_nxt[v_i_to] != self.NIL:
                    flg_success = False
                    break
                # check edge collision
                j = self.occupied_now[v_i_to]
                if j != self.NIL and j != i and Q_to[j] == v_i_from:
                    flg_success = False
                    break
                self.occupied_nxt[v_i_to] = i

        # perform PIBT
        if flg_success:
            for i in order:
                if Q_to[i] == self.NIL_COORD:
                    flg_success = self.funcPIBT(Q_from, Q_to, i)
                    if not flg_success:
                        break

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            if q_to != self.NIL_COORD:
                self.occupied_nxt[q_to] = self.NIL
        if self.load is not None:
            self.load.restore()

        return flg_success
