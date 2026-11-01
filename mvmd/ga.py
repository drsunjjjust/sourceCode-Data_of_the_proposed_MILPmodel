"""
Genetic algorithm baseline (manuscript Section 6.2), in the spirit of the hybrid GA of
Peng et al. (2019): permutation chromosomes with a split decoder.

Chromosome = one permutation per pool:
    pool "S0"      : independent-drone customers  -> split into trips (capacity Qd, range Ld, <= L0 trips)
    pool "cluster c": members of cluster c        -> split into sorties (capacity Qd, range Ld, <= nL1)
    pool "trucks"  : cluster ids                  -> split into truck routes (capacity Qt, range Lt, <= K)
Decoding is a first-fit split in chromosome order; a chromosome whose split needs more vehicles
than available is penalised.  Operators: order crossover (OX) per pool, swap / inversion mutation,
binary tournament selection, elitism.  The exact evaluation of Section 5 is used for the fitness,
so GA, VNS and the ALNS variants optimise the same objective under the same feasibility rules.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import numpy as np

from .alns import RunResult
from .instance import tour_length
from .solution import Solution

BIG = 1e6


def split(seq: List[int], start: int, inst, Q: int, L: float, max_v: int):
    routes: List[List[int]] = []
    cur: List[int] = []; load = 0
    for s in seq:
        if cur and (load + inst.node_demand[s] > Q or tour_length(cur + [s], start, inst.dist) > L):
            routes.append(cur); cur = []; load = 0
        cur.append(s); load += int(inst.node_demand[s])
    if cur:
        routes.append(cur)
    penalty = BIG * max(0, len(routes) - max_v)
    return routes, penalty


class GA:
    def __init__(self, inst, init_sol: Solution, seed: int = 0, pop_size: int = 50, p_cross: float = 0.9,
                 p_mut: float = 0.2, elite: int = 2):
        self.inst = inst; self.rng = random.Random(seed); self.init = init_sol
        self.pop_size = pop_size; self.p_cross = p_cross; self.p_mut = p_mut; self.elite = elite
        self.pools = ["S0"] + [f"c{c}" for c in range(inst.m)] + ["trucks"]

    # ---- encoding -----------------------------------------------------------
    def encode(self, sol: Solution) -> Dict[str, List[int]]:
        ch = {"S0": [s for t in sol.indep_trips for s in t]}
        for c in range(self.inst.m):
            ch[f"c{c}"] = [s for so in sol.sorties[c] for s in so]
        ch["trucks"] = [c for r in sol.truck_routes for c in r]
        return ch

    def random_chromosome(self) -> Dict[str, List[int]]:
        inst = self.inst; rng = self.rng
        ch = {"S0": list(inst.S0)}; rng.shuffle(ch["S0"])
        for c in range(inst.m):
            ch[f"c{c}"] = list(inst.cluster_members[c]); rng.shuffle(ch[f"c{c}"])
        ch["trucks"] = list(range(inst.m)); rng.shuffle(ch["trucks"])
        return ch

    def decode(self, ch: Dict[str, List[int]]):
        inst = self.inst
        sol = Solution(inst); pen = 0.0
        trips, p = split(ch["S0"], 0, inst, inst.Qd, inst.Ld, inst.n_indep_drones); pen += p
        sol.indep_trips = trips
        for c in range(inst.m):
            so, p = split(ch[f"c{c}"], inst.cluster_node(c), inst, inst.Qd, inst.Ld, inst.nL1); pen += p
            sol.sorties[c] = so
        if inst.m:
            cl = [inst.cluster_node(c) for c in ch["trucks"]]
            routes, p = split(cl, 0, inst, inst.Qt, inst.Lt, inst.n_trucks); pen += p
            sol.truck_routes = [[cn - inst.n - 1 for cn in r] for r in routes]
        sol.refresh_all()
        return sol, sol.cost() + pen, pen

    # ---- operators ------------------------------------------------------------
    def ox(self, a: List[int], b: List[int]) -> List[int]:
        n = len(a)
        if n < 2:
            return list(a)
        i, j = sorted(self.rng.sample(range(n), 2))
        child = [None] * n; child[i:j + 1] = a[i:j + 1]
        fill = [x for x in b if x not in child[i:j + 1]]
        q = 0
        for k in range(n):
            if child[k] is None:
                child[k] = fill[q]; q += 1
        return child

    def mutate(self, s: List[int]):
        n = len(s)
        if n < 2:
            return
        if self.rng.random() < 0.5:
            i, j = self.rng.sample(range(n), 2); s[i], s[j] = s[j], s[i]
        else:
            i, j = sorted(self.rng.sample(range(n), 2)); s[i:j + 1] = s[i:j + 1][::-1]

    def run(self, time_limit: float, max_gens: Optional[int] = None) -> RunResult:
        rng = self.rng
        t0 = time.perf_counter(); deadline = t0 + time_limit
        pop = [self.encode(self.init)] + [self.random_chromosome() for _ in range(self.pop_size - 1)]
        dec = [self.decode(ch) for ch in pop]
        fit = [d[1] for d in dec]; pens = [d[2] for d in dec]
        init_cost = fit[0]
        feas = [i for i in range(len(pop)) if pens[i] == 0]
        bi = min(feas, key=lambda i: fit[i]) if feas else int(np.argmin(fit)); best_ch = pop[bi]; best_cost = fit[bi]
        trace = [(0, 0.0, init_cost), (0, time.perf_counter() - t0, best_cost)]
        gen = 0; evals = len(pop)
        while time.perf_counter() < deadline and (max_gens is None or gen < max_gens):
            order = np.argsort(fit)
            new_pop = [pop[i] for i in order[:self.elite]]; new_fit = [fit[i] for i in order[:self.elite]]
            while len(new_pop) < self.pop_size:
                i, j = rng.sample(range(self.pop_size), 2); pa = pop[i] if fit[i] < fit[j] else pop[j]
                i, j = rng.sample(range(self.pop_size), 2); pb = pop[i] if fit[i] < fit[j] else pop[j]
                child = {}
                for pool in self.pools:
                    child[pool] = self.ox(pa[pool], pb[pool]) if rng.random() < self.p_cross else list(pa[pool])
                    if rng.random() < self.p_mut:
                        self.mutate(child[pool])
                sol, f, pen = self.decode(child); evals += 1
                new_pop.append(child); new_fit.append(f)
                if pen == 0 and f < best_cost - 1e-9:
                    best_cost = f; best_ch = child
                    trace.append((evals, time.perf_counter() - t0, best_cost))
                if time.perf_counter() >= deadline:
                    break
            pop, fit = new_pop, new_fit
            gen += 1
        elapsed = time.perf_counter() - t0
        best_sol, _, _ = self.decode(best_ch)
        trace.append((evals, elapsed, best_cost))
        return RunResult(best_cost=best_cost, best=best_sol.to_dict(), ops=evals, iterations=gen, time=elapsed, trace=trace,
                         init_cost=init_cost, op_time=elapsed / max(evals, 1))
