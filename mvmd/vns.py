"""
Variable neighbourhood search baseline (manuscript Section 6.2).

Basic VNS (Hansen and Mladenovic) with
  shaking N_k   : random removal of k * step requests followed by greedy re-insertion with
                  random tie breaking (k = 1..k_max);
  local search  : first-improvement relocate of every request to its best position (customers
                  between sorties / trips, clusters between trucks) followed by intra-route 2-opt
                  and relocate with the exact cost, repeated until no improvement;
  move or not   : accept if improved (k <- 1), else k <- k + 1 (cyclic).
The same solution representation, feasibility checks and exact evaluation are used as in OSD-ALNS.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import numpy as np

from .alns import RunResult
from .construct import local_search_sortie, local_search_trip
from .operators import (RequestInfo, assigned_requests, remove_request, removal_gain, repair, cluster_req, is_cluster,
                        cluster_id, _cluster_options, _customer_options, _insert_cluster, _insert_customer)
from .solution import Solution


def relocate_pass(sol: Solution, rng: random.Random) -> bool:
    """One pass over all requests: move each to its best position if that improves the cost."""
    improved = False
    reqs = assigned_requests(sol); rng.shuffle(reqs)
    for req in reqs:
        gain = removal_gain(sol, req)
        trial = sol.copy()
        remove_request(trial, req)
        if is_cluster(req):
            c = cluster_id(req)
            opts = _cluster_options(trial, c)
            if not opts:
                continue
            d, k, pos = min(opts)
            if d - gain < -1e-6:
                _insert_cluster(trial, c, k, pos)
                sol.indep_trips, sol.truck_routes, sol.sorties = trial.indep_trips, trial.truck_routes, trial.sorties
                sol.unassigned_clusters = []; sol.unassigned_customers = []
                sol.compact(); improved = True
        else:
            opts = _customer_options(trial, req)
            if not opts:
                continue
            d, desc = min(opts.values(), key=lambda o: o[0])
            if d - gain < -1e-6:
                _insert_customer(trial, req, desc)
                sol.indep_trips, sol.truck_routes, sol.sorties = trial.indep_trips, trial.truck_routes, trial.sorties
                sol.unassigned_clusters = []; sol.unassigned_customers = []
                sol.compact(); improved = True
    return improved


def intra_route_pass(sol: Solution):
    for k in range(len(sol.truck_routes)):
        for c in sol.truck_routes[k]:
            for r in range(len(sol.sorties[c])):
                local_search_sortie(sol, k, c, r)
    for l in range(len(sol.indep_trips)):
        local_search_trip(sol, l)
    sol.refresh_all()


def local_search(sol: Solution, rng: random.Random, deadline: float):
    while time.perf_counter() < deadline:
        c0 = sol.cost()
        relocate_pass(sol, rng)
        intra_route_pass(sol)
        if sol.cost() > c0 - 1e-6:
            break


def shake(sol: Solution, k: int, step: int, rng: random.Random, info: RequestInfo) -> Optional[Solution]:
    s = sol.copy()
    reqs = assigned_requests(s)
    q = min(len(reqs), k * step)
    for req in rng.sample(reqs, q):
        remove_request(s, req)
    if not repair(s, "greedy", rng):
        return None
    s.compact()
    return s


class VNS:
    def __init__(self, inst, init_sol: Solution, seed: int = 0, k_max: int = 5, step_frac: float = 0.05):
        self.inst = inst; self.init = init_sol; self.rng = random.Random(seed)
        self.k_max = k_max; self.step = max(1, int(round(step_frac * (inst.n + inst.m))))
        self.info = RequestInfo(inst)

    def run(self, time_limit: float, max_iters: Optional[int] = None) -> RunResult:
        rng = self.rng
        t0 = time.perf_counter(); deadline = t0 + time_limit
        cur = self.init.copy(); cur.compact()
        init_cost = cur.cost()
        local_search(cur, rng, deadline)
        best = cur.copy(); best_cost = best.cost()
        trace = [(0, 0.0, init_cost), (0, time.perf_counter() - t0, best_cost)]
        it = 0; ops = 0; k = 1
        while time.perf_counter() < deadline and (max_iters is None or it < max_iters):
            s = shake(cur, k, self.step, rng, self.info)
            ops += 1
            if s is None:
                k = k % self.k_max + 1; it += 1; continue
            local_search(s, rng, deadline)
            if s.cost() < cur.cost() - 1e-9:
                cur = s; k = 1
                if cur.cost() < best_cost - 1e-9:
                    best = cur.copy(); best_cost = best.cost()
                    trace.append((ops, time.perf_counter() - t0, best_cost))
            else:
                k = k % self.k_max + 1
            it += 1
        elapsed = time.perf_counter() - t0
        trace.append((ops, elapsed, best_cost))
        return RunResult(best_cost=best_cost, best=best.to_dict(), ops=ops, iterations=it, time=elapsed, trace=trace,
                         init_cost=init_cost, op_time=elapsed / max(ops, 1))
