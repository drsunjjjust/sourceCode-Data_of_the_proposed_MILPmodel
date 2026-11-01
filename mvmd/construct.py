"""
TWCC-GRH: time-window and capacity-constrained greedy routing heuristic (manuscript Section 5.1,
Figure 4).

Stage I   drone routing construction
          (1) customers in S0 are assigned greedily (ascending demand) to independent drone trips
              subject to payload and range; each trip is improved by 2-opt on distance;
          (2) at every cluster, members are assigned greedily to the truck-mounted drones subject
              to payload and range; each sortie is improved by 2-opt on distance (time windows
              ignored at this stage).
Stage II  truck routing construction: clusters (ascending total demand) are assigned greedily to
          trucks subject to capacity and range; each truck route is improved by 2-opt.
Integration: time is propagated along the routes and every drone sequence is re-optimised with
          the exact cost (including time-window penalties) by a first-improvement local search
          over 2-opt and relocate moves.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from .instance import Instance, two_opt, tour_length, nn_tour, pack_into_sorties
from .solution import Solution, trip_cost, truck_route_cost


def _greedy_pack(custs: List[int], start: int, inst: Instance, Q: int, L: float, max_vehicles: int,
                 order: str = "asc") -> Optional[List[List[int]]]:
    """Greedy assignment of customers to vehicles (first vehicle filled before the next is opened)."""
    key = (lambda s: inst.node_demand[s]) if order == "asc" else (lambda s: -inst.node_demand[s])
    pending = sorted(custs, key=key)
    vehicles: List[List[int]] = []
    while pending:
        if len(vehicles) >= max_vehicles:
            return None
        route: List[int] = []; load = 0
        rest = []
        for s in pending:
            if load + inst.node_demand[s] <= Q:
                cand = route + [s]
                seq = nn_tour(cand, start, inst.dist)
                if tour_length(seq, start, inst.dist) <= L or tour_length(two_opt(seq, start, inst.dist), start, inst.dist) <= L:
                    route = cand; load += int(inst.node_demand[s]); continue
            rest.append(s)
        if not route:          # single customer infeasible
            return None
        vehicles.append(two_opt(nn_tour(route, start, inst.dist), start, inst.dist))
        pending = rest
    return vehicles


def _first_fit_decreasing(custs, start, inst, Q, L, max_vehicles):
    """First-fit decreasing over all open vehicles (the procedure that defines the fleet bounds)."""
    vehicles: List[List[int]] = []
    for s in sorted(custs, key=lambda s: -inst.node_demand[s]):
        placed = False
        for v in vehicles:
            if sum(inst.node_demand[x] for x in v) + inst.node_demand[s] <= Q:
                seq = two_opt(nn_tour(v + [s], start, inst.dist), start, inst.dist)
                if tour_length(seq, start, inst.dist) <= L:
                    v.append(s); placed = True; break
        if not placed:
            vehicles.append([s])
    if len(vehicles) > max_vehicles:
        return None
    return [two_opt(nn_tour(v, start, inst.dist), start, inst.dist) for v in vehicles]


def _pack(custs, start, inst, Q, L, max_vehicles):
    if not custs:
        return []
    for order in ("asc", "desc"):
        v = _greedy_pack(custs, start, inst, Q, L, max_vehicles, order)
        if v is not None:
            return v
    v = _first_fit_decreasing(custs, start, inst, Q, L, max_vehicles)
    if v is not None:
        return v
    if len(custs) <= 12:   # exhaustive packing (used by the clustering) for small sets
        v = pack_into_sorties(list(custs), start, inst.dist, inst.node_demand, Q, L, max_vehicles)
        if v is not None:
            return v
    raise RuntimeError("TWCC-GRH could not pack customers %s from node %d" % (custs, start))


def local_search_sortie(sol: Solution, k: int, c: int, r: int):
    """First-improvement 2-opt + relocate on sortie r of cluster c (truck k), exact cost."""
    inst = sol.inst
    improved = True
    while improved:
        improved = False
        seq = sol.sorties[c][r]
        base = sol._truck_cost[k]
        n = len(seq)
        if n < 2:
            return
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                sol.sorties[c][r] = cand; sol.refresh_truck(k)
                if sol._truck_cost[k] < base - 1e-9 and sol.sortie_feasible(c, r):
                    improved = True; break
                sol.sorties[c][r] = seq
            if improved:
                break
        if improved:
            continue
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = list(seq); s = cand.pop(i); cand.insert(j, s)
                sol.sorties[c][r] = cand; sol.refresh_truck(k)
                if sol._truck_cost[k] < base - 1e-9 and sol.sortie_feasible(c, r):
                    improved = True; break
                sol.sorties[c][r] = seq
            if improved:
                break
        sol.refresh_truck(k)


def local_search_trip(sol: Solution, l: int):
    inst = sol.inst
    improved = True
    while improved:
        improved = False
        seq = sol.indep_trips[l]; base = sol._trip_cost[l]; n = len(seq)
        if n < 2:
            return
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                sol.indep_trips[l] = cand; sol.refresh_trip(l)
                if sol._trip_cost[l] < base - 1e-9 and sol.trip_feasible(l):
                    improved = True; break
                sol.indep_trips[l] = seq
            if improved:
                break
        if improved:
            continue
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = list(seq); s = cand.pop(i); cand.insert(j, s)
                sol.indep_trips[l] = cand; sol.refresh_trip(l)
                if sol._trip_cost[l] < base - 1e-9 and sol.trip_feasible(l):
                    improved = True; break
                sol.indep_trips[l] = seq
            if improved:
                break
        sol.refresh_trip(l)


def twcc_grh(inst: Instance, seed: int = 0, polish: bool = True) -> Solution:
    sol = Solution(inst)
    # ---- Stage I (1): independent drones
    sol.indep_trips = _pack(list(inst.S0), 0, inst, inst.Qd, inst.Ld, inst.n_indep_drones)
    # ---- Stage I (2): truck-mounted drones at each cluster
    for c in range(inst.m):
        sol.sorties[c] = _pack(list(inst.cluster_members[c]), inst.cluster_node(c), inst, inst.Qd, inst.Ld, inst.nL1)
    # ---- Stage II: truck routes over clusters
    if inst.m:
        routes = _pack([inst.cluster_node(c) for c in range(inst.m)], 0, inst, inst.Qt, inst.Lt, inst.n_trucks)
        sol.truck_routes = [[cn - inst.n - 1 for cn in r] for r in routes]
    sol.refresh_all()
    # ---- Integration: time propagation + exact-cost local search of drone sequences
    if polish:
        for k in range(len(sol.truck_routes)):
            for c in sol.truck_routes[k]:
                for r in range(len(sol.sorties[c])):
                    local_search_sortie(sol, k, c, r)
        for l in range(len(sol.indep_trips)):
            local_search_trip(sol, l)
    sol.refresh_all()
    return sol


def random_initial(inst: Instance, seed: int) -> Solution:
    """Randomised construction (used in the initialisation ablation): random order greedy packing."""
    rng = random.Random(seed)
    sol = Solution(inst)

    def pack_random(custs, start, Q, L, maxv):
        custs = list(custs); rng.shuffle(custs)
        vehicles = []
        for s in custs:
            placed = False
            for v in vehicles:
                if sum(inst.node_demand[x] for x in v) + inst.node_demand[s] <= Q:
                    cand = v + [s]
                    if tour_length(cand, start, inst.dist) <= L:
                        v.append(s); placed = True; break
            if not placed:
                vehicles.append([s])
        if len(vehicles) > maxv:
            return None
        return vehicles

    for attempt in range(200):
        t = pack_random(inst.S0, 0, inst.Qd, inst.Ld, inst.n_indep_drones) if inst.S0 else []
        if t is None:
            continue
        ok = True; sorties = []
        for c in range(inst.m):
            sc = pack_random(inst.cluster_members[c], inst.cluster_node(c), inst.Qd, inst.Ld, inst.nL1)
            if sc is None:
                ok = False; break
            sorties.append(sc)
        if not ok:
            continue
        routes = pack_random([inst.cluster_node(c) for c in range(inst.m)], 0, inst.Qt, inst.Lt, inst.n_trucks) if inst.m else []
        if routes is None:
            continue
        sol.indep_trips = t; sol.sorties = sorties
        sol.truck_routes = [[cn - inst.n - 1 for cn in r] for r in routes]
        sol.refresh_all()
        return sol
    return twcc_grh(inst, seed, polish=False)
