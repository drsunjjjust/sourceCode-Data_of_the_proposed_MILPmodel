"""
Destroy and repair operators (manuscript Section 5.2, Table 3).

Requests are of two kinds:
    customer s (1..n)        -- served inside an independent trip (s in S0) or inside a sortie of
                                its cluster (s in S1);
    cluster  c (encoded -c-1) -- a launch point visited by a truck; when removed, its sorties are
                                kept and the cluster is re-inserted into a truck route by the repair.

Destroy:  random removal, Shaw (relatedness) removal, cost-based (worst) removal,
          segment removal (all requests of a randomly chosen trip, sortie or truck route)
Repair:   greedy insertion, regret-2 insertion, regret-3 insertion, noisy greedy insertion
All repairs first re-insert removed clusters (truck level) and then removed customers (drone level).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .solution import (INF, Solution, trip_cost, truck_route_cost, eval_insert_customer_truck,
                       eval_insert_trip, eval_insert_cluster_truck)


# ----------------------------------------------------------------------------
# request helpers
# ----------------------------------------------------------------------------
def is_cluster(req: int) -> bool:
    return req < 0


def cluster_id(req: int) -> int:
    return -req - 1


def cluster_req(c: int) -> int:
    return -c - 1


class RequestInfo:
    """Static relatedness data of an instance (computed once)."""

    def __init__(self, inst):
        self.inst = inst
        n, m = inst.n, inst.m
        reqs = list(range(1, n + 1)) + [cluster_req(c) for c in range(m)]
        self.reqs = reqs
        self.index = {r: i for i, r in enumerate(reqs)}
        node = np.zeros(len(reqs), dtype=np.int64); tt = np.zeros(len(reqs)); grp = np.zeros(len(reqs), dtype=np.int64)
        for i, r in enumerate(reqs):
            if is_cluster(r):
                c = cluster_id(r); node[i] = inst.cluster_node(c); grp[i] = c
                tt[i] = np.mean([inst.tw_a[s] for s in inst.cluster_members[c]])
            else:
                node[i] = r; grp[i] = inst.cluster_of[r]; tt[i] = inst.tw_a[r]
        self.node = node; self.time = tt; self.group = grp
        self.dmax = float(inst.dist.max()) if inst.dist.max() > 0 else 1.0
        self.trange = float(max(1.0, inst.tw_a[1:].max() - inst.tw_a[1:].min())) if n > 0 else 1.0

    def relatedness(self, r: int, cand_idx: np.ndarray, phi=(1.0, 0.5, 1.0)) -> np.ndarray:
        i = self.index[r]
        d = self.inst.dist[self.node[i], self.node[cand_idx]] / self.dmax
        t = np.abs(self.time[i] - self.time[cand_idx]) / self.trange
        g = (self.group[cand_idx] != self.group[i]).astype(float)
        return phi[0] * d + phi[1] * t + phi[2] * g


# ----------------------------------------------------------------------------
# removal / insertion primitives on a Solution
# ----------------------------------------------------------------------------
def locate_customer(sol: Solution, s: int):
    inst = sol.inst
    if inst.cluster_of[s] < 0:
        for l, t in enumerate(sol.indep_trips):
            if s in t:
                return ("T", l, t.index(s))
    else:
        c = int(inst.cluster_of[s])
        for r, so in enumerate(sol.sorties[c]):
            if s in so:
                return ("S", c, r)
    return None


def remove_request(sol: Solution, req: int):
    inst = sol.inst
    if is_cluster(req):
        c = cluster_id(req)
        k = sol.truck_of_cluster(c)
        if k < 0:
            return
        sol.truck_routes[k].remove(c); sol.refresh_truck(k)
        sol.unassigned_clusters.append(c)
    else:
        s = req
        loc = locate_customer(sol, s)
        if loc is None:
            return
        if loc[0] == "T":
            l = loc[1]; sol.indep_trips[l].remove(s); sol.refresh_trip(l)
        else:
            c, r = loc[1], loc[2]
            sol.sorties[c][r].remove(s)
            k = sol.truck_of_cluster(c)
            if k >= 0:
                sol.refresh_truck(k)
        sol.unassigned_customers.append(s)


def removal_gain(sol: Solution, req: int) -> float:
    """Cost decrease obtained by removing the request (positive = saving)."""
    inst = sol.inst
    if is_cluster(req):
        c = cluster_id(req); k = sol.truck_of_cluster(c)
        if k < 0:
            return 0.0
        route = [x for x in sol.truck_routes[k] if x != c]
        tmp_routes = sol.truck_routes[k]; sol.truck_routes[k] = route
        arrs = sol.truck_arrays(k); sol.truck_routes[k] = tmp_routes
        new = truck_route_cost(*arrs, *sol._kernel_args())[0] if len(arrs[0]) else 0.0
        return sol._truck_cost[k] - new
    s = req; loc = locate_customer(sol, s)
    if loc is None:
        return 0.0
    if loc[0] == "T":
        l = loc[1]; seq = [x for x in sol.indep_trips[l] if x != s]
        new = trip_cost(np.array(seq, dtype=np.int64), *sol._trip_args())[0] if seq else 0.0
        return sol._trip_cost[l] - new
    c, r = loc[1], loc[2]; k = sol.truck_of_cluster(c)
    if k < 0:
        return 0.0
    tmp = sol.sorties[c][r]; sol.sorties[c][r] = [x for x in tmp if x != s]
    arrs = sol.truck_arrays(k); sol.sorties[c][r] = tmp
    new = truck_route_cost(*arrs, *sol._kernel_args())[0] if len(arrs[0]) else 0.0
    return sol._truck_cost[k] - new


def assigned_requests(sol: Solution) -> List[int]:
    reqs = []
    for t in sol.indep_trips:
        reqs.extend(t)
    for c in range(sol.inst.m):
        for so in sol.sorties[c]:
            reqs.extend(so)
    for r in sol.truck_routes:
        reqs.extend(cluster_req(c) for c in r)
    return reqs


# ----------------------------------------------------------------------------
# destroy operators
# ----------------------------------------------------------------------------
def shaw_removal(sol: Solution, q: int, rng: random.Random, info: RequestInfo, p: float = 6.0, phi=(1.0, 0.5, 1.0)):
    reqs = assigned_requests(sol)
    if not reqs:
        return
    q = min(q, len(reqs))
    first = rng.choice(reqs)
    remove_request(sol, first)
    removed = [first]
    remaining = [r for r in reqs if r != first]
    while len(removed) < q and remaining:
        ref = rng.choice(removed)
        idx = np.array([info.index[r] for r in remaining])
        rel = info.relatedness(ref, idx, phi)
        order = np.argsort(rel, kind="stable")
        pick = order[min(len(order) - 1, int(rng.random() ** p * len(order)))]
        req = remaining.pop(int(pick))
        remove_request(sol, req); removed.append(req)


def cost_removal(sol: Solution, q: int, rng: random.Random, info: RequestInfo, p: float = 3.0, phi=None):
    reqs = assigned_requests(sol)
    if not reqs:
        return
    q = min(q, len(reqs))
    gains: Dict[int, float] = {r: removal_gain(sol, r) for r in reqs}
    for _ in range(q):
        if not gains:
            break
        order = sorted(gains.items(), key=lambda kv: -kv[1])
        pick = order[min(len(order) - 1, int(rng.random() ** p * len(order)))][0]
        # which route is affected?
        affected = _route_members(sol, pick)
        remove_request(sol, pick); del gains[pick]
        for r in affected:
            if r in gains:
                gains[r] = removal_gain(sol, r)


def random_removal(sol: Solution, q: int, rng: random.Random, info: RequestInfo, p: float = 0.0, phi=None):
    reqs = assigned_requests(sol)
    if not reqs:
        return
    for req in rng.sample(reqs, min(q, len(reqs))):
        remove_request(sol, req)


def segment_removal(sol: Solution, q: int, rng: random.Random, info: RequestInfo, p: float = 0.0, phi=None):
    """Remove whole segments (an independent trip, a sortie or the clusters of a truck route) until
    q requests have been removed (route removal of Nagata and Braysy / Pisinger and Ropke)."""
    inst = sol.inst
    removed = 0
    guard = 0
    while removed < q and guard < 50:
        guard += 1
        segs = []
        for l, t in enumerate(sol.indep_trips):
            if t:
                segs.append(("T", l))
        for c in range(inst.m):
            for r, so in enumerate(sol.sorties[c]):
                if so:
                    segs.append(("S", c, r))
        for k, rt in enumerate(sol.truck_routes):
            if rt:
                segs.append(("K", k))
        if not segs:
            return
        seg = rng.choice(segs)
        if seg[0] == "T":
            members = list(sol.indep_trips[seg[1]])
        elif seg[0] == "S":
            members = list(sol.sorties[seg[1]][seg[2]])
        else:
            members = [cluster_req(c) for c in sol.truck_routes[seg[1]]]
        rng.shuffle(members)
        for req in members:
            if removed >= q:
                break
            remove_request(sol, req); removed += 1


def _route_members(sol: Solution, req: int) -> List[int]:
    """Requests sharing the route of `req` (whose gains change when req is removed)."""
    inst = sol.inst
    if is_cluster(req):
        c = cluster_id(req); k = sol.truck_of_cluster(c)
        if k < 0:
            return []
        out = []
        for cc in sol.truck_routes[k]:
            out.append(cluster_req(cc))
            for so in sol.sorties[cc]:
                out.extend(so)
        return out
    loc = locate_customer(sol, req)
    if loc is None:
        return []
    if loc[0] == "T":
        return list(sol.indep_trips[loc[1]])
    c = loc[1]; k = sol.truck_of_cluster(c)
    if k < 0:
        return []
    out = []
    for cc in sol.truck_routes[k]:
        out.append(cluster_req(cc))
        for so in sol.sorties[cc]:
            out.extend(so)
    return out


# ----------------------------------------------------------------------------
# repair operators
# ----------------------------------------------------------------------------
def _cluster_options(sol: Solution, c: int) -> List[Tuple[float, int, int]]:
    """(delta, truck k or -1 for new truck, position) for every truck."""
    inst = sol.inst
    csp, cc = sol.cluster_arrays(c)
    cn = inst.cluster_node(c)
    opts = []
    for k in range(len(sol.truck_routes)):
        if not sol.truck_routes[k]:
            continue
        arrs = sol.truck_arrays(k)
        d, pos = eval_insert_cluster_truck(*arrs, cn, csp, cc, inst.node_demand, sol._truck_cost[k], inst.Qt, inst.Lt,
                                           inst.demand, *sol._kernel_args())
        if d < INF / 2:
            opts.append((float(d), k, int(pos)))
    n_used = len([r for r in sol.truck_routes if r])
    if n_used < inst.n_trucks:
        clusters = np.array([cn], dtype=np.int64); clsp = np.array([0, len(csp) - 1], dtype=np.int64)
        cost = truck_route_cost(clusters, clsp, csp, cc, *sol._kernel_args())
        if cost[4] <= inst.Lt + 1e-9 and inst.cluster_demand[c] <= inst.Qt:
            opts.append((float(cost[0]), -1, 0))
    return opts


def _insert_cluster(sol: Solution, c: int, k: int, pos: int):
    if k < 0:
        # reuse an empty slot if any
        for kk, r in enumerate(sol.truck_routes):
            if not r:
                k = kk; break
        if k < 0:
            sol.truck_routes.append([]); k = len(sol.truck_routes) - 1
        pos = 0
    sol.truck_routes[k].insert(pos, c); sol.refresh_truck(k)
    sol.unassigned_clusters.remove(c)


def _trip_option(sol: Solution, s: int, l: int):
    inst = sol.inst
    t = sol.indep_trips[l]
    if not t:
        return None
    d, pos = eval_insert_trip(np.array(t, dtype=np.int64), s, sol._trip_cost[l], inst.Qd, inst.Ld, inst.demand,
                              *sol._trip_args())
    return (float(d), ("T", l, int(pos))) if d < INF / 2 else None


def _new_trip_option(sol: Solution, s: int):
    inst = sol.inst
    n_used = len([t for t in sol.indep_trips if t])
    if n_used < inst.n_indep_drones and 2 * inst.dist[0, s] <= inst.Ld + 1e-9:
        cost = trip_cost(np.array([s], dtype=np.int64), *sol._trip_args())[0]
        return (float(cost), ("T", -1, 0))
    return None


def _customer_options(sol: Solution, s: int) -> Dict:
    """Feasible insertion options of customer s: dict key -> (delta, descriptor)."""
    inst = sol.inst
    opts: Dict = {}
    if inst.cluster_of[s] < 0:
        for l in range(len(sol.indep_trips)):
            o = _trip_option(sol, s, l)
            if o is not None:
                opts[l] = o
        o = _new_trip_option(sol, s)
        if o is not None:
            opts["new"] = o
    else:
        c = int(inst.cluster_of[s]); k = sol.truck_of_cluster(c)
        if k < 0:
            return opts   # cluster not yet in a truck route: evaluated later
        j = sol.truck_routes[k].index(c)
        arrs = sol.truck_arrays(k)
        deltas, positions = eval_insert_customer_truck(*arrs, j, s, sol._truck_cost[k], inst.Qd, inst.Ld, inst.nL1,
                                                       inst.demand, *sol._kernel_args())
        nr = len(sol.sorties[c])
        for r in range(nr):
            if deltas[r] < INF / 2:
                opts[r] = (float(deltas[r]), ("S", c, r, int(positions[r])))
        if deltas[nr] < INF / 2:
            opts["new"] = (float(deltas[nr]), ("S", c, -1, 0))
    return opts


def _insert_customer(sol: Solution, s: int, desc: Tuple):
    inst = sol.inst
    if desc[0] == "T":
        l, pos = desc[1], desc[2]
        if l < 0:
            for ll, t in enumerate(sol.indep_trips):
                if not t:
                    l = ll; break
            if l < 0:
                sol.indep_trips.append([]); l = len(sol.indep_trips) - 1
            pos = 0
        sol.indep_trips[l].insert(pos, s); sol.refresh_trip(l)
    else:
        c, r, pos = desc[1], desc[2], desc[3]
        if r < 0:
            sol.sorties[c].append([s])
        else:
            sol.sorties[c][r].insert(pos, s)
        k = sol.truck_of_cluster(c)
        if k >= 0:
            sol.refresh_truck(k)
    sol.unassigned_customers.remove(s)


def _choose(options: Dict, mode: str, rng: random.Random, noise: float = 0.0):
    """Pick the request to insert next. mode = 'greedy' | 'regret2' | 'regret3' | 'noise'.
    options: req -> {key: (delta, desc)}.  'noise' is greedy insertion with an additive uniform noise of
    amplitude `noise` on every insertion cost (Ropke and Pisinger, 2006)."""
    best_req = None; best_key = None; best_choice = None
    for req, opts in options.items():
        if not opts:
            return req, None   # infeasible request -> signal failure
        vals = sorted(opts.values(), key=lambda o: o[0])
        if mode == "greedy":
            key = (vals[0][0], rng.random())
        elif mode == "noise":
            key = (vals[0][0] + noise * rng.uniform(-1.0, 1.0), rng.random())
        elif mode == "regret3":
            second = vals[1][0] if len(vals) > 1 else INF
            third = vals[2][0] if len(vals) > 2 else INF
            reg = (min(second, INF) - vals[0][0]) + (min(third, INF) - vals[0][0])
            key = (-reg, vals[0][0], rng.random())
        else:
            second = vals[1][0] if len(vals) > 1 else INF
            key = (-(second - vals[0][0]), vals[0][0], rng.random())
        if best_key is None or key < best_key:
            best_key = key; best_req = req; best_choice = vals[0]
    return best_req, best_choice


def repair(sol: Solution, mode: str, rng: random.Random, eta: float = 0.025) -> bool:
    """Greedy ('greedy'), noisy greedy ('noise'), regret-2 ('regret2') or regret-3 ('regret3') repair.
    Returns False if some request cannot be inserted."""
    inst = sol.inst
    noise = eta * inst.params["ct"] * float(inst.dist[:inst.n + 1, :inst.n + 1].max()) / inst.params["vt"] if mode == "noise" else 0.0
    # ---- 1. clusters (truck level)
    while sol.unassigned_clusters:
        options = {cluster_req(c): {i: (d, (k, pos)) for i, (d, k, pos) in enumerate(_cluster_options(sol, c))}
                   for c in sol.unassigned_clusters}
        req, choice = _choose(options, mode, rng, noise)
        if choice is None:
            return False
        k, pos = choice[1]
        _insert_cluster(sol, cluster_id(req), k, pos)
    # ---- 2. customers (drone level)
    if not sol.unassigned_customers:
        return True
    options = {s: _customer_options(sol, s) for s in sol.unassigned_customers}
    while options:
        req, choice = _choose(options, mode, rng, noise)
        if choice is None:
            return False
        desc = choice[1]
        n_trips_before = len([t for t in sol.indep_trips if t])
        _insert_customer(sol, req, desc)
        del options[req]
        # update affected customers
        if desc[0] == "T":
            l = desc[1]
            if l < 0:   # a new trip was opened: locate it
                l = next(ll for ll, t in enumerate(sol.indep_trips) if t == [req])
            new_opened = len([t for t in sol.indep_trips if t]) > n_trips_before
            for s in list(options):
                if inst.cluster_of[s] < 0:
                    o = _trip_option(sol, s, l)
                    if o is not None:
                        options[s][l] = o
                    else:
                        options[s].pop(l, None)
                    if new_opened:
                        o = _new_trip_option(sol, s)
                        if o is not None:
                            options[s]["new"] = o
                        else:
                            options[s].pop("new", None)
        else:
            c = desc[1]
            for s in list(options):
                if inst.cluster_of[s] == c:
                    options[s] = _customer_options(sol, s)
    return True


def greedy_insertion(sol: Solution, rng: random.Random) -> bool:
    return repair(sol, "greedy", rng)


def regret2_insertion(sol: Solution, rng: random.Random) -> bool:
    return repair(sol, "regret2", rng)


def regret3_insertion(sol: Solution, rng: random.Random) -> bool:
    return repair(sol, "regret3", rng)


def noisy_insertion(sol: Solution, rng: random.Random) -> bool:
    return repair(sol, "noise", rng)


DESTROY_OPERATORS = {"random": random_removal, "shaw": shaw_removal, "cost": cost_removal, "segment": segment_removal}
REPAIR_OPERATORS = {"greedy": greedy_insertion, "noise": noisy_insertion, "regret2": regret2_insertion,
                    "regret3": regret3_insertion}
