"""
Solution representation and exact evaluation.

A solution consists of
    indep_trips  : list of customer lists, one per independent drone trip (depot -> ... -> depot)
    truck_routes : list of cluster-id lists, one per truck (depot -> c1 -> ... -> depot)
    sorties[c]   : list of customer lists, one per truck-mounted drone sortie launched at cluster c

Objective (manuscript Eqs. (1)-(4), (38)): fixed costs (cT per used truck, cD per sortie / per
independent trip) + travel costs (ct * truck travel time, cd * drone flight time) + soft
time-window penalties (cA per second of earliness, cB per second of tardiness).

Timing convention (identical to the MILP in Section 4):
  * drones follow earliest-start schedules: the time stamp of customer j is
        t_j = t_i + d_ij / v_d + e_j   (service time included, as in constraints (28)-(29));
  * a truck launches its drones on arrival at the cluster and leaves once the last drone has
    returned (temporary waiting, constraints (30)-(34));
  * departure times from the depot and voluntary waiting of trucks at clusters are free
    decision variables (time variables are lower-bounded by the resource extension in the MILP).
    For a fixed set of routes, the optimal waiting policy of a truck is obtained exactly by the
    pool-adjacent-violators algorithm on the chain of clusters (isotonic regression with convex
    piecewise-linear losses), see `_truck_penalty_pav`.
Hence, for identical routes, the heuristic objective equals the MILP objective.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

INF = 1e18


# ----------------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------------
@njit(cache=True)
def _block_minimizer(alpha, beta, lo, hi, cA, cB):
    """
    Minimiser over S >= 0 of  sum_{s in [lo,hi)} cA*max(alpha_s - S, 0) + cB*max(S - beta_s, 0).
    alpha_s = a_s - tau_s, beta_s = b_s - tau_s. Returns (S*, F(S*)).
    """
    n = hi - lo
    if n == 0:
        return 0.0, 0.0
    pts = np.empty(2 * n)
    inc = np.empty(2 * n)
    for i in range(n):
        pts[2 * i] = alpha[lo + i]; inc[2 * i] = cA
        pts[2 * i + 1] = beta[lo + i]; inc[2 * i + 1] = cB
    order = np.argsort(pts)
    slope = -cA * n
    S = 0.0
    found = False
    for k in range(2 * n):
        p = pts[order[k]]
        if slope >= 0.0:
            S = pts[order[k - 1]] if k > 0 else p
            found = True
            break
        slope += inc[order[k]]
    if not found:
        S = pts[order[2 * n - 1]]
    if S < 0.0:
        S = 0.0
    F = 0.0
    for i in range(n):
        e = alpha[lo + i] - S
        if e > 0.0:
            F += cA * e
        l = S - beta[lo + i]
        if l > 0.0:
            F += cB * l
    return S, F


@njit(cache=True)
def _block_value(alpha, beta, lo, hi, S, cA, cB):
    F = 0.0
    for i in range(lo, hi):
        e = alpha[i] - S
        if e > 0.0:
            F += cA * e
        l = S - beta[i]
        if l > 0.0:
            F += cB * l
    return F


@njit(cache=True)
def _truck_penalty_pav(alpha, beta, cl_cust_ptr, ncl, cA, cB):
    """
    Pool-adjacent-violators over the chain of clusters.  cl_cust_ptr[j]..cl_cust_ptr[j+1] is the
    range of customers of the j-th visited cluster in the flattened arrays.  Returns the minimum
    total penalty and the waiting shifts S_j (non-decreasing along the route).
    """
    blk_lo = np.empty(ncl, dtype=np.int64)
    blk_hi = np.empty(ncl, dtype=np.int64)
    blk_val = np.empty(ncl)
    nb = 0
    for j in range(ncl):
        lo = cl_cust_ptr[j]; hi = cl_cust_ptr[j + 1]
        S, _ = _block_minimizer(alpha, beta, lo, hi, cA, cB)
        blk_lo[nb] = lo; blk_hi[nb] = hi; blk_val[nb] = S; nb += 1
        while nb > 1 and blk_val[nb - 1] < blk_val[nb - 2] - 1e-9:
            # pool the last two blocks
            lo2 = blk_lo[nb - 2]; hi2 = blk_hi[nb - 1]
            S2, _ = _block_minimizer(alpha, beta, lo2, hi2, cA, cB)
            nb -= 1
            blk_lo[nb - 1] = lo2; blk_hi[nb - 1] = hi2; blk_val[nb - 1] = S2
    total = 0.0
    shifts = np.empty(ncl)
    jj = 0
    for b in range(nb):
        total += _block_value(alpha, beta, blk_lo[b], blk_hi[b], blk_val[b], cA, cB)
        # assign shift to every cluster inside the block
        while jj < ncl and cl_cust_ptr[jj] < blk_hi[b]:
            shifts[jj] = blk_val[b]; jj += 1
        # clusters without customers (cannot happen) would be skipped
    while jj < ncl:
        shifts[jj] = blk_val[nb - 1] if nb > 0 else 0.0; jj += 1
    return total, shifts


@njit(cache=True)
def truck_route_cost(clusters, cl_sortie_ptr, sortie_ptr, cust, dist, t_truck, t_drone, tw_a, tw_b,
                     service, cA, cB, cT, cD, ct, cd):
    """
    Exact cost of one truck route.
    clusters      : node indices of visited clusters in order (length m)
    cl_sortie_ptr : (m+1,) sortie index range of each cluster
    sortie_ptr    : (S+1,) customer index range of each sortie
    cust          : flattened customers
    Returns array [total, fixed, travel, penalty, truck_distance, max_sortie_distance].
    """
    out = np.zeros(6)
    m = clusters.shape[0]
    if m == 0:
        return out
    nsort = sortie_ptr.shape[0] - 1
    ncust = cust.shape[0]
    alpha = np.empty(ncust)
    beta = np.empty(ncust)
    cl_cust_ptr = np.empty(m + 1, dtype=np.int64)
    fixed = cT
    travel = 0.0
    truck_dist = 0.0
    max_sortie = 0.0
    cur = 0.0
    prev = 0
    for j in range(m):
        c = clusters[j]
        truck_dist += dist[prev, c]
        travel += ct * t_truck[prev, c]
        arr = cur + t_truck[prev, c]
        cl_cust_ptr[j] = sortie_ptr[cl_sortie_ptr[j]]
        W = 0.0
        for r in range(cl_sortie_ptr[j], cl_sortie_ptr[j + 1]):
            lo = sortie_ptr[r]; hi = sortie_ptr[r + 1]
            if hi > lo:
                fixed += cD
                tau = arr
                p = c
                sd = 0.0
                for q in range(lo, hi):
                    s = cust[q]
                    tau += t_drone[p, s] + service
                    sd += dist[p, s]
                    travel += cd * t_drone[p, s]
                    alpha[q] = tw_a[s] - tau
                    beta[q] = tw_b[s] - tau
                    p = s
                tau += t_drone[p, c]
                sd += dist[p, c]
                travel += cd * t_drone[p, c]
                if sd > max_sortie:
                    max_sortie = sd
                if tau - arr > W:
                    W = tau - arr
        cur = arr + W
        prev = c
    cl_cust_ptr[m] = ncust
    truck_dist += dist[prev, 0]
    travel += ct * t_truck[prev, 0]
    pen, _ = _truck_penalty_pav(alpha, beta, cl_cust_ptr, m, cA, cB)
    out[0] = fixed + travel + pen
    out[1] = fixed; out[2] = travel; out[3] = pen; out[4] = truck_dist; out[5] = max_sortie
    return out


@njit(cache=True)
def truck_route_schedule(clusters, cl_sortie_ptr, sortie_ptr, cust, dist, t_truck, t_drone, tw_a, tw_b,
                         service, cA, cB):
    """Absolute time stamps of the customers (after optimal waiting) and cluster arrival times."""
    m = clusters.shape[0]
    ncust = cust.shape[0]
    alpha = np.empty(ncust); beta = np.empty(ncust); tau_s = np.empty(ncust)
    cl_cust_ptr = np.empty(m + 1, dtype=np.int64)
    arr_t = np.empty(m); dep_t = np.empty(m)
    cur = 0.0; prev = 0
    for j in range(m):
        c = clusters[j]
        arr = cur + t_truck[prev, c]
        cl_cust_ptr[j] = sortie_ptr[cl_sortie_ptr[j]]
        W = 0.0
        for r in range(cl_sortie_ptr[j], cl_sortie_ptr[j + 1]):
            lo = sortie_ptr[r]; hi = sortie_ptr[r + 1]
            if hi > lo:
                tau = arr; p = c
                for q in range(lo, hi):
                    s = cust[q]
                    tau += t_drone[p, s] + service
                    alpha[q] = tw_a[s] - tau; beta[q] = tw_b[s] - tau; tau_s[q] = tau
                    p = s
                tau += t_drone[p, c]
                if tau - arr > W:
                    W = tau - arr
        arr_t[j] = arr; dep_t[j] = arr + W
        cur = arr + W; prev = c
    cl_cust_ptr[m] = ncust
    pen, shifts = _truck_penalty_pav(alpha, beta, cl_cust_ptr, m, cA, cB)
    times = np.empty(ncust)
    for j in range(m):
        for q in range(cl_cust_ptr[j], cl_cust_ptr[j + 1]):
            times[q] = tau_s[q] + shifts[j]
        arr_t[j] += shifts[j]; dep_t[j] += shifts[j]
    return pen, times, arr_t, dep_t, shifts


@njit(cache=True)
def trip_cost(seq, dist, t_drone, tw_a, tw_b, service, cA, cB, cD, cd):
    """Exact cost of one independent drone trip depot -> seq -> depot (free departure time)."""
    out = np.zeros(6)
    n = seq.shape[0]
    if n == 0:
        return out
    alpha = np.empty(n); beta = np.empty(n)
    tau = 0.0; p = 0; sd = 0.0; travel = 0.0
    for q in range(n):
        s = seq[q]
        tau += t_drone[p, s] + service
        sd += dist[p, s]
        travel += cd * t_drone[p, s]
        alpha[q] = tw_a[s] - tau; beta[q] = tw_b[s] - tau
        p = s
    sd += dist[p, 0]
    travel += cd * t_drone[p, 0]
    S, pen = _block_minimizer(alpha, beta, 0, n, cA, cB)
    out[0] = cD + travel + pen; out[1] = cD; out[2] = travel; out[3] = pen; out[4] = sd; out[5] = sd
    return out


@njit(cache=True)
def trip_length(seq, dist):
    n = seq.shape[0]
    if n == 0:
        return 0.0
    L = dist[0, seq[0]]
    for q in range(n - 1):
        L += dist[seq[q], seq[q + 1]]
    return L + dist[seq[n - 1], 0]


@njit(cache=True)
def eval_insert_customer_truck(clusters, cl_sortie_ptr, sortie_ptr, cust, j, s, base_cost, Qd, Ld, nL1, demand,
                               dist, t_truck, t_drone, tw_a, tw_b, service, cA, cB, cT, cD, ct, cd):
    """
    Evaluate inserting customer s into every sortie of the j-th cluster of a truck route, and as a
    new sortie.  Returns (deltas, positions): entry r < nsorties_j refers to sortie r of the cluster
    (best position), the last entry refers to a new sortie.  Infeasible options get delta = INF.
    """
    r0 = cl_sortie_ptr[j]; r1 = cl_sortie_ptr[j + 1]
    nr = r1 - r0
    deltas = np.full(nr + 1, INF)
    positions = np.full(nr + 1, -1, dtype=np.int64)
    c = clusters[j]
    ncust = cust.shape[0]
    new_cust = np.empty(ncust + 1, dtype=np.int64)
    new_sptr = sortie_ptr.copy()
    for rr in range(nr):
        r = r0 + rr
        lo = sortie_ptr[r]; hi = sortie_ptr[r + 1]
        load = 0
        L = 0.0
        p = c
        for q in range(lo, hi):
            load += demand[cust[q]]
            L += dist[p, cust[q]]; p = cust[q]
        L += dist[p, c]
        if load + demand[s] > Qd:
            continue
        # try every position
        for pos in range(hi - lo + 1):
            prev = c if pos == 0 else cust[lo + pos - 1]
            nxt = c if pos == hi - lo else cust[lo + pos]
            Lnew = L - dist[prev, nxt] + dist[prev, s] + dist[s, nxt]
            if Lnew > Ld + 1e-9:
                continue
            # build modified arrays
            ins = lo + pos
            for q in range(ins):
                new_cust[q] = cust[q]
            new_cust[ins] = s
            for q in range(ins, ncust):
                new_cust[q + 1] = cust[q]
            for rr2 in range(sortie_ptr.shape[0]):
                new_sptr[rr2] = sortie_ptr[rr2] + (1 if rr2 > r else 0)
            cost = truck_route_cost(clusters, cl_sortie_ptr, new_sptr, new_cust, dist, t_truck, t_drone, tw_a, tw_b,
                                    service, cA, cB, cT, cD, ct, cd)[0]
            d = cost - base_cost
            if d < deltas[rr]:
                deltas[rr] = d; positions[rr] = pos
    # new sortie
    if nr < nL1 and 2.0 * dist[c, s] <= Ld + 1e-9 and demand[s] <= Qd:
        ins = sortie_ptr[r1]  # customers of the new sortie go after the last sortie of cluster j
        for q in range(ins):
            new_cust[q] = cust[q]
        new_cust[ins] = s
        for q in range(ins, ncust):
            new_cust[q + 1] = cust[q]
        nsp = sortie_ptr.shape[0]
        new_sptr2 = np.empty(nsp + 1, dtype=np.int64)
        for rr2 in range(r1 + 1):
            new_sptr2[rr2] = sortie_ptr[rr2]
        new_sptr2[r1 + 1] = sortie_ptr[r1] + 1
        for rr2 in range(r1 + 1, nsp):
            new_sptr2[rr2 + 1] = sortie_ptr[rr2] + 1
        new_clsp = cl_sortie_ptr.copy()
        for jj in range(j + 1, cl_sortie_ptr.shape[0]):
            new_clsp[jj] += 1
        cost = truck_route_cost(clusters, new_clsp, new_sptr2, new_cust, dist, t_truck, t_drone, tw_a, tw_b,
                                service, cA, cB, cT, cD, ct, cd)[0]
        deltas[nr] = cost - base_cost; positions[nr] = 0
    return deltas, positions


@njit(cache=True)
def eval_insert_trip(seq, s, base_cost, Qd, Ld, demand, dist, t_drone, tw_a, tw_b, service, cA, cB, cD, cd):
    """Best insertion position of customer s in an independent trip. Returns (delta, pos)."""
    n = seq.shape[0]
    load = 0
    for q in range(n):
        load += demand[seq[q]]
    if load + demand[s] > Qd:
        return INF, -1
    L = trip_length(seq, dist)
    best = INF; bpos = -1
    new_seq = np.empty(n + 1, dtype=np.int64)
    for pos in range(n + 1):
        prev = 0 if pos == 0 else seq[pos - 1]
        nxt = 0 if pos == n else seq[pos]
        Lnew = L - dist[prev, nxt] + dist[prev, s] + dist[s, nxt]
        if Lnew > Ld + 1e-9:
            continue
        for q in range(pos):
            new_seq[q] = seq[q]
        new_seq[pos] = s
        for q in range(pos, n):
            new_seq[q + 1] = seq[q]
        cost = trip_cost(new_seq, dist, t_drone, tw_a, tw_b, service, cA, cB, cD, cd)[0]
        d = cost - base_cost
        if d < best:
            best = d; bpos = pos
    return best, bpos


@njit(cache=True)
def eval_insert_cluster_truck(clusters, cl_sortie_ptr, sortie_ptr, cust, c, c_sortie_ptr, c_cust, c_demand, base_cost,
                              Qt, Lt, demand, dist, t_truck, t_drone, tw_a, tw_b, service, cA, cB, cT, cD, ct, cd):
    """
    Evaluate inserting cluster node c (with its sorties: c_sortie_ptr / c_cust) at every position of
    a truck route.  Returns (best_delta, best_pos).
    """
    m = clusters.shape[0]
    load = 0
    for j in range(m):
        load += c_demand[clusters[j]]
    if load + c_demand[c] > Qt:
        return INF, -1
    # route distance check
    L = 0.0; prev = 0
    for j in range(m):
        L += dist[prev, clusters[j]]; prev = clusters[j]
    L += dist[prev, 0]
    best = INF; bpos = -1
    ncl_s = c_sortie_ptr.shape[0] - 1
    ncl_c = c_cust.shape[0]
    nS = sortie_ptr.shape[0] - 1
    ncust = cust.shape[0]
    new_clusters = np.empty(m + 1, dtype=np.int64)
    new_clsp = np.empty(m + 2, dtype=np.int64)
    new_sptr = np.empty(nS + ncl_s + 1, dtype=np.int64)
    new_cust = np.empty(ncust + ncl_c, dtype=np.int64)
    for pos in range(m + 1):
        prev = 0 if pos == 0 else clusters[pos - 1]
        nxt = 0 if pos == m else clusters[pos]
        Lnew = L - dist[prev, nxt] + dist[prev, c] + dist[c, nxt]
        if Lnew > Lt + 1e-9:
            continue
        # clusters
        for j in range(pos):
            new_clusters[j] = clusters[j]
        new_clusters[pos] = c
        for j in range(pos, m):
            new_clusters[j + 1] = clusters[j]
        # sortie pointers per cluster
        rins = cl_sortie_ptr[pos]           # sortie index where the new cluster's sorties start
        qins = sortie_ptr[rins]             # customer index where they start
        for j in range(pos + 1):
            new_clsp[j] = cl_sortie_ptr[j]
        new_clsp[pos + 1] = cl_sortie_ptr[pos] + ncl_s
        for j in range(pos + 1, m + 1):
            new_clsp[j + 1] = cl_sortie_ptr[j] + ncl_s
        for r in range(rins + 1):
            new_sptr[r] = sortie_ptr[r]
        for r in range(ncl_s):
            new_sptr[rins + 1 + r] = qins + c_sortie_ptr[r + 1]
        for r in range(rins + 1, nS + 1):
            new_sptr[r + ncl_s] = sortie_ptr[r] + ncl_c
        for q in range(qins):
            new_cust[q] = cust[q]
        for q in range(ncl_c):
            new_cust[qins + q] = c_cust[q]
        for q in range(qins, ncust):
            new_cust[q + ncl_c] = cust[q]
        cost = truck_route_cost(new_clusters, new_clsp, new_sptr, new_cust, dist, t_truck, t_drone, tw_a, tw_b,
                                service, cA, cB, cT, cD, ct, cd)[0]
        d = cost - base_cost
        if d < best:
            best = d; bpos = pos
    return best, bpos


# ----------------------------------------------------------------------------
# Solution object
# ----------------------------------------------------------------------------
class Solution:
    """Mutable solution with cached per-route costs."""

    def __init__(self, inst):
        self.inst = inst
        self.indep_trips: List[List[int]] = []
        self.truck_routes: List[List[int]] = []      # cluster ids (0..m-1)
        self.sorties: List[List[List[int]]] = [[] for _ in range(inst.m)]
        self._trip_cost: List[float] = []
        self._truck_cost: List[float] = []
        self._truck_arrays: List[Tuple] = []
        self.unassigned_customers: List[int] = []
        self.unassigned_clusters: List[int] = []

    # ---- helpers --------------------------------------------------------------
    @property
    def p(self):
        return self.inst.params

    def _kernel_args(self):
        i = self.inst; p = i.params
        return (i.dist, i.t_truck, i.t_drone, i.tw_a, i.tw_b, p["service_time"], p["cA"], p["cB"],
                p["cT"], p["cD"], p["ct"], p["cd"])

    def _trip_args(self):
        i = self.inst; p = i.params
        return (i.dist, i.t_drone, i.tw_a, i.tw_b, p["service_time"], p["cA"], p["cB"], p["cD"], p["cd"])

    def truck_arrays(self, k: int):
        """Flattened arrays (clusters, cl_sortie_ptr, sortie_ptr, cust) of truck k."""
        route = self.truck_routes[k]
        clusters = np.array([self.inst.cluster_node(c) for c in route], dtype=np.int64)
        cl_sortie_ptr = [0]; sortie_ptr = [0]; cust = []
        for c in route:
            for sortie in self.sorties[c]:
                cust.extend(sortie); sortie_ptr.append(len(cust))
            cl_sortie_ptr.append(len(sortie_ptr) - 1)
        return (clusters, np.array(cl_sortie_ptr, dtype=np.int64), np.array(sortie_ptr, dtype=np.int64),
                np.array(cust, dtype=np.int64))

    def cluster_arrays(self, c: int):
        sortie_ptr = [0]; cust = []
        for sortie in self.sorties[c]:
            cust.extend(sortie); sortie_ptr.append(len(cust))
        return np.array(sortie_ptr, dtype=np.int64), np.array(cust, dtype=np.int64)

    def refresh_truck(self, k: int):
        arrs = self.truck_arrays(k)
        while len(self._truck_arrays) <= k:
            self._truck_arrays.append(None); self._truck_cost.append(0.0)
        self._truck_arrays[k] = arrs
        self._truck_cost[k] = truck_route_cost(*arrs, *self._kernel_args())[0] if len(arrs[0]) else 0.0

    def refresh_trip(self, l: int):
        while len(self._trip_cost) <= l:
            self._trip_cost.append(0.0)
        seq = np.array(self.indep_trips[l], dtype=np.int64)
        self._trip_cost[l] = trip_cost(seq, *self._trip_args())[0] if len(seq) else 0.0

    def refresh_all(self):
        self._truck_arrays = []; self._truck_cost = []; self._trip_cost = []
        for k in range(len(self.truck_routes)):
            self.refresh_truck(k)
        for l in range(len(self.indep_trips)):
            self.refresh_trip(l)

    def cost(self) -> float:
        return float(sum(self._truck_cost) + sum(self._trip_cost))

    def sortie_feasible(self, c: int, r: int) -> bool:
        i = self.inst; s_ = self.sorties[c][r]
        if not s_:
            return True
        cn = i.cluster_node(c)
        if sum(i.demand[s] for s in s_) > i.Qd:
            return False
        L = i.dist[cn, s_[0]] + sum(i.dist[s_[q], s_[q + 1]] for q in range(len(s_) - 1)) + i.dist[s_[-1], cn]
        return L <= i.Ld + 1e-9

    def trip_feasible(self, l: int) -> bool:
        i = self.inst; t = self.indep_trips[l]
        if not t:
            return True
        if sum(i.demand[s] for s in t) > i.Qd:
            return False
        return trip_length(np.array(t, dtype=np.int64), i.dist) <= i.Ld + 1e-9

    def truck_of_cluster(self, c: int) -> int:
        for k, r in enumerate(self.truck_routes):
            if c in r:
                return k
        return -1

    def copy(self) -> "Solution":
        s = Solution(self.inst)
        s.indep_trips = [list(t) for t in self.indep_trips]
        s.truck_routes = [list(r) for r in self.truck_routes]
        s.sorties = [[list(x) for x in sl] for sl in self.sorties]
        s._trip_cost = list(self._trip_cost); s._truck_cost = list(self._truck_cost)
        s._truck_arrays = list(self._truck_arrays)
        s.unassigned_customers = list(self.unassigned_customers)
        s.unassigned_clusters = list(self.unassigned_clusters)
        return s

    def compact(self):
        """Drop empty trips / sorties / routes and refresh caches."""
        self.indep_trips = [t for t in self.indep_trips if t]
        self.truck_routes = [r for r in self.truck_routes if r]
        for c in range(self.inst.m):
            self.sorties[c] = [s for s in self.sorties[c] if s]
        self.refresh_all()

    # ---- full evaluation / reporting -----------------------------------------
    def breakdown(self) -> Dict[str, float]:
        i = self.inst
        tot = np.zeros(6)
        for k in range(len(self.truck_routes)):
            arrs = self.truck_arrays(k)
            if len(arrs[0]):
                tot += truck_route_cost(*arrs, *self._kernel_args())
        tr = np.zeros(6)
        for t in self.indep_trips:
            if t:
                tr += trip_cost(np.array(t, dtype=np.int64), *self._trip_args())
        return dict(total=float(tot[0] + tr[0]), fixed=float(tot[1] + tr[1]), travel=float(tot[2] + tr[2]),
                    penalty=float(tot[3] + tr[3]), n_trucks=len([r for r in self.truck_routes if r]),
                    n_sorties=sum(len([s for s in sl if s]) for sl in self.sorties),
                    n_indep_trips=len([t for t in self.indep_trips if t]))

    def is_feasible(self, verbose=False) -> bool:
        i = self.inst; ok = True
        served = []
        for t in self.indep_trips:
            served += t
            if sum(i.demand[s] for s in t) > i.Qd + 1e-9:
                ok = False; verbose and print("trip capacity", t)
            if trip_length(np.array(t, dtype=np.int64), i.dist) > i.Ld + 1e-6:
                ok = False; verbose and print("trip range", t)
            for s in t:
                if s not in i.S0:
                    ok = False; verbose and print("non-S0 customer in trip", s)
        if len([t for t in self.indep_trips if t]) > i.n_indep_drones:
            ok = False; verbose and print("too many independent trips")
        if len([r for r in self.truck_routes if r]) > i.n_trucks:
            ok = False; verbose and print("too many trucks")
        visited = []
        for k, r in enumerate(self.truck_routes):
            visited += r
            if sum(i.cluster_demand[c] for c in r) > i.Qt + 1e-9:
                ok = False; verbose and print("truck capacity", k)
            arrs = self.truck_arrays(k)
            if len(arrs[0]):
                res = truck_route_cost(*arrs, *self._kernel_args())
                if res[4] > i.Lt + 1e-6:
                    ok = False; verbose and print("truck range", k)
        if sorted(visited) != list(range(i.m)):
            ok = False; verbose and print("clusters visited", sorted(visited))
        for c in range(i.m):
            if len(self.sorties[c]) > i.nL1:
                ok = False; verbose and print("too many sorties at", c)
            cn = i.cluster_node(c)
            for s_ in self.sorties[c]:
                served += s_
                if sum(i.demand[s] for s in s_) > i.Qd + 1e-9:
                    ok = False; verbose and print("sortie capacity", c, s_)
                L = i.dist[cn, s_[0]] + sum(i.dist[s_[q], s_[q + 1]] for q in range(len(s_) - 1)) + i.dist[s_[-1], cn] if s_ else 0
                if L > i.Ld + 1e-6:
                    ok = False; verbose and print("sortie range", c, s_)
                for s in s_:
                    if i.cluster_of[s] != c:
                        ok = False; verbose and print("customer in wrong cluster", s, c)
        if sorted(served) != list(range(1, i.n + 1)):
            ok = False; verbose and print("customers served", sorted(served))
        return ok

    def to_dict(self) -> Dict:
        return dict(indep_trips=[list(map(int, t)) for t in self.indep_trips],
                    truck_routes=[list(map(int, r)) for r in self.truck_routes],
                    sorties=[[list(map(int, s)) for s in sl] for sl in self.sorties],
                    cost=self.cost())

    @staticmethod
    def from_dict(inst, d: Dict) -> "Solution":
        s = Solution(inst)
        s.indep_trips = [list(t) for t in d["indep_trips"]]
        s.truck_routes = [list(r) for r in d["truck_routes"]]
        s.sorties = [[list(x) for x in sl] for sl in d["sorties"]]
        s.refresh_all()
        return s

    def schedule(self) -> Dict:
        """Detailed schedule (times) for reporting."""
        i = self.inst; out = dict(trucks=[], trips=[])
        for k in range(len(self.truck_routes)):
            arrs = self.truck_arrays(k)
            if not len(arrs[0]):
                continue
            pen, times, arr_t, dep_t, shifts = truck_route_schedule(*arrs, i.dist, i.t_truck, i.t_drone, i.tw_a, i.tw_b,
                                                                     i.params["service_time"], i.params["cA"], i.params["cB"])
            out["trucks"].append(dict(route=list(self.truck_routes[k]), arrival=arr_t.tolist(), departure=dep_t.tolist(),
                                      customer_times=dict(zip(map(int, arrs[3]), times.tolist())), penalty=float(pen)))
        for t in self.indep_trips:
            if t:
                out["trips"].append(dict(trip=list(t), cost=float(trip_cost(np.array(t, dtype=np.int64), *self._trip_args())[0])))
        return out
