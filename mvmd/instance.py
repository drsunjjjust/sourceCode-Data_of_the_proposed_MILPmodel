"""
Instance definition and generation.

Units: metres, seconds, CNY, demand units (as in the manuscript, Section 6.1).

Node indexing used throughout the code
    0                : depot (origin O and destination D share this location)
    1..n             : customers
    n+1..n+m         : cluster centres (launch/retrieval points of truck-mounted drones)

Pre-processing (manuscript Section 3.1, Figure 2)
    1. customers within radius gamma*Ld of the depot form S0 (independent drones),
       the others form S1 (truck-supported drones);
    2. S1 is partitioned into clusters by a capacity- and distance-constrained k-means:
       the demand of a cluster may not exceed min(Qt, nL1*Qd), every member must be
       reachable by a feasible set of at most nL1 sorties (capacity Qd, range Ld).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Baseline parameters (manuscript Section 6.1)
# ----------------------------------------------------------------------------
BASE_PARAMS = dict(
    cT=50.0,        # fixed dispatching cost of a truck            [CNY]
    cD=10.0,        # fixed dispatching cost of a drone sortie     [CNY]
    ct=0.083,       # truck operating cost per unit travel time    [CNY/s]
    cd=0.0017,      # drone operating cost per unit flight time    [CNY/s]
    Lt=100_000.0,   # maximum travel distance of a truck           [m]
    Ld=10_000.0,    # maximum flight distance of a drone (sortie)  [m]
    Qt=30,          # truck capacity                               [units]
    Qd=5,           # drone capacity                               [units]
    vt=15_000.0 / 3600.0,   # truck speed 15 km/h                  [m/s]
    vd=45_000.0 / 3600.0,   # drone speed 45 km/h                  [m/s]
    cA=0.001,       # earliness penalty                            [CNY/s]
    cB=0.1,         # tardiness penalty                            [CNY/s]
    gamma=0.2,      # scaling factor of the independent-drone service radius
    service_time=60.0,      # service time at a customer           [s]
    nL1=2,          # drones carried by each truck
)

SCENARIOS = {
    "A-1": dict(gamma=0.2),
    "A-2": dict(gamma=0.4),
    "B-1": dict(Ld=8_000.0),
    "B-2": dict(Ld=12_000.0),
    "C-1": dict(Qd=4),
    "C-2": dict(Qd=6),
}

# Toy network (manuscript Figure 7(a)); coordinates in metres, demands in units.
TOY_DEPOT = (2000.0, 4000.0)
TOY_CUSTOMERS = [  # id, x, y, demand
    (1, 1000, 5000, 4), (2, 2000, 5000, 2), (3, 3000, 4000, 2), (4, 2000, 3000, 3),
    (5, 1000, 4000, 1), (6, 4000, 5000, 2), (7, 5000, 6000, 3), (8, 6000, 5000, 1),
    (9, 6000, 3000, 1), (10, 5000, 2000, 3), (11, 7000, 1000, 1), (12, 0, 2000, 2),
    (13, 1000, 0, 2), (14, 2000, 1000, 4),
]
SOLOMON_TIME_SCALE = 10.0   # one Solomon time unit = 10 s (C101: horizon 1236 units = 3.4 h)
# the R101 / RC101 horizons are 230 / 240 units; they are scaled to the same ~3.3 h working day
SOLOMON_SOURCES = {"C": dict(file="solomon_c101.txt", scale=10.0),
                   "R": dict(file="solomon_r101.txt", scale=50.0),
                   "RC": dict(file="solomon_rc101.txt", scale=50.0)}


@dataclass
class Instance:
    name: str
    params: Dict[str, float]
    depot: Tuple[float, float]
    cust_xy: np.ndarray          # (n+1, 2), row 0 = depot
    demand: np.ndarray           # (n+1,), demand[0] = 0
    tw_a: np.ndarray             # (n+1,)
    tw_b: np.ndarray             # (n+1,)
    S0: List[int]                # independent-drone customers
    S1: List[int]                # truck-supported customers
    cluster_xy: np.ndarray       # (m, 2)
    cluster_members: List[List[int]]
    n_trucks: int                # |K|  (upper bound on trucks)
    n_indep_drones: int          # |L0| (upper bound on independent drone trips)
    meta: Dict = field(default_factory=dict)

    # ---- derived ------------------------------------------------------------
    def __post_init__(self):
        self.n = len(self.demand) - 1
        self.m = len(self.cluster_members)
        xy = np.vstack([self.cust_xy, self.cluster_xy]) if self.m else self.cust_xy.copy()
        self.xy = xy.astype(float)
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        self.dist = d                                   # metres
        self.t_truck = d / self.params["vt"]            # seconds
        self.t_drone = d / self.params["vd"]            # seconds
        self.cluster_of = np.full(self.n + 1, -1, dtype=np.int64)
        for c, mem in enumerate(self.cluster_members):
            for s in mem:
                self.cluster_of[s] = c
        self.cluster_demand = np.array([sum(int(self.demand[s]) for s in mem) for mem in self.cluster_members], dtype=np.int64)
        self.node_demand = np.concatenate([self.demand.astype(np.int64), self.cluster_demand]) if self.m else self.demand.astype(np.int64)
        self.nL1 = int(self.params["nL1"])
        self.Qd = int(self.params["Qd"]); self.Qt = int(self.params["Qt"])
        self.Ld = float(self.params["Ld"]); self.Lt = float(self.params["Lt"])
        self.cnode = lambda c: self.n + 1 + c       # node index of cluster c

    def cluster_node(self, c: int) -> int:
        return self.n + 1 + c

    # ---- JSON I/O -----------------------------------------------------------
    def to_dict(self) -> Dict:
        return dict(
            name=self.name, params=self.params, depot=list(self.depot),
            customers=[dict(id=i, x=float(self.cust_xy[i, 0]), y=float(self.cust_xy[i, 1]),
                            demand=int(self.demand[i]), a=float(self.tw_a[i]), b=float(self.tw_b[i]))
                       for i in range(1, self.n + 1)],
            S0=list(map(int, self.S0)), S1=list(map(int, self.S1)),
            clusters=[dict(id=c, x=float(self.cluster_xy[c, 0]), y=float(self.cluster_xy[c, 1]),
                           members=list(map(int, self.cluster_members[c])))
                      for c in range(self.m)],
            n_trucks=int(self.n_trucks), n_indep_drones=int(self.n_indep_drones), meta=self.meta,
        )

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "Instance":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Instance.from_dict(d)

    @staticmethod
    def from_dict(d: Dict) -> "Instance":
        cust = sorted(d["customers"], key=lambda r: r["id"])
        n = len(cust)
        xy = np.zeros((n + 1, 2)); dem = np.zeros(n + 1, dtype=np.int64)
        a = np.zeros(n + 1); b = np.zeros(n + 1)
        xy[0] = d["depot"]
        for r in cust:
            i = r["id"]; xy[i] = (r["x"], r["y"]); dem[i] = r["demand"]; a[i] = r["a"]; b[i] = r["b"]
        cl = sorted(d["clusters"], key=lambda r: r["id"])
        cxy = np.array([[r["x"], r["y"]] for r in cl]).reshape(-1, 2)
        mem = [list(r["members"]) for r in cl]
        return Instance(d["name"], d["params"], tuple(d["depot"]), xy, dem, a, b, list(d["S0"]), list(d["S1"]),
                        cxy, mem, int(d["n_trucks"]), int(d["n_indep_drones"]), d.get("meta", {}))


# ----------------------------------------------------------------------------
# Helpers for feasibility of a set of customers served from one launch point
# ----------------------------------------------------------------------------
def tour_length(seq: List[int], start: int, dist: np.ndarray) -> float:
    if not seq:
        return 0.0
    L = dist[start, seq[0]]
    for i in range(len(seq) - 1):
        L += dist[seq[i], seq[i + 1]]
    return L + dist[seq[-1], start]


def two_opt(seq: List[int], start: int, dist: np.ndarray) -> List[int]:
    """Classical 2-opt on an open-closed tour start -> seq -> start."""
    best = list(seq)
    if len(best) < 3:
        return best
    improved = True
    bestL = tour_length(best, start, dist)
    while improved:
        improved = False
        for i in range(0, len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                L = tour_length(cand, start, dist)
                if L < bestL - 1e-9:
                    best, bestL, improved = cand, L, True
                    break
            if improved:
                break
    return best


def nn_tour(members: List[int], start: int, dist: np.ndarray) -> List[int]:
    rem = list(members); seq = []; cur = start
    while rem:
        nxt = min(rem, key=lambda s: dist[cur, s]); seq.append(nxt); rem.remove(nxt); cur = nxt
    return seq


def pack_into_sorties(members: List[int], start: int, dist: np.ndarray, demand: np.ndarray,
                      Qd: int, Ld: float, nL1: int) -> Optional[List[List[int]]]:
    """
    Try to serve `members` from launch point `start` with at most nL1 sorties, each of
    capacity Qd and flight range Ld.  Exhaustive over bin assignments (members are few),
    tour per bin = nearest neighbour + 2-opt.  Returns the sortie list or None.
    """
    if not members:
        return []
    mem = sorted(members, key=lambda s: -demand[s])
    if any(demand[s] > Qd for s in mem):
        return None
    bins: List[List[int]] = []
    loads: List[int] = []

    def feasible_bin(b: List[int]) -> Optional[List[int]]:
        seq = two_opt(nn_tour(b, start, dist), start, dist)
        return seq if tour_length(seq, start, dist) <= Ld + 1e-9 else None

    best: Optional[List[List[int]]] = None

    def rec(idx: int):
        nonlocal best
        if best is not None:
            return
        if idx == len(mem):
            seqs = []
            for b in bins:
                s = feasible_bin(b)
                if s is None:
                    return
                seqs.append(s)
            best = seqs
            return
        s = mem[idx]
        for k in range(len(bins)):
            if loads[k] + demand[s] <= Qd:
                bins[k].append(s); loads[k] += demand[s]
                # quick range pre-check with NN tour
                if tour_length(nn_tour(bins[k], start, dist), start, dist) <= 1.25 * Ld:
                    rec(idx + 1)
                bins[k].pop(); loads[k] -= demand[s]
                if best is not None:
                    return
        if len(bins) < nL1:
            bins.append([s]); loads.append(int(demand[s]))
            rec(idx + 1)
            bins.pop(); loads.pop()

    rec(0)
    return best


# ----------------------------------------------------------------------------
# Capacity- and distance-constrained k-means clustering of S1
# ----------------------------------------------------------------------------
def constrained_kmeans(inst_xy: np.ndarray, S1: List[int], demand: np.ndarray, Qcl: int, Qd: int,
                       Ld: float, nL1: int, seed: int, r_max: float,
                       max_iter: int = 60) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Returns cluster centres and member lists.  k starts at ceil(total demand / Qcl) and is
    increased until a feasible partition is found.  Assignment step: customer-centre pairs are
    processed in increasing distance; a customer joins the nearest centre whose cluster stays
    (i) within capacity Qcl, (ii) within radius r_max and (iii) serviceable by <= nL1 feasible
    sorties.  Centres are then recomputed as centroids until the partition is stable.
    """
    rng = np.random.default_rng(seed)
    pts = inst_xy[S1]
    total = int(sum(demand[s] for s in S1))
    k = max(1, math.ceil(total / Qcl))

    def xy_of(s):
        return inst_xy[s]

    while True:
        # k-means++ initialisation
        centres = [pts[rng.integers(len(pts))]]
        while len(centres) < k:
            d2 = np.min([((pts - c) ** 2).sum(1) for c in centres], axis=0)
            prob = d2 / d2.sum() if d2.sum() > 0 else np.ones(len(pts)) / len(pts)
            centres.append(pts[rng.choice(len(pts), p=prob)])
        centres = np.array(centres, dtype=float)
        prev = None
        members: List[List[int]] = [[] for _ in range(k)]
        ok = False
        for _ in range(max_iter):
            members = [[] for _ in range(k)]
            loads = [0] * k
            # temporary distance matrix including the current centres as extra nodes
            allxy = np.vstack([inst_xy, centres])
            dist = np.sqrt(((allxy[:, None, :] - allxy[None, :, :]) ** 2).sum(-1))
            cnode = lambda c: len(inst_xy) + c
            pairs = sorted(((dist[s, cnode(c)], s, c) for s in S1 for c in range(k)))
            assigned = set()
            for d, s, c in pairs:
                if s in assigned or d > r_max:
                    continue
                if loads[c] + demand[s] > Qcl:
                    continue
                trial = members[c] + [s]
                if pack_into_sorties(trial, cnode(c), dist, demand, Qd, Ld, nL1) is None:
                    continue
                members[c] = trial; loads[c] += int(demand[s]); assigned.add(s)
            if len(assigned) < len(S1):
                ok = False
                break
            ok = True
            new_centres = np.array([pts[[S1.index(s) for s in mem]].mean(0) if mem else centres[c]
                                    for c, mem in enumerate(members)])
            key = tuple(tuple(sorted(mem)) for mem in members)
            if key == prev:
                break
            prev = key
            centres = new_centres
        if ok:
            keep = [c for c in range(k) if members[c]]
            centres = np.array([np.round(centres[c]) for c in keep]).reshape(-1, 2)
            members = [sorted(members[c]) for c in keep]
            return centres, members
        k += 1
        if k > len(S1):
            raise RuntimeError("clustering failed: even singleton clusters infeasible")


# ----------------------------------------------------------------------------
# Fleet upper bounds from a greedy construction (so that the MILP has finite sets K, L0)
# ----------------------------------------------------------------------------
def greedy_trip_count(custs: List[int], start: int, dist: np.ndarray, demand: np.ndarray, Q: int, L: float) -> int:
    """First-fit (by decreasing demand) packing into trips with capacity Q and range L."""
    trips: List[List[int]] = []
    for s in sorted(custs, key=lambda s: -demand[s]):
        placed = False
        for tr in trips:
            if sum(demand[x] for x in tr) + demand[s] <= Q:
                seq = two_opt(nn_tour(tr + [s], start, dist), start, dist)
                if tour_length(seq, start, dist) <= L:
                    tr.append(s); placed = True; break
        if not placed:
            trips.append([s])
    return len(trips)


# ----------------------------------------------------------------------------
# Instance builders
# ----------------------------------------------------------------------------
def _build(name: str, params: Dict, depot, cust_xy, demand, tw_a, tw_b, seed: int, meta: Dict) -> Instance:
    n = len(demand) - 1
    xy = np.vstack([np.array(depot, dtype=float)[None, :], cust_xy]) if cust_xy.shape[0] == n else cust_xy
    radius = params["gamma"] * params["Ld"]
    d0 = np.sqrt(((xy[1:] - xy[0]) ** 2).sum(1))
    S0 = [i + 1 for i in range(n) if d0[i] <= radius + 1e-9]
    S1 = [i + 1 for i in range(n) if d0[i] > radius + 1e-9]
    Qcl = min(int(params["Qt"]), int(params["nL1"]) * int(params["Qd"]))
    if S1:
        cxy, members = constrained_kmeans(xy, S1, demand, Qcl, int(params["Qd"]), params["Ld"], int(params["nL1"]),
                                          seed=seed, r_max=0.35 * params["Ld"])
    else:
        cxy, members = np.zeros((0, 2)), []
    # fleet upper bounds
    full_xy = np.vstack([xy, cxy]) if len(members) else xy
    dist = np.sqrt(((full_xy[:, None, :] - full_xy[None, :, :]) ** 2).sum(-1))
    n_indep = greedy_trip_count(S0, 0, dist, demand, int(params["Qd"]), params["Ld"]) + 1 if S0 else 0
    if members:
        cl_dem = np.zeros(len(full_xy), dtype=np.int64)
        for c, mem in enumerate(members):
            cl_dem[n + 1 + c] = sum(int(demand[s]) for s in mem)
        n_trucks = greedy_trip_count([n + 1 + c for c in range(len(members))], 0, dist, cl_dem,
                                     int(params["Qt"]), params["Lt"]) + 1
    else:
        n_trucks = 0
    return Instance(name, dict(params), tuple(depot), xy, np.asarray(demand, dtype=np.int64), np.asarray(tw_a, float),
                    np.asarray(tw_b, float), S0, S1, cxy, members, n_trucks, n_indep, meta)


def make_toy(scenario: str = "A-1", seed: int = 2026, extra: Optional[Dict] = None) -> Instance:
    params = dict(BASE_PARAMS); params.update(SCENARIOS[scenario]); params["Qt"] = 15
    if extra:
        params.update(extra)
    n = len(TOY_CUSTOMERS)
    xy = np.zeros((n + 1, 2)); dem = np.zeros(n + 1, dtype=np.int64)
    xy[0] = TOY_DEPOT
    for i, x, y, d in TOY_CUSTOMERS:
        xy[i] = (x, y); dem[i] = d
    # time windows: reproducible, spread over a two-hour horizon (seconds)
    rng = np.random.default_rng(seed)
    a = np.zeros(n + 1); b = np.zeros(n + 1)
    for i in range(1, n + 1):
        a[i] = float(rng.integers(600, 5400)); b[i] = a[i] + float(rng.integers(900, 1800))
    name = f"toy-{scenario}"
    meta = dict(source="toy", scenario=scenario, tw_rule="a~U[600,5400], width~U[900,1800] s, seed=%d" % seed)
    return _build(name, params, TOY_DEPOT, xy, dem, a, b, seed, meta)


def load_solomon(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 7 and p[0].isdigit():
                rows.append([int(v) for v in p])
    return rows


def make_solomon(n_customers: int, scenario: str = "A-1", seed: int = 2026,
                 raw_path: Optional[str] = None, extra: Optional[Dict] = None,
                 name: Optional[str] = None, source: str = "C") -> Instance:
    """
    Solomon based instance (source "C" = C101 clustered, "R" = R101 random, "RC" = RC101 mixed) with
    the first `n_customers` customers.  Coordinates x100 (metres, 10 km x 10 km area);
    demand = min(4, solomon_demand // 10); time windows = Solomon windows x time scale (seconds),
    the scale being chosen so that every source has a working day of about 3.3 h.
    """
    params = dict(BASE_PARAMS); params.update(SCENARIOS[scenario])
    if extra:
        params.update(extra)
    src = SOLOMON_SOURCES[source]
    raw_path = raw_path or os.path.join(os.path.dirname(__file__), "..", "data", "raw", src["file"])
    scale = src["scale"]
    rows = load_solomon(raw_path)
    depot = (rows[0][1] * 100.0, rows[0][2] * 100.0)
    n = n_customers
    xy = np.zeros((n + 1, 2)); dem = np.zeros(n + 1, dtype=np.int64); a = np.zeros(n + 1); b = np.zeros(n + 1)
    xy[0] = depot
    for r in rows[1:n + 1]:
        i = r[0]; xy[i] = (r[1] * 100.0, r[2] * 100.0); dem[i] = max(1, min(4, r[3] // 10)) if source != "C" else min(4, r[3] // 10)
        a[i] = r[4] * scale; b[i] = r[5] * scale
    name = name or (f"{scenario}-{n}" if source == "C" else f"{scenario}-{source}-{n}")
    meta = dict(source=f"solomon_{src['file'][8:-4]}", geometry=source, scenario=scenario, n=n, time_scale=scale, seed=seed)
    return _build(name, params, depot, xy, dem, a, b, seed, meta)


def make_merged(sources=("C", "R"), scenario: str = "A-1", seed: int = 2026, extra: Optional[Dict] = None,
                name: Optional[str] = None) -> Instance:
    """Large instance: union of the customer sets of several Solomon instances (same 10 km x 10 km
    area, C101 depot); time windows scaled per source as in make_solomon."""
    params = dict(BASE_PARAMS); params.update(SCENARIOS[scenario])
    if extra:
        params.update(extra)
    xs = []; ds = []; As = []; Bs = []
    depot = None
    for src in sources:
        sd = SOLOMON_SOURCES[src]
        rows = load_solomon(os.path.join(os.path.dirname(__file__), "..", "data", "raw", sd["file"]))
        if depot is None:
            depot = (rows[0][1] * 100.0, rows[0][2] * 100.0)
        for r in rows[1:]:
            xs.append((r[1] * 100.0, r[2] * 100.0)); ds.append(max(1, min(4, r[3] // 10)))
            As.append(r[4] * sd["scale"]); Bs.append(r[5] * sd["scale"])
    n = len(xs)
    xy = np.zeros((n + 1, 2)); dem = np.zeros(n + 1, dtype=np.int64); a = np.zeros(n + 1); b = np.zeros(n + 1)
    xy[0] = depot
    for i in range(n):
        xy[i + 1] = xs[i]; dem[i + 1] = ds[i]; a[i + 1] = As[i]; b[i + 1] = Bs[i]
    name = name or f"{scenario}-M-{n}"
    meta = dict(source="+".join(sources), geometry="M", scenario=scenario, n=n, seed=seed)
    return _build(name, params, depot, xy, dem, a, b, seed, meta)


def instance_path(root: str, name: str) -> str:
    return os.path.join(root, f"{name}.json")


def generate_all(root: str, scales=(15, 30, 50, 75, 100), rc_scales=(50, 75, 100), seed: int = 2026, verbose=True) -> List[str]:
    os.makedirs(root, exist_ok=True)
    out = []
    for sc in SCENARIOS:
        inst = make_toy(sc, seed=seed)
        p = instance_path(root, inst.name); inst.save(p); out.append(p)
        if verbose:
            print(f"{inst.name:10s} n={inst.n:3d} |S0|={len(inst.S0):3d} |S1|={len(inst.S1):3d} m={inst.m:2d} K={inst.n_trucks} L0={inst.n_indep_drones}")
        for n in scales:
            inst = make_solomon(n, sc, seed=seed)
            p = instance_path(root, inst.name); inst.save(p); out.append(p)
            if verbose:
                print(f"{inst.name:10s} n={inst.n:3d} |S0|={len(inst.S0):3d} |S1|={len(inst.S1):3d} m={inst.m:2d} K={inst.n_trucks} L0={inst.n_indep_drones}")
        for src in ("R", "RC"):
            for n in rc_scales:
                inst = make_solomon(n, sc, seed=seed, source=src)
                p = instance_path(root, inst.name); inst.save(p); out.append(p)
                if verbose:
                    print(f"{inst.name:10s} n={inst.n:3d} |S0|={len(inst.S0):3d} |S1|={len(inst.S1):3d} m={inst.m:2d} K={inst.n_trucks} L0={inst.n_indep_drones}")
    # large instances: union of the C101 and R101 (200) and of the C101, R101 and RC101 (300) customer sets
    for sc in ("A-1", "B-1", "C-1"):
        for srcs in (("C", "R"), ("C", "R", "RC")):
            inst = make_merged(srcs, sc, seed=seed)
            p = instance_path(root, inst.name); inst.save(p); out.append(p)
            if verbose:
                print(f"{inst.name:10s} n={inst.n:3d} |S0|={len(inst.S0):3d} |S1|={len(inst.S1):3d} m={inst.m:2d} K={inst.n_trucks} L0={inst.n_indep_drones}")
    return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "instances")
    generate_all(root)
