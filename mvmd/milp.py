"""
Arc-based MILP of the manuscript (Section 4) with optional strengthening (Section 4.5).

The problem decomposes into two independent models (Assumption 5):
    * the truck-supported model  (variables x, y, t, t_hat, u, u_hat, w, p)   constraints (5)-(6),
      (8)-(14), (18)-(20), (22)-(23), (25)-(26), (28), (30)-(34), (40)-(41)
    * the independent-drone model (variables z, t_check, u_check, p)          constraints (7),
      (15)-(17), (21), (24), (27), (29), (42)-(43)
Constraint names in the solver carry the manuscript numbers (e.g. "C12_k0_l1_c3").

Timing convention: drones follow earliest-start schedules, i.e. the resource-extension constraints
(28) and (29) are imposed as equalities whenever an arc is used (two big-M inequalities).  Trucks may
wait at the depot and at clusters (constraints (32)-(34) remain one-sided), exactly as evaluated by
the heuristic (solution.py).

Strengthening options (all valid inequalities, Section 4.5):
    prep       : elimination of drone arcs that cannot belong to any feasible sortie
    rounding   : rounded capacity / distance lower bounds on the number of trucks, sorties, trips
    two_cycle  : x_ij + x_ji <= 1 type inequalities
    lifted_mtz : Desrochers-Laporte lifting of the MTZ constraints
    symmetry   : symmetry breaking among identical trucks / drones (usage order + lexicographic)
    disagg     : disaggregated truck-drone linking y_ij <= sum_i x_ic  (replaces aggregated (18))

Backends: HiGHS (highspy) and Gurobi (gurobipy).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .solution import Solution

ALL_STRENGTH = dict(prep=True, rounding=True, two_cycle=True, lifted_mtz=True, symmetry=True, disagg=True)
NO_STRENGTH = dict(prep=False, rounding=False, two_cycle=False, lifted_mtz=False, symmetry=False, disagg=False)


# ----------------------------------------------------------------------------
# generic model container
# ----------------------------------------------------------------------------
class Model:
    def __init__(self, name: str):
        self.name = name
        self.var_names: List[str] = []; self.lb: List[float] = []; self.ub: List[float] = []
        self.obj: List[float] = []; self.is_int: List[bool] = []
        self.rows: List[Tuple[Dict[int, float], float, float, str]] = []
        self.index: Dict[str, int] = {}

    def var(self, name: str, lb=0.0, ub=1.0, obj=0.0, integer=True) -> int:
        i = len(self.var_names)
        self.var_names.append(name); self.lb.append(lb); self.ub.append(ub); self.obj.append(obj); self.is_int.append(integer)
        self.index[name] = i
        return i

    def add(self, coefs: Dict[int, float], lb: float, ub: float, name: str):
        c = {k: v for k, v in coefs.items() if abs(v) > 1e-12}
        self.rows.append((c, lb, ub, name))

    def eq(self, coefs, rhs, name):
        self.add(coefs, rhs, rhs, name)

    def le(self, coefs, rhs, name):
        self.add(coefs, -math.inf, rhs, name)

    def ge(self, coefs, rhs, name):
        self.add(coefs, rhs, math.inf, name)

    @property
    def n_vars(self):
        return len(self.var_names)

    @property
    def n_int(self):
        return sum(self.is_int)

    @property
    def n_rows(self):
        return len(self.rows)


def _lin(*terms):
    """terms: (idx, coef) pairs or dicts -> merged dict"""
    d: Dict[int, float] = {}
    for t in terms:
        if isinstance(t, dict):
            for k, v in t.items():
                d[k] = d.get(k, 0.0) + v
        else:
            k, v = t; d[k] = d.get(k, 0.0) + v
    return d


# ----------------------------------------------------------------------------
# solver backends
# ----------------------------------------------------------------------------
def solve_model(model: Model, backend: str = "highs", time_limit: float = 3600.0, mip_gap: float = 1e-4,
                threads: int = 1, verbose: bool = False, relax: bool = False, log_file: Optional[str] = None) -> Dict:
    if backend == "highs":
        return _solve_highs(model, time_limit, mip_gap, threads, verbose, relax, log_file)
    elif backend == "gurobi":
        return _solve_gurobi(model, time_limit, mip_gap, threads, verbose, relax, log_file)
    raise ValueError(backend)


def _solve_highs(model: Model, time_limit, mip_gap, threads, verbose, relax, log_file):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(verbose))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.setOptionValue("threads", int(threads))
    if log_file:
        h.setOptionValue("log_file", log_file); h.setOptionValue("log_to_console", bool(verbose)); h.setOptionValue("output_flag", True)
    lp = highspy.HighsLp()
    nv = model.n_vars; nr = model.n_rows
    lp.num_col_ = nv; lp.num_row_ = nr
    lp.col_cost_ = np.array(model.obj, dtype=float)
    lp.col_lower_ = np.array(model.lb, dtype=float); lp.col_upper_ = np.array(model.ub, dtype=float)
    lp.row_lower_ = np.array([r[1] for r in model.rows], dtype=float); lp.row_upper_ = np.array([r[2] for r in model.rows], dtype=float)
    starts = [0]; idx = []; vals = []
    for coefs, lo, up, name in model.rows:
        for k, v in coefs.items():
            idx.append(k); vals.append(v)
        starts.append(len(idx))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = np.array(starts, dtype=np.int32); lp.a_matrix_.index_ = np.array(idx, dtype=np.int32)
    lp.a_matrix_.value_ = np.array(vals, dtype=float)
    lp.a_matrix_.num_col_ = nv; lp.a_matrix_.num_row_ = nr
    if not relax:
        lp.integrality_ = [highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous for b in model.is_int]
    h.passModel(lp)
    for i, nm in enumerate(model.var_names):
        h.passColName(i, nm)
    for i, r in enumerate(model.rows):
        h.passRowName(i, r[3])
    t0 = time.perf_counter()
    h.run()
    el = time.perf_counter() - t0
    st = h.getModelStatus()
    status = h.modelStatusToString(st)
    info = h.getInfo()
    sol = h.getSolution()
    x = np.array(sol.col_value) if len(sol.col_value) else None
    obj = float(info.objective_function_value) if x is not None else None
    bound = float(info.mip_dual_bound) if not relax else obj
    gap = float(info.mip_gap) if not relax else 0.0
    nodes = int(info.mip_node_count) if not relax else 0
    has_sol = x is not None and (status in ("Optimal", "Time limit reached", "Interrupt", "Objective target", "Feasible")) and info.primal_solution_status == 2
    if status == "Time limit reached" and not has_sol:
        obj = None
    return dict(status=status, obj=obj if has_sol or relax else None, bound=bound, gap=gap, nodes=nodes, time=el,
                x=x if (has_sol or relax) else None, n_vars=model.n_vars, n_int=model.n_int, n_rows=model.n_rows,
                optimal=(status == "Optimal"))


def _solve_gurobi(model: Model, time_limit, mip_gap, threads, verbose, relax, log_file):
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True); env.setParam("OutputFlag", int(verbose)); env.start()
    m = gp.Model(model.name, env=env)
    m.Params.TimeLimit = time_limit; m.Params.MIPGap = mip_gap; m.Params.Threads = threads
    if log_file:
        m.Params.LogFile = log_file
    vs = []
    for i, nm in enumerate(model.var_names):
        vt = GRB.CONTINUOUS if (relax or not model.is_int[i]) else (GRB.BINARY if model.ub[i] == 1.0 and model.lb[i] == 0.0 else GRB.INTEGER)
        vs.append(m.addVar(lb=model.lb[i], ub=model.ub[i], obj=model.obj[i], vtype=vt, name=nm))
    m.update()
    for coefs, lo, up, name in model.rows:
        expr = gp.LinExpr([(v, vs[k]) for k, v in coefs.items()])
        if lo == up:
            m.addLConstr(expr == lo, name=name)
        else:
            if lo > -math.inf:
                m.addLConstr(expr >= lo, name=name + "_lo" if up < math.inf else name)
            if up < math.inf:
                m.addLConstr(expr <= up, name=name + "_up" if lo > -math.inf else name)
    m.ModelSense = GRB.MINIMIZE
    t0 = time.perf_counter(); m.optimize(); el = time.perf_counter() - t0
    has_sol = m.SolCount > 0
    status = {GRB.OPTIMAL: "Optimal", GRB.TIME_LIMIT: "Time limit reached", GRB.INFEASIBLE: "Infeasible"}.get(m.Status, str(m.Status))
    x = np.array([v.X for v in vs]) if has_sol else None
    return dict(status=status, obj=float(m.ObjVal) if has_sol else None, bound=float(m.ObjBound) if not relax else (float(m.ObjVal) if has_sol else None),
                gap=float(m.MIPGap) if (has_sol and not relax) else 0.0, nodes=int(m.NodeCount), time=el, x=x,
                n_vars=model.n_vars, n_int=model.n_int, n_rows=model.n_rows, optimal=(m.Status == GRB.OPTIMAL))


# ----------------------------------------------------------------------------
# model builders
# ----------------------------------------------------------------------------
def _horizon(inst) -> float:
    """Safe upper bound on any time stamp in an optimal schedule (used for big-M and bounds)."""
    p = inst.params
    tw_max = float(inst.tw_b[1:].max()) if inst.n else 0.0
    truck = inst.Lt / p["vt"]
    drones = sum(inst.Ld / p["vd"] + len(mem) * p["service_time"] for mem in inst.cluster_members)
    return tw_max + truck + drones + inst.Ld / p["vd"] + inst.n * p["service_time"]


def build_truck_model(inst, strengthen: Optional[Dict] = None, fleet: Optional[int] = None) -> Model:
    """Truck-supported drone model (x, y, t, t_hat, u, u_hat, w, p)."""
    S = dict(NO_STRENGTH); S.update(strengthen or {})
    p = inst.params; m = inst.m
    K = fleet if fleet is not None else inst.n_trucks
    L1 = inst.nL1
    cT, cD, ct, cd, cA, cB, e = p["cT"], p["cD"], p["ct"], p["cd"], p["cA"], p["cB"], p["service_time"]
    vt, vd = p["vt"], p["vd"]
    M = Model("truck_supported")
    if m == 0 or K == 0:
        return M
    Tmax = _horizon(inst)
    Wmax = inst.Ld / vd + max(len(mem) for mem in inst.cluster_members) * e   # longest possible sortie
    # ---- node sets (MILP-local names)
    # truck nodes: "O", "D", ("cm", c), ("cp", c)
    cnode = inst.cluster_node
    def d_truck(i, j):
        if i == "O":
            return inst.dist[0, cnode(j[1])] if isinstance(j, tuple) else 0.0
        if j == "D":
            return inst.dist[cnode(i[1]), 0]
        if i[0] == "cm" and j[0] == "cp" and i[1] == j[1]:
            return 0.0
        return inst.dist[cnode(i[1]), cnode(j[1])]
    truck_arcs = []
    for c in range(m):
        truck_arcs.append(("O", ("cm", c))); truck_arcs.append((("cm", c), ("cp", c))); truck_arcs.append((("cp", c), "D"))
        for c2 in range(m):
            if c2 != c:
                truck_arcs.append((("cp", c), ("cm", c2)))
    truck_arcs.append(("O", "D"))
    # drone arcs per cluster (with optional preprocessing)
    drone_arcs: List[List[Tuple]] = []
    for c in range(m):
        cn = cnode(c); mem = inst.cluster_members[c]; arcs = []
        for s in mem:
            arcs.append((("cm", c), s)); arcs.append((s, ("cp", c)))
            for s2 in mem:
                if s2 != s:
                    if S["prep"]:
                        if inst.dist[cn, s] + inst.dist[s, s2] + inst.dist[s2, cn] > inst.Ld + 1e-9:
                            continue
                        if inst.demand[s] + inst.demand[s2] > inst.Qd:
                            continue
                    arcs.append((s, s2))
        arcs.append((("cm", c), ("cp", c)))
        drone_arcs.append(arcs)

    def dn(i, c):  # printable node name ("O", "D", "cm3", "cp3", or the customer id)
        if isinstance(i, str):
            return i
        return i if isinstance(i, int) else f"{i[0]}{i[1]}"

    def d_drone(i, j, c):
        cn = cnode(c)
        a = cn if isinstance(i, tuple) else i
        b = cn if isinstance(j, tuple) else j
        if isinstance(i, tuple) and isinstance(j, tuple):
            return 0.0
        return inst.dist[a, b]

    # ---- variables
    x = {}
    for k in range(K):
        for (i, j) in truck_arcs:
            x[k, i, j] = M.var(f"x_{k}_{dn(i,0)}_{dn(j,0)}", 0, 1, obj=(cT if (i == "O" and j != "D") else 0.0) + ct * d_truck(i, j) / vt)
    y = {}
    for k in range(K):
        for l in range(L1):
            for c in range(m):
                for (i, j) in drone_arcs[c]:
                    fixed = cD if (isinstance(i, tuple) and isinstance(j, int)) else 0.0
                    y[k, l, c, i, j] = M.var(f"y_{k}_{l}_{dn(i,c)}_{dn(j,c)}", 0, 1, obj=fixed + cd * d_drone(i, j, c) / vd)
    truck_nodes = ["O", "D"] + [("cm", c) for c in range(m)] + [("cp", c) for c in range(m)]
    t = {(k, i): M.var(f"t_{k}_{dn(i,0)}", 0, Tmax, 0.0, False) for k in range(K) for i in truck_nodes}
    NT = 2 * m   # non-depot truck nodes
    u = {(k, i): M.var(f"u_{k}_{dn(i,0)}", 0 if i in ("O", "D") else 1, NT + 1 if i in ("O", "D") else NT, 0.0, False)
         for k in range(K) for i in truck_nodes}
    th = {}; uh = {}; w = {}; pA = {}; pB = {}
    for k in range(K):
        for c in range(m):
            w[k, c] = M.var(f"w_{k}_{c}", 0, Wmax, 0.0, False)
            for l in range(L1):
                nodes = [("cm", c), ("cp", c)] + list(inst.cluster_members[c])
                Nc = len(inst.cluster_members[c])
                for i in nodes:
                    th[k, l, c, i] = M.var(f"that_{k}_{l}_{dn(i,c)}", 0, Tmax, 0.0, False)
                    lo_u, up_u = (0, Nc + 1) if isinstance(i, tuple) else (1, Nc)
                    uh[k, l, c, i] = M.var(f"uhat_{k}_{l}_{dn(i,c)}", lo_u, up_u, 0.0, False)
                for s in inst.cluster_members[c]:
                    pA[k, l, s] = M.var(f"pA_{k}_{l}_{s}", 0, math.inf, cA, False)
                    pB[k, l, s] = M.var(f"pB_{k}_{l}_{s}", 0, math.inf, cB, False)

    def xin_c(k, c):   # sum_i x^k_{i, c-}
        return {x[k, i, ("cm", c)]: 1.0 for (i, j) in truck_arcs if j == ("cm", c)}

    def xout_cp(k, c):  # sum_j x^k_{c+, j}
        return {x[k, ("cp", c), j]: 1.0 for (i, j) in truck_arcs if i == ("cp", c)}

    # ---- constraints
    for c in range(m):   # (5)
        M.eq(_lin(*[(x[k, i, ("cm", c)], 1.0) for k in range(K) for (i, j) in truck_arcs if j == ("cm", c)]), 1.0, f"C5_c{c}")
    for c in range(m):   # (6)
        for s in inst.cluster_members[c]:
            M.eq(_lin(*[(y[k, l, c, i, j], 1.0) for k in range(K) for l in range(L1) for (i, j) in drone_arcs[c] if j == s]), 1.0, f"C6_s{s}")
    for k in range(K):   # (8), (9)
        M.eq({x[k, "O", j]: 1.0 for (i, j) in truck_arcs if i == "O"}, 1.0, f"C8_k{k}")
        M.eq({x[k, i, "D"]: 1.0 for (i, j) in truck_arcs if j == "D"}, 1.0, f"C9_k{k}")
        for c in range(m):   # (10), (11)
            M.eq(_lin(xin_c(k, c), (x[k, ("cm", c), ("cp", c)], -1.0)), 0.0, f"C10_k{k}_c{c}")
            M.eq(_lin((x[k, ("cm", c), ("cp", c)], 1.0), {v: -1.0 for v in xout_cp(k, c)}), 0.0, f"C11_k{k}_c{c}")
            for l in range(L1):   # (12), (13)
                M.eq(_lin({y[k, l, c, i, j]: 1.0 for (i, j) in drone_arcs[c] if i == ("cm", c)}, {v: -1.0 for v in xin_c(k, c)}), 0.0, f"C12_k{k}_l{l}_c{c}")
                M.eq(_lin({y[k, l, c, i, j]: 1.0 for (i, j) in drone_arcs[c] if j == ("cp", c)}, {v: -1.0 for v in xout_cp(k, c)}), 0.0, f"C13_k{k}_l{l}_c{c}")
                for s in inst.cluster_members[c]:   # (14)
                    M.eq(_lin({y[k, l, c, i, j]: 1.0 for (i, j) in drone_arcs[c] if j == s}, {y[k, l, c, i, j]: -1.0 for (i, j) in drone_arcs[c] if i == s}), 0.0, f"C14_k{k}_l{l}_s{s}")
            # (18) aggregated linking (or disaggregated VI)
            if S["disagg"]:
                for l in range(L1):
                    for (i, j) in drone_arcs[c]:
                        if isinstance(i, tuple):
                            continue   # implied by (12)
                        M.le(_lin((y[k, l, c, i, j], 1.0), {v: -1.0 for v in xin_c(k, c)}), 0.0, f"VI_disagg_k{k}_l{l}_{dn(i,c)}_{dn(j,c)}")
            else:
                big = sum(len(drone_arcs[c]) for _ in range(L1))
                M.ge(_lin({v: float(big) for v in xin_c(k, c)}, {y[k, l, c, i, j]: -1.0 for l in range(L1) for (i, j) in drone_arcs[c]}), 0.0, f"C18_k{k}_c{c}")
        # (19) truck range, (22) truck capacity
        M.le({x[k, i, j]: d_truck(i, j) for (i, j) in truck_arcs}, inst.Lt, f"C19_k{k}")
        M.le({x[k, i, ("cm", c)]: float(inst.cluster_demand[c]) for (i, j) in truck_arcs if isinstance(j, tuple) and j[0] == "cm" for c in [j[1]]}, float(inst.Qt), f"C22_k{k}")
        for l in range(L1):
            for c in range(m):   # (20) range, (23) capacity per sortie
                M.le({y[k, l, c, i, j]: d_drone(i, j, c) for (i, j) in drone_arcs[c]}, inst.Ld, f"C20_k{k}_l{l}_c{c}")
                M.le({y[k, l, c, i, j]: float(inst.demand[j]) for (i, j) in drone_arcs[c] if isinstance(j, int)}, float(inst.Qd), f"C23_k{k}_l{l}_c{c}")
        # (25) MTZ truck layer
        for (i, j) in truck_arcs:
            if i == "O" or j == "D":
                continue
            if S["lifted_mtz"] and not (i[0] == "cm" and j[0] == "cp" and i[1] == j[1]):
                # Desrochers-Laporte lifting: u_i - u_j + N x_ij + (N-2) x_ji <= N - 1
                coefs = _lin((u[k, i], 1.0), (u[k, j], -1.0), (x[k, i, j], float(NT)))
                if (k, j, i) in x:
                    coefs = _lin(coefs, (x[k, j, i], float(NT - 2)))
                M.le(coefs, float(NT - 1), f"C25L_k{k}_{dn(i,0)}_{dn(j,0)}")
            else:
                M.le(_lin((u[k, i], 1.0), (u[k, j], -1.0), (x[k, i, j], float(NT))), float(NT - 1), f"C25_k{k}_{dn(i,0)}_{dn(j,0)}")
        # (26) MTZ drone layers, (28) drone time (two-sided), (30) waiting, (40)-(41) penalties
        for l in range(L1):
            for c in range(m):
                Nc = len(inst.cluster_members[c])
                for (i, j) in drone_arcs[c]:
                    if isinstance(i, tuple) and isinstance(j, tuple):
                        # c- -> c+ : drone stays on the truck; time at c+ >= time at c-
                        M.ge(_lin((th[k, l, c, j], 1.0), (th[k, l, c, i], -1.0)), 0.0, f"C28_k{k}_l{l}_c{c}_stay")
                        continue
                    if isinstance(i, int) and isinstance(j, int):
                        if S["lifted_mtz"]:
                            coefs = _lin((uh[k, l, c, i], 1.0), (uh[k, l, c, j], -1.0), (y[k, l, c, i, j], float(Nc)))
                            if (k, l, c, j, i) in y:
                                coefs = _lin(coefs, (y[k, l, c, j, i], float(Nc - 2)))
                            M.le(coefs, float(Nc - 1), f"C26L_k{k}_l{l}_{i}_{j}")
                        else:
                            M.le(_lin((uh[k, l, c, i], 1.0), (uh[k, l, c, j], -1.0), (y[k, l, c, i, j], float(Nc + 1))), float(Nc), f"C26_k{k}_l{l}_{i}_{j}")
                    elif isinstance(i, tuple):   # c- -> s : u_s >= 1 = u_{c-} + 1 holds by bounds; keep MTZ form
                        M.le(_lin((uh[k, l, c, i], 1.0), (uh[k, l, c, j], -1.0), (y[k, l, c, i, j], float(Nc + 1))), float(Nc), f"C26_k{k}_l{l}_{dn(i,c)}_{j}")
                    else:                       # s -> c+
                        M.le(_lin((uh[k, l, c, i], 1.0), (uh[k, l, c, j], -1.0), (y[k, l, c, i, j], float(Nc + 1))), float(Nc), f"C26_k{k}_l{l}_{i}_{dn(j,c)}")
                    tau = d_drone(i, j, c) / vd + (e if isinstance(j, int) else 0.0)
                    bigM = Tmax + tau
                    # (28): t_j >= t_i + tau - M(1-y)  and  t_j <= t_i + tau + M(1-y)
                    M.ge(_lin((th[k, l, c, j], 1.0), (th[k, l, c, i], -1.0), (y[k, l, c, i, j], -bigM)), tau - bigM, f"C28_k{k}_l{l}_{dn(i,c)}_{dn(j,c)}")
                    M.le(_lin((th[k, l, c, j], 1.0), (th[k, l, c, i], -1.0), (y[k, l, c, i, j], bigM)), tau + bigM, f"C28b_k{k}_l{l}_{dn(i,c)}_{dn(j,c)}")
                # (30) waiting time of the truck at c >= sortie duration of every drone
                M.ge(_lin((w[k, c], 1.0), (th[k, l, c, ("cp", c)], -1.0), (th[k, l, c, ("cm", c)], 1.0)), 0.0, f"C30_k{k}_l{l}_c{c}")
                # (33)-(34) synchronisation of the launch time with the truck arrival
                #   t_hat_{c-} >= t_{c-} - M(1 - sum x)   and   t_hat_{c-} <= t_{c-} + M(1 - sum x)
                M.ge(_lin((th[k, l, c, ("cm", c)], 1.0), (t[k, ("cm", c)], -1.0), {v: -Tmax for v in xin_c(k, c)}), -Tmax, f"C33_k{k}_l{l}_c{c}")
                M.le(_lin((th[k, l, c, ("cm", c)], 1.0), (t[k, ("cm", c)], -1.0), {v: Tmax for v in xin_c(k, c)}), Tmax, f"C34_k{k}_l{l}_c{c}")
                for s in inst.cluster_members[c]:   # (40), (41)
                    served = {y[k, l, c, i, j]: 1.0 for (i, j) in drone_arcs[c] if i == s}
                    bigM = Tmax + float(inst.tw_a[s])
                    M.ge(_lin((pA[k, l, s], 1.0), (th[k, l, c, s], 1.0), {v: -bigM for v in served}), float(inst.tw_a[s]) - bigM, f"C40_k{k}_l{l}_s{s}")
                    M.ge(_lin((pB[k, l, s], 1.0), (th[k, l, c, s], -1.0)), -float(inst.tw_b[s]), f"C41_k{k}_l{l}_s{s}")
        # (32) truck time propagation with waiting at c+
        for (i, j) in truck_arcs:
            tau = d_truck(i, j) / vt
            bigM = Tmax + tau + Wmax
            coefs = _lin((t[k, j], 1.0), (t[k, i], -1.0), (x[k, i, j], -bigM))
            if isinstance(j, tuple) and j[0] == "cp":
                coefs = _lin(coefs, (w[k, j[1]], -1.0))
            M.ge(coefs, tau - bigM, f"C32_k{k}_{dn(i,0)}_{dn(j,0)}")
    # ---- valid inequalities
    if S["rounding"]:
        tot = int(inst.cluster_demand.sum())
        M.ge({x[k, "O", ("cm", c)]: 1.0 for k in range(K) for c in range(m)}, float(math.ceil(tot / inst.Qt)), "VI_fleet")
        for c in range(m):
            cn = cnode(c); mem = inst.cluster_members[c]
            lb_cap = math.ceil(inst.cluster_demand[c] / inst.Qd)
            nodes_c = [cn] + list(mem)
            lb_dist = 0.0
            for s in mem:
                din = min(inst.dist[i, s] for i in nodes_c if i != s); dout = min(inst.dist[s, j] for j in nodes_c if j != s)
                lb_dist += 0.5 * (din + dout)
            lb = max(lb_cap, math.ceil(lb_dist / inst.Ld - 1e-9))
            M.ge({y[k, l, c, i, j]: 1.0 for k in range(K) for l in range(L1) for (i, j) in drone_arcs[c] if i == ("cm", c) and isinstance(j, int)}, float(lb), f"VI_sorties_c{c}")
    if S["two_cycle"]:
        for k in range(K):
            for c in range(m):
                for c2 in range(c + 1, m):
                    M.le(_lin((x[k, ("cp", c), ("cm", c2)], 1.0), (x[k, ("cp", c2), ("cm", c)], 1.0)), 1.0, f"VI_2cyc_k{k}_{c}_{c2}")
            for l in range(L1):
                for c in range(m):
                    mem = inst.cluster_members[c]
                    for a_ in range(len(mem)):
                        for b_ in range(a_ + 1, len(mem)):
                            i, j = mem[a_], mem[b_]
                            if (k, l, c, i, j) in y and (k, l, c, j, i) in y:
                                M.le(_lin((y[k, l, c, i, j], 1.0), (y[k, l, c, j, i], 1.0)), 1.0, f"VI_2cyc_k{k}_l{l}_{i}_{j}")
    if S["symmetry"]:
        for k in range(K - 1):   # truck usage order
            M.ge(_lin({x[k, "O", ("cm", c)]: 1.0 for c in range(m)}, {x[k + 1, "O", ("cm", c)]: -1.0 for c in range(m)}), 0.0, f"VI_symT_k{k}")
        # lexicographic cluster assignment: cluster 0 on truck 0; truck k visits c only if truck k-1 visits some c' < c
        if K >= 1 and m >= 1:
            M.eq(xin_c(0, 0), 1.0, "VI_symT_c0")
        for k in range(1, K):
            for c in range(1, m):
                M.le(_lin(xin_c(k, c), {v: -1.0 for c2 in range(c) for v in xin_c(k - 1, c2)}), 0.0, f"VI_symT_k{k}_c{c}")
        for k in range(K):
            for c in range(m):
                mem = inst.cluster_members[c]
                for l in range(L1 - 1):   # drone usage order at every cluster
                    M.ge(_lin({y[k, l, c, i, j]: 1.0 for (i, j) in drone_arcs[c] if i == ("cm", c) and isinstance(j, int)},
                              {y[k, l + 1, c, i, j]: -1.0 for (i, j) in drone_arcs[c] if i == ("cm", c) and isinstance(j, int)}), 0.0, f"VI_symD_k{k}_l{l}_c{c}")
                    # lexicographic: drone l+1 serves s only if drone l serves a lower-index customer
                    for idx_s, s in enumerate(mem):
                        lower = [s2 for s2 in mem[:idx_s]]
                        M.le(_lin({y[k, l + 1, c, i, j]: 1.0 for (i, j) in drone_arcs[c] if j == s},
                                  {y[k, l, c, i, j]: -1.0 for (i, j) in drone_arcs[c] if j in lower}), 0.0, f"VI_symD_k{k}_l{l}_s{s}")
    M.meta = dict(x=x, y=y, t=t, th=th, u=u, uh=uh, w=w, pA=pA, pB=pB, truck_arcs=truck_arcs, drone_arcs=drone_arcs, K=K, L1=L1)
    return M


def build_indep_model(inst, strengthen: Optional[Dict] = None, fleet: Optional[int] = None) -> Model:
    """Independent-drone model (z, t_check, u_check, p)."""
    S = dict(NO_STRENGTH); S.update(strengthen or {})
    p = inst.params
    L0 = fleet if fleet is not None else inst.n_indep_drones
    cD, cd, cA, cB, e, vd = p["cD"], p["cd"], p["cA"], p["cB"], p["service_time"], p["vd"]
    M = Model("independent")
    S0 = list(inst.S0)
    if not S0 or L0 == 0:
        return M
    Tmax = _horizon(inst)
    arcs = []
    for s in S0:
        arcs.append(("O", s)); arcs.append((s, "D"))
        for s2 in S0:
            if s2 != s:
                if S["prep"]:
                    if inst.dist[0, s] + inst.dist[s, s2] + inst.dist[s2, 0] > inst.Ld + 1e-9 or inst.demand[s] + inst.demand[s2] > inst.Qd:
                        continue
                arcs.append((s, s2))
    arcs.append(("O", "D"))

    def dd(i, j):
        a = 0 if isinstance(i, str) else i; b = 0 if isinstance(j, str) else j
        return 0.0 if (isinstance(i, str) and isinstance(j, str)) else inst.dist[a, b]

    z = {}
    for l in range(L0):
        for (i, j) in arcs:
            z[l, i, j] = M.var(f"z_{l}_{i}_{j}", 0, 1, obj=(cD if (i == "O" and j != "D") else 0.0) + cd * dd(i, j) / vd)
    nodes = ["O", "D"] + S0; n0 = len(S0)
    tc = {(l, i): M.var(f"tcheck_{l}_{i}", 0, Tmax, 0.0, False) for l in range(L0) for i in nodes}
    uc = {(l, i): M.var(f"ucheck_{l}_{i}", 0 if isinstance(i, str) else 1, n0 + 1 if isinstance(i, str) else n0, 0.0, False) for l in range(L0) for i in nodes}
    pA = {(l, s): M.var(f"pA_{l}_{s}", 0, math.inf, cA, False) for l in range(L0) for s in S0}
    pB = {(l, s): M.var(f"pB_{l}_{s}", 0, math.inf, cB, False) for l in range(L0) for s in S0}
    for s in S0:   # (7)
        M.eq({z[l, i, j]: 1.0 for l in range(L0) for (i, j) in arcs if j == s}, 1.0, f"C7_s{s}")
    for l in range(L0):
        M.eq({z[l, i, j]: 1.0 for (i, j) in arcs if i == "O"}, 1.0, f"C15_l{l}")      # (15)
        M.eq({z[l, i, j]: 1.0 for (i, j) in arcs if j == "D"}, 1.0, f"C16_l{l}")      # (16)
        for s in S0:                                                                  # (17)
            M.eq(_lin({z[l, i, j]: 1.0 for (i, j) in arcs if j == s}, {z[l, i, j]: -1.0 for (i, j) in arcs if i == s}), 0.0, f"C17_l{l}_s{s}")
        M.le({z[l, i, j]: dd(i, j) for (i, j) in arcs}, inst.Ld, f"C21_l{l}")        # (21)
        M.le({z[l, i, j]: float(inst.demand[j]) for (i, j) in arcs if isinstance(j, int)}, float(inst.Qd), f"C24_l{l}")  # (24)
        for (i, j) in arcs:
            if i == "O" and j == "D":
                continue
            if isinstance(i, int) and isinstance(j, int):                              # (27)
                if S["lifted_mtz"]:
                    coefs = _lin((uc[l, i], 1.0), (uc[l, j], -1.0), (z[l, i, j], float(n0)))
                    if (l, j, i) in z:
                        coefs = _lin(coefs, (z[l, j, i], float(n0 - 2)))
                    M.le(coefs, float(n0 - 1), f"C27L_l{l}_{i}_{j}")
                else:
                    M.le(_lin((uc[l, i], 1.0), (uc[l, j], -1.0), (z[l, i, j], float(n0 + 1))), float(n0), f"C27_l{l}_{i}_{j}")
            else:
                M.le(_lin((uc[l, i], 1.0), (uc[l, j], -1.0), (z[l, i, j], float(n0 + 1))), float(n0), f"C27_l{l}_{i}_{j}")
            tau = dd(i, j) / vd + (e if isinstance(j, int) else 0.0)                 # (29) two-sided
            bigM = Tmax + tau
            M.ge(_lin((tc[l, j], 1.0), (tc[l, i], -1.0), (z[l, i, j], -bigM)), tau - bigM, f"C29_l{l}_{i}_{j}")
            M.le(_lin((tc[l, j], 1.0), (tc[l, i], -1.0), (z[l, i, j], bigM)), tau + bigM, f"C29b_l{l}_{i}_{j}")
        for s in S0:                                                                  # (42), (43)
            served = {z[l, i, j]: 1.0 for (i, j) in arcs if i == s}
            bigM = Tmax + float(inst.tw_a[s])
            M.ge(_lin((pA[l, s], 1.0), (tc[l, s], 1.0), {v: -bigM for v in served}), float(inst.tw_a[s]) - bigM, f"C42_l{l}_s{s}")
            M.ge(_lin((pB[l, s], 1.0), (tc[l, s], -1.0)), -float(inst.tw_b[s]), f"C43_l{l}_s{s}")
    if S["rounding"]:
        lb_cap = math.ceil(sum(int(inst.demand[s]) for s in S0) / inst.Qd)
        nodes0 = [0] + S0; lb_dist = 0.0
        for s in S0:
            din = min(inst.dist[i, s] for i in nodes0 if i != s); dout = min(inst.dist[s, j] for j in nodes0 if j != s)
            lb_dist += 0.5 * (din + dout)
        lb = max(lb_cap, math.ceil(lb_dist / inst.Ld - 1e-9))
        M.ge({z[l, "O", s]: 1.0 for l in range(L0) for s in S0}, float(lb), "VI_trips")
    if S["two_cycle"]:
        for l in range(L0):
            for a_ in range(n0):
                for b_ in range(a_ + 1, n0):
                    i, j = S0[a_], S0[b_]
                    if (l, i, j) in z and (l, j, i) in z:
                        M.le(_lin((z[l, i, j], 1.0), (z[l, j, i], 1.0)), 1.0, f"VI_2cyc_l{l}_{i}_{j}")
    if S["symmetry"]:
        for l in range(L0 - 1):
            M.ge(_lin({z[l, "O", s]: 1.0 for s in S0}, {z[l + 1, "O", s]: -1.0 for s in S0}), 0.0, f"VI_symI_l{l}")
            for idx_s, s in enumerate(S0):
                lower = S0[:idx_s]
                M.le(_lin({z[l + 1, i, j]: 1.0 for (i, j) in arcs if j == s}, {z[l, i, j]: -1.0 for (i, j) in arcs if j in lower}), 0.0, f"VI_symI_l{l}_s{s}")
    M.meta = dict(z=z, tc=tc, uc=uc, pA=pA, pB=pB, arcs=arcs, L0=L0)
    return M


# ----------------------------------------------------------------------------
# solution extraction
# ----------------------------------------------------------------------------
def extract_solution(inst, truck_model: Optional[Model], truck_res: Optional[Dict],
                     indep_model: Optional[Model], indep_res: Optional[Dict]) -> Solution:
    sol = Solution(inst)
    if truck_model is not None and truck_res is not None and truck_res.get("x") is not None and truck_model.n_vars:
        xv = truck_res["x"]; meta = truck_model.meta
        K, L1 = meta["K"], meta["L1"]
        for k in range(K):
            succ = {}
            for (i, j) in meta["truck_arcs"]:
                if xv[meta["x"][k, i, j]] > 0.5:
                    succ[i] = j
            route = []; cur = "O"
            while cur != "D" and cur in succ:
                nxt = succ[cur]
                if isinstance(nxt, tuple) and nxt[0] == "cm":
                    route.append(nxt[1])
                cur = nxt
            if route:
                sol.truck_routes.append(route)
            for c in route:
                for l in range(L1):
                    succ_d = {}
                    for (i, j) in meta["drone_arcs"][c]:
                        if xv[meta["y"][k, l, c, i, j]] > 0.5:
                            succ_d[i] = j
                    seq = []; cur = ("cm", c)
                    while cur in succ_d:
                        nxt = succ_d[cur]
                        if isinstance(nxt, int):
                            seq.append(nxt)
                        cur = nxt
                        if isinstance(cur, tuple):
                            break
                    if seq:
                        sol.sorties[c].append(seq)
    if indep_model is not None and indep_res is not None and indep_res.get("x") is not None and indep_model.n_vars:
        zv = indep_res["x"]; meta = indep_model.meta
        for l in range(meta["L0"]):
            succ = {}
            for (i, j) in meta["arcs"]:
                if zv[meta["z"][l, i, j]] > 0.5:
                    succ[i] = j
            seq = []; cur = "O"
            while cur in succ and succ[cur] != "D":
                cur = succ[cur]; seq.append(cur)
            if seq:
                sol.indep_trips.append(seq)
    sol.refresh_all()
    return sol


def solve_instance(inst, backend="highs", time_limit=3600.0, strengthen: Optional[Dict] = None, mip_gap=1e-4,
                   threads=1, verbose=False, relax=False, log_prefix: Optional[str] = None) -> Dict:
    """Solve both sub-models; returns a summary dict and the extracted solution (if any)."""
    out = {}
    tm = build_truck_model(inst, strengthen)
    im = build_indep_model(inst, strengthen)
    res_t = solve_model(tm, backend, time_limit, mip_gap, threads, verbose, relax, (log_prefix + "_truck.log") if log_prefix else None) if tm.n_vars else dict(status="Empty", obj=0.0, bound=0.0, gap=0.0, nodes=0, time=0.0, x=None, n_vars=0, n_int=0, n_rows=0, optimal=True)
    res_i = solve_model(im, backend, time_limit, mip_gap, threads, verbose, relax, (log_prefix + "_indep.log") if log_prefix else None) if im.n_vars else dict(status="Empty", obj=0.0, bound=0.0, gap=0.0, nodes=0, time=0.0, x=None, n_vars=0, n_int=0, n_rows=0, optimal=True)
    out["truck"] = {k: v for k, v in res_t.items() if k != "x"}
    out["indep"] = {k: v for k, v in res_i.items() if k != "x"}
    has_obj = (res_t["obj"] is not None) and (res_i["obj"] is not None)
    out["obj"] = (res_t["obj"] + res_i["obj"]) if has_obj else None
    out["bound"] = (res_t["bound"] or 0.0) + (res_i["bound"] or 0.0)
    out["time"] = res_t["time"] + res_i["time"]
    out["optimal"] = bool(res_t["optimal"] and res_i["optimal"])
    out["n_vars"] = res_t["n_vars"] + res_i["n_vars"]; out["n_int"] = res_t["n_int"] + res_i["n_int"]; out["n_rows"] = res_t["n_rows"] + res_i["n_rows"]
    out["nodes"] = res_t["nodes"] + res_i["nodes"]
    out["gap"] = (out["obj"] - out["bound"]) / max(1e-9, abs(out["obj"])) if (out["obj"] is not None and not relax) else None
    if not relax and has_obj:
        sol = extract_solution(inst, tm, res_t, im, res_i)
        out["solution"] = sol.to_dict()
        out["heuristic_eval"] = sol.cost()
        out["feasible"] = sol.is_feasible()
    return out
