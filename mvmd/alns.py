"""
Adaptive large neighbourhood search variants (manuscript Section 5).

The *elementary operation* is one destroy-repair move; the *unit of adaptation* is a destroy-repair
pair u = (destroy operator, repair operator) (optionally extended by a destroy-size class, see
DEFAULT_PARAMS["sizes"]).

mode = "I"    ALNS-I  : conventional ALNS with *independent* roulette wheels for the destroy
                        operator, the repair operator and the size class (Ropke and Pisinger, 2006);
                        acceptance after every elementary operation.
mode = "P"    ALNS-P  : one weight per *unit* (pair-coupled weights); acceptance after every
                        elementary operation.  Identical to OSD-ALNS with Len = 1.
mode = "OSD"  OSD-ALNS: selection over ordered *sequences* sigma = (u_1, ..., u_Len) of units; the
                        units of a sequence are applied consecutively and the acceptance test is
                        applied to the composite move (look-ahead acceptance).  The sequence weights
                        are *hierarchical*:  w(sigma) = v(sigma) * prod_{u in sigma} w(u),
                        where the unit weights w(u) are learned from the outcome of every unit and
                        the sequence factors v(sigma) from the outcome of the composite move
                        (back-off to unit weights for sequences that have not been tried yet).
                        hierarchical=False gives flat sequence weights (one independent weight
                        per sequence); accept_each=True applies the acceptance test after every
                        unit (ablation variants of Section 6).

All variants share the operators, the reward scheme, the smoothing updates
    theta <- lam*theta + (1-lam)*psi ,   w <- (1-rho)*w + rho*theta   (floored at w_min),
the simulated-annealing acceptance, the same cooling schedule *per elementary operation*
(T <- alpha*T after every destroy-repair move) and the same restart rule (the current solution is
reset to the incumbent best after `restart_ops` elementary operations without improvement), so that
equal numbers of elementary operations correspond to identical annealing schedules.

Effort accounting: `ops` counts elementary destroy-repair operations; `trace` records
(ops, cpu_seconds, best_cost) at every improvement of the incumbent best solution.
"""
from __future__ import annotations

import itertools
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .operators import RequestInfo, DESTROY_OPERATORS, REPAIR_OPERATORS
from .solution import Solution

DEFAULT_PARAMS = dict(
    T0=500.0, alpha=0.995, rho=0.20, lam=0.5, omegas=(8.0, 4.0, 2.0, 0.0),
    q_min=0.10, q_mid=0.20, q_max=0.30, w_min=0.05, v_min=0.05, p_shaw=6.0, p_cost=3.0, phi=(1.0, 0.5, 1.0),
    destroy=("random", "shaw", "cost", "segment"), repair=("greedy", "noise", "regret2", "regret3"),
    sizes=("m",), restart_ops=500, variable_len=True,
)
# sizes=("m",): the number of removed requests is drawn uniformly from [q_min R, q_max R] for every
# unit (equal expected effort per elementary operation); sizes=("s","l") makes the size class part of
# the adaptive unit (small: [q_min, q_mid), large: [q_mid, q_max]) -- studied in the ablation only.


@dataclass
class RunResult:
    best_cost: float
    best: Dict
    ops: int
    iterations: int
    time: float
    trace: List[Tuple[int, float, float]]     # (ops, seconds, best_cost)
    init_cost: float
    accepted: int = 0
    improved: int = 0
    weights: Dict = field(default_factory=dict)
    usage: Dict = field(default_factory=dict)
    op_time: float = 0.0   # mean seconds per elementary operation


def unit_name(u) -> str:
    return "%s+%s/%s" % u


def seq_name(seq) -> str:
    return "->".join(unit_name(u) for u in seq)


class ALNS:
    def __init__(self, inst, init_sol: Solution, mode: str = "OSD", Len: int = 2, seed: int = 0,
                 params: Optional[Dict] = None, accept_each: bool = False, hierarchical: bool = True,
                 prefix_best: bool = False):
        self.inst = inst
        self.mode = mode
        self.Len = 1 if mode in ("I", "P") else Len
        self.accept_each = accept_each if mode == "OSD" else True
        self.hierarchical = hierarchical if mode == "OSD" else False
        # prefix_best=True (ablation): the candidate submitted to the acceptance test is the best of the
        # solutions obtained after each prefix (u_1), (u_1,u_2), ...; default: the final solution only
        # (pure look-ahead acceptance, the intermediate solutions only update the incumbent best)
        self.prefix_best = prefix_best if mode == "OSD" else False
        self.p = dict(DEFAULT_PARAMS); self.p.update(params or {})
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.info = RequestInfo(inst)
        self.init = init_sol
        self.destroy_names = list(self.p["destroy"]); self.repair_names = list(self.p["repair"])
        self.size_names = list(self.p["sizes"])
        self.units = [(d, r, s) for d in self.destroy_names for r in self.repair_names for s in self.size_names]
        self.uidx = {u: i for i, u in enumerate(self.units)}
        nU = len(self.units)
        # unit-level weights (ALNS-P and the back-off level of OSD-ALNS)
        self.w_u = np.ones(nU); self.th_u = np.zeros(nU)
        if mode == "I":
            self.w_d = {d: 1.0 for d in self.destroy_names}; self.th_d = {d: 0.0 for d in self.destroy_names}
            self.w_r = {r: 1.0 for r in self.repair_names}; self.th_r = {r: 0.0 for r in self.repair_names}
            self.w_s = {s: 1.0 for s in self.size_names}; self.th_s = {s: 0.0 for s in self.size_names}
            self.seqs = None
        else:
            # variable_len=True: the sequence space contains every ordered sequence of 1..Len units and a
            # top-level roulette over the lengths is learned as well (ALNS-P is the special case Len = 1)
            self.variable_len = bool(self.p.get("variable_len", True)) and mode == "OSD" and self.Len > 1
            lens = range(1, self.Len + 1) if self.variable_len else [self.Len]
            self.seqs_by_len = {L: list(itertools.product(range(nU), repeat=L)) for L in lens}
            self.seq_units_by_len = {L: np.array(self.seqs_by_len[L], dtype=np.int64) for L in lens}
            self.v = {L: np.ones(len(self.seqs_by_len[L])) for L in lens}; self.th_v = {L: np.zeros(len(self.seqs_by_len[L])) for L in lens}
            self.w_seq = {L: np.ones(len(self.seqs_by_len[L])) for L in lens}; self.th_seq = {L: np.zeros(len(self.seqs_by_len[L])) for L in lens}
            self.w_len = {L: 1.0 for L in lens}; self.th_len = {L: 0.0 for L in lens}
            self.lens = list(lens)
        self.usage: Dict = {}

    # ---- selection ------------------------------------------------------------
    def _roulette(self, weights: Dict):
        keys = list(weights); tot = sum(weights[k] for k in keys)
        x = self.rng.random() * tot; acc = 0.0
        for k in keys:
            acc += weights[k]
            if x <= acc:
                return k
        return keys[-1]

    def _seq_weights(self, L: int) -> np.ndarray:
        if self.mode == "P":
            return self.w_u
        if self.hierarchical:
            return self.v[L] * np.prod(self.w_u[self.seq_units_by_len[L]], axis=1)
        return self.w_seq[L]

    def select(self):
        """returns ((length, sequence index) or None, tuple of units)"""
        if self.mode == "I":
            return None, ((self._roulette(self.w_d), self._roulette(self.w_r), self._roulette(self.w_s)),)
        L = self._roulette(self.w_len) if len(self.lens) > 1 else self.lens[0]
        w = self._seq_weights(L)
        c = np.cumsum(w)
        i = int(np.searchsorted(c, self.rng.random() * c[-1], side="right"))
        i = min(i, len(c) - 1)
        return (L, i), tuple(self.units[j] for j in self.seqs_by_len[L][i])

    # ---- weight update ----------------------------------------------------------
    def _smooth(self, th, w, idx, psi, floor):
        lam, rho = self.p["lam"], self.p["rho"]
        th[idx] = lam * th[idx] + (1 - lam) * psi
        w[idx] = max(floor, (1 - rho) * w[idx] + rho * th[idx])

    def update(self, seq_idx, units, psi_units: List[float], psi_seq: float):
        wmin, vmin = self.p["w_min"], self.p["v_min"]
        if self.mode == "I":
            d, r, s = units[0]; psi = psi_units[0]
            self._smooth(self.th_d, self.w_d, d, psi, wmin)
            self._smooth(self.th_r, self.w_r, r, psi, wmin)
            self._smooth(self.th_s, self.w_s, s, psi, wmin)
            return
        if self.mode == "P":
            self._smooth(self.th_u, self.w_u, self.uidx[units[0]], psi_units[0], wmin)
            return
        L, i = seq_idx
        if len(self.lens) > 1:
            self._smooth(self.th_len, self.w_len, L, psi_seq, wmin)
        if self.hierarchical:
            for u, psi in zip(units, psi_units):
                self._smooth(self.th_u, self.w_u, self.uidx[u], psi, wmin)
            self._smooth(self.th_v[L], self.v[L], i, psi_seq, vmin)
        else:
            self._smooth(self.th_seq[L], self.w_seq[L], i, psi_seq, wmin)

    # ---- main loop --------------------------------------------------------------
    def run(self, budget_ops: Optional[int] = None, time_limit: Optional[float] = None,
            max_no_improve: Optional[int] = None, max_iters: Optional[int] = None, verbose: bool = False) -> RunResult:
        inst = self.inst; p = self.p; rng = self.rng
        cur = self.init.copy(); cur.compact()
        best = cur.copy(); best_cost = best.cost(); cur_cost = best_cost
        init_cost = best_cost
        T = p["T0"]; alpha = p["alpha"]; om = p["omegas"]
        R = inst.n + inst.m
        qlo = max(1, int(round(p["q_min"] * R))); qmid = max(qlo + 1, int(round(p["q_mid"] * R)))
        qhi = max(qmid, int(round(p["q_max"] * R)))
        if p.get("q_cap"):   # optional absolute cap on the destroy size (Ropke and Pisinger, 2006)
            cap = int(p["q_cap"]); qhi = min(qhi, cap); qlo = min(qlo, max(1, cap // 2)); qmid = min(qmid, max(qlo + 1, (qlo + qhi) // 2))
        qrange = {"s": (qlo, max(qlo, qmid - 1)), "l": (qmid, qhi), "m": (qlo, qhi)}
        ops = 0; iters = 0; no_imp = 0; accepted = 0; improved = 0; last_best_ops = 0
        t0 = time.perf_counter()
        trace = [(0, 0.0, best_cost)]
        destroy = {d: DESTROY_OPERATORS[d] for d in self.destroy_names}
        repair = {r: REPAIR_OPERATORS[r] for r in self.repair_names}
        pp = {"shaw": p["p_shaw"], "cost": p["p_cost"], "random": 0.0, "segment": 0.0}
        restart_ops = p.get("restart_ops")

        def stop():
            if budget_ops is not None and ops >= budget_ops:
                return True
            if time_limit is not None and time.perf_counter() - t0 >= time_limit:
                return True
            if max_no_improve is not None and no_imp >= max_no_improve:
                return True
            if max_iters is not None and iters >= max_iters:
                return True
            return False

        while not stop():
            seq_idx, units = self.select()
            key = seq_name(units)
            self.usage[key] = self.usage.get(key, 0) + 1
            s_tmp = cur.copy()
            psi_units: List[float] = []
            composite_ok = True
            found_best = False
            pre_cost = cur_cost
            cand = None; cand_cost = math.inf
            for (d, r, sz) in units:
                lo, hi = qrange[sz]
                q = rng.randint(lo, hi)
                best_before = best_cost
                destroy[d](s_tmp, q, rng, self.info, pp[d], p["phi"])
                ok = repair[r](s_tmp, rng)
                ops += 1
                T *= alpha
                if not ok:
                    composite_ok = False
                    psi_units.append(om[3])
                    break
                s_tmp.compact()
                new_cost = s_tmp.cost()
                if self.accept_each:
                    delta = new_cost - cur_cost
                    acc = delta <= 1e-9 or rng.random() < math.exp(-delta / max(T, 1e-12))
                    if new_cost < best_cost - 1e-9:
                        psi = om[0]; best = s_tmp.copy(); best_cost = new_cost; found_best = True
                        trace.append((ops, time.perf_counter() - t0, best_cost))
                    elif acc and delta < -1e-9:
                        psi = om[1]
                    elif acc:
                        psi = om[2]
                    else:
                        psi = om[3]
                    psi_units.append(psi)
                    if acc:
                        cur = s_tmp; cur_cost = new_cost; accepted += 1
                        if delta < -1e-9:
                            improved += 1
                    s_tmp = cur.copy()
                else:
                    # look-ahead acceptance: the incumbent best is tracked over every evaluated
                    # solution (also the intermediate solutions of a sequence); the acceptance test
                    # is applied only to the complete composite move
                    if new_cost < best_before - 1e-9:
                        best = s_tmp.copy(); best_cost = new_cost; found_best = True
                        trace.append((ops, time.perf_counter() - t0, best_cost))
                        psi_units.append(om[0])
                    elif new_cost < pre_cost - 1e-9:
                        psi_units.append(om[1])
                    else:
                        psi_units.append(None)   # decided by the composite acceptance below
                    pre_cost = new_cost
                    if self.prefix_best and new_cost < cand_cost - 1e-9:
                        cand = s_tmp.copy(); cand_cost = new_cost
                if stop():
                    break
            if self.accept_each:
                psi_seq = max(psi_units) if psi_units else om[3]
            else:
                if self.prefix_best and cand is not None:
                    composite_ok = True
                    s_tmp = cand
                if composite_ok:
                    new_cost = s_tmp.cost()
                    delta = new_cost - cur_cost
                    acc = delta <= 1e-9 or rng.random() < math.exp(-delta / max(T, 1e-12))
                    if found_best:
                        psi_seq = om[0]
                    elif acc and delta < -1e-9:
                        psi_seq = om[1]
                    elif acc:
                        psi_seq = om[2]
                    else:
                        psi_seq = om[3]
                    if acc:
                        cur = s_tmp; cur_cost = new_cost; accepted += 1
                        if delta < -1e-9:
                            improved += 1
                    fill = om[2] if acc else om[3]
                else:
                    psi_seq = om[3]; fill = om[3]
                psi_units = [fill if x is None else x for x in psi_units]
            while len(psi_units) < len(units):
                psi_units.append(om[3])
            self.update(seq_idx, units, psi_units, psi_seq)
            iters += 1
            if found_best:
                no_imp = 0; last_best_ops = ops
            else:
                no_imp += 1
            if restart_ops and ops - last_best_ops >= restart_ops:
                cur = best.copy(); cur_cost = best_cost; last_best_ops = ops
            if verbose and iters % 50 == 0:
                print(f"iter {iters} ops {ops} cur {cur_cost:.2f} best {best_cost:.2f} T {T:.3f}")
        elapsed = time.perf_counter() - t0
        trace.append((ops, elapsed, best_cost))
        if self.mode == "I":
            weights = {"destroy": dict(self.w_d), "repair": dict(self.w_r), "size": dict(self.w_s)}
        elif self.mode == "P":
            weights = {unit_name(u): float(self.w_u[i]) for i, u in enumerate(self.units)}
        else:
            weights = {"units": {unit_name(u): float(self.w_u[i]) for i, u in enumerate(self.units)},
                       "lengths": {int(L): float(w) for L, w in self.w_len.items()}, "sequences": {}}
            for L in self.lens:
                wv = self._seq_weights(L)
                weights["sequences"].update({seq_name(tuple(self.units[j] for j in sq)): float(wv[i]) for i, sq in enumerate(self.seqs_by_len[L])})
        return RunResult(best_cost=best_cost, best=best.to_dict(), ops=ops, iterations=iters, time=elapsed, trace=trace,
                         init_cost=init_cost, accepted=accepted, improved=improved, weights=weights, usage=dict(self.usage),
                         op_time=elapsed / max(ops, 1))
