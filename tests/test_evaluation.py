"""
Tests of the exact evaluation kernels against an LP (scipy) formulation of the optimal waiting
policy, and of the insertion kernels against full re-evaluation.
Run:  python -m pytest tests/ -q      or     python tests/test_evaluation.py
"""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.optimize import linprog

from mvmd.instance import make_solomon, make_toy
from mvmd.solution import (Solution, truck_route_cost, trip_cost, eval_insert_customer_truck,
                           eval_insert_trip, eval_insert_cluster_truck, truck_route_schedule)


def random_solution(inst, rng):
    """Random feasible-structure solution (capacity/range may be violated; evaluation only)."""
    sol = Solution(inst)
    # sorties: random split of each cluster's members
    for c in range(inst.m):
        mem = list(inst.cluster_members[c]); rng.shuffle(mem)
        k = rng.randint(1, min(inst.nL1, len(mem)))
        parts = [[] for _ in range(k)]
        for i, s in enumerate(mem):
            parts[i % k].append(s)
        sol.sorties[c] = parts
    cl = list(range(inst.m)); rng.shuffle(cl)
    k = max(1, min(inst.n_trucks, 2))
    sol.truck_routes = [[] for _ in range(k)]
    for i, c in enumerate(cl):
        sol.truck_routes[i % k].append(c)
    s0 = list(inst.S0); rng.shuffle(s0)
    k = max(1, len(s0) // 3)
    sol.indep_trips = [[] for _ in range(k)]
    for i, s in enumerate(s0):
        sol.indep_trips[i % k].append(s)
    sol.compact()
    return sol


def lp_truck_penalty(inst, sol, k):
    """Optimal waiting policy of truck k by LP: variables S_j (cluster shifts), e_s, l_s."""
    p = inst.params
    route = sol.truck_routes[k]
    taus = []   # (cluster position j, customer, nominal time)
    cur = 0.0; prev = 0
    for j, c in enumerate(route):
        cn = inst.cluster_node(c)
        arr = cur + inst.t_truck[prev, cn]
        W = 0.0
        for sortie in sol.sorties[c]:
            tau = arr; pp = cn
            for s in sortie:
                tau += inst.t_drone[pp, s] + p["service_time"]; taus.append((j, s, tau)); pp = s
            tau += inst.t_drone[pp, cn]
            W = max(W, tau - arr)
        cur = arr + W; prev = cn
    m = len(route); nc = len(taus)
    # variables: S_0..S_{m-1}, e_0..e_{nc-1}, l_0..l_{nc-1}
    nv = m + 2 * nc
    cobj = np.zeros(nv); cobj[m:m + nc] = p["cA"]; cobj[m + nc:] = p["cB"]
    A = []; b = []
    for i, (j, s, tau) in enumerate(taus):
        row = np.zeros(nv); row[j] = 1.0; row[m + i] = 1.0; A.append(-row); b.append(-(inst.tw_a[s] - tau))     # S + e >= a - tau
        row = np.zeros(nv); row[j] = -1.0; row[m + nc + i] = 1.0; A.append(-row); b.append(-(tau - inst.tw_b[s]))  # -S + l >= tau - b
    for j in range(m - 1):
        row = np.zeros(nv); row[j] = 1.0; row[j + 1] = -1.0; A.append(row); b.append(0.0)   # S_j <= S_{j+1}
    res = linprog(cobj, A_ub=np.array(A) if A else None, b_ub=np.array(b) if b else None, bounds=[(0, None)] * nv, method="highs")
    return res.fun


def test_pav_vs_lp():
    rng = random.Random(1)
    for inst in [make_toy("A-1"), make_toy("B-2"), make_solomon(30, "A-1"), make_solomon(50, "C-1"), make_solomon(75, "B-1")]:
        for rep in range(6):
            sol = random_solution(inst, rng)
            for k in range(len(sol.truck_routes)):
                arrs = sol.truck_arrays(k)
                res = truck_route_cost(*arrs, *sol._kernel_args())
                lp = lp_truck_penalty(inst, sol, k)
                assert abs(res[3] - lp) <= 1e-6 * max(1.0, abs(lp)) + 1e-6, (inst.name, k, res[3], lp)
                # schedule consistency
                pen, times, arr_t, dep_t, shifts = truck_route_schedule(*arrs, inst.dist, inst.t_truck, inst.t_drone,
                                                                         inst.tw_a, inst.tw_b, inst.params["service_time"],
                                                                         inst.params["cA"], inst.params["cB"])
                assert abs(pen - res[3]) < 1e-6
                assert np.all(np.diff(shifts) >= -1e-9)
    print("PAV == LP on all random routes")


def test_insert_kernels():
    rng = random.Random(7)
    inst = make_solomon(50, "A-1")
    for rep in range(5):
        sol = random_solution(inst, rng)
        p = inst.params
        # customer insertion: remove a random customer and compare kernel delta with re-evaluation
        for k in range(len(sol.truck_routes)):
            for j, c in enumerate(sol.truck_routes[k]):
                if not sol.sorties[c] or not sol.sorties[c][0]:
                    continue
                s = sol.sorties[c][0][0]
                sol.sorties[c][0].remove(s)
                sol.refresh_truck(k)
                arrs = sol.truck_arrays(k)
                base = sol._truck_cost[k]
                deltas, positions = eval_insert_customer_truck(*arrs, j, s, base, inst.Qd, inst.Ld, inst.nL1, inst.demand,
                                                               *sol._kernel_args())
                for r in range(len(sol.sorties[c])):
                    if deltas[r] < 1e17:
                        sol2 = sol.copy(); sol2.sorties[c][r].insert(positions[r], s); sol2.refresh_truck(k)
                        assert abs(sol2._truck_cost[k] - base - deltas[r]) < 1e-6
                if deltas[-1] < 1e17:
                    sol2 = sol.copy(); sol2.sorties[c].append([s]); sol2.refresh_truck(k)
                    assert abs(sol2._truck_cost[k] - base - deltas[-1]) < 1e-6, (sol2._truck_cost[k] - base, deltas[-1])
                sol.sorties[c][0].insert(0, s); sol.refresh_truck(k)
        # cluster insertion
        for k in range(len(sol.truck_routes)):
            if len(sol.truck_routes[k]) < 2:
                continue
            c = sol.truck_routes[k].pop(0); sol.refresh_truck(k)
            arrs = sol.truck_arrays(k); base = sol._truck_cost[k]
            csp, cc = sol.cluster_arrays(c)
            d, pos = eval_insert_cluster_truck(*arrs, inst.cluster_node(c), csp, cc, inst.node_demand, base, inst.Qt, inst.Lt,
                                               inst.demand, *sol._kernel_args())
            if d < 1e17:
                sol2 = sol.copy(); sol2.truck_routes[k].insert(pos, c); sol2.refresh_truck(k)
                assert abs(sol2._truck_cost[k] - base - d) < 1e-6
                # best over all positions
                best = min(sol.copy_with_cluster(k, c, q)._truck_cost[k] for q in range(len(sol.truck_routes[k]) + 1)) if hasattr(sol, "copy_with_cluster") else None
            sol.truck_routes[k].insert(0, c); sol.refresh_truck(k)
        # trip insertion
        for l in range(len(sol.indep_trips)):
            if len(sol.indep_trips[l]) < 2:
                continue
            s = sol.indep_trips[l].pop(); sol.refresh_trip(l)
            seq = np.array(sol.indep_trips[l], dtype=np.int64)
            d, pos = eval_insert_trip(seq, s, sol._trip_cost[l], inst.Qd, inst.Ld, inst.demand, *sol._trip_args())
            if d < 1e17:
                t2 = list(sol.indep_trips[l]); t2.insert(pos, s)
                c2 = trip_cost(np.array(t2, dtype=np.int64), *sol._trip_args())[0]
                assert abs(c2 - sol._trip_cost[l] - d) < 1e-6
            sol.indep_trips[l].append(s); sol.refresh_trip(l)
    print("insertion kernels consistent with re-evaluation")


if __name__ == "__main__":
    test_pav_vs_lp()
    test_insert_kernels()
    print("all tests passed")
