"""
Heuristic experiments of the manuscript (Section 6).

usage: python experiments/run_heuristics.py <experiment> [--workers 2] [--params results/calibrated_params.json]

experiments
  calib     parameter calibration of OSD-ALNS (one factor at a time)           -> results/calib.csv
  protoE    protocol E: equal number of elementary operations, I / P / OSD      -> results/protoE.csv (+ traces)
  protoT    protocol T: equal CPU time (= mean OSD time of protoE), VNS / GA   -> results/protoT.csv (+ traces)
  len       sequence-length study (fixed Len = 1, 2, 3; variable <= 2, <= 3)   -> results/len.csv
  ablation  component ablation of OSD-ALNS                                      -> results/ablation.csv
  sens      sensitivity analyses (gamma, Ld, Qd) with OSD-ALNS                  -> results/sens.csv

Seeds: 10 per instance on the toy and C101-based instances, 5 on the R101/RC101-based and on the
200/300-customer instances (common.SEEDS); calibration, len and ablation use 5 seeds.
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa
from mvmd.alns import DEFAULT_PARAMS

FIELDS = ["experiment", "instance", "algo", "variant", "seed", "best", "init", "ops", "iters", "time", "op_time", "accepted",
          "improved", "budget_ops", "time_limit", "params", "n", "geometry", "scenario"]
KEY = ["experiment", "instance", "algo", "variant", "seed"]
ABL_INSTANCES = ["A-1-R-100", "B-1-R-100", "A-1-RC-100", "C-1-RC-100", "A-1-M-200"]
ROB_INSTANCES = ["A-1-R-100", "B-1-R-100", "A-1-RC-100", "C-1-RC-100", "A-1-M-200", "B-1-M-200"]


def scenario_of(name):
    return name.split("-", 1)[1] if name.startswith("toy") else "-".join(name.split("-")[:2])


def _job(args):
    (exp, name, algo, variant, seed, budget_ops, time_limit, params, init_kind, Len, accept_each, save_tr) = args
    inst = load_instance(name)
    res = run_algorithm(inst, algo, seed, budget_ops=budget_ops, time_limit=time_limit, params=params,
                        init_kind=init_kind, Len=Len, accept_each=accept_each)
    sol = Solution.from_dict(inst, res.best)
    assert sol.is_feasible(), f"infeasible result {name} {algo} {seed}"
    assert abs(sol.cost() - res.best_cost) < 1e-6
    row = dict(experiment=exp, instance=name, algo=algo, variant=variant, seed=seed, best=f"{res.best_cost:.6f}",
               init=f"{res.init_cost:.6f}", ops=res.ops, iters=res.iterations, time=f"{res.time:.3f}", op_time=f"{res.op_time:.6f}",
               accepted=res.accepted, improved=res.improved, budget_ops=budget_ops or "", time_limit=f"{time_limit:.3f}" if time_limit else "",
               params=json.dumps(params or {}), n=scale_of(name), geometry=geometry_of(name), scenario=scenario_of(name))
    tr = res.trace if save_tr else None
    return row, tr, f"{exp}_{name}_{algo}_{variant}_{seed}"


def run_jobs(jobs, writer, workers):
    todo = [j for j in jobs if not writer.has((j[0], j[1], j[2], j[3], j[4]))]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time(); done = 0
    if workers > 1:
        with Pool(workers) as pool:
            for row, tr, tag in pool.imap_unordered(_job, todo):
                writer.write(row); done += 1
                if tr is not None:
                    save_trace(tag, tr)
                if done % 10 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    else:
        for j in todo:
            row, tr, tag = _job(j); writer.write(row); done += 1
            if tr is not None:
                save_trace(tag, tr)
            print(f"  {done}/{len(todo)} {row['instance']} {row['algo']} {row['variant']} best={row['best']} t={row['time']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--params", type=str, default="", help="json file with calibrated parameters")
    a = ap.parse_args()
    params = json.load(open(a.params)) if a.params else None
    exp = a.experiment
    writer = CsvWriter(os.path.join(RES_DIR, f"{exp}.csv"), FIELDS, KEY)
    jobs = []
    if exp == "calib":
        names = ["A-1-50", "C-1-50", "A-1-75", "B-1-75", "A-1-R-50", "B-2-RC-75"]
        base = dict(DEFAULT_PARAMS)
        grid = {"T0": [50.0, 500.0, 5000.0], "alpha": [0.99, 0.995, 0.999], "rho": [0.1, 0.2, 0.5], "lam": [0.3, 0.5, 0.8],
                "q": [(0.05, 0.20), (0.10, 0.30), (0.05, 0.35), (0.20, 0.40)], "restart_ops": [0, 250, 500, 1000]}
        variants = {"default": {}}
        for k, vals in grid.items():
            for v in vals:
                if k == "q":
                    if v == (base["q_min"], base["q_max"]):
                        continue
                    variants[f"q={v[0]}-{v[1]}"] = dict(q_min=v[0], q_max=v[1])
                else:
                    if v == base[k]:
                        continue
                    variants[f"{k}={v}"] = {k: v}
        for name in names:
            for var, pr in variants.items():
                for s in range(1, 6):
                    jobs.append((exp, name, "OSD", var, s, budget_of(name), None, pr, "twcc", 2, False, False))
    elif exp == "protoE":
        for name in all_instance_names():
            for algo in ["I", "P", "OSD"]:
                for s in range(1, seeds_of(name) + 1):
                    jobs.append((exp, name, algo, "-", s, budget_of(name), None, params, "twcc", 2, False, True))
    elif exp == "protoT":
        import csv as _csv
        tl = {}
        with open(os.path.join(RES_DIR, "protoE.csv")) as f:
            rows = [r for r in _csv.DictReader(f) if r["algo"] == "OSD"]
        for name in all_instance_names():
            ts = [float(r["time"]) for r in rows if r["instance"] == name]
            if ts:
                tl[name] = max(0.5, sum(ts) / len(ts))
        for name in all_instance_names():
            if name not in tl or scale_of(name) == 300:
                continue
            for algo in ["VNS", "GA"]:
                for s in range(1, (4 if geometry_of(name) == "M" else 6)):
                    jobs.append((exp, name, algo, "-", s, None, tl[name], None, "twcc", 2, False, True))
    elif exp == "len":
        pr = dict(params or {})
        variants = {"Len=1": (1, dict(pr, variable_len=False)), "Len=2": (2, dict(pr, variable_len=False)),
                    "Len<=2": (2, dict(pr, variable_len=True)), "Len<=3": (3, dict(pr, variable_len=True))}
        names = ["A-1-75", "B-1-75", "A-1-R-75", "B-1-R-75", "A-1-RC-75", "B-1-RC-75"]
        for name in names:
            for var, (Len, p2) in variants.items():
                for s in range(1, 6):
                    if var == "Len<=2":
                        continue   # = the protocol-E runs of OSD-ALNS (same parameters and seeds)
                    jobs.append((exp, name, "OSD", var, s, budget_of(name), None, p2, "twcc", Len, False, False))
    elif exp == "ablation":
        # the full algorithm is the protocol-E run of OSD-ALNS (same parameters and seeds 1-5)
        pr = dict(params or {})
        variants = {
            "flat-weights": dict(params=dict(pr, hierarchical=False), init="twcc", accept_each=False),
            "accept-each": dict(params=pr, init="twcc", accept_each=True),
            "fixed-Len=2": dict(params=dict(pr, variable_len=False), init="twcc", accept_each=False),
            "no-adaptation": dict(params=dict(pr, rho=0.0), init="twcc", accept_each=False),
            "no-restart": dict(params=dict(pr, restart_ops=0), init="twcc", accept_each=False),
            "random-init": dict(params=pr, init="random", accept_each=False),
            "small-pool": dict(params=dict(pr, destroy=("shaw", "cost"), repair=("greedy", "regret2")), init="twcc", accept_each=False),
        }
        for name in ABL_INSTANCES:
            for var, cfg in variants.items():
                for s in range(1, 6):
                    jobs.append((exp, name, "OSD", var, s, budget_of(name), None, cfg["params"], cfg["init"], 2, cfg["accept_each"], False))
    elif exp == "robust":
        # robustness of the three layers to the acceptance criterion / cooling schedule
        pr = dict(params or {})
        # the calibrated setting (fast cooling, alpha = 0.99) is the protocol-E run itself; the robustness
        # study brackets it by a slower schedule and by a pure descent
        settings = {"slow": dict(pr, alpha=0.995), "descent": dict(pr, T0=1e-6)}
        for name in ROB_INSTANCES:
            for st, p2 in settings.items():
                for algo in ["I", "P", "OSD"]:
                    for s in range(1, 6):
                        jobs.append((exp, name, algo, st, s, budget_of(name), None, p2, "twcc", 2, False, True))
    elif exp == "qcap":
        # destroy size capped at 60 requests per elementary operation on the 300-customer instances
        pr = dict(params or {}, q_cap=60)
        for name in [n for n in LARGE if scale_of(n) == 300]:
            for algo in ["I", "P", "OSD"]:
                for s in range(1, 4):
                    jobs.append((exp, name, algo, "qcap=60", s, budget_of(name), None, pr, "twcc", 2, False, True))
    elif exp == "sens":
        # instances generated on the fly with modified parameters (A-1 baseline)
        from mvmd.instance import make_solomon
        sweeps = {"gamma": [round(0.05 * i, 2) for i in range(0, 11)], "Ld": [1000.0 * v for v in range(5, 16)],
                  "Qd": list(range(4, 14))}   # Qd = 3 is infeasible (customer demands up to 4 units)
        for n in (15, 30):
            for par, vals in sweeps.items():
                for v in vals:
                    name = f"sens-{par}-{v}-{n}"
                    path = os.path.join(INST_DIR, f"{name}.json")
                    if not os.path.exists(path):
                        inst = make_solomon(n, "A-1", extra={par: v}, name=name); inst.save(path)
                    for s in range(1, 6):
                        jobs.append((exp, name, "OSD", f"{par}={v}", s, budget_of(f"x-{n}"), None, params, "twcc", 2, False, False))
    else:
        raise SystemExit("unknown experiment")
    run_jobs(jobs, writer, a.workers)


if __name__ == "__main__":
    main()
