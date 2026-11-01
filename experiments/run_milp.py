"""
Exact experiments (Section 6.2.1 and Section 4.5).

usage: python experiments/run_milp.py main  [--backend highs|gurobi] [--time 1800]
       python experiments/run_milp.py vi    [--backend highs|gurobi] [--time 600]

main : strengthened MILP on the toy and the 15-/30-customer instances         -> results/milp_main.csv
vi   : effect of the strengthening components (toy and 15-customer instances) -> results/vi.csv
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa
from mvmd.milp import solve_instance, ALL_STRENGTH, NO_STRENGTH

FIELDS = ["experiment", "instance", "config", "backend", "time_limit", "obj", "bound", "gap", "optimal", "time", "nodes",
          "n_vars", "n_int", "n_rows", "lp_bound", "truck_obj", "truck_status", "truck_time", "truck_nodes", "indep_obj",
          "indep_status", "indep_time", "indep_nodes", "feasible", "heuristic_eval", "solution"]
KEY = ["experiment", "instance", "config", "backend"]

CONFIGS = {
    "base": dict(NO_STRENGTH),
    "all": dict(ALL_STRENGTH),
    "+prep": dict(NO_STRENGTH, prep=True),
    "+rounding": dict(NO_STRENGTH, rounding=True),
    "+two_cycle": dict(NO_STRENGTH, two_cycle=True),
    "+lifted_mtz": dict(NO_STRENGTH, lifted_mtz=True),
    "+symmetry": dict(NO_STRENGTH, symmetry=True),
    "+disagg": dict(NO_STRENGTH, disagg=True),
    "all-symmetry": dict(ALL_STRENGTH, symmetry=False),
    "all-rounding": dict(ALL_STRENGTH, rounding=False),
}


def run_one(exp, name, cfg, backend, tl, writer, with_lp=True):
    if writer.has((exp, name, cfg, backend)):
        return
    inst = load_instance(name)
    st = CONFIGS[cfg]
    lp = ""
    if with_lp:
        try:
            r_lp = solve_instance(inst, backend, time_limit=min(tl, 600), strengthen=st, relax=True)
            lp = f"{r_lp['obj']:.6f}" if r_lp["obj"] is not None else ""
        except Exception as ex:  # pragma: no cover
            lp = ""
    r = solve_instance(inst, backend, time_limit=tl, strengthen=st, log_prefix=os.path.join(RES_DIR, "milp_logs", f"{exp}_{name}_{cfg}"))
    row = dict(experiment=exp, instance=name, config=cfg, backend=backend, time_limit=tl,
               obj=f"{r['obj']:.6f}" if r["obj"] is not None else "", bound=f"{r['bound']:.6f}",
               gap=f"{r['gap']:.6f}" if r["gap"] is not None else "", optimal=int(r["optimal"]), time=f"{r['time']:.2f}",
               nodes=r["nodes"], n_vars=r["n_vars"], n_int=r["n_int"], n_rows=r["n_rows"], lp_bound=lp,
               truck_obj=r["truck"]["obj"] if r["truck"]["obj"] is not None else "", truck_status=r["truck"]["status"],
               truck_time=f"{r['truck']['time']:.2f}", truck_nodes=r["truck"]["nodes"],
               indep_obj=r["indep"]["obj"] if r["indep"]["obj"] is not None else "", indep_status=r["indep"]["status"],
               indep_time=f"{r['indep']['time']:.2f}", indep_nodes=r["indep"]["nodes"],
               feasible=int(r.get("feasible", False)), heuristic_eval=f"{r['heuristic_eval']:.6f}" if "heuristic_eval" in r else "",
               solution=json.dumps(r.get("solution", {})))
    writer.write(row)
    print(f"{name:9s} {cfg:12s} obj={row['obj']} bound={row['bound']} gap={row['gap']} time={row['time']} nodes={row['nodes']} lp={lp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--backend", default="highs")
    ap.add_argument("--time", type=float, default=1800.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES_DIR, "milp_logs"), exist_ok=True)
    exp = a.experiment
    if exp == "main":
        writer = CsvWriter(os.path.join(RES_DIR, "milp_main.csv"), FIELDS, KEY)
        for name in instance_names(scales=("toy", 15, 30)):
            run_one(exp, name, "all", a.backend, a.time, writer, with_lp=True)
    elif exp == "vi":
        writer = CsvWriter(os.path.join(RES_DIR, "vi.csv"), FIELDS, KEY)
        for name in instance_names(scales=("toy", 15)):
            for cfg in ["base", "all"]:
                run_one(exp, name, cfg, a.backend, a.time, writer)
        for name in instance_names(scales=("toy",)):
            for cfg in ["+prep", "+rounding", "+two_cycle", "+lifted_mtz", "+symmetry", "+disagg"]:
                run_one(exp, name, cfg, a.backend, a.time, writer)
        # (single-component configurations on the 15-customer instances are omitted for time reasons;
        #  run them with: python experiments/run_milp.py vi15)
    elif exp == "lp":
        # LP relaxation bounds of every configuration on the toy and 15-customer instances (cheap)
        fields = ["instance", "config", "backend", "lp_bound", "lp_time", "n_vars", "n_rows"]
        writer = CsvWriter(os.path.join(RES_DIR, "vi_lp.csv"), fields, ["instance", "config", "backend"])
        for name in instance_names(scales=("toy", 15)):
            inst = load_instance(name)
            for cfg in ["base", "+prep", "+rounding", "+two_cycle", "+lifted_mtz", "+symmetry", "+disagg", "all"]:
                if writer.has((name, cfg, a.backend)):
                    continue
                r = solve_instance(inst, a.backend, time_limit=600, strengthen=CONFIGS[cfg], relax=True)
                writer.write(dict(instance=name, config=cfg, backend=a.backend, lp_bound=f"{r['obj']:.6f}" if r["obj"] is not None else "",
                                  lp_time=f"{r['time']:.2f}", n_vars=r["n_vars"], n_rows=r["n_rows"]))
                print(name, cfg, r["obj"], round(r["time"], 2), flush=True)
    elif exp == "vi15":
        writer = CsvWriter(os.path.join(RES, "vi.csv"), FIELDS, KEY)
        for name in instance_names(scales=(15,)):
            for cfg in ["+prep", "+rounding", "+two_cycle", "+lifted_mtz", "+symmetry", "+disagg"]:
                run_one("vi", name, cfg, a.backend, min(a.time, 600.0), writer)
    else:
        raise SystemExit("unknown experiment")


if __name__ == "__main__":
    main()
