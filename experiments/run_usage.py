"""
Operator-usage statistics of ALNS-P and OSD-ALNS (Section 6.4 of the manuscript): which pairs /
sequences are selected and whether the learned sequence distribution departs from the product of its
marginals (dependence between consecutive pairs).

usage: python experiments/run_usage.py [--seeds 3]      -> results/usage.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *   # noqa

INSTANCES = ["A-1-100", "A-1-R-100", "B-1-R-100", "A-1-RC-100", "A-1-M-200"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--params", default=os.path.join(RES_DIR, "calibrated_params.json"))
    a = ap.parse_args()
    params = json.load(open(a.params)) if os.path.exists(a.params) else None
    path = os.path.join(RES_DIR, "usage.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in INSTANCES:
        inst = load_instance(name)
        for algo in ("P", "OSD"):
            for seed in range(1, a.seeds + 1):
                key = f"{name}|{algo}|{seed}"
                if key in out:
                    continue
                res = run_algorithm(inst, algo, seed, budget_ops=budget_of(name), params=params)
                out[key] = dict(best=res.best_cost, usage=res.usage, weights=res.weights, ops=res.ops)
                print(name, algo, seed, res.best_cost, flush=True)
                with open(path, "w") as f:
                    json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
