"""Summarize the parameter calibration (results/calib.csv) and write the calibrated parameter file."""
import json, os, sys
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
df = pd.read_csv(os.path.join(ROOT, "results", "calib.csv"))
df["best"] = df["best"].astype(float)
g = df.groupby(["instance", "variant"])["best"].agg(["mean", "std", "min", "count"]).reset_index()
# relative deviation from the best mean over variants per instance
best_per_inst = g.groupby("instance")["mean"].min()
g["rel"] = 100 * (g["mean"] - g["instance"].map(best_per_inst)) / g["instance"].map(best_per_inst)
summary = g.groupby("variant").agg(mean_rel=("rel", "mean"), max_rel=("rel", "max"), n_inst=("instance", "nunique"), runs=("count", "sum")).sort_values("mean_rel")
pd.set_option("display.width", 200)
print(summary.round(4))
summary.to_csv(os.path.join(ROOT, "results", "calib_summary.csv"))
# choose: for every factor, the level with the smallest mean_rel (default level has variant 'default')
from mvmd.alns import DEFAULT_PARAMS
chosen = {}
factors = {"T0": ["T0=50.0", "default", "T0=5000.0"], "alpha": ["alpha=0.99", "default", "alpha=0.999"],
           "rho": ["rho=0.1", "default", "rho=0.5"], "lam": ["lam=0.3", "default", "lam=0.8"],
           "q": ["q=0.05-0.2", "default", "q=0.05-0.35", "q=0.2-0.4"],
           "restart_ops": ["restart_ops=0", "restart_ops=250", "default", "restart_ops=1000"]}
levels = {"T0": {"T0=50.0": 50.0, "default": 500.0, "T0=5000.0": 5000.0},
          "alpha": {"alpha=0.99": 0.99, "default": 0.995, "alpha=0.999": 0.999},
          "rho": {"rho=0.1": 0.1, "default": 0.2, "rho=0.5": 0.5},
          "lam": {"lam=0.3": 0.3, "default": 0.5, "lam=0.8": 0.8},
          "q": {"q=0.05-0.2": (0.05, 0.2), "default": (0.1, 0.3), "q=0.05-0.35": (0.05, 0.35), "q=0.2-0.4": (0.2, 0.4)},
          "restart_ops": {"restart_ops=0": 0, "restart_ops=250": 250, "default": 500, "restart_ops=1000": 1000}}
for f, vars_ in factors.items():
    sub = summary.loc[[v for v in vars_ if v in summary.index]]
    bestv = sub["mean_rel"].idxmin()
    # keep the default unless the alternative is better by a margin of 0.02 percentage points
    if bestv != "default" and sub.loc["default", "mean_rel"] - sub.loc[bestv, "mean_rel"] < 0.02:
        bestv = "default"
    chosen[f] = levels[f][bestv]
    print(f"{f}: chosen {bestv} -> {chosen[f]}")
params = {"T0": chosen["T0"], "alpha": chosen["alpha"], "rho": chosen["rho"], "lam": chosen["lam"],
          "q_min": chosen["q"][0], "q_max": chosen["q"][1], "restart_ops": chosen["restart_ops"]}
with open(os.path.join(ROOT, "results", "calibrated_params.json"), "w") as f:
    json.dump(params, f, indent=1)
print("written", params)
