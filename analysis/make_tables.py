"""
Build every table, figure and numeric macro of the manuscript from results/*.csv (study v2).

Outputs (paper/):  tab_instances.tex, tab_vi.tex, tab_milp.tex, tab_calib.tex, tab_protoE_summary.tex,
                   tab_protoE.tex (C101 family, appendix), tab_protoE_rc.tex (R101/RC101/large, appendix),
                   tab_tests.tex, tab_efficiency.tex, tab_len.tex, tab_ablation.tex, tab_protoT.tex (appendix),
                   tab_protoT_summary.tex, tab_usage.tex, figures/fig_convergence.pdf, figures/fig_sensitivity.pdf,
                   numbers.tex, and results/summary_*.csv
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
RES = os.path.join(ROOT, "results"); PAPER = os.path.join(ROOT, "paper"); FIG = os.path.join(PAPER, "figures")
os.makedirs(FIG, exist_ok=True)
from mvmd.instance import Instance, SCENARIOS
from common import instance_names, all_instance_names, scale_of, geometry_of, BUDGET, LARGE, seeds_of  # noqa

SCALES = ["toy", 15, 30, 50, 75, 100]
FAMILY_LABEL = {"C": "C101", "R": "R101", "RC": "RC101", "M": "large", "toy": "toy"}
NUMBERS = {}
ALGOS = ["I", "P", "OSD"]; NAMES = {"I": "ALNS-I", "P": "ALNS-P", "OSD": "OSD-ALNS"}
TARGET_TOL = 0.0005   # ops-to-target: within 0.05 % of the best known solution


def num(name, value, fmt="{:.2f}"):
    NUMBERS[name] = fmt.format(value) if not isinstance(value, str) else value


def inst_name(sc, n):
    return f"toy-{sc}" if n == "toy" else f"{sc}-{n}"


def group_of(name):
    """(family, scale) used for grouping"""
    return (geometry_of(name), scale_of(name))


GROUPS = [("toy", "toy")] + [("C", n) for n in (15, 30, 50, 75, 100)] + [("R", n) for n in (50, 75, 100)] + [("RC", n) for n in (50, 75, 100)] + [("M", 200), ("M", 300)]


def group_label(g):
    fam, n = g
    if fam == "toy":
        return "toy"
    if fam == "M":
        return f"large, {n} customers"
    return f"{FAMILY_LABEL[fam]}, {n} customers"


def load(name):
    p = os.path.join(RES, f"{name}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    for c in ("best", "init", "time", "op_time", "obj", "bound", "gap", "lp_bound", "heuristic_eval"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if name == "protoE":
        # use exactly seeds 1..seeds_of(instance) (runs with additional seeds from an earlier plan are ignored)
        df = df[df["seed"] <= df["instance"].map(seeds_of)].copy()
    return df


WIDE = {"tab_protoE.tex", "tab_protoE_rc.tex", "tab_protoT.tex", "tab_tests.tex", "tab_ablation.tex", "tab_usage.tex", "tab_robust.tex",
        "tab_instances.tex", "tab_milp.tex", "tab_protoE_summary.tex", "tab_efficiency.tex", "tab_protoT_summary.tex"}


def write(name, text):
    if name in WIDE and "\\begin{tabular}" in text:
        # scale wide tables to the text width (no-op for tables that already fit)
        text = text.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}", 1)
        text = text.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
    with open(os.path.join(PAPER, name), "w") as f:
        f.write(text)


def fmt(v, d=2):
    return "--" if v is None or (isinstance(v, float) and (math.isnan(v))) else f"{v:,.{d}f}".replace(",", "\\,")


def ordered_instances(names):
    order = {n: i for i, n in enumerate(all_instance_names())}
    return sorted(names, key=lambda x: order.get(x, 10 ** 6))


# ----------------------------------------------------------------------------
# Table: instances
# ----------------------------------------------------------------------------
def table_instances():
    rows = []
    for name in all_instance_names():
        p = os.path.join(ROOT, "data", "instances", name + ".json")
        if not os.path.exists(p):
            continue
        inst = Instance.load(p)
        rows.append((inst.name, inst.n, len(inst.S0), len(inst.S1), inst.m, inst.n_trucks, inst.n_indep_drones, int(inst.demand.sum())))
    half = (len(rows) + 1) // 2
    lines = [r"\begin{table}[p]\centering\scriptsize", r"\caption{Benchmark instances after pre-processing: customers ($n$), independent-drone customers ($|S_0|$), truck-supported customers ($|S_1|$), clusters ($|C|$), available trucks ($|K|$), available independent trips ($|L_0|$) and total demand. Names: scenario (A-1 to C-2), geometry (none = C101, R = R101, RC = RC101, M = union of the C101, R101 and, for 300 customers, RC101 customer sets) and number of customers}\label{tab:instances}",
             r"\begin{tabular}{lrrrrrrr@{\hspace{1.2em}}lrrrrrrr}", r"\toprule",
             r"Instance & $n$ & $|S_0|$ & $|S_1|$ & $|C|$ & $|K|$ & $|L_0|$ & $\sum d_s$ & Instance & $n$ & $|S_0|$ & $|S_1|$ & $|C|$ & $|K|$ & $|L_0|$ & $\sum d_s$ \\", r"\midrule"]
    for i in range(half):
        a = rows[i]; b = rows[i + half] if i + half < len(rows) else None
        la = " & ".join(str(v) for v in a)
        lb = " & ".join(str(v) for v in b) if b else " & " * 7
        lines.append(la + " & " + lb + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_instances.tex", "\n".join(lines) + "\n")
    num("n_instances", len(rows), "{:d}")


# ----------------------------------------------------------------------------
# Table: valid inequalities (unchanged: exact study on the C101 family)
# ----------------------------------------------------------------------------
def table_vi():
    lp = load("vi_lp"); bb = load("vi")
    if lp is None or lp.empty:
        write("tab_vi.tex", "% vi results not available yet\n"); return
    cfgs = ["base", "+prep", "+rounding", "+two_cycle", "+lifted_mtz", "+symmetry", "+disagg", "all"]
    labels = {"base": "Base model", "+prep": "Base + arc elimination", "+rounding": "Base + rounded fleet-size (Prop.~1)", "+two_cycle": "Base + two-cycle (Prop.~2)",
              "+lifted_mtz": "Base + lifted MTZ (Prop.~3)", "+symmetry": "Base + symmetry breaking (Prop.~6)", "+disagg": "Base + disaggregated linking (Prop.~4)", "all": "Strengthened model (all)"}
    ref = {}
    mm = load("milp_main"); pe = load("protoE")
    for i in lp["instance"].unique():
        cands = []
        if mm is not None:
            cands += list(mm[mm["instance"] == i]["obj"].dropna())
        if bb is not None:
            cands += list(bb[bb["instance"] == i]["obj"].dropna())
        if pe is not None:
            cands += list(pe[pe["instance"] == i]["best"])
        ref[i] = min(cands)
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Effect of the strengthening. Top: LP root gap $(z^{*}-z_{LP})/z^{*}$ (\%, $z^{*}$ the best known objective) of every configuration, averaged over the six toy and the six 15-customer instances. Bottom: branch-and-bound effort of the base and the strengthened model (HiGHS, one thread, 300~s per sub-model; averages over the instances of each group)}\label{tab:vi}",
             r"\begin{tabular}{lrr}", r"\toprule", r"Configuration & Root gap toy (\%) & Root gap 15 cust.\ (\%) \\", r"\midrule"]
    for cfg in cfgs:
        cells = [labels[cfg]]
        for scale in ("toy", 15):
            sub = lp[(lp["config"] == cfg) & (lp["instance"].apply(scale_of) == scale)]
            g = np.mean([100 * (ref[i] - b) / ref[i] for i, b in zip(sub["instance"], sub["lp_bound"]) if not np.isnan(b)]) if len(sub) else np.nan
            cells.append(fmt(g, 2))
            num(f"vi_{scale}_{cfg.replace('+','').replace('_','')}_rootgap", g)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\midrule", r"\multicolumn{3}{l}{\textit{Branch-and-bound effort}}\\", r"\end{tabular}", "", r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Instances & Model & Optimal & Final gap (\%) & B\&B nodes & Time (s) \\", r"\midrule"]
    if bb is not None:
        for scale, title in (("toy", "Toy (6)"), (15, "15 customers (6)")):
            for cfg in ("base", "all"):
                sub = bb[(bb["config"] == cfg) & (bb["instance"].apply(scale_of) == scale)]
                if sub.empty:
                    continue
                nopt = int(sub["optimal"].sum()); gap = 100 * sub["gap"].mean(); nodes = sub["nodes"].mean(); t = sub["time"].mean()
                lines.append(f"{title} & {'base' if cfg == 'base' else 'strengthened'} & {nopt}/{len(sub)} & {fmt(gap,2)} & {fmt(nodes,0)} & {fmt(t,1)} \\\\")
                num(f"vi_{scale}_{cfg}_time", t, "{:.1f}"); num(f"vi_{scale}_{cfg}_nodes", nodes, "{:.0f}"); num(f"vi_{scale}_{cfg}_finalgap", gap); num(f"vi_{scale}_{cfg}_nopt", nopt, "{:d}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_vi.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Table: MILP main results
# ----------------------------------------------------------------------------
def milp_reference():
    df = load("milp_main")
    ref = {}
    if df is None:
        return ref
    for _, r in df.iterrows():
        ref[r["instance"]] = dict(obj=r["obj"], bound=r["bound"], optimal=bool(r["optimal"]), time=r["time"], gap=r["gap"])
    return ref


def table_milp():
    df = load("milp_main")
    if df is None or df.empty:
        write("tab_milp.tex", "% milp results not available yet\n"); return
    lines = [r"\begin{table}[t]\centering\scriptsize", r"\caption{Exact results of the strengthened model (HiGHS, one thread, 600~s per sub-model). Obj.: objective of the best solution found; LB: best lower bound; Gap $=(\text{Obj.}-\text{LB})/\text{Obj.}$; Opt.: proven optimal; the last two columns give the OSD-ALNS mean objective (protocol E, ten seeds) and its gap to the exact objective}\label{tab:milp}",
             r"\begin{tabular}{lrrrrrrrrr}", r"\toprule", r"Instance & Vars. & Rows & Obj. (CNY) & LB (CNY) & Gap (\%) & Opt. & Nodes & Time (s) & OSD-ALNS mean / gap (\%) \\", r"\midrule"]
    pe = load("protoE")
    for scale in ("toy", 15, 30):
        sub = df[df["instance"].apply(scale_of) == scale]
        for _, r in sub.iterrows():
            osd = ""
            if pe is not None:
                v = pe[(pe["instance"] == r["instance"]) & (pe["algo"] == "OSD")]["best"]
                if len(v):
                    m = v.mean(); g = 100 * (m - r["obj"]) / r["obj"] if not np.isnan(r["obj"]) else np.nan
                    osd = f"{fmt(m)} / {fmt(g)}"
            lines.append(f"{r['instance']} & {fmt(r['n_vars'],0)} & {fmt(r['n_rows'],0)} & {fmt(r['obj'])} & {fmt(r['bound'])} & {fmt(100*r['gap'] if not np.isnan(r['gap']) else np.nan)} & {'yes' if r['optimal'] else 'no'} & {fmt(r['nodes'],0)} & {fmt(r['time'],1)} & {osd} \\\\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_milp.tex", "\n".join(lines) + "\n")
    num("milp_n_opt", int(df["optimal"].sum()), "{:d}"); num("milp_n", len(df), "{:d}")
    for scale in ("toy", 15, 30):
        sub = df[df["instance"].apply(scale_of) == scale]
        if len(sub):
            num(f"milp_{scale}_nopt", int(sub["optimal"].sum()), "{:d}"); num(f"milp_{scale}_time", sub["time"].mean(), "{:.1f}")
            num(f"milp_{scale}_gap", 100 * sub["gap"].mean())


def table_calib():
    p = os.path.join(RES, "calib_summary.csv")
    if not os.path.exists(p):
        write("tab_calib.tex", "% calibration not available\n"); return
    s = pd.read_csv(p, index_col=0)
    order = ["default", "T0=50.0", "T0=5000.0", "alpha=0.99", "alpha=0.999", "rho=0.1", "rho=0.5", "lam=0.3", "lam=0.8", "q=0.05-0.2", "q=0.05-0.35", "q=0.2-0.4",
             "restart_ops=0", "restart_ops=250", "restart_ops=1000"]
    lab = {"default": "Default ($T_0{=}500$, $\\alpha{=}0.995$, $\\rho{=}0.2$, $\\lambda{=}0.5$, $q\\in[0.1,0.3]$, $\\kappa{=}500$)", "T0=50.0": "$T_0=50$", "T0=5000.0": "$T_0=5000$",
           "alpha=0.99": "$\\alpha=0.99$", "alpha=0.999": "$\\alpha=0.999$", "rho=0.1": "$\\rho=0.1$", "rho=0.5": "$\\rho=0.5$",
           "lam=0.3": "$\\lambda=0.3$", "lam=0.8": "$\\lambda=0.8$", "q=0.05-0.2": "$q\\in[0.05,0.20]$", "q=0.05-0.35": "$q\\in[0.05,0.35]$", "q=0.2-0.4": "$q\\in[0.20,0.40]$",
           "restart_ops=0": "no restart ($\\kappa=\\infty$)", "restart_ops=250": "restart after $\\kappa=250$ operations", "restart_ops=1000": "restart after $\\kappa=1000$ operations"}
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Parameter calibration of OSD-ALNS (one factor at a time, six instances, five seeds, protocol E): mean and maximum relative deviation of the mean objective from the best mean objective obtained by any setting on the same instance}\label{tab:calib}",
             r"\begin{tabular}{lrr}", r"\toprule", r"Setting & Mean dev. (\%) & Max dev. (\%) \\", r"\midrule"]
    for k in order:
        if k in s.index:
            lines.append(f"{lab[k]} & {s.loc[k,'mean_rel']:.3f} & {s.loc[k,'max_rel']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_calib.tex", "\n".join(lines) + "\n")
    with open(os.path.join(RES, "calibrated_params.json")) as f:
        pr = json.load(f)
    num("calibparams", f"$T_0={pr['T0']:g}$, $\\alpha={pr['alpha']}$, $\\rho={pr['rho']}$, $\\lambda={pr['lam']}$, $[q_{{\\min}},q_{{\\max}}]=[{pr['q_min']},{pr['q_max']}]$, $\\kappa={pr.get('restart_ops', 500)}$")
    num("calib_max_dev", s["mean_rel"].max(), "{:.3f}")
    worst = s["mean_rel"].idxmax(); num("calib_worst", lab.get(worst, worst))


# ----------------------------------------------------------------------------
# Protocol E: per-instance tables (appendix), summary table, tests, efficiency
# ----------------------------------------------------------------------------
def best_known_values(de=None, dt=None):
    ref = milp_reference()
    bk = {}
    for df in (de, dt):
        if df is None:
            continue
        for i, v in df.groupby("instance")["best"].min().items():
            bk[i] = min(bk.get(i, v), v)
    for i, v in ref.items():
        if not np.isnan(v["obj"]):
            bk[i] = min(bk.get(i, v["obj"]), v["obj"])
    return bk


def ops_to_target(trace_path, target):
    if not os.path.exists(trace_path):
        return None
    tr = json.load(open(trace_path))
    for o, t, c in tr:
        if c <= target + 1e-9:
            return int(o), float(t)
    return math.inf, math.inf


def table_protoE():
    df = load("protoE")
    if df is None or df.empty:
        for f in ("tab_protoE.tex", "tab_protoE_rc.tex", "tab_protoE_summary.tex", "tab_tests.tex", "tab_efficiency.tex"):
            write(f, "% protoE not available\n")
        return
    ref = milp_reference()
    g = df.groupby(["instance", "algo"]).agg(mean=("best", "mean"), best=("best", "min"), std=("best", "std"), time=("time", "mean"), optime=("op_time", "mean"), n=("best", "count")).reset_index()
    piv = g.pivot(index="instance", columns="algo")
    insts_all = [i for i in all_instance_names() if i in piv.index and all(not np.isnan(piv.loc[i, ("mean", a)]) for a in ALGOS)]
    bk = best_known_values(df, load("protoT"))

    def ref_of(i):
        allbest = min(piv.loc[i, ("best", a)] for a in ALGOS)
        if i in ref and not np.isnan(ref[i]["obj"]):
            rv = min(ref[i]["obj"], allbest)
            rtxt = (r"\textbf{" + fmt(rv) + "}") if (ref[i]["optimal"] and ref[i]["obj"] <= allbest + 1e-6) else (r"\textit{" + fmt(rv) + "}")
        else:
            rv = allbest; rtxt = fmt(rv)
        return rv, rtxt

    rows_stats = []
    for i in insts_all:
        rv, _ = ref_of(i)
        means = {a: piv.loc[i, ("mean", a)] for a in ALGOS}
        rows_stats.append(dict(instance=i, family=geometry_of(i), scale=scale_of(i), ref=rv, seeds=int(piv.loc[i, ("n", "OSD")]),
                               **{f"mean_{a}": means[a] for a in ALGOS}, **{f"best_{a}": piv.loc[i, ("best", a)] for a in ALGOS},
                               **{f"gap_{a}": 100 * (means[a] - rv) / rv for a in ALGOS}, **{f"time_{a}": piv.loc[i, ("time", a)] for a in ALGOS},
                               **{f"optime_{a}": piv.loc[i, ("optime", a)] for a in ALGOS}))
    st = pd.DataFrame(rows_stats); st.to_csv(os.path.join(RES, "summary_protoE.csv"), index=False)

    # ---- per-instance tables (appendix)
    def per_instance_table(fname, groups, label, caption):
        lines = [r"\begin{table}[p]\centering\scriptsize", caption,
                 r"\begin{tabular}{lr|rrr|rrr|rrr|rrr}", r"\toprule",
                 r"Instance & Ref. & \multicolumn{3}{c|}{Mean obj.} & \multicolumn{3}{c|}{Best obj.} & \multicolumn{3}{c|}{Gap of mean (\%)} & \multicolumn{3}{c}{Time (s)} \\",
                 r" & & I & P & OSD & I & P & OSD & I & P & OSD & I & P & OSD \\", r"\midrule"]
        for grp in groups:
            sub = st[(st["family"] == grp[0]) & (st["scale"] == grp[1])]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                i = r["instance"]; rv, rtxt = ref_of(i)
                means = [r[f"mean_{a}"] for a in ALGOS]; mmin = min(means)
                cells = [i, rtxt] + [(r"\textbf{" + fmt(m) + "}") if abs(m - mmin) < 1e-6 else fmt(m) for m in means]
                cells += [fmt(r[f"best_{a}"]) for a in ALGOS] + [fmt(r[f"gap_{a}"]) for a in ALGOS] + [fmt(r[f"time_{a}"], 1) for a in ALGOS]
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\multicolumn{2}{l}{\textit{Average (" + group_label(grp) + ")}} & & & & & & & " + " & ".join(fmt(sub[f"gap_{a}"].mean()) for a in ALGOS) + " & " + " & ".join(fmt(sub[f"time_{a}"].mean(), 1) for a in ALGOS) + r" \\")
            lines.append(r"\midrule")
        lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        write(fname, "\n".join(lines) + "\n")

    per_instance_table("tab_protoE.tex", [("toy", "toy")] + [("C", n) for n in (15, 30, 50, 75, 100)], "tab:protoE",
                       r"\caption{Protocol E on the toy and C101-based instances (equal number of elementary operations, ten seeds): mean and best objective (CNY), CPU time (s) of ALNS-I, ALNS-P and OSD-ALNS; Ref.: exact optimum (bold), best objective of the exact solver within the time limit (italic) or best known value; Gap: deviation of the mean objective from Ref.}\label{tab:protoE}")
    per_instance_table("tab_protoE_rc.tex", [("R", n) for n in (50, 75, 100)] + [("RC", n) for n in (50, 75, 100)] + [("M", 200), ("M", 300)], "tab:protoErc",
                       r"\caption{Protocol E on the R101-, RC101-based and large instances (equal number of elementary operations, five seeds): mean and best objective (CNY), CPU time (s) of ALNS-I, ALNS-P and OSD-ALNS; Ref.: best known value; Gap: deviation of the mean objective from Ref.}\label{tab:protoErc}")

    # ---- summary table by group
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Protocol E (equal number of elementary operations): by instance group, mean gap of the mean objective to the reference value (\%), number of instances on which each layer attains the best mean objective (ties counted for every layer), and mean CPU time per run (s)}\label{tab:protoEsum}",
             r"\begin{tabular}{lr|rrr|rrr|rrr}", r"\toprule", r"Group & Inst. & \multicolumn{3}{c|}{Mean gap (\%)} & \multicolumn{3}{c|}{Best mean (\#)} & \multicolumn{3}{c}{Time (s)} \\",
             r" & & I & P & OSD & I & P & OSD & I & P & OSD \\", r"\midrule"]
    fam_rows = []
    for grp in GROUPS:
        sub = st[(st["family"] == grp[0]) & (st["scale"] == grp[1])]
        if sub.empty:
            continue
        mmin = sub[[f"mean_{a}" for a in ALGOS]].min(axis=1)
        nbest = {a: int((sub[f"mean_{a}"] <= mmin + 1e-6).sum()) for a in ALGOS}
        cells = [group_label(grp), str(len(sub))] + [fmt(sub[f"gap_{a}"].mean(), 3) for a in ALGOS] + [str(nbest[a]) for a in ALGOS] + [fmt(sub[f"time_{a}"].mean(), 1) for a in ALGOS]
        lines.append(" & ".join(cells) + r" \\")
        for a in ALGOS:
            num(f"pE_{a}_gap_{grp[0]}{grp[1]}", sub[f"gap_{a}"].mean(), "{:.3f}"); num(f"pE_{a}_nbest_{grp[0]}{grp[1]}", nbest[a], "{:d}")
    lines.append(r"\midrule")
    mmin = st[[f"mean_{a}" for a in ALGOS]].min(axis=1)
    nbest_all = {a: int((st[f"mean_{a}"] <= mmin + 1e-6).sum()) for a in ALGOS}
    lines.append(" & ".join(["All instances", str(len(st))] + [fmt(st[f"gap_{a}"].mean(), 3) for a in ALGOS] + [str(nbest_all[a]) for a in ALGOS] + [fmt(st[f"time_{a}"].mean(), 1) for a in ALGOS]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_protoE_summary.tex", "\n".join(lines) + "\n")
    for a in ALGOS:
        num(f"pE_{a}_gap", st[f"gap_{a}"].mean(), "{:.3f}"); num(f"pE_{a}_time", st[f"time_{a}"].mean(), "{:.1f}")
        num(f"pE_{a}_optime", 1000 * st[f"optime_{a}"].mean(), "{:.2f}"); num(f"pE_{a}_nbest", nbest_all[a], "{:d}")
        for fam in ("C", "R", "RC", "M"):
            sub = st[st["family"] == fam]
            if len(sub):
                num(f"pE_{a}_gap_fam{fam}", sub[f"gap_{a}"].mean(), "{:.3f}")
    num("pE_n", len(st), "{:d}")
    d_oi = 100 * (st["mean_OSD"] - st["mean_I"]) / st["mean_I"]; d_op = 100 * (st["mean_OSD"] - st["mean_P"]) / st["mean_P"]; d_pi = 100 * (st["mean_P"] - st["mean_I"]) / st["mean_I"]
    num("pE_OSDvsI_meandiff", d_oi.mean(), "{:+.3f}"); num("pE_OSDvsP_meandiff", d_op.mean(), "{:+.3f}"); num("pE_PvsI_meandiff", d_pi.mean(), "{:+.3f}")
    num("pE_OSDvsI_min", d_oi.min(), "{:+.3f}"); num("pE_OSDvsI_max", d_oi.max(), "{:+.3f}")
    num("pE_OSD_wins", int((d_oi < -1e-6).sum()), "{:d}"); num("pE_OSD_losses", int((d_oi > 1e-6).sum()), "{:d}"); num("pE_OSD_ties", int((d_oi.abs() <= 1e-6).sum()), "{:d}")
    num("pE_OSD_within01", int((d_oi.abs() <= 0.1).sum()), "{:d}")
    worst = st.loc[d_oi.abs().idxmax(), "instance"]; num("pE_OSDvsI_extreme_inst", worst)
    num("pE_OSDvsI_extreme", d_oi[d_oi.abs().idxmax()], "{:+.3f}")
    # large instances
    lg = st[st["family"] == "M"]
    for _, r in lg.iterrows():
        for a in ALGOS:
            num(f"pE_{a}_gap_{r['instance'].replace('-', '')}", r[f"gap_{a}"], "{:.3f}")

    # ---- statistical tests (all instances and per family)
    tl = [r"\begin{table}[t]\centering\small", r"\caption{Statistical tests on the instance-wise mean objectives under protocol E. Wilcoxon: two-sided signed-rank test on the paired differences; W$+$/W$-$: numbers of instances on which the first layer is better/worse (ties excluded); Holm: adjusted $p$-value over the three pairwise comparisons; Friedman: test over the three layers with mean ranks}\label{tab:tests}",
          r"\begin{tabular}{llrrrrr}", r"\toprule", r"Instances & Comparison & Mean diff. (\%) & W$+$ & W$-$ & Wilcoxon $p$ & Holm $p$ \\", r"\midrule"]
    pairs = [("OSD", "I"), ("OSD", "P"), ("P", "I")]
    for setname, sub in (("all (%d)" % len(st), st), ("C101 family (%d)" % int((st["family"].isin(["toy", "C"])).sum()), st[st["family"].isin(["toy", "C"])]),
                         ("R101/RC101/large (%d)" % int((st["family"].isin(["R", "RC", "M"])).sum()), st[st["family"].isin(["R", "RC", "M"])])):
        if len(sub) < 6:
            continue
        pvals = []; rows_t = []
        for a, b in pairs:
            x = sub[f"mean_{a}"].values; y = sub[f"mean_{b}"].values
            diff = x - y; nz = diff[np.abs(diff) > 1e-9]
            p = stats.wilcoxon(x, y, zero_method="wilcox").pvalue if len(nz) >= 5 else 1.0
            pvals.append(p)
            rows_t.append((a, b, 100 * np.mean(diff / y), int((diff < -1e-9).sum()), int((diff > 1e-9).sum()), p))
        order = np.argsort(pvals); adj = [0] * len(pvals); m = len(pvals); running = 0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pvals[idx]); adj[idx] = min(1.0, running)
        tag = setname.split(" ")[0].replace("/", "")
        for k, ((a, b, md, wp, wm, p), pa) in enumerate(zip(rows_t, adj)):
            tl.append(f"{setname if k == 0 else ''} & {NAMES[a]} vs.\\ {NAMES[b]} & {md:+.3f} & {wp} & {wm} & {p:.3f} & {pa:.3f} \\\\")
            num(f"pE_wilcoxon_{a}_{b}_{tag}", p, "{:.3f}"); num(f"pE_holm_{a}_{b}_{tag}", pa, "{:.3f}")
            num(f"pE_wplus_{a}_{b}_{tag}", wp, "{:d}"); num(f"pE_wminus_{a}_{b}_{tag}", wm, "{:d}")
        fr = stats.friedmanchisquare(sub["mean_I"], sub["mean_P"], sub["mean_OSD"])
        ranks = sub[[f"mean_{a}" for a in ALGOS]].rank(axis=1).mean().values
        tl.append(f" & Friedman & \\multicolumn{{5}}{{l}}{{$\\chi^2={fr.statistic:.2f}$, $p={fr.pvalue:.3f}$; mean ranks: " + ", ".join(f"{NAMES[a]} {r:.2f}" for a, r in zip(ALGOS, ranks)) + r"} \\")
        tl.append(r"\midrule")
        num(f"pE_friedman_p_{tag}", fr.pvalue, "{:.3f}"); num(f"pE_friedman_chi_{tag}", fr.statistic, "{:.2f}")
        for a, r in zip(ALGOS, ranks):
            num(f"pE_rank_{a}_{tag}", r, "{:.2f}")
    tl = tl[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_tests.tex", "\n".join(tl) + "\n")
    # non-inferiority: bootstrap CI of the mean gap difference (percentage points) and one-sided Wilcoxon
    # test of H0: gap_OSD - gap_b >= delta against H1: < delta
    rng = np.random.default_rng(2026)
    for b in ("I", "P"):
        d = (st["gap_OSD"] - st[f"gap_{b}"]).values
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(20000)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        num(f"ni_{b}_lo", lo, "{:+.3f}"); num(f"ni_{b}_hi", hi, "{:+.3f}"); num(f"ni_{b}_mean", d.mean(), "{:+.3f}")
        for delta, tag in ((0.02, "A"), (0.05, "B")):
            pv = stats.wilcoxon(d - delta, alternative="less").pvalue
            num(f"ni_{b}_p{tag}", pv, "{:.3f}")
            num(f"ni_{b}_p{tag}_txt", "p<0.001" if pv < 0.001 else f"p={pv:.3f}")

    # ---- efficiency: hit rate and operations / time to target (within TARGET_TOL of the best known)
    rows_e = []
    for i in insts_all:
        target = bk[i] * (1 + TARGET_TOL)
        for a in ALGOS:
            seeds = df[(df["instance"] == i) & (df["algo"] == a)]["seed"].tolist()
            hits = []; o2t = []; t2t = []
            for s in seeds:
                r = ops_to_target(os.path.join(RES, "traces", f"protoE_{i}_{a}_-_{s}.json"), target)
                if r is None:
                    continue
                hits.append(1.0 if r[0] < math.inf else 0.0)
                o2t.append(r[0] if r[0] < math.inf else BUDGET[scale_of(i)]); t2t.append(r[1] if r[1] < math.inf else float(df[(df["instance"] == i) & (df["algo"] == a) & (df["seed"] == s)]["time"].iloc[0]))
            if hits:
                rows_e.append(dict(instance=i, family=geometry_of(i), scale=scale_of(i), algo=a, hit=np.mean(hits), ops=np.median(o2t), time=np.median(t2t)))
    ef = pd.DataFrame(rows_e); ef.to_csv(os.path.join(RES, "summary_efficiency.csv"), index=False)
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Efficiency under protocol E: hit rate (share of runs that reach a solution within 0.05\,\% of the best known value within the budget) and median number of elementary operations and CPU time (s) needed to reach it (runs that fail are counted at the budget), averaged over the instances of each group}\label{tab:efficiency}",
             r"\begin{tabular}{l|rrr|rrr|rrr}", r"\toprule", r"Group & \multicolumn{3}{c|}{Hit rate} & \multicolumn{3}{c|}{Operations to target} & \multicolumn{3}{c}{Time to target (s)} \\",
             r" & I & P & OSD & I & P & OSD & I & P & OSD \\", r"\midrule"]
    if not ef.empty:
        for grp in GROUPS:
            sub = ef[(ef["family"] == grp[0]) & (ef["scale"] == grp[1])]
            if sub.empty:
                continue
            cells = [group_label(grp)]
            for key, d in (("hit", 2), ("ops", 0), ("time", 1)):
                cells += [fmt(sub[sub["algo"] == a][key].mean(), d) for a in ALGOS]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        cells = ["All instances"]
        for key, d in (("hit", 2), ("ops", 0), ("time", 1)):
            cells += [fmt(ef[ef["algo"] == a][key].mean(), d) for a in ALGOS]
            for a in ALGOS:
                num(f"eff_{a}_{key}", ef[ef["algo"] == a][key].mean(), "{:.%df}" % d)
        lines.append(" & ".join(cells) + r" \\")
        for fam in ("C", "R", "RC", "M"):
            sub = ef[ef["family"] == fam]
            for a in ALGOS:
                if len(sub):
                    num(f"eff_{a}_hit_fam{fam}", sub[sub["algo"] == a]["hit"].mean(), "{:.2f}"); num(f"eff_{a}_ops_fam{fam}", sub[sub["algo"] == a]["ops"].mean(), "{:.0f}")
        # paired test on ops-to-target OSD vs I
        pv = ef.pivot(index="instance", columns="algo", values="ops")
        d = (pv["OSD"] - pv["I"]).values
        num("eff_wilcoxon_OSD_I", stats.wilcoxon(pv["OSD"], pv["I"]).pvalue if (np.abs(d) > 1e-9).sum() >= 5 else 1.0, "{:.3f}")
        pv2 = ef.pivot(index="instance", columns="algo", values="hit")
        for a in ALGOS:
            num(f"eff_{a}_hit_all", pv2[a].mean(), "{:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_efficiency.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Convergence profiles
# ----------------------------------------------------------------------------
def fig_convergence():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load("protoE")
    if df is None or df.empty:
        return
    bk = best_known_values(df, load("protoT"))
    colors = {"I": "#4C72B0", "P": "#55A868", "OSD": "#C44E52"}
    panels = [("C", 100), ("R", 100), ("RC", 100), ("M", 300)]
    panels = [p for p in panels if any((geometry_of(i), scale_of(i)) == p for i in df["instance"].unique())]
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(7.4, 2.3 * len(panels)), squeeze=False)
    ops_curve = {}
    for row, grp in enumerate(panels):
        budget = BUDGET[grp[1]]
        for a in ALGOS:
            sub = df[(df["algo"] == a) & (df["instance"].apply(group_of) == grp)]
            grid_ops = np.linspace(0, budget, 121); curves_ops = []
            tmax = sub["time"].max(); grid_t = np.linspace(0, tmax, 121); curves_t = []
            for _, r in sub.iterrows():
                p = os.path.join(RES, "traces", f"protoE_{r['instance']}_{a}_-_{r['seed']}.json")
                if not os.path.exists(p):
                    continue
                tr = json.load(open(p)); b = bk[r["instance"]]
                ops = np.array([t[0] for t in tr]); tim = np.array([t[1] for t in tr]); val = np.array([t[2] for t in tr])

                def at(grid, key):
                    out = np.empty(len(grid))
                    for q, gg in enumerate(grid):
                        idx = np.searchsorted(key, gg, side="right") - 1
                        out[q] = val[max(idx, 0)]
                    return out
                curves_ops.append(100 * (at(grid_ops, ops) - b) / b); curves_t.append(100 * (at(grid_t, tim) - b) / b)
            if curves_ops:
                m_ops = np.mean(curves_ops, axis=0); m_t = np.mean(curves_t, axis=0)
                axes[row, 0].plot(grid_ops, m_ops, color=colors[a], label=NAMES[a], lw=1.6)
                axes[row, 1].plot(grid_t, m_t, color=colors[a], label=NAMES[a], lw=1.6)
                ops_curve[(grp, a)] = (grid_ops, m_ops)
        axes[row, 0].set_title(f"{group_label(grp)} -- elementary operations", fontsize=9); axes[row, 0].set_xlabel("elementary operations", fontsize=8)
        axes[row, 1].set_title(f"{group_label(grp)} -- CPU time", fontsize=9); axes[row, 1].set_xlabel("CPU time (s)", fontsize=8)
        for ax in (axes[row, 0], axes[row, 1]):
            ax.set_yscale("symlog", linthresh=0.1); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            ax.set_ylabel("excess over best known (%)", fontsize=8)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_convergence.pdf")); fig.savefig(os.path.join(FIG, "fig_convergence.png"), dpi=200); plt.close(fig)
    for (grp, a), (g, m) in ops_curve.items():
        tag = f"{grp[0]}{grp[1]}"
        for thr, lab in ((0.5, "05"), (0.1, "01")):
            idx = np.where(m <= thr)[0]
            num(f"conv_{a}_{tag}_ops{lab}", g[idx[0]] if len(idx) else float("nan"), "{:.0f}")
        num(f"conv_{a}_{tag}_final", m[-1], "{:.3f}")


# ----------------------------------------------------------------------------
# Len, ablation, protoT, sensitivity
# ----------------------------------------------------------------------------
def table_len():
    df = load("len"); de = load("protoE")
    if df is None or df.empty or de is None:
        write("tab_len.tex", "% len not available\n"); return
    full = de[(de["algo"] == "OSD") & (de["instance"].isin(df["instance"].unique())) & (de["seed"] <= 5)].copy(); full["variant"] = "Len<=2"
    df = pd.concat([full, df])
    g = df.groupby(["instance", "variant"]).agg(mean=("best", "mean"), time=("time", "mean"), iters=("iters", "mean")).reset_index()
    g["group"] = g["instance"].apply(geometry_of)
    bestv = g.groupby("instance")["mean"].min()
    g["dev"] = 100 * (g["mean"] - g["instance"].map(bestv)) / g["instance"].map(bestv)
    variants = ["Len=1", "Len=2", "Len<=2", "Len<=3"]
    lab = {"Len=1": "fixed $L=1$ (16 sequences)", "Len=2": "fixed $L=2$ (256)", "Len<=2": "variable $L\\leq2$ (272, default)", "Len<=3": "variable $L\\leq3$ (4\\,368)"}
    groups = [("C", "C101, 75 cust."), ("R", "R101, 75 cust."), ("RC", "RC101, 75 cust.")]
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Effect of the sequence space (OSD-ALNS with hierarchical weights, protocol E, five seeds, A-1 and B-1 instances with 75 customers): mean relative deviation of the mean objective from the best mean over the four variants (\%) and mean CPU time (s)}\label{tab:len}",
             r"\begin{tabular}{l" + "rr" * len(groups) + "}", r"\toprule", " & " + " & ".join(r"\multicolumn{2}{c}{" + lb + "}" for _, lb in groups) + r" \\",
             r"Sequence space & " + " & ".join("Dev. & Time" for _ in groups) + r" \\", r"\midrule"]
    for v in variants:
        cells = [lab[v]]
        for key, _ in groups:
            s = g[(g["variant"] == v) & (g["group"] == key)]
            cells += [fmt(s["dev"].mean(), 3), fmt(s["time"].mean(), 1)]
            num(f"len_{v.replace('<=','le').replace('=','')}_{key}_dev", s["dev"].mean(), "{:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_len.tex", "\n".join(lines) + "\n")
    base = g[g["variant"] == "Len<=2"].set_index("instance")["mean"]
    for v in variants:
        if v == "Len<=2":
            continue
        x = g[g["variant"] == v].set_index("instance")["mean"]; common = x.index.intersection(base.index)
        d = (x[common] - base[common]).values
        p = stats.wilcoxon(x[common], base[common]).pvalue if (np.abs(d) > 1e-9).sum() >= 5 else 1.0
        num(f"len_wilcoxon_{v.replace('<=','le').replace('=','')}", p, "{:.3f}")
        num(f"len_{v.replace('<=','le').replace('=','')}_meandiff", 100 * np.mean(d / base[common].values), "{:+.3f}")
    allmean = g.groupby("variant")["dev"].mean()
    for v in variants:
        if v in allmean.index:
            num(f"len_{v.replace('<=','le').replace('=','')}_dev_all", allmean[v], "{:.3f}")


def table_ablation():
    df = load("ablation"); de = load("protoE")
    if df is None or df.empty or de is None:
        write("tab_ablation.tex", "% ablation not available\n"); return
    full = de[(de["algo"] == "OSD") & (de["instance"].isin(df["instance"].unique())) & (de["seed"] <= 5)].copy(); full["variant"] = "OSD"
    df = pd.concat([full, df])
    g = df.groupby(["instance", "variant"]).agg(mean=("best", "mean"), time=("time", "mean")).reset_index()
    g["family"] = g["instance"].apply(geometry_of)
    base = g[g["variant"] == "OSD"].set_index("instance")["mean"]
    bk_all = pd.concat([de, df]).groupby("instance")["best"].min()
    df = df.copy(); df["rdev"] = 100 * (df["best"] - df["instance"].map(bk_all)) / df["instance"].map(bk_all)
    trapped = df.assign(t=df["rdev"] > 0.1).groupby("variant")["t"].sum()
    nruns = df.groupby("variant")["best"].count()
    g["dev"] = 100 * (g["mean"] - g["instance"].map(base)) / g["instance"].map(base)
    order = ["OSD", "flat-weights", "fixed-Len=2", "accept-each", "size-classes", "no-adaptation", "no-restart", "random-init", "small-pool"]
    lab = {"OSD": "OSD-ALNS (full)", "flat-weights": "flat sequence weights", "fixed-Len=2": "fixed length $L=2$",
           "accept-each": "acceptance after every pair", "size-classes": "destroy-size class in the unit",
           "no-adaptation": "no adaptation ($\\rho=0$)", "no-restart": "no restart ($\\kappa=\\infty$)", "random-init": "random initial solution",
           "small-pool": "reduced $2\\times2$ operator pool"}
    ninst = g["instance"].nunique()
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Component ablation of OSD-ALNS (protocol E, five seeds, %d instances): mean relative deviation of the mean objective from the full algorithm (\%%, positive = worse) on the R101/RC101-based instances with 100 customers and on the 200-customer instance, number of runs that end more than $0.1\%%$ above the best known value, wins/losses against the full algorithm and Wilcoxon $p$-value of the paired comparison (-- : fewer than five non-tied instances)}\label{tab:ablation}" % ninst,
             r"\begin{tabular}{lrrrrrrr}", r"\toprule", r"Variant & Dev. R/RC (100) & Dev. large (200) & Dev. all & Runs $>0.1\%$ & Better & Worse & Wilcoxon $p$ \\", r"\midrule"]
    for v in order:
        s = g[g["variant"] == v]
        if s.empty:
            continue
        dC = s[s["family"] == "C"]["dev"].mean(); dR = s[s["family"].isin(["R", "RC"])]["dev"].mean(); dM = s[s["family"] == "M"]["dev"].mean(); dall = s["dev"].mean()
        if v == "OSD":
            lines.append(f"{lab[v]} & 0.000 & 0.000 & 0.000 & {int(trapped[v])}/{int(nruns[v])} & -- & -- & -- \\\\"); num("abl_OSD_trapped", int(trapped[v]), "{:d}"); continue
        x = s.set_index("instance")["mean"]; y = base[x.index]
        better = int((x < y - 1e-9).sum()); worse = int((x > y + 1e-9).sum())
        d = (x - y).values
        p = stats.wilcoxon(x, y).pvalue if (np.abs(d) > 1e-9).sum() >= 5 else float("nan")
        lines.append(f"{lab[v]} & {dR:+.3f} & {fmt(dM,3) if np.isnan(dM) else '%+.3f' % dM} & {dall:+.3f} & {int(trapped[v])}/{int(nruns[v])} & {better} & {worse} & {fmt(p,3)} \\\\")
        num(f"abl_{v.replace('-', '').replace('=', '')}_trapped", int(trapped[v]), "{:d}")
        tag = v.replace("-", "").replace("=", "")
        num(f"abl_{tag}_dev", dall, "{:+.3f}"); num(f"abl_{tag}_p", p, "{:.3f}"); num(f"abl_{tag}_devM", dM, "{:+.3f}"); num(f"abl_{tag}_devC", dC, "{:+.3f}"); num(f"abl_{tag}_devR", dR, "{:+.3f}")
        num(f"abl_{tag}_better", better, "{:d}"); num(f"abl_{tag}_worse", worse, "{:d}")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_ablation.tex", "\n".join(lines) + "\n")
    num("abl_n", ninst, "{:d}")


def table_protoT():
    dt = load("protoT"); de = load("protoE")
    if dt is None or dt.empty or de is None:
        write("tab_protoT.tex", "% protoT not available\n"); write("tab_protoT_summary.tex", "% protoT not available\n"); return
    bk = best_known_values(de, dt)
    gt = dt.groupby(["instance", "algo"]).agg(mean=("best", "mean"), best=("best", "min"), time=("time", "mean"), ops=("ops", "mean")).reset_index()
    ge = de.groupby(["instance", "algo"]).agg(mean=("best", "mean"), best=("best", "min"), time=("time", "mean")).reset_index()
    keys = ["VNS", "GA", "I", "P", "OSD"]
    rows = []
    for i in all_instance_names():
        s = gt[gt["instance"] == i]; e = ge[ge["instance"] == i]
        if (s["algo"] == "VNS").sum() == 0 or (s["algo"] == "GA").sum() == 0 or e.empty:
            continue
        m = {"VNS": s[s["algo"] == "VNS"]["mean"].values[0], "GA": s[s["algo"] == "GA"]["mean"].values[0]}
        for a in ALGOS:
            m[a] = e[e["algo"] == a]["mean"].values[0]
        rows.append(dict(instance=i, family=geometry_of(i), scale=scale_of(i), time=s["time"].mean(), **m, **{f"dev_{k}": 100 * (m[k] - bk[i]) / bk[i] for k in keys}))
    st = pd.DataFrame(rows); st.to_csv(os.path.join(RES, "summary_protoT.csv"), index=False)
    # full table (appendix)
    lines = [r"\begin{table}[p]\centering\scriptsize", r"\caption{Protocol T (equal CPU time per run, = mean OSD-ALNS time under protocol E; ten seeds on the toy and C101-based instances, five otherwise): mean objective (CNY) of VNS and GA, of the three adaptive layers (from protocol E), and deviation of the mean from the best known solution (\%)}\label{tab:protoT}",
             r"\begin{tabular}{lrrrrr|rrrrr|r}", r"\toprule", r"Instance & \multicolumn{5}{c|}{Mean objective} & \multicolumn{5}{c|}{Deviation from best known (\%)} & Time (s) \\",
             r" & VNS & GA & ALNS-I & ALNS-P & OSD-ALNS & VNS & GA & ALNS-I & ALNS-P & OSD-ALNS & \\", r"\midrule"]
    for grp in GROUPS:
        sub = st[(st["family"] == grp[0]) & (st["scale"] == grp[1])]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            means = [r[k] for k in keys]; mmin = min(means)
            cells = [r["instance"]] + [(r"\textbf{" + fmt(v) + "}") if abs(v - mmin) < 1e-6 else fmt(v) for v in means] + [fmt(r[f"dev_{k}"]) for k in keys] + [fmt(r["time"], 1)]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\multicolumn{6}{l}{\textit{Average (" + group_label(grp) + ")}} & " + " & ".join(fmt(sub[f"dev_{k}"].mean()) for k in keys) + r" & \\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_protoT.tex", "\n".join(lines) + "\n")
    # summary table
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Protocol T (equal CPU time): by instance group, mean deviation of the mean objective from the best known solution (\%) of VNS, GA and the three adaptive layers, and mean time budget per run (s)}\label{tab:protoTsum}",
             r"\begin{tabular}{lr|rrrrr|r}", r"\toprule", r"Group & Inst. & VNS & GA & ALNS-I & ALNS-P & OSD-ALNS & Time (s) \\", r"\midrule"]
    for grp in GROUPS:
        sub = st[(st["family"] == grp[0]) & (st["scale"] == grp[1])]
        if sub.empty:
            continue
        lines.append(" & ".join([group_label(grp), str(len(sub))] + [fmt(sub[f"dev_{k}"].mean(), 3) for k in keys] + [fmt(sub["time"].mean(), 1)]) + r" \\")
        for k in keys:
            num(f"pT_{k}_dev_{grp[0]}{grp[1]}", sub[f"dev_{k}"].mean(), "{:.3f}")
    lines.append(r"\midrule")
    lines.append(" & ".join(["All instances", str(len(st))] + [fmt(st[f"dev_{k}"].mean(), 3) for k in keys] + [fmt(st["time"].mean(), 1)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_protoT_summary.tex", "\n".join(lines) + "\n")
    for k in keys:
        num(f"pT_{k}_dev", st[f"dev_{k}"].mean(), "{:.3f}")
        for fam in ("C", "R", "RC", "M"):
            sub = st[st["family"] == fam]
            if len(sub):
                num(f"pT_{k}_dev_fam{fam}", sub[f"dev_{k}"].mean(), "{:.3f}")
    for k in ("VNS", "GA", "I", "P"):
        d = st["OSD"] - st[k]
        p = stats.wilcoxon(st["OSD"], st[k]).pvalue if (np.abs(d) > 1e-9).sum() >= 5 else float("nan")
        num(f"pT_wilcoxon_OSD_{k}", p, "{:.3f}"); num(f"pT_OSD_vs_{k}_meandiff", 100 * np.mean(d / st[k]), "{:+.3f}")
        num(f"pT_OSD_wins_{k}", int((d < -1e-9).sum()), "{:d}"); num(f"pT_OSD_losses_{k}", int((d > 1e-9).sum()), "{:d}")
    fr = stats.friedmanchisquare(*[st[k] for k in keys])
    num("pT_friedman_p", fr.pvalue, "{:.3f}")
    ranks = st[keys].rank(axis=1).mean()
    for k in ranks.index:
        num(f"pT_rank_{k}", ranks[k], "{:.2f}")
    num("pT_n", len(st), "{:d}")
    num("pT_VNS_max", st["dev_VNS"].max(), "{:.2f}"); num("pT_GA_max", st["dev_GA"].max(), "{:.2f}")


def fig_sensitivity():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load("sens")
    if df is None or df.empty:
        return
    df["par"] = df["variant"].str.split("=").str[0]; df["val"] = df["variant"].str.split("=").str[1].astype(float)
    df["n"] = df["instance"].str.split("-").str[-1].astype(int)
    g = df.groupby(["par", "val", "n"]).agg(best=("best", "min"), mean=("best", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    labels = {"gamma": ("(a) radius factor $\\gamma$", 1.0), "Ld": ("(b) drone endurance $L_d$ (km)", 1000.0), "Qd": ("(c) drone capacity $Q_d$ (units)", 1.0)}
    for ax, par in zip(axes, ["gamma", "Ld", "Qd"]):
        for n, mk in ((15, "o"), (30, "s")):
            s = g[(g["par"] == par) & (g["n"] == n)].sort_values("val")
            ax.plot(s["val"] / labels[par][1], s["best"], marker=mk, ms=4, lw=1.4, label=f"{n} customers")
        ax.set_xlabel(labels[par][0], fontsize=9); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("objective (CNY)", fontsize=9); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_sensitivity.pdf")); fig.savefig(os.path.join(FIG, "fig_sensitivity.png"), dpi=200); plt.close(fig)
    g.to_csv(os.path.join(RES, "summary_sens.csv"), index=False)
    for par in ["gamma", "Ld", "Qd"]:
        for n in (15, 30):
            s = g[(g["par"] == par) & (g["n"] == n)].sort_values("val")
            if s.empty:
                continue
            imin = s["best"].idxmin()
            num(f"sens_{par}_{n}_argmin", s.loc[imin, "val"] / labels[par][1], "{:g}")
            num(f"sens_{par}_{n}_min", s.loc[imin, "best"]); num(f"sens_{par}_{n}_first", s["best"].iloc[0]); num(f"sens_{par}_{n}_last", s["best"].iloc[-1])
            num(f"sens_{par}_{n}_firstval", s["val"].iloc[0] / labels[par][1], "{:g}"); num(f"sens_{par}_{n}_lastval", s["val"].iloc[-1] / labels[par][1], "{:g}")


# ----------------------------------------------------------------------------
# Operator usage statistics (sequence dependence, length adaptation)
# ----------------------------------------------------------------------------
def table_usage():
    p = os.path.join(RES, "usage.json")
    if not os.path.exists(p):
        write("tab_usage.tex", "% usage not available\n"); return
    data = json.load(open(p))
    rows = {}
    for key, v in data.items():
        inst, algo, seed = key.split("|")
        usage = v["usage"]; tot = sum(usage.values())
        if algo == "P":
            shares = {k: c / tot for k, c in usage.items()}
            rows.setdefault(inst, {}).setdefault("P_top", []).append(max(shares.values()))
            rows[inst].setdefault("P_topname", []).append(max(shares, key=shares.get))
        else:
            two = {k: c for k, c in usage.items() if "->" in k}; one = {k: c for k, c in usage.items() if "->" not in k}
            n2 = sum(two.values()); n1 = sum(one.values())
            rows.setdefault(inst, {}).setdefault("O_share2", []).append(n2 / tot)
            if n2:
                joint = {}
                for k, c in two.items():
                    a, b = k.split("->"); joint[(a, b)] = joint.get((a, b), 0) + c / n2
                m1 = {}; m2 = {}
                for (a, b), pr in joint.items():
                    m1[a] = m1.get(a, 0) + pr; m2[b] = m2.get(b, 0) + pr
                tv = 0.5 * sum(abs(joint.get((a, b), 0) - m1[a] * m2[b]) for a in m1 for b in m2)
                # null value: TV of n2 independent draws from the product of the marginals (sampling noise)
                rng = np.random.default_rng(12345)
                A = list(m1); Bk = list(m2); pa = np.array([m1[x] for x in A]); pb = np.array([m2[x] for x in Bk])
                nulls = []
                for _ in range(200):
                    xa = rng.choice(len(A), size=n2, p=pa / pa.sum()); xb = rng.choice(len(Bk), size=n2, p=pb / pb.sum())
                    J = np.zeros((len(A), len(Bk))); np.add.at(J, (xa, xb), 1); J /= n2
                    nulls.append(0.5 * np.abs(J - np.outer(J.sum(1), J.sum(0))).sum())
                rows[inst].setdefault("O_tvnull", []).append(float(np.mean(nulls)))
                rows[inst].setdefault("O_tvq95", []).append(float(np.quantile(nulls, 0.95)))
                rows[inst].setdefault("O_top2", []).append(max(joint.values())); rows[inst].setdefault("O_tv", []).append(tv)
                rows[inst].setdefault("O_top2name", []).append("->".join(max(joint, key=joint.get)))
            if n1:
                rows[inst].setdefault("O_top1", []).append(max(one.values()) / n1)
            wl = v["weights"].get("lengths", {})
            if wl:
                rows[inst].setdefault("O_wlen2", []).append(wl.get("2", 0) / (wl.get("1", 1e-9) + wl.get("2", 0)))
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Operator-usage statistics (two seeds, protocol E). ALNS-P: share of the most frequently selected destroy--repair pair (16 pairs; uniform selection gives 0.063). OSD-ALNS: share of iterations that apply a sequence of two pairs and share of the most frequent such sequence (256 sequences; uniform selection gives 0.004)}\label{tab:usage}",
             r"\begin{tabular}{lrrr}", r"\toprule", r"Instance & ALNS-P: top pair & OSD-ALNS: two-pair share & OSD-ALNS: top sequence \\", r"\midrule"]
    tvs = []; s2 = []; tvn = []
    for inst in rows:
        r = rows[inst]
        lines.append(f"{inst} & {np.mean(r.get('P_top',[np.nan])):.3f} & {np.mean(r.get('O_share2',[np.nan])):.2f} & {np.mean(r.get('O_top2',[np.nan])):.3f} \\\\")
        tvs += r.get("O_tv", []); s2 += r.get("O_share2", []); tvn += r.get("O_tvnull", [])
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_usage.tex", "\n".join(lines) + "\n")
    if tvs:
        num("usage_tv_mean", np.mean(tvs), "{:.3f}"); num("usage_tv_max", np.max(tvs), "{:.3f}")
        num("usage_tvnull_mean", np.mean(tvn), "{:.3f}")
        ptop = [x for r in rows.values() for x in r.get("P_top", [])]; otop = [x for r in rows.values() for x in r.get("O_top2", [])]
        num("usage_Ptop_min", min(ptop), "{:.3f}"); num("usage_Ptop_max", max(ptop), "{:.3f}")
        num("usage_Otop_min", min(otop), "{:.3f}"); num("usage_Otop_max", max(otop), "{:.3f}")
    if s2:
        num("usage_share2_mean", np.mean(s2), "{:.2f}"); num("usage_share2_min", np.min(s2), "{:.2f}"); num("usage_share2_max", np.max(s2), "{:.2f}")
    # most frequent pairs / sequences overall
    from collections import Counter
    cp = Counter(x for r in rows.values() for x in r.get("P_topname", [])); cs = Counter(x for r in rows.values() for x in r.get("O_top2name", []))
    if cp:
        num("usage_P_topname", cp.most_common(1)[0][0].replace("+", "$+$").replace("/m", ""))
    if cs:
        num("usage_O_topname", cs.most_common(1)[0][0].replace("+", "$+$").replace("/m", "").replace("->", "$\\to$"))
    with open(os.path.join(RES, "summary_usage.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}, f, indent=1)


def table_robust():
    """Robustness of the three layers to the cooling schedule / acceptance criterion."""
    dr = load("robust"); de = load("protoE")
    if dr is None or dr.empty or de is None:
        write("tab_robust.tex", "% robust not available\n"); return
    bk = best_known_values(de, load("protoT"))
    for i, v in dr.groupby("instance")["best"].min().items():
        bk[i] = min(bk.get(i, v), v)
    insts = [i for i in all_instance_names() if i in set(dr["instance"])]
    de2 = de[de["instance"].isin(insts)].copy(); de2["variant"] = "default"
    d = pd.concat([de2[["instance", "algo", "variant", "seed", "best"]], dr[["instance", "algo", "variant", "seed", "best"]]])
    d["gap"] = 100 * (d["best"] - d["instance"].map(bk)) / d["instance"].map(bk)
    g = d.groupby(["variant", "instance", "algo"])["gap"].mean().reset_index()
    settings = [("default", r"calibrated ($\alpha=0.99$)"), ("slow", r"slower cooling ($\alpha=0.995$)"), ("descent", r"descent ($T_0\approx0$)")]
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Robustness to the cooling schedule (protocol E, %d instances, five seeds; ten for the calibrated setting on the 200-customer instances): mean gap of the mean objective to the best known solution (\%%) by layer and setting, number of instances on which OSD-ALNS is better/worse than ALNS-I, and Wilcoxon $p$-value of OSD-ALNS vs.\ ALNS-I over the instances}\label{tab:robust}" % len(insts),
             r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Setting & ALNS-I & ALNS-P & OSD-ALNS & OSD better & OSD worse & Wilcoxon $p$ \\", r"\midrule"]
    allrows = []
    for key, lab in settings:
        sub = g[g["variant"] == key]
        if sub.empty:
            continue
        pv = sub.pivot(index="instance", columns="algo", values="gap")
        if not all(a in pv.columns for a in ALGOS):
            continue
        pv = pv.dropna()
        dd = (pv["OSD"] - pv["I"]).values
        better = int((dd < -1e-6).sum()); worse = int((dd > 1e-6).sum())
        p = stats.wilcoxon(pv["OSD"], pv["I"]).pvalue if (np.abs(dd) > 1e-9).sum() >= 5 else float("nan")
        lines.append(f"{lab} & {pv['I'].mean():.3f} & {pv['P'].mean():.3f} & {pv['OSD'].mean():.3f} & {better} & {worse} & {fmt(p,3)} \\\\")
        for a in ALGOS:
            num(f"rob_{key}_{a}", pv[a].mean(), "{:.3f}")
        num(f"rob_{key}_better", better, "{:d}"); num(f"rob_{key}_worse", worse, "{:d}"); num(f"rob_{key}_p", p, "{:.3f}")
        if key != "default":
            allrows.append(pv.assign(setting=key))
    if allrows:
        pooled = pd.concat(allrows)
        dd = (pooled["OSD"] - pooled["I"]).values
        p = stats.wilcoxon(pooled["OSD"], pooled["I"]).pvalue if (np.abs(dd) > 1e-9).sum() >= 5 else float("nan")
        lines.append(r"\midrule")
        lines.append(f"non-default settings pooled & {pooled['I'].mean():.3f} & {pooled['P'].mean():.3f} & {pooled['OSD'].mean():.3f} & {int((dd < -1e-6).sum())} & {int((dd > 1e-6).sum())} & {fmt(p,3)} \\\\")
        for a in ALGOS:
            num(f"rob_pooled_{a}", pooled[a].mean(), "{:.3f}")
        num("rob_pooled_better", int((dd < -1e-6).sum()), "{:d}"); num("rob_pooled_worse", int((dd > 1e-6).sum()), "{:d}"); num("rob_pooled_p", p, "{:.3f}")
        dp = (pooled["OSD"] - pooled["P"]).values
        num("rob_pooled_p_OSD_P", stats.wilcoxon(pooled["OSD"], pooled["P"]).pvalue if (np.abs(dp) > 1e-9).sum() >= 5 else float("nan"), "{:.3f}")
        # worst-case over settings per layer
        wc = pd.concat([g[g["variant"] == k] for k, _ in settings]).groupby(["instance", "algo"])["gap"].max().reset_index().pivot(index="instance", columns="algo", values="gap")
        for a in ALGOS:
            num(f"rob_worst_{a}", wc[a].mean(), "{:.3f}")
        lines.append(f"worst setting per instance & {wc['I'].mean():.3f} & {wc['P'].mean():.3f} & {wc['OSD'].mean():.3f} & & & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_robust.tex", "\n".join(lines) + "\n")
    num("rob_n", len(insts), "{:d}")


def table_qcap():
    dq = load("qcap"); de = load("protoE")
    if dq is None or dq.empty or de is None:
        write("tab_qcap.tex", "% qcap not available\n"); return
    insts = sorted(dq["instance"].unique(), key=lambda x: all_instance_names().index(x))
    seeds = sorted(dq["seed"].unique())
    e = de[de["instance"].isin(insts) & de["seed"].isin(seeds)]
    bk = pd.concat([de[de["instance"].isin(insts)], dq]).groupby("instance")["best"].min()
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Destroy size on the 300-customer instances (protocol E, %d seeds): mean gap of the mean objective to the best known solution (\%%) with the calibrated destroy size ($q\in[0.2R,0.4R]$, up to about 140 requests per operation) and with the destroy size capped at 60 requests per operation, and mean CPU time per run (s)}\label{tab:qcap}" % len(seeds),
             r"\begin{tabular}{l|rrr|rrr}", r"\toprule", r" & \multicolumn{3}{c|}{calibrated $q$} & \multicolumn{3}{c}{$q\leq60$} \\", r"Instance & ALNS-I & ALNS-P & OSD-ALNS & ALNS-I & ALNS-P & OSD-ALNS \\", r"\midrule"]
    acc = {("cal", a): [] for a in ALGOS}; acc.update({("cap", a): [] for a in ALGOS}); tm = {("cal", a): [] for a in ALGOS}; tm.update({("cap", a): [] for a in ALGOS})
    for i in insts:
        cells = [i]
        for key, d in (("cal", e), ("cap", dq)):
            for a in ALGOS:
                s = d[(d["instance"] == i) & (d["algo"] == a)]
                gv = 100 * (s["best"].mean() - bk[i]) / bk[i]
                cells.append(fmt(gv, 3)); acc[(key, a)].append(gv); tm[(key, a)].append(s["time"].mean())
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(["Average"] + [fmt(np.mean(acc[(k, a)]), 3) for k in ("cal", "cap") for a in ALGOS]) + r" \\")
    lines.append(" & ".join(["Time (s)"] + [fmt(np.mean(tm[(k, a)]), 0) for k in ("cal", "cap") for a in ALGOS]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_qcap.tex", "\n".join(lines) + "\n")
    for k in ("cal", "cap"):
        for a in ALGOS:
            num(f"qcap_{k}_{a}", np.mean(acc[(k, a)]), "{:.3f}"); num(f"qcap_{k}_{a}_time", np.mean(tm[(k, a)]), "{:.0f}")
    num("qcap_seeds", len(seeds), "{:d}")


def write_numbers():
    lines = ["% auto-generated by analysis/make_tables.py -- do not edit"]
    for k, v in NUMBERS.items():
        key = "".join(ch for ch in k if ch.isalnum())
        key = key.translate(str.maketrans("0123456789", "ABCDEFGHIJ"))
        lines.append(f"\\newcommand{{\\n{key}}}{{{v}}}")
    write("numbers.tex", "\n".join(lines) + "\n")
    with open(os.path.join(RES, "numbers.json"), "w") as f:
        json.dump(NUMBERS, f, indent=1)


if __name__ == "__main__":
    table_instances(); table_vi(); table_milp(); table_calib(); table_protoE(); fig_convergence(); table_robust(); table_qcap(); table_len(); table_ablation(); table_protoT(); fig_sensitivity(); table_usage()
    write_numbers()
    print("done;", len(NUMBERS), "numbers")
