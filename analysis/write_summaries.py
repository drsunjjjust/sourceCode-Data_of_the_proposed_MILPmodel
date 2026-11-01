"""Write paper/summaries.tex: result-dependent prose that references the macros of numbers.tex."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N = json.load(open(os.path.join(ROOT, "results", "numbers.json")))


def m(k):
    if k not in N:
        raise KeyError(k)
    key = "".join(ch for ch in k if ch.isalnum()).translate(str.maketrans("0123456789", "ABCDEFGHIJ"))
    return "\\n" + key + "{}"


def v(k):
    return N[k]


S = {}
S["calibparams"] = "\\ncalibparams"

S["viSummary"] = (r"The LP relaxation of the base model is weak: its root gap averages " + m("vi_toy_base_rootgap") + r"\% on the toy instances and " + m("vi_15_base_rootgap") + r"\% on the 15-customer instances (on the pure independent-drone instance A-2-15 the LP bound is essentially zero, because the fixed trip costs can be spread over fractional departures of many identical drones). The strengthened model reduces the root gap to " + m("vi_toy_all_rootgap") + r"\% and " + m("vi_15_all_rootgap") + r"\%, and the effect on the search is correspondingly large: on the toy instances the average branch-and-bound tree shrinks from " + m("vi_toy_base_nodes") + " to " + m("vi_toy_all_nodes") + r" nodes and the time from " + m("vi_toy_base_time") + " to " + m("vi_toy_all_time") + r"~s, and on the 15-customer instances the base model solves none of the six instances within $2\times300$~s (final gaps between $8\%$ and $17\%$, $98\%$ on A-2-15) whereas the strengthened model proves optimality on four of them in 13--237~s. The component analysis identifies the rounded fleet-size inequalities of Proposition~\ref{prop:round} as the decisive ingredient: added alone they reduce the root gap to " + m("vi_toy_rounding_rootgap") + r"\% and " + m("vi_15_rounding_rootgap") + r"\%, because the relaxation otherwise ``shares'' trucks, sorties and trips fractionally. The symmetry-breaking constraints of Proposition~\ref{prop:sym} also tighten the root bound (" + m("vi_toy_symmetry_rootgap") + r"\% and " + m("vi_15_symmetry_rootgap") + r"\%), since the lexicographic constraints cut off fractional solutions that spread a route over several identical vehicles, and they act above all on the tree: on toy-A-1 they alone reduce the search from 836 to 6 nodes. Arc elimination improves the bound marginally, whereas the two-cycle inequalities, the lifted MTZ constraints and the disaggregated linking inequalities leave the LP bound of these instances unchanged (the fleet-size relaxation dominates the subtour and linking structure at the root) and contribute, if at all, inside the tree (on toy-A-1 the disaggregated linking alone reduces the tree from 836 to 137 nodes, whereas the two-cycle inequalities alone slow the solver on toy-A-2); their value appears in combination, where the strengthened model is the fastest configuration on every instance.")

S["milpSummary"] = (r"The strengthened model proves optimality on " + m("milp_n_opt") + r" of the 18 instances: on all six toy instances within $1.2$~s and on four of the six 15-customer instances within 15--250~s. The two remaining 15-customer instances reach the time limit with final gaps of $5.0\%$ (A-2-15, a pure independent-drone instance with seven trips) and $0.7\%$ (B-2-15), and none of the 30-customer instances is solved to optimality within $2\times600$~s (gaps between $2.3\%$ and $11.9\%$). On every instance with a proven optimum the mean objective of OSD-ALNS---and, by Table~\ref{tab:protoE}, of ALNS-I and ALNS-P---equals the optimum in every run, and on the 30-customer instances and on A-2-15 the metaheuristics return better solutions than the incumbent of the solver (by up to $10.3\%$ on A-2-30), which is why the reference values of Section~\ref{sec:protoE} use the best known solutions whenever the solver is not conclusive. The difficulty of the exact model is concentrated in the independent-drone sub-model, whose set-partitioning-like structure with many identical trips keeps large symmetric search trees even after symmetry breaking; a commercial solver or a branch-and-price approach would push the frontier, but the qualitative picture---exact solutions for up to 15 customers, heuristic reference values beyond---is what the metaheuristic study requires.")

S["calibSummary"] = (r"The search is insensitive to most parameters: nine of the fifteen settings lie within $0.015\%$ of the best mean objective. The calibration is decided on instance A-1-R-50, which has deep local optima ($968.25$ and $973.27$ against $947.41$~CNY, with one additional independent trip): settings that keep the search diversified---a hotter start, faster restarts, larger destroy sizes, a slower reaction of the weights---escape them in all five runs, whereas the default setting is trapped in one run and the variant without restarts in three. The largest mean deviation (" + m("calib_max_dev") + r"\%) is that of the search without restarts. Each factor was set to its best level when this improved on the default by more than $0.02$ percentage points, which gives the calibrated setting " + "\\ncalibparams" + r"; it is used for all three layers in the remaining experiments.")

S["protoESummary"] = (r"All three layers return the exact optimum in every run on the toy and the 15-customer instances with a proven optimum, and on " + m("pE_OSD_within01") + r" of the " + m("pE_n") + r" instances the mean objectives of OSD-ALNS and ALNS-I differ by less than $0.1\%$. Averaged over all instances OSD-ALNS has the smallest gap to the reference values (" + m("pE_OSD_gap") + r"\% against " + m("pE_I_gap") + r"\% for ALNS-I and " + m("pE_P_gap") + r"\% for ALNS-P). Its advantage is most pronounced on the clustered C101-based and the random R101-based instances, where its mean gap is " + m("pE_OSD_gap_famC") + r"\% and " + m("pE_OSD_gap_famR") + r"\% against " + m("pE_I_gap_famC") + r"\% and " + m("pE_I_gap_famR") + r"\% for ALNS-I and " + m("pE_P_gap_famC") + r"\% and " + m("pE_P_gap_famR") + r"\% for ALNS-P: on the 75-customer C101 instances it reduces the gap of ALNS-I from " + m("pE_I_gap_C75") + r"\% to " + m("pE_OSD_gap_C75") + r"\%, on the 100-customer R101 instances from " + m("pE_I_gap_R100") + r"\% to " + m("pE_OSD_gap_R100") + r"\%, and the largest single improvement is " + f"{abs(float(v('pE_OSDvsI_extreme'))):.3f}" + r"\% on " + m("pE_OSDvsI_extreme_inst") + r", the instance with the largest independent-drone component of its size. On the mixed RC101-based and on the 300-customer instances the three layers are close, with small differences in favour of the conventional layers. These gains come at no extra cost: the mean time per elementary operation is " + m("pE_I_optime") + ", " + m("pE_P_optime") + " and " + m("pE_OSD_optime") + r"~ms for ALNS-I, ALNS-P and OSD-ALNS (Proposition~\ref{prop:effort}).")

S["testsSummary"] = (r"Table~\ref{tab:tests} reports the nonparametric tests on the instance-wise means. The differences between the layers are small compared with the run-to-run variation, so that they are not significant at the $5\%$ level (Friedman $p=" + m("pE_friedman_p_all") + r"$). A non-inferiority analysis makes the comparison precise: the $95\%$ bootstrap confidence interval of the mean gap difference between OSD-ALNS and ALNS-I is $[" + m("ni_I_lo") + ", " + m("ni_I_hi") + r"]$ percentage points (against ALNS-P: $[" + m("ni_P_lo") + ", " + m("ni_P_hi") + r"]$), and the one-sided Wilcoxon test rejects that OSD-ALNS is worse than ALNS-I by $0.02$ percentage points or more ($" + v("ni_I_pA_txt") + r"$; against ALNS-P, $" + v("ni_P_pA_txt") + r"$). OSD-ALNS is therefore non-inferior to conventional ALNS at identical cost per operation, with the lowest average gap and clear gains on the clustered and random instances.")

S["convSummary"] = (r"During the first few hundred operations the three curves coincide: the early, steep descent is driven by the operators. The layers separate in the second half of the budget. On the 100-customer C101 and R101 instances OSD-ALNS keeps improving after the conventional layers have stalled and ends at " + m("conv_OSD_C100_final") + r"\% and " + m("conv_OSD_R100_final") + r"\% above the best known solution, against " + m("conv_I_C100_final") + r"\% and " + m("conv_I_R100_final") + r"\% for ALNS-I and " + m("conv_P_C100_final") + r"\% and " + m("conv_P_R100_final") + r"\% for ALNS-P; on the R101 instances it reaches the $0.1\%$ level after " + m("conv_OSD_R100_ops01") + r" operations, against " + m("conv_I_R100_ops01") + " and " + m("conv_P_R100_ops01") + r". Table~\ref{tab:efficiency} confirms this picture on the C101 family, where OSD-ALNS has the highest hit rate (" + m("eff_OSD_hit_famC") + r" against " + m("eff_I_hit_famC") + r" and " + m("eff_P_hit_famC") + r") and needs the fewest operations to reach the target (" + m("eff_OSD_ops_famC") + r" against " + m("eff_I_ops_famC") + r" and " + m("eff_P_ops_famC") + r"); on the RC101 and the large instances the conventional layers reach the target slightly earlier, because a sequence of two pairs must be completed before its outcome is accepted.")


for k in ("ablationSummary", "lenSummary", "usageSummary", "protoTSummary", "sensSummary"):
    S.setdefault(k, r"\hl{[results pending]}")

extra = os.path.join(ROOT, "analysis", "summaries_extra.py")
if os.path.exists(extra):
    exec(open(extra, encoding="utf-8").read())

for k in ("conclVI", "conclAlgo", "conclMgmt"):
    S.setdefault(k, r"\hl{[pending]}")

with open(os.path.join(ROOT, "paper", "summaries.tex"), "w", encoding="utf-8") as f:
    f.write("% result-dependent prose, generated by analysis/write_summaries.py from results/numbers.json\n")
    for k, t in S.items():
        if k == "calibparams":
            f.write("\\newcommand{\\calibparams}{\\ncalibparams}\n")
        else:
            f.write("\\newcommand{\\" + k + "}{" + t + "}\n")
print("written", len(S), "summaries")
