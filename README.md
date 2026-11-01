# Multi-Vehicle Multi-Drone Delivery with Independent and Truck-Supported Drone Operations

Companion code and data of the manuscript

> *Multi-Vehicle Multi-Drone Delivery with Independent and Truck-Supported Drone Operations: A Strengthened Formulation and a Sequence-Aware Adaptive Large Neighborhood Search* (Sun, Ye, Wang, Zhang)

Everything reported in the paper — instances, exact model, heuristics, experiments, statistical tests, tables and figures — is produced by the scripts in this repository.

> **Repository note.** To keep the number of files small, the benchmark instances, the search traces and the
> solver logs are stored as zip archives. Unzip them in place before running any script:
> `data/instances.zip` → `data/instances/`, `results/traces.zip` → `results/traces/`,
> `results/milp_logs.zip` → `results/milp_logs/` (e.g. `cd data && unzip instances.zip`).

## 1. Installation

```bash
pip install numpy scipy pandas matplotlib highspy numba
# optional: pip install gurobipy   (a licence is needed for instances larger than the trial size)
python -m pytest tests/ -q      # kernel tests (evaluation = LP, insertion kernels = re-evaluation)
```

Python ≥ 3.10. `numba` is optional (the code runs without it, ~20× slower).

## 2. Layout

| Path | Content | Manuscript |
|---|---|---|
| `mvmd/instance.py` | instance class, toy network, Solomon C101 / R101 / RC101 instances, 200/300-customer merged instances, capacity- and distance-constrained k-means clustering, JSON I/O | Sec. 3.1, 6.1 |
| `mvmd/solution.py` | solution representation, exact evaluation (PAV optimal waiting), insertion kernels | Sec. 5.2, Prop. 7 |
| `mvmd/construct.py` | TWCC-GRH two-stage constructive heuristic | Sec. 5.1, Alg. 1 |
| `mvmd/operators.py` | random, Shaw, cost-based and route-based removal; greedy, noisy greedy, regret-2 and regret-3 insertion (two-level requests) | Sec. 5.3 |
| `mvmd/alns.py` | ALNS-I (independent weights), ALNS-P (pair weights), OSD-ALNS (variable-length sequences, hierarchical sequence weights, look-ahead acceptance), restart rule, effort accounting, traces | Sec. 5.4, Alg. 2 |
| `mvmd/vns.py`, `mvmd/ga.py` | VNS and GA baselines (same representation and evaluation) | Sec. 6.7 |
| `mvmd/milp.py` | arc-based MILP (two sub-models), valid inequalities, big-M tightening, symmetry breaking; HiGHS / Gurobi backends | Sec. 4, 4.5 |
| `data/raw/solomon_{c101,r101,rc101}.txt` | Solomon C101, R101 and RC101 data | Sec. 6.1 |
| `data/instances/*.json` | the 78 benchmark instances (+ sensitivity instances) | Sec. 6.1 |
| `experiments/run_heuristics.py` | calibration, protocol E (equal operations), sequence-space study, component ablation, protocol T (equal time), sensitivity; additional runs (cooling schedule, destroy-size cap) | Sec. 6.3–6.8 |
| `experiments/pipeline_v2.sh`, `experiments/ensure.sh` | the complete (resumable) heuristic study in the order of the paper | Sec. 6 |
| `experiments/run_usage.py` | operator-usage statistics of ALNS-P / OSD-ALNS | Sec. 6.6 |
| `experiments/run_milp.py` | exact runs and the valid-inequality study | Sec. 6.2 |
| `analysis/*.py` | tables, statistical tests, convergence profiles, figures, `paper/numbers.tex` | Sec. 6 |
| `results/` | raw results (CSV) and traces (JSON) of the reported runs | — |
| `paper/` | generated tables (`tab_*.tex`), result macros (`numbers.tex`) and figures (created by `analysis/`) | — |

## 3. Reproducing the paper

```bash
python -m mvmd.instance data/instances                 # regenerate the 78 instances (deterministic, seed 2026)
experiments/pipeline_v2.sh                             # complete heuristic study (about 15 h on two cores), or step by step:
python experiments/run_heuristics.py calib             # parameter calibration            -> results/calib.csv
python analysis/calib.py                               # -> results/calibrated_params.json
python experiments/run_heuristics.py protoE   --params results/calibrated_params.json
python experiments/run_heuristics.py robust   --params results/calibrated_params.json
python experiments/run_heuristics.py ablation --params results/calibrated_params.json
python experiments/run_heuristics.py len      --params results/calibrated_params.json
python experiments/run_usage.py --seeds 2     --params results/calibrated_params.json
python experiments/run_heuristics.py protoT            # VNS / GA with the OSD-ALNS time of protocol E
python experiments/run_heuristics.py sens     --params results/calibrated_params.json
python experiments/run_heuristics.py qcap     --params results/calibrated_params.json
python experiments/run_milp.py main --backend highs --time 600    # or --backend gurobi
python experiments/run_milp.py vi   --backend highs --time 300
python experiments/run_milp.py lp   --backend highs
python analysis/make_tables.py                         # all tables, tests, figures, numbers.tex
python analysis/write_summaries.py                     # result-dependent prose of Section 6 (summaries.tex)
python analysis/make_figures_springer.py               # Figs. 5-8 as vector PDF/EPS in the journal format
```

All experiment scripts are resumable (finished runs are skipped).

## 4. Mapping between the manuscript and the code

### 4.1 Model (`mvmd/milp.py`)

Constraint names in the solver carry the equation numbers of the manuscript:

| Manuscript | Solver row name | Note |
|---|---|---|
| (5) cluster visited once | `C5_c*` | |
| (6) customer served once (truck-supported) | `C6_s*` | |
| (7) customer served once (independent) | `C7_s*` | independent sub-model |
| (8), (9) truck leaves O / enters D | `C8_k*`, `C9_k*` | |
| (10), (11) truck flow at c−, c+ | `C10_*`, `C11_*` | |
| (12), (13), (14) drone flow at c−, c+, customers | `C12_*`, `C13_*`, `C14_*` | |
| (15), (16), (17) independent drone flow | `C15_*`, `C16_*`, `C17_*` | |
| (18) aggregated truck–drone link | `C18_*` | replaced by `VI_disagg_*` when `disagg=True` (Prop. 4) |
| (19)–(21) distance limits | `C19_*`, `C20_*`, `C21_*` | |
| (22)–(24) capacity limits | `C22_*`, `C23_*`, `C24_*` | |
| (25)–(27) MTZ | `C25_*`, `C26_*`, `C27_*` | lifted versions `C25L_*`, `C26L_*`, `C27L_*` when `lifted_mtz=True` (Prop. 3) |
| (28), (29) drone time propagation (two-sided) | `C28_*` (≥), `C28b_*` (≤); `C29_*`, `C29b_*` | |
| (30) waiting ≥ sortie duration | `C30_*` | |
| (31) drone staying on the truck | `C28_*_stay` | |
| (32) truck time propagation | `C32_*` | |
| (33), (34) launch synchronisation | `C33_*`, `C34_*` | |
| (40)–(43) earliness / tardiness linearisation | `C40_*`, `C41_*`, `C42_*`, `C43_*` | |
| (44)–(46) rounded fleet-size inequalities | `VI_fleet`, `VI_sorties_c*`, `VI_trips` | Prop. 1 |
| (47) two-cycle inequalities | `VI_2cyc_*` | Prop. 2 |
| (48)–(50) lifted MTZ | `C25L_*`, `C26L_*`, `C27L_*` | Prop. 3 |
| (51) disaggregated truck–drone linking | `VI_disagg_*` | Prop. 4 |
| (52)–(54) symmetry breaking | `VI_symT_*`, `VI_symD_*`, `VI_symI_*` | Prop. 6 |
| big-M values | `_horizon()`, `Wmax` | Prop. 5 |
| arc elimination | `strengthen['prep']` | Sec. 4.5 |

Variables: `x_k_i_j`, `y_k_l_i_j`, `z_l_i_j`, `t_k_i`, `that_k_l_i`, `tcheck_l_i`, `u_*`, `uhat_*`, `ucheck_*`, `w_k_c`, `pA_*`, `pB_*`; dummy nodes are named `cm<c>` (c−) and `cp<c>` (c+).

### 4.2 Heuristics

| Manuscript | Code |
|---|---|
| Algorithm 1 (TWCC-GRH) | `construct.twcc_grh` |
| Proposition 7 (optimal waiting, PAV) | `solution._truck_penalty_pav`, `solution.truck_route_cost` |
| random / Shaw / cost-based / route-based removal | `operators.random_removal`, `shaw_removal`, `cost_removal`, `segment_removal` |
| greedy / noisy greedy / regret-2 / regret-3 insertion | `operators.repair(mode='greedy'|'noise'|'regret2'|'regret3')` |
| ALNS-I / ALNS-P / OSD-ALNS, Eq. (reward), Eq. (hierarchical weights), Algorithm 2 | `alns.ALNS(mode='I'|'P'|'OSD', Len=L_max)` |
| flat sequence weights / fixed length (ablation) | `ALNS(..., hierarchical=False)`, `params={'variable_len': False}` |
| accept-each variant (Lemma 2) | `ALNS(..., accept_each=True)` |
| no-adaptation / no-restart variants | `params={'rho': 0.0}`, `params={'restart_ops': 0}` |
| destroy-size cap | `params={'q_cap': 60}` |
| elementary operations / traces | `RunResult.ops`, `RunResult.trace` |
| VNS, GA | `vns.VNS`, `ga.GA` |

### 4.3 Instances (`data/instances/*.json`)

`toy-<scenario>.json` (14 customers), `<scenario>-<n>.json` (first *n* customers of Solomon C101, coordinates ×100 m, demand = min(4, ⌊d/10⌋), time windows ×10 s), `<scenario>-R-<n>.json` and `<scenario>-RC-<n>.json` (first *n* customers of R101 / RC101, demand = max(1, min(4, ⌊d/10⌋)), time windows ×50 s) and `<scenario>-M-200/300.json` (union of the C101 and R101, resp. C101, R101 and RC101 customer sets). Scenarios: A-1 (γ=0.2), A-2 (γ=0.4), B-1 (L_d=8 km), B-2 (L_d=12 km), C-1 (Q_d=4), C-2 (Q_d=6); other parameters at their baseline values (Table 4 of the paper). Each file contains the depot, the customers (coordinates, demand, window), the sets S0/S1, the clusters (centre, members) and the fleet bounds.

## 5. 中文说明

* 论文中的每一个表格和图都由 `analysis/make_tables.py` 从 `results/` 下的原始结果生成；
* `mvmd/milp.py` 中约束的名称即论文公式编号（见上表），审稿人可逐条核对；
* OSD-ALNS 包含 4×4 算子池、可变长度序列、分层序列权重、前瞻接受和重启规则；基准包括 C101/R101/RC101 及 200/300 客户实例；`experiments/pipeline_v2.sh` 可按论文顺序一次跑完全部启发式实验（可断点续跑）；
* 启发式的目标值与 MILP 完全一致（`tests/test_evaluation.py` 用 LP 验证了最优等待策略的 PAV 算法）；
* 精确求解默认使用开源 HiGHS；安装 gurobipy 并加 `--backend gurobi` 即可用 Gurobi 重跑。
