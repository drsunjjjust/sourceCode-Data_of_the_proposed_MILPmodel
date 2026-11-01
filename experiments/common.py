"""Shared helpers for the experiment scripts."""
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from mvmd.instance import Instance, make_solomon, make_toy, SCENARIOS
from mvmd.construct import twcc_grh, random_initial
from mvmd.alns import ALNS
from mvmd.vns import VNS
from mvmd.ga import GA
from mvmd.solution import Solution

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INST_DIR = os.path.join(ROOT, "data", "instances")
RES_DIR = os.path.join(ROOT, "results")
TRACE_DIR = os.path.join(RES_DIR, "traces")
os.makedirs(RES_DIR, exist_ok=True); os.makedirs(TRACE_DIR, exist_ok=True)

SCALES = [15, 30, 50, 75, 100]
# elementary-operation budgets of protocol E (equal number of destroy-repair operations)
BUDGET = {"toy": 2000, 15: 2000, 30: 3000, 50: 4000, 75: 5000, 100: 6000, 200: 8000, 300: 10000}
LARGE = ["A-1-M-200", "B-1-M-200", "C-1-M-200", "A-1-M-300", "B-1-M-300", "C-1-M-300"]


GEOMETRIES = ("C", "R", "RC")
RC_SCALES = [50, 75, 100]


def instance_names(scales=("toy",) + tuple(SCALES), scenarios=tuple(SCENARIOS), geometries=GEOMETRIES):
    """toy and C101-based instances of every scale, R101/RC101-based instances of 50/75/100 customers"""
    names = []
    for sc in scenarios:
        for n in scales:
            if n == "toy":
                if "C" in geometries:
                    names.append(f"toy-{sc}")
                continue
            for g in geometries:
                if g == "C":
                    names.append(f"{sc}-{n}")
                elif n in RC_SCALES:
                    names.append(f"{sc}-{g}-{n}")
    return names


def scale_of(name: str):
    return "toy" if name.startswith("toy") else int(name.split("-")[-1])


def geometry_of(name: str) -> str:
    if name.startswith("toy"):
        return "toy"
    if name.startswith("sens"):
        return "C"
    parts = name.split("-")
    return parts[2] if len(parts) == 4 else "C"


def seeds_of(name: str) -> int:
    """ten seeds on the toy, 15- and 30-customer C101 instances (exact comparison) and on the
    200-customer instances (largest run-to-run variance), five seeds otherwise"""
    g = geometry_of(name); n = scale_of(name)
    if g == "toy" or (g == "C" and n <= 30) or (g == "M" and n == 200):
        return 10
    return 5


def all_instance_names():
    return instance_names() + LARGE


def budget_of(name: str) -> int:
    return BUDGET[scale_of(name)]


_inst_cache = {}


def load_instance(name: str) -> Instance:
    if name not in _inst_cache:
        _inst_cache[name] = Instance.load(os.path.join(INST_DIR, f"{name}.json"))
    return _inst_cache[name]


_init_cache = {}


def initial_solution(inst: Instance, kind: str = "twcc", seed: int = 0) -> Solution:
    key = (inst.name, kind, seed if kind != "twcc" else 0)
    if key not in _init_cache:
        _init_cache[key] = twcc_grh(inst) if kind == "twcc" else random_initial(inst, seed)
    return _init_cache[key].copy()


def run_algorithm(inst: Instance, algo: str, seed: int, budget_ops=None, time_limit=None, params=None, init_kind="twcc",
                  Len=2, accept_each=False, max_no_improve=None):
    """algo in {'I','P','OSD','VNS','GA'}. Returns RunResult."""
    init = initial_solution(inst, init_kind, seed)
    if algo in ("I", "P", "OSD"):
        params = dict(params or {})
        hierarchical = params.pop("hierarchical", True)
        prefix_best = params.pop("prefix_best", False)
        a = ALNS(inst, init, mode=algo, Len=Len, seed=seed, params=params, accept_each=accept_each, hierarchical=hierarchical,
                 prefix_best=prefix_best)
        return a.run(budget_ops=budget_ops, time_limit=time_limit, max_no_improve=max_no_improve)
    if algo == "VNS":
        return VNS(inst, init, seed=seed).run(time_limit=time_limit)
    if algo == "GA":
        return GA(inst, init, seed=seed).run(time_limit=time_limit)
    raise ValueError(algo)


class CsvWriter:
    """Append rows to a CSV; remembers keys already present to allow resuming."""

    def __init__(self, path: str, fields, key_fields):
        self.path = path; self.fields = list(fields); self.key_fields = list(key_fields)
        self.done = set()
        if os.path.exists(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    self.done.add(tuple(row[k] for k in self.key_fields))
        else:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def has(self, key) -> bool:
        return tuple(str(k) for k in key) in self.done

    def write(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow({k: row.get(k, "") for k in self.fields})
        self.done.add(tuple(str(row[k]) for k in self.key_fields))


def save_trace(tag: str, trace):
    with open(os.path.join(TRACE_DIR, tag + ".json"), "w") as f:
        json.dump([[int(o), float(t), float(c)] for o, t, c in trace], f)


def best_at(trace, ops=None, t=None, init=None):
    v = init if init is not None else trace[0][2]
    for o, tt, c in trace:
        if (ops is not None and o <= ops) or (t is not None and tt <= t):
            v = c
    return v
