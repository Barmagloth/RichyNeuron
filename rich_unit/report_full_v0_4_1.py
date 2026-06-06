"""FULL ablation report (v0.4.1) — curves, params@Q, SYN(Q), reliability, verdict.

Reads results/full_ablation.csv. Computes, per task: param->quality curves for the
4 points (lr* by median val), Q grid calibrated on `neither`, params@Q (log-linear
interpolation), log2 savings, SYN(Q) = save(both)-[save(linear)+save(gate)], and the
§2.4 verdict. Also: reliability per point, A1 (W_h=0) on gate/both, B2 anchor, and
the pilot-2 reproducibility check (gate@d64 ~0.80, A1 Δ~0.67).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "full_ablation.csv"
SUMMARY_PATH = RESULTS_DIR / "full_summary.json"

SYN_MIN = 0.3
RELIABILITY_BAR = 0.7
Q_CANDIDATES = [0.6, 0.7, 0.8, 0.9]
D_MODELS = [16, 32, 48, 64]
D_STATE = 16
POINTS = ["neither", "linear", "gate", "both"]
TASKS = ["sc", "ar"]
med = statistics.median


def load():
    with open(CSV_PATH, newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def _cell(rows, point, task, dm, ablate=0):
    """All rows for one (point,task,dm,d_state=16,ablate) cell, grouped by lr."""
    by_lr = {}
    for r in rows:
        if (r["point"], r["task"], int(r["d_model"]), int(r["d_state"]), int(r["ablate"])) \
           == (point, task, dm, D_STATE, ablate):
            by_lr.setdefault(float(r["lr"]), []).append(r)
    return by_lr


def cell_stats(rows, point, task, dm, ablate=0):
    by_lr = _cell(rows, point, task, dm, ablate)
    if not by_lr:
        return None
    lr_star = max(by_lr, key=lambda lr: med(float(r["best_val"]) for r in by_lr[lr]))
    rs = by_lr[lr_star]
    tests = [float(r["test_at_best"]) for r in rs]
    return {
        "lr": lr_star, "params": int(rs[0]["params"]),
        "median_test": med(tests), "tests": sorted(tests),
        "reliability": sum(1 for t in tests if t >= RELIABILITY_BAR) / len(tests),
    }


def curve(rows, point, task):
    out = []
    for dm in D_MODELS:
        c = cell_stats(rows, point, task, dm)
        if c:
            out.append((c["params"], c["median_test"], c["reliability"], dm, c["lr"]))
    return sorted(out, key=lambda x: x[0])


def params_at_Q(crv, Q):
    """Min params reaching median_test >= Q (log-linear interpolation). None if never."""
    pts = [(p, a) for (p, a, *_ ) in crv]
    # direct hit at the smallest width already >= Q
    for i, (p, a) in enumerate(pts):
        if a >= Q:
            if i == 0:
                return float(p)
            p0, a0 = pts[i - 1]
            if a == a0:
                return float(p)
            frac = (Q - a0) / (a - a0)
            lg = math.log2(p0) + frac * (math.log2(p) - math.log2(p0))
            return 2.0 ** lg
    return None


def q_grid(rows, task):
    """Calibrate Q on `neither`: candidates within its achievable median-test range."""
    crv = curve(rows, "neither", task)
    accs = [a for (_, a, *_ ) in crv]
    lo, hi = min(accs), max(accs)
    return [q for q in Q_CANDIDATES if lo + 0.03 <= q <= hi - 0.03], (lo, hi)


def synergy(rows, task):
    crv = {p: curve(rows, p, task) for p in POINTS}
    qs, neither_range = q_grid(rows, task)
    rows_out = []
    syns = []
    for Q in qs:
        pq = {p: params_at_Q(crv[p], Q) for p in POINTS}
        if any(pq[p] is None for p in POINTS):
            rows_out.append({"Q": Q, "achievable": False, "params_at_Q": pq})
            continue
        save = {p: math.log2(pq["neither"]) - math.log2(pq[p]) for p in POINTS}
        syn = save["both"] - (save["linear"] + save["gate"])
        syns.append(syn)
        rows_out.append({"Q": Q, "achievable": True,
                         "params_at_Q": {p: round(pq[p], 1) for p in POINTS},
                         "save": {p: round(save[p], 3) for p in POINTS},
                         "SYN": round(syn, 3)})
    return {"neither_range": neither_range, "q_levels": qs, "rows": rows_out,
            "median_SYN": (round(med(syns), 3) if syns else None),
            "n_syn_ge_min": sum(1 for s in syns if s >= SYN_MIN),
            "n_achievable": len(syns)}


def task_verdict(syn):
    if not syn["n_achievable"]:
        return "NO ACHIEVABLE Q (curves too low/high) — uninterpretable"
    m = syn["median_SYN"]
    half = syn["n_syn_ge_min"] >= math.ceil(syn["n_achievable"] / 2)
    if m >= SYN_MIN and half:
        return "synergy"
    if m <= -SYN_MIN:
        return "anti-synergy"
    return "additive (H0 holds)"


def a1_block(rows):
    out = {}
    for point in ("gate", "both"):
        for task in TASKS:
            normal = cell_stats(rows, point, task, 64, ablate=0)
            abl = cell_stats(rows, point, task, 64, ablate=1)
            if normal and abl:
                out[f"{point}/{task}"] = {
                    "normal_median": round(normal["median_test"], 3),
                    "ablated_median": round(abl["median_test"], 3),
                    "delta": round(normal["median_test"] - abl["median_test"], 3),
                }
    return out


def main():
    rows = load()
    summary = {"spec": "FULL_ABLATION_SPEC_v0_4_1 (A-fix-off)", "SYN_MIN": SYN_MIN,
               "tasks": {}, "verdict": None}

    for task in TASKS:
        syn = synergy(rows, task)
        curves = {p: [(p_, round(a, 3), round(rel, 2)) for (p_, a, rel, *_) in curve(rows, p, task)]
                  for p in POINTS}
        reliab = {p: [round(c[2], 2) for c in curve(rows, p, task)] for p in POINTS}
        summary["tasks"][task] = {"curves": curves, "reliability": reliab,
                                  "synergy": syn, "task_verdict": task_verdict(syn)}

    # overall verdict (§2.4): synergy only if BOTH tasks show synergy
    tvs = [summary["tasks"][t]["task_verdict"] for t in TASKS]
    if any(v.startswith("NO ACHIEVABLE") for v in tvs):
        summary["verdict"] = "UNINTERPRETABLE: >=1 task has no achievable Q band"
    elif all(v == "synergy" for v in tvs):
        summary["verdict"] = "H0 REFUTED: synergy on both tasks"
    elif any(v == "synergy" for v in tvs):
        summary["verdict"] = "PARTIAL (task-dependent) — not a win"
    elif any(v == "anti-synergy" for v in tvs):
        summary["verdict"] = "ANTI-SYNERGY on >=1 task"
    else:
        summary["verdict"] = "H0 HOLDS: axes additive (negative result)"

    summary["A1"] = a1_block(rows)
    # pilot-2 reproducibility anchor: gate@sc d64
    g = cell_stats(rows, "gate", "sc", 64)
    summary["pilot2_reproduction"] = {
        "gate_sc_d64_median_test": round(g["median_test"], 3) if g else None,
        "expected": "~0.80 test, A1 delta ~0.67 (pilot 2)",
        "A1_delta_gate_sc": summary["A1"].get("gate/sc", {}).get("delta"),
    }
    # B2 anchor
    summary["anchor_B2"] = {}
    for task in TASKS:
        for dm in (32, 64):
            c = cell_stats(rows, "anchor", task, dm)
            if c:
                summary["anchor_B2"][f"{task}/d{dm}"] = round(c["median_test"], 3)
    # alpha collapse check (fixed alpha)
    a_means = [float(r["alpha_mean"]) for r in rows if r["point"] != "anchor"]
    summary["alpha_fixed_mean_range"] = [round(min(a_means), 3), round(max(a_means), 3)] if a_means else None

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2))
    for task in TASKS:
        t = summary["tasks"][task]
        print(f"\n== {task}: {t['task_verdict']} | median SYN={t['synergy']['median_SYN']} "
              f"over Q={t['synergy']['q_levels']} ==")
        for r in t["synergy"]["rows"]:
            if r.get("achievable"):
                print(f"   Q={r['Q']}: SYN={r['SYN']} save={r['save']}")
    print(f"\nVERDICT: {summary['verdict']}")


if __name__ == "__main__":
    main()
