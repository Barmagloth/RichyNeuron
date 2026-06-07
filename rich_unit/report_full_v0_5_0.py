"""FULL ablation report (v0.5.0, A-fix-off) — Pareto-dominance margin verdict.

Reads results/full_ablation.csv. Primary criterion (spec v0.5.0, PREREG_FULL):
    save(X)   = -log2(params@Q[X])
    margin(Q) = save(both) - max(save(linear), save(gate))
              = log2(params@Q[best_single]) - log2(params@Q[both])
margin >= MARGIN_MIN  =>  `both` Pareto-dominates the best single readout by params.

`neither` is CONTROL ONLY (must sit at chance; base of no metric). Verdict uses only
linear/gate/both. Inconclusive if any of them never reaches Q. Strictly "Pareto-
dominance", never "synergy"/"emergence" (those are deferred to EMERGENCE_BRANCH.md).
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

MARGIN_MIN = 0.5                 # bits (~1.4x fewer params)
RELIABILITY_BAR = 0.7
NEITHER_CHANCE_MAX = 0.25        # neither above this on a task -> "may not need memory"
Q_GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]   # 0.30..0.90 step 0.05
D_MODELS = [16, 32, 48, 64]
D_STATE = 16
VERDICT_POINTS = ["linear", "gate", "both"]
TASKS = ["sc", "ar"]
med = statistics.median


def load():
    with open(CSV_PATH, newline="") as fh:
        return [r for r in csv.DictReader(fh)]


def cell_stats(rows, point, task, dm, ablate=0):
    by_lr = {}
    for r in rows:
        if (r["point"], r["task"], int(r["d_model"]), int(r["d_state"]), int(r["ablate"])) \
           == (point, task, dm, D_STATE, ablate):
            by_lr.setdefault(float(r["lr"]), []).append(r)
    if not by_lr:
        return None
    lr_star = max(by_lr, key=lambda lr: med(float(r["best_val"]) for r in by_lr[lr]))
    rs = by_lr[lr_star]
    tests = [float(r["test_at_best"]) for r in rs]
    return {"lr": lr_star, "params": int(rs[0]["params"]), "median_test": med(tests),
            "reliability": sum(1 for t in tests if t >= RELIABILITY_BAR) / len(tests)}


def curve(rows, point, task):
    out = []
    for dm in D_MODELS:
        c = cell_stats(rows, point, task, dm)
        if c:
            out.append((c["params"], c["median_test"], c["reliability"]))
    return sorted(out, key=lambda x: x[0])


def params_at_Q(crv, Q):
    pts = [(p, a) for (p, a, _) in crv]
    for i, (p, a) in enumerate(pts):
        if a >= Q:
            if i == 0:
                return float(p)
            p0, a0 = pts[i - 1]
            if a == a0:
                return float(p)
            frac = (Q - a0) / (a - a0)
            return 2.0 ** (math.log2(p0) + frac * (math.log2(p) - math.log2(p0)))
    return None


def margin_at(crvs, Q):
    pq = {p: params_at_Q(crvs[p], Q) for p in VERDICT_POINTS}
    if any(pq[p] is None for p in VERDICT_POINTS):
        return None
    best_single = min(pq["linear"], pq["gate"])
    best_axis = "linear" if pq["linear"] <= pq["gate"] else "gate"
    return {"Q": Q, "params_at_Q": {p: round(pq[p], 1) for p in VERDICT_POINTS},
            "best_single_axis": best_axis,
            "margin": round(math.log2(best_single) - math.log2(pq["both"]), 3)}


def task_block(rows, task):
    crvs = {p: curve(rows, p, task) for p in VERDICT_POINTS}
    reachable = [Q for Q in Q_GRID if margin_at(crvs, Q) is not None]
    neither_crv = curve(rows, "neither", task)
    neither_max = max((a for (_, a, _) in neither_crv), default=None)

    block = {
        "curves": {p: [(pp, round(a, 3), round(r, 2)) for (pp, a, r) in curve(rows, p, task)]
                   for p in (VERDICT_POINTS + ["neither"])},
        "neither_max_test": round(neither_max, 3) if neither_max is not None else None,
        "neither_is_control_ok": (neither_max is not None and neither_max < NEITHER_CHANCE_MAX),
        "reachable_Q": reachable,
    }
    if not reachable:
        block["verdict"] = "INCONCLUSIVE: linear/gate/both do not jointly reach any Q band"
        return block

    primary = reachable[len(reachable) // 2]                  # median achievable Q
    neighbors = [q for q in (round(primary - 0.05, 2), round(primary + 0.05, 2)) if q in reachable]
    m_primary = margin_at(crvs, primary)
    m_neighbors = [margin_at(crvs, q) for q in neighbors]
    block.update({"primary_Q": primary, "primary": m_primary,
                  "neighbors": {q: margin_at(crvs, q) for q in neighbors},
                  "all_margins": {m["Q"]: m["margin"] for m in (margin_at(crvs, q) for q in reachable)}})

    mp = m_primary["margin"]
    sign_flip = any((mn["margin"] > 0) != (mp > 0) for mn in m_neighbors)
    if mp >= MARGIN_MIN and not sign_flip:
        v = f"both Pareto-dominates best single ({m_primary['best_single_axis']}) by params, margin {mp} bits"
    elif mp >= MARGIN_MIN and sign_flip:
        v = f"band-dependent: margin {mp}>=MIN at primary but sign flips on a neighbor"
    elif mp <= -MARGIN_MIN:
        v = f"best single ({m_primary['best_single_axis']}) dominates both, margin {mp} bits"
    else:
        v = f"no Pareto-dominance (|margin| {mp} < {MARGIN_MIN})"
    block["verdict"] = v
    return block


def a1_block(rows):
    out = {}
    for point in ("gate", "both"):
        for task in TASKS:
            n = cell_stats(rows, point, task, 64, ablate=0)
            a = cell_stats(rows, point, task, 64, ablate=1)
            if n and a:
                out[f"{point}/{task}"] = {"normal": round(n["median_test"], 3),
                                          "ablated": round(a["median_test"], 3),
                                          "delta": round(n["median_test"] - a["median_test"], 3)}
    return out


def main():
    rows = load()
    summary = {"spec": "FULL_ABLATION_SPEC_v0_5_0 (A-fix-off)", "MARGIN_MIN": MARGIN_MIN,
               "terminology": "Pareto-dominance (NOT synergy/emergence)", "tasks": {}}

    for task in TASKS:
        summary["tasks"][task] = task_block(rows, task)

    verdicts = {t: summary["tasks"][t]["verdict"] for t in TASKS}
    dom = {t: v.startswith("both Pareto-dominates") for t, v in verdicts.items()}
    inconcl = {t: v.startswith("INCONCLUSIVE") for t, v in verdicts.items()}
    if all(dom.values()):
        overall = "both PARETO-DOMINATES the best single on BOTH tasks"
    elif any(inconcl.values()):
        other = [t for t in TASKS if not inconcl[t]]
        overall = ("INCONCLUSIVE on >=1 task; "
                   + "; ".join(f"{t}: {verdicts[t]}" for t in other))
    elif any(dom.values()):
        overall = "TASK-DEPENDENT: dominance on one task only — not a win"
    else:
        overall = "NO Pareto-dominance: combining does not beat the best single"
    summary["overall_verdict"] = overall

    # controls
    summary["neither_control"] = {t: {"max_test": summary["tasks"][t]["neither_max_test"],
                                      "at_chance_ok": summary["tasks"][t]["neither_is_control_ok"]}
                                  for t in TASKS}
    summary["A1"] = a1_block(rows)
    g = cell_stats(rows, "gate", "sc", 64)
    summary["pilot2_reproduction"] = {
        "gate_sc_d64_median_test": round(g["median_test"], 3) if g else None,
        "expected": "~0.80 test, A1 delta ~0.67",
        "A1_delta_gate_sc": summary["A1"].get("gate/sc", {}).get("delta")}
    summary["anchor_B2"] = {f"{t}/d{dm}": (round(c["median_test"], 3) if c else None)
                            for t in TASKS for dm in (32, 64)
                            for c in [cell_stats(rows, "anchor", t, dm)]}
    a_means = [float(r["alpha_mean"]) for r in rows if r["point"] != "anchor"]
    summary["alpha_fixed_mean_range"] = ([round(min(a_means), 3), round(max(a_means), 3)]
                                         if a_means else None)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2))
    for t in TASKS:
        b = summary["tasks"][t]
        print(f"\n== {t}: {b['verdict']} ==")
        if b.get("primary"):
            print(f"   primary Q={b['primary_Q']} margins={b.get('all_margins')}")
        print(f"   neither control max_test={b['neither_max_test']} "
              f"(at-chance ok: {b['neither_is_control_ok']})")
    print(f"\nOVERALL: {summary['overall_verdict']}")


if __name__ == "__main__":
    main()
