"""PILOT 2 (reconnaissance) — does the rich-vs-B1 gap survive a fair budget & lr?

NOT the H0 test. No verdict, PREREG stays unfrozen. Pilot 1 measured the gap in
an under-trained regime (both far from ceiling, GRU already at 1.0) — the least
trustworthy place to compare. Pilot 2 lets each model actually converge and asks
whether the gap holds.

Design (agreed):
  * Budget: early-stop on a val plateau (patience PATIENCE evals, MIN_DELTA margin),
    cap MAX_STEPS. Identical criterion for every model (fair budget, SPEC §6 A3).
    steps_to_best is logged — convergence speed is itself a result.
  * lr: per-CELL sweep LR_GRID; the cell's lr is chosen by the MEDIAN-over-seed
    val (not the luckiest lr*seed pair — that would overfit the sweep).
  * Honesty split: lr-selection and early-stop run on VAL; the reported number is
    on a disjoint TEST split at the best-val step.
  * Grid trimmed (pilot 1 showed the edges): d_model {16,32,48,64}, d_state {8,16}.
  * 5 seeds (pilot 1's "11x" leaned on a point with one collapsed seed).
  * B2/GRU kept only as a cheap anchor (task still solvable), not in the sweep.
  * A1/A2 repeated on pilot 2's converged winning config.

ACC_TARGET is still NOT frozen — pilot 2 is the last calibration step before FULL.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

import torch

from .models.rich_unit_v0_2_0 import RichUnitLayer
from .models.baselines_v0_2_0 import StackedTemporalChannel, GRUCore
from .models.wrapper_v0_2_0 import SequenceModel
from .tasks.selective_copy_v0_2_0 import make_batch, val_seeds, test_seeds, VOCAB_SIZE
from .train_v0_2_0 import train_one, TrainConfig
from .runner_v0_2_0 import CheckpointedCSV, git_checkpoint

SEEDS = [0, 1, 2, 3, 4]
D_MODELS = [16, 32, 48, 64]
D_STATES = [8, 16]
LR_GRID = [1e-3, 3e-3, 1e-2]
MAX_STEPS = 8000
EVAL_EVERY = 200
PATIENCE = 5          # 5 * 200 = 1000 steps without a >MIN_DELTA val gain -> stop
MIN_DELTA = 0.005
BATCH = 64

B2_WIDTHS = [32, 64]
B2_LR = 3e-3
B2_SEEDS = [0, 1, 2]

CHECKPOINT_EVERY = 10
PUSH_CHECKPOINTS = os.environ.get("RICH_UNIT_CHECKPOINT_PUSH", "1") == "1"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "pilot2_selective_copy.csv"
SUMMARY_PATH = RESULTS_DIR / "pilot2_summary.json"
VAL = val_seeds()
TEST = test_seeds()
FIELDS = ["model", "d_model", "d_state", "lr", "seed", "params",
          "best_val", "test_at_best", "steps_to_best", "stopped_step"]


def build(model_name, d_model, d_state, ablate=False) -> SequenceModel:
    if model_name == "rich":
        core = RichUnitLayer(d_model, d_state, ablate_state=ablate)
    elif model_name == "B1":
        core = StackedTemporalChannel(d_model, d_state)
    elif model_name == "B2":
        core = GRUCore(d_model)
    else:
        raise ValueError(model_name)
    return SequenceModel(core, VOCAB_SIZE, d_model)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_cell(model_name, d_model, d_state, lr, seed, ablate=False):
    model = build(model_name, d_model, d_state, ablate=ablate)
    cfg = TrainConfig(max_steps=MAX_STEPS, lr=lr, batch_size=BATCH,
                      eval_every=EVAL_EVERY, seed=seed,
                      patience=PATIENCE, min_delta=MIN_DELTA)
    res = train_one(model, make_batch, VAL, cfg, test_seeds=TEST)
    return model, res


def _jobs():
    """(model, d_model, d_state, lr, seeds) tuples for the whole sweep."""
    jobs = []
    for model in ("rich", "B1"):
        for dm in D_MODELS:
            for ds in D_STATES:
                for lr in LR_GRID:
                    jobs.append((model, dm, ds, lr, SEEDS))
    for dm in B2_WIDTHS:
        jobs.append(("B2", dm, 0, B2_LR, B2_SEEDS))
    return jobs


def run_sweep():
    log = CheckpointedCSV(CSV_PATH, FIELDS,
                          key_fields=["model", "d_model", "d_state", "lr", "seed"])
    if log.n_done():
        print(f"== resuming: {log.n_done()} runs already done ==", flush=True)
    since = 0
    for (model, dm, ds, lr, seeds) in _jobs():
        for seed in seeds:
            key = dict(model=model, d_model=dm, d_state=ds, lr=lr, seed=seed)
            if log.is_done(key):
                continue
            t0 = time.time()
            m, res = run_cell(model, dm, ds, lr, seed)
            p = n_params(m)
            log.append(dict(model=model, d_model=dm, d_state=ds, lr=lr, seed=seed,
                            params=p, best_val=round(res.best_val_acc, 4),
                            test_at_best=round(res.test_at_best, 4),
                            steps_to_best=res.steps_to_best,
                            stopped_step=res.stopped_step))
            print(f"  {model} dm={dm} ds={ds} lr={lr} seed={seed}: "
                  f"val={res.best_val_acc:.3f} test={res.test_at_best:.3f} "
                  f"stop@{res.stopped_step} best@{res.steps_to_best} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            since += 1
            if PUSH_CHECKPOINTS and since >= CHECKPOINT_EVERY:
                git_checkpoint([CSV_PATH], f"chore(pilot2): checkpoint "
                               f"({log.n_done()} runs)", push=True)
                since = 0
    if PUSH_CHECKPOINTS and since:
        git_checkpoint([CSV_PATH], f"chore(pilot2): checkpoint ({log.n_done()} runs)",
                       push=True)


def _read_rows():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(model=r["model"], d_model=int(r["d_model"]),
                             d_state=int(r["d_state"]), lr=float(r["lr"]),
                             seed=int(r["seed"]), params=int(r["params"]),
                             best_val=float(r["best_val"]),
                             test_at_best=float(r["test_at_best"]),
                             steps_to_best=int(r["steps_to_best"])))
    return rows


def aggregate(rows):
    """Per cell (model,d_model,d_state): pick lr by MEDIAN val, report TEST stats."""
    by_lr = {}          # (model,dm,ds,lr) -> list of rows
    for r in rows:
        by_lr.setdefault((r["model"], r["d_model"], r["d_state"], r["lr"]), []).append(r)
    cells = {}
    for (model, dm, ds, lr), rs in by_lr.items():
        cells.setdefault((model, dm, ds), {})[lr] = rs
    out = []
    for (model, dm, ds), lrmap in cells.items():
        # choose lr by median val (not test, not best-seed)
        best_lr = max(lrmap, key=lambda lr: statistics.median(x["best_val"] for x in lrmap[lr]))
        rs = lrmap[best_lr]
        tests = [x["test_at_best"] for x in rs]
        out.append(dict(
            model=model, d_model=dm, d_state=ds, chosen_lr=best_lr,
            params=rs[0]["params"],
            median_test=statistics.median(tests), min_test=min(tests), max_test=max(tests),
            median_val=statistics.median(x["best_val"] for x in rs),
            median_steps=statistics.median(x["steps_to_best"] for x in rs),
        ))
    return sorted(out, key=lambda a: (a["model"], a["d_model"], a["d_state"]))


def ablation_A1A2(dm, ds, lr):
    normal, ablated, alphas, steps = [], [], [], []
    for seed in SEEDS:
        m, r = run_cell("rich", dm, ds, lr, seed, ablate=False)
        normal.append(r.test_at_best); steps.append(r.steps_to_best)
        alphas.append(m.core.alpha.detach().tolist())
        _, ra = run_cell("rich", dm, ds, lr, seed, ablate=True)
        ablated.append(ra.test_at_best)
    flat = [a for row in alphas for a in row]
    return {
        "config": {"d_model": dm, "d_state": ds, "lr": lr},
        "normal_median_test": statistics.median(normal), "normal": normal,
        "ablated_median_test": statistics.median(ablated), "ablated": ablated,
        "delta_median": statistics.median(normal) - statistics.median(ablated),
        "median_steps_to_best": statistics.median(steps),
        "alpha_min": min(flat), "alpha_max": max(flat),
        "alpha_mean": sum(flat) / len(flat),
        "alpha_frac_near_0": sum(1 for a in flat if a < 0.05) / len(flat),
        "alpha_frac_near_1": sum(1 for a in flat if a > 0.95) / len(flat),
    }


def analyse():
    agg = aggregate(_read_rows())
    rich = [a for a in agg if a["model"] == "rich"]
    b1 = [a for a in agg if a["model"] == "B1"]
    b2 = [a for a in agg if a["model"] == "B2"]
    b1_ceiling = max(a["median_test"] for a in b1)
    rich_winner = max(rich, key=lambda a: a["median_test"])
    # smallest rich cell whose median test matches B1's ceiling
    matched = [a for a in rich if a["median_test"] >= b1_ceiling]
    rich_match = min(matched, key=lambda a: a["params"]) if matched else None

    w = rich_winner
    a1a2 = ablation_A1A2(w["d_model"], w["d_state"], w["chosen_lr"])

    summary = {
        "design": {"seeds": SEEDS, "d_models": D_MODELS, "d_states": D_STATES,
                   "lr_grid": LR_GRID, "max_steps": MAX_STEPS, "patience": PATIENCE,
                   "min_delta": MIN_DELTA, "metric": "test_at_best"},
        "rich_ceiling": rich_winner, "b1_ceiling_median_test": b1_ceiling,
        "b1_ceiling_cell": max(b1, key=lambda a: a["median_test"]),
        "rich_match_to_b1_ceiling": rich_match,
        "param_ratio_b1ceil_over_richmatch": (
            round(max(b1, key=lambda a: a["median_test"])["params"] / rich_match["params"], 2)
            if rich_match else None),
        "b2_anchor": b2,
        "A1A2": a1a2,
        "cells": agg,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n== SUMMARY ==", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2), flush=True)
    if PUSH_CHECKPOINTS:
        git_checkpoint([CSV_PATH, SUMMARY_PATH], "chore(pilot2): final results", push=True)


def main():
    torch.manual_seed(0)
    t0 = time.time()
    run_sweep()
    analyse()
    print(f"\nwall time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
