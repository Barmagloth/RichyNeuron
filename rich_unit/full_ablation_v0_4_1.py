"""FULL ablation (v0.4.1, A-fix-off) — synergy of two state-readout axes.

FROZEN spec: docs/FULL_ABLATION_SPEC_v0_4_1.md, PREREG_FULL.md. First verdict run.
4 points (neither/linear/gate/both) x 2 tasks (sc/ar) x width grid x lr x 5 seed,
fixed alpha. Then A1 (W_h=0) on gate/both, and a B2/GRU reproducibility anchor.
Checkpoint/resume keyed by (point,task,d_model,d_state,lr,seed,ablate); curves to a
jsonl sidecar. Verdict is computed by report_full_v0_4_1.py.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

import torch

from .models.ablation_unit_v0_4_1 import AblationUnit
from .models.baselines_v0_2_0 import GRUCore
from .models.wrapper_v0_2_0 import SequenceModel
from .tasks import selective_copy_v0_2_0 as sc
from .tasks import assoc_recall_v0_2_0 as ar
from .train_v0_2_0 import train_one, TrainConfig
from .runner_v0_2_0 import CheckpointedCSV, git_checkpoint

POINTS = {"neither": (False, False), "linear": (True, False),
          "gate": (False, True), "both": (True, True)}
TASKS = {"sc": sc, "ar": ar}
D_MODELS = [16, 32, 48, 64]
D_STATES = [16]                      # deviation: narrowed from {8,16} (PREREG_FULL)
LR_GRID = [1e-3, 3e-3, 1e-2]
SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 8000
EVAL_EVERY = 200
PATIENCE = 5
MIN_DELTA = 0.005
BATCH = 64
A1_CONFIG = (64, 16)                 # gate/both A1 + pilot-2 reproducibility anchor
ANCHOR_WIDTHS = [32, 64]
ANCHOR_LR = 3e-3
ANCHOR_SEEDS = [0, 1, 2]

CHECKPOINT_EVERY = 15
PUSH_CHECKPOINTS = os.environ.get("RICH_UNIT_CHECKPOINT_PUSH", "1") == "1"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "full_ablation.csv"
JSONL_PATH = RESULTS_DIR / "full_curves.jsonl"
FIELDS = ["point", "task", "d_model", "d_state", "lr", "seed", "ablate", "params",
          "best_val", "test_at_best", "steps_to_best", "stopped_step",
          "alpha_mean", "alpha_std"]
med = statistics.median


def build(point, task_mod, dm, ds, ablate=False):
    if point == "anchor":
        core = GRUCore(dm)
        alpha = None
    else:
        lin, gate = POINTS[point]
        core = AblationUnit(dm, ds, linear_readout=lin, gate_readout=gate, ablate_state=ablate)
        alpha = core.alpha
    return SequenceModel(core, task_mod.VOCAB_SIZE, dm), alpha


def run_cell(point, task, dm, ds, lr, seed, ablate=False):
    torch.manual_seed(seed)                          # seed BEFORE build (paired ablation)
    task_mod = TASKS[task]
    model, alpha = build(point, task_mod, dm, ds, ablate=ablate)
    cfg = TrainConfig(max_steps=MAX_STEPS, lr=lr, batch_size=BATCH, eval_every=EVAL_EVERY,
                      seed=seed, patience=PATIENCE, min_delta=MIN_DELTA)
    res = train_one(model, task_mod.make_batch, task_mod.val_seeds(), cfg,
                    test_seeds=task_mod.test_seeds())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a_mean = float(alpha.mean()) if alpha is not None else -1.0
    a_std = float(alpha.std()) if alpha is not None else -1.0
    return res, params, a_mean, a_std


def _append_jsonl(rec):
    with open(JSONL_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _do(jobs, log):
    since = 0
    for (point, task, dm, ds, lr, seed, ablate) in jobs:
        key = dict(point=point, task=task, d_model=dm, d_state=ds, lr=lr, seed=seed,
                   ablate=int(ablate))
        if log.is_done(key):
            continue
        t0 = time.time()
        res, params, a_mean, a_std = run_cell(point, task, dm, ds, lr, seed, ablate)
        log.append(dict(point=point, task=task, d_model=dm, d_state=ds, lr=lr, seed=seed,
                        ablate=int(ablate), params=params,
                        best_val=round(res.best_val_acc, 4),
                        test_at_best=round(res.test_at_best, 4),
                        steps_to_best=res.steps_to_best, stopped_step=res.stopped_step,
                        alpha_mean=round(a_mean, 4), alpha_std=round(a_std, 4)))
        _append_jsonl(dict(point=point, task=task, d_model=dm, d_state=ds, lr=lr,
                           seed=seed, ablate=int(ablate), history=res.history))
        print(f"  {point}/{task} dm={dm} ds={ds} lr={lr} s={seed} abl={int(ablate)}: "
              f"test={res.test_at_best:.3f} val={res.best_val_acc:.3f} "
              f"stop@{res.stopped_step} ({time.time()-t0:.0f}s)", flush=True)
        since += 1
        if PUSH_CHECKPOINTS and since >= CHECKPOINT_EVERY:
            git_checkpoint([CSV_PATH, JSONL_PATH],
                           f"chore(full): checkpoint ({log.n_done()} runs)", push=True)
            since = 0
    if PUSH_CHECKPOINTS and since:
        git_checkpoint([CSV_PATH, JSONL_PATH],
                       f"chore(full): checkpoint ({log.n_done()} runs)", push=True)


def _rows():
    rows = []
    if not CSV_PATH.exists():
        return rows
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _lr_star(point, task, dm, ds):
    """lr with the best median val over seeds for a (point,task,width) cell."""
    by_lr = {}
    for r in _rows():
        if (r["point"], r["task"], int(r["d_model"]), int(r["d_state"]), int(r["ablate"])) \
           == (point, task, dm, ds, 0):
            by_lr.setdefault(float(r["lr"]), []).append(float(r["best_val"]))
    by_lr = {k: v for k, v in by_lr.items() if v}
    return max(by_lr, key=lambda lr: med(by_lr[lr])) if by_lr else None


def main():
    torch.manual_seed(0)
    t0 = time.time()
    log = CheckpointedCSV(CSV_PATH, FIELDS,
                          key_fields=["point", "task", "d_model", "d_state", "lr", "seed", "ablate"])
    if log.n_done():
        print(f"== resuming: {log.n_done()} runs done ==", flush=True)

    # Phase 1: main sweep. `neither` FIRST (Q is calibrated on it) then the rest.
    order = ["neither", "linear", "gate", "both"]
    main_jobs = [(p, task, dm, ds, lr, s, False)
                 for p in order for task in TASKS for dm in D_MODELS for ds in D_STATES
                 for lr in LR_GRID for s in SEEDS]
    print("== main sweep ==", flush=True)
    _do(main_jobs, log)

    # Phase 2: A1 (W_h=0) on gate/both at A1_CONFIG, at each cell's lr*.
    dm, ds = A1_CONFIG
    a1_jobs = []
    for point in ("gate", "both"):
        for task in TASKS:
            lr = _lr_star(point, task, dm, ds)
            if lr is None:
                continue
            a1_jobs += [(point, task, dm, ds, lr, s, True) for s in SEEDS]
    print("== A1 ablation (W_h=0) ==", flush=True)
    _do(a1_jobs, log)

    # Phase 3: B2/GRU reproducibility anchor (not in synergy).
    anchor_jobs = [("anchor", task, dm, 0, ANCHOR_LR, s, False)
                   for task in TASKS for dm in ANCHOR_WIDTHS for s in ANCHOR_SEEDS]
    print("== B2/GRU anchor ==", flush=True)
    _do(anchor_jobs, log)

    print(f"\nFULL sweep done. wall {time.time()-t0:.0f}s. "
          f"Run report_full_v0_4_1.py for the verdict.", flush=True)
    if PUSH_CHECKPOINTS:
        git_checkpoint([CSV_PATH, JSONL_PATH], "chore(full): sweep complete", push=True)


if __name__ == "__main__":
    main()
