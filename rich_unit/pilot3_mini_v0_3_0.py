"""PILOT 3 mini (v0.3.0) — do the two axes collapse into each other?

FROZEN spec: PILOT3_MINI_SPEC.md. Calibration reconnaissance, NOT the H0 test, NO
synergy verdict. One question (§1): with selectivity ON (input-dependent alpha_t,
axis 1), does the readout path W_h·s (axis 2) still do work, or become redundant?

Method (§3): train RichSel (both axes) on Selective Copy at d_model=64, d_state=16;
per-cell lr sweep {1e-3,3e-3,1e-2} chosen by median val; 5 seeds; early-stop budget
identical to pilot 2; metric test_at_best. Then ablation A1 (W_h=0 frozen) on the
winning lr. Decide per §5 against the pre-registered threshold 0.10 and Δ_rich=0.674.

Trainability controls (§4): log val curves, the trained alpha_t distribution
(does it vary or degenerate to a constant), and check rich+sel reaches ~rich level.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

import torch

from .models.rich_sel_v0_3_0 import RichSelLayer
from .models.wrapper_v0_2_0 import SequenceModel
from .tasks.selective_copy_v0_2_0 import make_batch, val_seeds, test_seeds, VOCAB_SIZE
from .train_v0_2_0 import train_one, TrainConfig
from .runner_v0_2_0 import CheckpointedCSV, git_checkpoint

D_MODEL = 64
D_STATE = 16
LR_GRID = [1e-3, 3e-3, 1e-2]
SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 8000
EVAL_EVERY = 200
PATIENCE = 5
MIN_DELTA = 0.005
BATCH = 64

# Pre-registered decision constants (§5) — fixed before the run.
DELTA_THRESH = 0.10          # readout "still carries work" if Δ_richsel >= this
DELTA_RICH_REF = 0.674       # pilot 2, same d64/s16 config
RICH_LEVEL = 0.80            # rich's pilot-2 test at this config (sanity target)
UNDERTRAINED_BELOW = 0.70    # normal_med clearly below ~0.80 -> uninterpretable
ALPHA_DEGEN_STD = 0.01       # alpha_t std (over batch,time) below this -> degenerate

CHECKPOINT_EVERY = 6
PUSH_CHECKPOINTS = os.environ.get("RICH_UNIT_CHECKPOINT_PUSH", "1") == "1"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "pilot3_selective_copy.csv"
JSONL_PATH = RESULTS_DIR / "pilot3_curves.jsonl"
SUMMARY_PATH = RESULTS_DIR / "pilot3_summary.json"
VAL = val_seeds()
TEST = test_seeds()
FIELDS = ["d_model", "d_state", "lr", "seed", "ablate", "params",
          "best_val", "test_at_best", "steps_to_best", "stopped_step",
          "alpha_mean", "alpha_std", "alpha_frac0", "alpha_frac1"]
med = statistics.median


@torch.no_grad()
def alpha_stats(model: SequenceModel) -> dict:
    """Distribution of trained alpha_t over a fixed test batch (§4 selectivity log).

    alpha_std = mean over state-channels of std over (batch, time); near 0 means
    alpha degenerated to a constant (selectivity not working).
    """
    model.eval()
    tokens, _ = make_batch(BATCH, TEST[0])
    x = model.embed(tokens)                       # [B, T, d_model] the core sees
    a = model.core.alpha_for(x)                   # [B, T, d_state]
    return {
        "alpha_mean": float(a.mean()),
        "alpha_std": float(a.std(dim=(0, 1)).mean()),
        "alpha_frac0": float((a < 0.05).float().mean()),
        "alpha_frac1": float((a > 0.95).float().mean()),
    }


def run_cell(lr, seed, ablate):
    core = RichSelLayer(D_MODEL, D_STATE, ablate_state=ablate)
    model = SequenceModel(core, VOCAB_SIZE, D_MODEL)
    cfg = TrainConfig(max_steps=MAX_STEPS, lr=lr, batch_size=BATCH,
                      eval_every=EVAL_EVERY, seed=seed,
                      patience=PATIENCE, min_delta=MIN_DELTA)
    res = train_one(model, make_batch, VAL, cfg, test_seeds=TEST)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, res, nparams


def _append_jsonl(rec):
    with open(JSONL_PATH, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_jobs(jobs, log):
    since = 0
    for (lr, seed, ablate) in jobs:
        key = dict(d_model=D_MODEL, d_state=D_STATE, lr=lr, seed=seed, ablate=int(ablate))
        if log.is_done(key):
            continue
        t0 = time.time()
        model, res, p = run_cell(lr, seed, ablate)
        ast = alpha_stats(model)
        log.append(dict(d_model=D_MODEL, d_state=D_STATE, lr=lr, seed=seed,
                        ablate=int(ablate), params=p,
                        best_val=round(res.best_val_acc, 4),
                        test_at_best=round(res.test_at_best, 4),
                        steps_to_best=res.steps_to_best, stopped_step=res.stopped_step,
                        **{k: round(v, 4) for k, v in ast.items()}))
        _append_jsonl(dict(lr=lr, seed=seed, ablate=int(ablate),
                           history=res.history, **ast))
        print(f"  lr={lr} seed={seed} ablate={int(ablate)}: "
              f"val={res.best_val_acc:.3f} test={res.test_at_best:.3f} "
              f"a_mean={ast['alpha_mean']:.3f} a_std={ast['alpha_std']:.3f} "
              f"stop@{res.stopped_step} ({time.time()-t0:.0f}s)", flush=True)
        since += 1
        if PUSH_CHECKPOINTS and since >= CHECKPOINT_EVERY:
            git_checkpoint([CSV_PATH, JSONL_PATH],
                           f"chore(pilot3): checkpoint ({log.n_done()} runs)", push=True)
            since = 0
    if PUSH_CHECKPOINTS and since:
        git_checkpoint([CSV_PATH, JSONL_PATH],
                       f"chore(pilot3): checkpoint ({log.n_done()} runs)", push=True)


def _rows():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (int(r[k]) if k in ("d_model", "d_state", "seed", "ablate",
                                                "params", "steps_to_best", "stopped_step")
                             else float(r[k])) for k in r})
    return rows


def analyse():
    rows = _rows()
    normal = [r for r in rows if r["ablate"] == 0]
    # lr* by median val over the 5 normal seeds
    bylr = {}
    for r in normal:
        bylr.setdefault(r["lr"], []).append(r)
    lr_star = max(bylr, key=lambda lr: med(r["best_val"] for r in bylr[lr]))

    norm_star = sorted(bylr[lr_star], key=lambda r: r["seed"])
    abl_star = sorted([r for r in rows if r["ablate"] == 1 and r["lr"] == lr_star],
                      key=lambda r: r["seed"])
    norm_tests = [r["test_at_best"] for r in norm_star]
    abl_tests = [r["test_at_best"] for r in abl_star]
    normal_med = med(norm_tests)
    ablated_med = med(abl_tests) if abl_tests else None
    delta = (normal_med - ablated_med) if ablated_med is not None else None

    alpha_std_med = med(r["alpha_std"] for r in norm_star)
    alpha_mean_med = med(r["alpha_mean"] for r in norm_star)
    alpha_degenerate = alpha_std_med < ALPHA_DEGEN_STD

    # §5 decision (pre-registered)
    if delta is None:
        outcome = "INCOMPLETE: ablation runs missing"
    elif alpha_degenerate:
        outcome = "UNINTERPRETABLE: alpha degenerated to ~constant (selectivity not working)"
    elif normal_med < UNDERTRAINED_BELOW:
        outcome = f"UNINTERPRETABLE: rich+sel undertrained (normal_med {normal_med:.3f} << {RICH_LEVEL})"
    elif delta >= DELTA_THRESH:
        outcome = "GREEN: axes do NOT collapse — readout still load-bearing — proceed to full 4-point ablation"
    else:
        outcome = "AXES COLLAPSE: selectivity absorbed the readout — no full ablation needed (negative result)"

    summary = {
        "spec": "PILOT3_MINI_SPEC.md v0.3.0", "config": {"d_model": D_MODEL, "d_state": D_STATE},
        "lr_star": lr_star,
        "normal_test_per_seed": norm_tests, "normal_median_test": normal_med,
        "ablated_test_per_seed": abl_tests, "ablated_median_test": ablated_med,
        "delta_richsel": delta, "delta_rich_ref": DELTA_RICH_REF,
        "threshold": DELTA_THRESH, "rich_level_ref": RICH_LEVEL,
        "alpha_mean_median": alpha_mean_med, "alpha_std_median": alpha_std_med,
        "alpha_degenerate": alpha_degenerate,
        "normal_steps_to_best": [r["steps_to_best"] for r in norm_star],
        "outcome": outcome,
        "all_lr_medians_val": {lr: med(r["best_val"] for r in bylr[lr]) for lr in bylr},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n== SUMMARY ==", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if PUSH_CHECKPOINTS:
        git_checkpoint([CSV_PATH, JSONL_PATH, SUMMARY_PATH],
                       "chore(pilot3): final results", push=True)


def main():
    torch.manual_seed(0)
    t0 = time.time()
    log = CheckpointedCSV(CSV_PATH, FIELDS, key_fields=["lr", "seed", "ablate"])
    if log.n_done():
        print(f"== resuming: {log.n_done()} runs done ==", flush=True)

    # 1) normal sweep (lr x seed)
    print("== rich+sel normal sweep ==", flush=True)
    run_jobs([(lr, s, False) for lr in LR_GRID for s in SEEDS], log)

    # 2) lr* by median val, then ablation A1 (5 seeds, W_h=0) at lr*
    normal = [r for r in _rows() if r["ablate"] == 0]
    bylr = {}
    for r in normal:
        bylr.setdefault(r["lr"], []).append(r)
    lr_star = max(bylr, key=lambda lr: med(r["best_val"] for r in bylr[lr]))
    print(f"== lr* = {lr_star}; ablation A1 (W_h=0) ==", flush=True)
    run_jobs([(lr_star, s, True) for s in SEEDS], log)

    analyse()
    print(f"\nwall time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
