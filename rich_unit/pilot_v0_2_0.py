"""PILOT (reconnaissance) — NOT the H0 test (SPEC §5 is untouched).

Scope agreed for this run:
  * Task: Selective Copy only (the cheaper of the two).
  * N_SEEDS = 3; d_model in {16,32,48,64,96,128}; d_state in {4,8,16}.
  * ACC_TARGET is NOT frozen here — we only observe achievable quality (to
    calibrate run 2's target on B1, SPEC §5.1).
  * Decides only: (a) is RichUnit no worse than B1 by minimal width? (b) does
    ablation A1 (W_h=0) drop accuracy? (c) ablation A2: has alpha NOT collapsed?
  * NO H0 verdict is issued (PROBLEM/SPEC discipline).

Fairness note (found during calibration): a single shared lr is unfair — B1
diverges to chance at lr=1e-2 while RichUnit tolerates it. So lr is calibrated
PER MODEL from a shared grid and logged below, which is the fair reading of
"each model at its best lr from the same sweep" (anti-A3-trap, SPEC §6).
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import torch

from .models.rich_unit_v0_2_0 import RichUnitLayer
from .models.baselines_v0_2_0 import StackedTemporalChannel, GRUCore
from .models.wrapper_v0_2_0 import SequenceModel
from .tasks.selective_copy_v0_2_0 import make_batch, val_seeds, VOCAB_SIZE
from .train_v0_2_0 import train_one, TrainConfig

SEEDS = [0, 1, 2]
D_MODELS = [16, 32, 48, 64, 96, 128]
D_STATES = [4, 8, 16]
LR_GRID = [3e-3, 1e-2]
MAX_STEPS = 1200
BATCH = 64
EVAL_EVERY = 200

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
VAL = val_seeds()


def build(model_name: str, d_model: int, d_state: int, ablate: bool = False) -> SequenceModel:
    if model_name == "rich":
        core = RichUnitLayer(d_model, d_state, ablate_state=ablate)
    elif model_name == "B1":
        core = StackedTemporalChannel(d_model, d_state)
    elif model_name == "B2":
        core = GRUCore(d_model)
    else:
        raise ValueError(model_name)
    return SequenceModel(core, VOCAB_SIZE, d_model)


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_cell(model_name, d_model, d_state, lr, seed, ablate=False):
    model = build(model_name, d_model, d_state, ablate=ablate)
    cfg = TrainConfig(max_steps=MAX_STEPS, lr=lr, batch_size=BATCH,
                      eval_every=EVAL_EVERY, seed=seed)
    res = train_one(model, make_batch, VAL, cfg)
    return model, res


def calibrate_lr() -> dict[str, float]:
    """Pick each model's lr from LR_GRID at a reference config (transparent)."""
    print("== lr calibration (reference d_model=64, d_state=8, seed=0) ==", flush=True)
    chosen = {}
    ref = {"rich": (64, 8), "B1": (64, 8), "B2": (64, 8)}
    for name, (dm, ds) in ref.items():
        best_lr, best_acc = None, -1.0
        for lr in LR_GRID:
            _, res = run_cell(name, dm, ds, lr, seed=0)
            print(f"  {name} lr={lr}: best_val={res.best_val_acc:.3f}", flush=True)
            if res.best_val_acc > best_acc:
                best_acc, best_lr = res.best_val_acc, lr
        chosen[name] = best_lr
        print(f"  -> {name} chosen lr={best_lr} (best_val={best_acc:.3f})", flush=True)
    return chosen


def median(xs):
    return statistics.median(xs)


def run_grid(lr_by_model: dict[str, float]):
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "pilot_selective_copy.csv"
    rows = []
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "d_model", "d_state", "lr", "seed",
                    "params", "best_val_acc", "steps_to_best"])
        grid = [("rich", dm, ds) for dm in D_MODELS for ds in D_STATES]
        grid += [("B1", dm, ds) for dm in D_MODELS for ds in D_STATES]
        grid += [("B2", dm, 0) for dm in D_MODELS]
        for (name, dm, ds) in grid:
            lr = lr_by_model[name]
            for seed in SEEDS:
                t0 = time.time()
                model, res = run_cell(name, dm, ds, lr, seed)
                p = n_params(model)
                w.writerow([name, dm, ds, lr, seed, p,
                            round(res.best_val_acc, 4), res.steps_to_best])
                fh.flush()
                rows.append(dict(model=name, d_model=dm, d_state=ds, lr=lr,
                                 seed=seed, params=p, best_val_acc=res.best_val_acc))
                print(f"  {name} dm={dm} ds={ds} seed={seed}: "
                      f"best_val={res.best_val_acc:.3f} params={p} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return rows


def aggregate(rows):
    """Median best_val over seeds, per (model, d_model, d_state)."""
    agg = {}
    for r in rows:
        key = (r["model"], r["d_model"], r["d_state"])
        agg.setdefault(key, {"accs": [], "params": r["params"]})
        agg[key]["accs"].append(r["best_val_acc"])
    out = []
    for (m, dm, ds), v in agg.items():
        out.append(dict(model=m, d_model=dm, d_state=ds, params=v["params"],
                        median_acc=median(v["accs"]),
                        min_acc=min(v["accs"]), max_acc=max(v["accs"])))
    return sorted(out, key=lambda x: (x["model"], x["d_model"], x["d_state"]))


def min_width_at(agg, model, bar):
    """Smallest (params) config of `model` whose median_acc >= bar."""
    cands = [a for a in agg if a["model"] == model and a["median_acc"] >= bar]
    if not cands:
        return None
    return min(cands, key=lambda a: a["params"])


def ablation_A1(lr_rich, config):
    """A1: re-train chosen RichUnit config with W_h=0 frozen; compare (SPEC §6)."""
    dm, ds = config
    normal, ablated, alphas = [], [], []
    for seed in SEEDS:
        m, r = run_cell("rich", dm, ds, lr_rich, seed, ablate=False)
        normal.append(r.best_val_acc)
        alphas.append(m.core.alpha.detach().tolist())
        _, ra = run_cell("rich", dm, ds, lr_rich, seed, ablate=True)
        ablated.append(ra.best_val_acc)
    return {
        "config": {"d_model": dm, "d_state": ds},
        "normal_median": median(normal), "normal": normal,
        "ablated_median": median(ablated), "ablated": ablated,
        "delta_median": median(normal) - median(ablated),
        "alphas": alphas,
    }


def main():
    torch.manual_seed(0)
    t_start = time.time()
    lr_by_model = calibrate_lr()
    print("\n== grid ==", flush=True)
    rows = run_grid(lr_by_model)
    agg = aggregate(rows)

    rich_max = max(a["median_acc"] for a in agg if a["model"] == "rich")
    b1_max = max(a["median_acc"] for a in agg if a["model"] == "B1")
    # reconnaissance bar: a modest fraction of the better achievable quality
    bar = round(0.75 * max(rich_max, b1_max), 3)

    rich_min = min_width_at(agg, "rich", bar)
    b1_min = min_width_at(agg, "B1", bar)

    # A1/A2 on RichUnit's min-width config (fallback to its best config).
    if rich_min:
        cfg = (rich_min["d_model"], rich_min["d_state"])
    else:
        best = max((a for a in agg if a["model"] == "rich"), key=lambda a: a["median_acc"])
        cfg = (best["d_model"], best["d_state"])
    print(f"\n== ablations A1/A2 on rich config {cfg} ==", flush=True)
    a1 = ablation_A1(lr_by_model["rich"], cfg)

    # A2 summary: fraction of alpha mass near the extremes.
    flat = [a for row in a1["alphas"] for a in row]
    near0 = sum(1 for a in flat if a < 0.05) / len(flat)
    near1 = sum(1 for a in flat if a > 0.95) / len(flat)

    summary = {
        "calibrated_lr": lr_by_model,
        "rich_max_median_acc": rich_max,
        "b1_max_median_acc": b1_max,
        "b2_max_median_acc": max((a["median_acc"] for a in agg if a["model"] == "B2"), default=None),
        "recon_bar": bar,
        "rich_min_width": rich_min,
        "b1_min_width": b1_min,
        "A1": a1,
        "A2": {"alpha_min": min(flat), "alpha_max": max(flat),
               "alpha_mean": sum(flat)/len(flat),
               "frac_near_0": near0, "frac_near_1": near1},
        "agg": agg,
        "wall_time_s": round(time.time() - t_start, 1),
    }
    (RESULTS_DIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n== SUMMARY ==", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "agg"}, indent=2), flush=True)
    print(f"\nwall time: {summary['wall_time_s']}s", flush=True)


if __name__ == "__main__":
    main()
