"""Single, model-agnostic training loop (SPEC §4, §5).

Contract (SPEC §0, §5.0, §6 A3):
  * IDENTICAL budget / optimizer (AdamW) / lr for ALL models.
  * CPU-only; runs without CUDA (SPEC §1).
  * Training draws fresh batches from an infinite seeded stream; validation uses
    fixed, disjoint val-seeds. No early stopping on a lucky val for selection
    (SPEC §0). We DO record the best val seen, for reconnaissance only.

Reused unchanged by every model variant — the core is the only moving part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks.selective_copy_v0_2_0 import IGNORE_INDEX

# Training seeds stay in a low namespace, disjoint from the val-seed base.
_TRAIN_SEED_BASE = 1_000


@dataclass(frozen=True)
class TrainConfig:
    """Shared training budget — IDENTICAL across all models (SPEC §5.0)."""

    max_steps: int = 400
    lr: float = 3e-3
    batch_size: int = 64
    weight_decay: float = 0.0
    eval_every: int = 25
    seed: int = 0          # run seed (selects train stream + init)
    patience: int | None = None   # early-stop: # evals w/o >min_delta val gain; None=off
    min_delta: float = 0.0        # min val improvement that counts (anti slow-creep)


@dataclass
class TrainResult:
    """Outcome of one (model, task, seed) run."""

    final_val_acc: float
    best_val_acc: float
    steps_to_best: int
    history: list[tuple[int, float]] = field(default_factory=list)
    test_at_best: float | None = None   # test acc at the best-val step (honest metric)
    stopped_step: int = 0               # step at which training actually stopped


MakeBatch = Callable[[int, int], tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def evaluate(model: nn.Module, make_batch: MakeBatch, seeds: list[int],
             batch_size: int) -> float:
    """Token-level accuracy over the fixed val-seeds (answer slots only)."""
    model.eval()
    correct = 0
    total = 0
    for s in seeds:
        tokens, targets = make_batch(batch_size, s)
        logits = model(tokens)
        pred = logits.argmax(dim=-1)
        mask = targets != IGNORE_INDEX
        correct += int((pred[mask] == targets[mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


def train_one(model: nn.Module, make_batch: MakeBatch, val_seeds: list[int],
              cfg: TrainConfig, test_seeds: list[int] | None = None) -> TrainResult:
    """Train one model; return val metrics + curve.

    Early-stop (if ``cfg.patience``): stop when val has not improved by more than
    ``cfg.min_delta`` for ``patience`` consecutive evals; ``max_steps`` is the cap.
    The criterion is identical for every model, so the budget is fair (SPEC §6 A3).

    If ``test_seeds`` is given, test accuracy is recorded at every eval and the
    value AT the best-val step is returned as ``test_at_best`` — lr selection and
    early-stop run on val, the reported number on a disjoint test split.
    """
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    history: list[tuple[int, float]] = []
    best_val = -1.0          # best-so-far val (drives reporting, test, AND patience)
    steps_to_best = 0
    test_at_best: float | None = None
    evals_since_improve = 0
    stopped_step = cfg.max_steps
    # Per-run, per-step training seed stream; disjoint from val-seed namespace.
    stream_base = _TRAIN_SEED_BASE + cfg.seed * 100_000

    for step in range(1, cfg.max_steps + 1):
        model.train()
        tokens, targets = make_batch(cfg.batch_size, stream_base + step)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            acc = evaluate(model, make_batch, val_seeds, cfg.batch_size)
            history.append((step, acc))
            # "improvement" = a > min_delta gain over best-so-far (computed BEFORE
            # best_val is updated). A slow monotone creep below min_delta is NOT an
            # improvement, so patience still fires (anti slow-creep).
            improved = acc > best_val + cfg.min_delta
            if acc > best_val:
                best_val = acc                       # true running max -> reporting
                steps_to_best = step
                if test_seeds is not None:           # test measured AT the best-val
                    test_at_best = evaluate(model, make_batch, test_seeds, cfg.batch_size)
            if improved:
                evals_since_improve = 0
            else:
                evals_since_improve += 1
                if cfg.patience is not None and evals_since_improve >= cfg.patience:
                    stopped_step = step
                    break

    final_val = history[-1][1] if history else 0.0
    return TrainResult(
        final_val_acc=final_val,
        best_val_acc=best_val,
        steps_to_best=steps_to_best,
        history=history,
        test_at_best=test_at_best,
        stopped_step=stopped_step,
    )
