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


@dataclass
class TrainResult:
    """Outcome of one (model, task, seed) run."""

    final_val_acc: float
    best_val_acc: float
    steps_to_best: int
    history: list[tuple[int, float]] = field(default_factory=list)


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
              cfg: TrainConfig) -> TrainResult:
    """Train one model on one task with one seed; return val metrics + curve."""
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    history: list[tuple[int, float]] = []
    best_val = 0.0
    steps_to_best = 0
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
            if acc > best_val:
                best_val = acc
                steps_to_best = step

    final_val = history[-1][1] if history else 0.0
    return TrainResult(
        final_val_acc=final_val,
        best_val_acc=best_val,
        steps_to_best=steps_to_best,
        history=history,
    )
