"""Single, model-agnostic training loop (SPEC §4, §5).

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

Contract (SPEC §0, §5.0, §6 A3):
  * IDENTICAL budget / optimizer (AdamW) / lr-sweep for ALL models — an
    under-trained baseline invalidates the result.
  * CPU-only; must run without CUDA (SPEC §1, §8).
  * No early stopping on a lucky val (SPEC §0, selection bias).

Reused unchanged by every model variant — the core is the only moving part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class TrainConfig:
    """Shared training budget — IDENTICAL across all models (SPEC §5.0)."""

    max_steps: int
    lr: float
    batch_size: int
    seed: int
    # ... AdamW betas / weight_decay etc. fixed here, never per-model.


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one (model, task, seed) run."""

    final_val_acc: float
    steps_to_target: int | None
    history: list[float]  # val-acc curve, for the B1 convergence proof (A3).


def train_one(
    model: nn.Module,
    make_batch: Callable[[int, int], tuple],
    val_seeds: list[int],
    cfg: TrainConfig,
) -> TrainResult:
    """Train one model on one task with one seed; return val metrics + curve."""
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §4/§5.")
