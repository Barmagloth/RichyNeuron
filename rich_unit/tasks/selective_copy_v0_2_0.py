"""Selective Copy task (SPEC §3.1).

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

Spec:
  * alphabet n_tokens=8 (+ blank=0, + marker), length T=64.
  * k=4 "significant" tokens scattered at random positions, rest = blank.
  * after the marker, the model must emit the significant tokens IN ORDER OF
    APPEARANCE.
  * input: one-hot or embedding [B, T, d_model]; output: per-position
    classification.
  * generator is DETERMINISTIC per seed; train/val split is BY SEED (infinite
    generator, fixed val-seeds), not by samples (SPEC §3.1).

Metric: token-level accuracy on the fixed val set (SPEC §3).
"""

from __future__ import annotations

import torch

N_TOKENS = 8
SEQ_LEN = 64
K_SIGNIFICANT = 4


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for ``seed``: ``(tokens [B, T], targets [B, T])``.

    Same ``seed`` -> identical batch (required by test_task, SPEC §7).
    """
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §3.1.")


def val_seeds() -> list[int]:
    """Fixed validation seeds, disjoint from training seeds (SPEC §3.1)."""
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §3.1.")
