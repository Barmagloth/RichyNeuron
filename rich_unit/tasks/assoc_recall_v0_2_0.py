"""Associative Recall task (SPEC §3.2).

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

Spec:
  * n_pairs key-value pairs (key, value) presented sequentially, then
    query = one of the keys; target = emit the corresponding value.
  * n_tokens=16, n_pairs=8, T derived from these.
  * same generator contract as Selective Copy: seeded, deterministic, split by
    seed (SPEC §3.2).

Metric: token-level accuracy on the fixed val set (SPEC §3).
"""

from __future__ import annotations

import torch

N_TOKENS = 16
N_PAIRS = 8


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for ``seed``: ``(tokens [B, T], targets [B, T])``.

    Same ``seed`` -> identical batch (required by test_task, SPEC §7).
    """
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §3.2.")


def val_seeds() -> list[int]:
    """Fixed validation seeds, disjoint from training seeds (SPEC §3.2)."""
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §3.2.")
