"""Selective Copy task (SPEC §3.1).

Spec:
  * alphabet n_tokens=8 data symbols, blank=0, plus a marker; length T=64.
  * k=4 "significant" data tokens scattered at random positions in the memory
    region, the rest blank.
  * after the marker, the model must emit the significant tokens IN ORDER OF
    APPEARANCE, one per answer slot.
  * input: token ids [B, T] (embedded to [B, T, d_model] by the wrapper);
    output: per-position classification.
  * generator is DETERMINISTIC per seed; train/val split is BY SEED (infinite
    generator, fixed val-seeds), not by samples.

Metric: token-level accuracy on the answer slots only (SPEC §3).

Vocabulary layout:
    0          = blank
    1 .. 8     = data symbols (n_tokens = 8)
    9          = marker
Total vocab = n_tokens + 2 = 10. Non-answer target positions use IGNORE_INDEX so
cross-entropy and accuracy are computed only on the k answer slots.
"""

from __future__ import annotations

import torch

N_TOKENS = 8                       # data symbols 1..8
SEQ_LEN = 64                       # T
K_SIGNIFICANT = 4                  # k

BLANK = 0
MARKER = N_TOKENS + 1              # = 9
VOCAB_SIZE = N_TOKENS + 2          # blank + data + marker = 10
IGNORE_INDEX = -100                # cross-entropy ignore for non-answer positions

# Memory region holds the scattered data; then 1 marker; then k answer slots.
_MEM_LEN = SEQ_LEN - K_SIGNIFICANT - 1
_MARKER_POS = _MEM_LEN             # index of the marker
_ANSWER_START = _MEM_LEN + 1       # first answer slot

# Val seeds live in a high, reserved namespace; training seeds stay well below it
# (see ``train_v0_2_0``), guaranteeing the split is by seed (SPEC §3.1).
_VAL_SEED_BASE = 2_000_000
_N_VAL_BATCHES = 8


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for ``seed``: ``(tokens [B, T], targets [B, T])``.

    Same ``seed`` -> identical batch. Targets are IGNORE_INDEX except on the k
    answer slots, where they are the data tokens in order of appearance.
    """
    g = torch.Generator().manual_seed(int(seed))

    tokens = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    targets = torch.full((batch_size, SEQ_LEN), IGNORE_INDEX, dtype=torch.long)

    for i in range(batch_size):
        # k distinct positions in the memory region, kept in increasing order so
        # "order of appearance" is well defined.
        perm = torch.randperm(_MEM_LEN, generator=g)[:K_SIGNIFICANT]
        positions, _ = torch.sort(perm)
        # k data symbols in 1..N_TOKENS (with replacement).
        symbols = torch.randint(1, N_TOKENS + 1, (K_SIGNIFICANT,), generator=g)

        tokens[i, positions] = symbols
        tokens[i, _MARKER_POS] = MARKER
        # answer slots stay BLANK in the input; targets carry the symbols in order.
        targets[i, _ANSWER_START:_ANSWER_START + K_SIGNIFICANT] = symbols

    return tokens, targets


def val_seeds() -> list[int]:
    """Fixed validation seeds, disjoint from training seeds (SPEC §3.1)."""
    return [_VAL_SEED_BASE + i for i in range(_N_VAL_BATCHES)]
