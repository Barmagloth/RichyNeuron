"""Majority / mode task (FULL v0.5.0, A-fix-off second task candidate).

Output the most frequent symbol seen in the sequence. Memory-via-integration (track
a distribution over the alphabet), NOT content-matching. d_model-bound by design: the
difficulty is holding/mixing counts over a LARGE alphabet (rich projections), not the
capacity of a scalar counter — so quality should scale with d_model (unlike counting).

A memoryless unit (`neither`) cannot aggregate -> chance (1/N_SYM). A leaky state can
integrate per-symbol evidence; a linear/gate readout maps it to the mode class.

Vocabulary: 0=blank(unused), 1..N_SYM symbols, marker=N_SYM+1. VOCAB = N_SYM+2.
Target (answer slot) = the unique most-frequent symbol class in {1..N_SYM}.

Sequence (T = M + 2):
    [ M symbols from 1..N_SYM ]  marker  answer
     0 .............. M-1          M       M+1
Unique mode guaranteed by rejection (ties resampled). Deterministic per seed;
train<1e6 / val 2e6+ / test 3e6+ (disjoint, same discipline as the other tasks).
"""

from __future__ import annotations

import torch

N_SYM = 12                             # alphabet size -> d_model-bound
M = 24                                 # sequence length to aggregate over

BLANK = 0
MARKER = N_SYM + 1                     # = 13
VOCAB_SIZE = N_SYM + 2                 # = 14
IGNORE_INDEX = -100

SEQ_LEN = M + 2                        # 26
_MARKER_POS = M                        # 24
_ANSWER_POS = M + 1                    # 25

_VAL_SEED_BASE = 2_000_000
_N_VAL_BATCHES = 8
_TEST_SEED_BASE = 3_000_000
_N_TEST_BATCHES = 8


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for ``seed``: ``(tokens [B, T], targets [B, T])``.

    Target is IGNORE_INDEX except at the answer slot = the unique most-frequent
    symbol (1..N_SYM). Sequences with a tied mode are resampled.
    """
    g = torch.Generator().manual_seed(int(seed))
    tokens = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    targets = torch.full((batch_size, SEQ_LEN), IGNORE_INDEX, dtype=torch.long)

    for i in range(batch_size):
        while True:
            syms = torch.randint(1, N_SYM + 1, (M,), generator=g)
            counts = torch.bincount(syms, minlength=N_SYM + 1)
            top = int(counts.max())
            if int((counts == top).sum()) == 1:        # unique mode
                break
        tokens[i, :M] = syms
        tokens[i, _MARKER_POS] = MARKER
        targets[i, _ANSWER_POS] = int(counts.argmax())

    return tokens, targets


def val_seeds() -> list[int]:
    return [_VAL_SEED_BASE + i for i in range(_N_VAL_BATCHES)]


def test_seeds() -> list[int]:
    return [_TEST_SEED_BASE + i for i in range(_N_TEST_BATCHES)]
