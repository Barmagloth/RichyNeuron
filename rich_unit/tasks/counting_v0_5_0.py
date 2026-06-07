"""Counting task (FULL v0.5.0, A-fix-off second task).

A memory-via-integration task that a leaky integrator CAN do — and that needs memory
but NOT content-based matching (unlike associative recall). k signal tokens are
scattered among blanks; after a marker the model outputs the COUNT k as a class.

A memoryless unit (`neither`) cannot count -> sits at chance (1/(K_MAX+1)). A leaky
state integrates the signal occurrences; a linear or gated readout of that state maps
the integrated magnitude to the count class. Distinct from selective copy (aggregate
statistic, not ordered recall).

Vocabulary: 0=blank, 1=signal, 2=marker; the answer class is the count in {0..K_MAX}
(class ids 0..K_MAX). VOCAB_SIZE = K_MAX+1 covers both input (0,1,2) and targets.

Sequence (T = M + 2):
    [ memory region: k signals among blanks ]  marker  answer
     0 ........................... M-1            M      M+1
Deterministic per seed; train/val/test split by seed (same namespaces as the other
tasks: train < 1e6, val 2e6+, test 3e6+).
"""

from __future__ import annotations

import torch

K_MAX = 8                              # counts 0..8 -> 9 classes
M = 16                                 # memory-region length (>= K_MAX to fit k signals)

BLANK = 0
SIGNAL = 1
MARKER = 2
VOCAB_SIZE = K_MAX + 1                  # 9 (covers input 0,1,2 and target classes 0..8)
IGNORE_INDEX = -100

SEQ_LEN = M + 2                        # 18
_MARKER_POS = M                        # 16
_ANSWER_POS = M + 1                    # 17

_VAL_SEED_BASE = 2_000_000
_N_VAL_BATCHES = 8
_TEST_SEED_BASE = 3_000_000
_N_TEST_BATCHES = 8


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for ``seed``: ``(tokens [B, T], targets [B, T])``.

    Same ``seed`` -> identical batch. Target is IGNORE_INDEX except at the answer
    slot, where it is the count k of signal tokens (a class in {0..K_MAX}).
    """
    g = torch.Generator().manual_seed(int(seed))
    tokens = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    targets = torch.full((batch_size, SEQ_LEN), IGNORE_INDEX, dtype=torch.long)

    for i in range(batch_size):
        k = int(torch.randint(0, K_MAX + 1, (1,), generator=g))     # count 0..K_MAX
        if k > 0:
            pos = torch.randperm(M, generator=g)[:k]
            tokens[i, pos] = SIGNAL
        tokens[i, _MARKER_POS] = MARKER
        targets[i, _ANSWER_POS] = k

    return tokens, targets


def val_seeds() -> list[int]:
    return [_VAL_SEED_BASE + i for i in range(_N_VAL_BATCHES)]


def test_seeds() -> list[int]:
    return [_TEST_SEED_BASE + i for i in range(_N_TEST_BATCHES)]
