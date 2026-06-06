"""Associative Recall task (SPEC §3.2).

n_pairs (key,value) pairs are presented interleaved, then a query marker and one
of the keys; the model must emit that key's value at the answer slot.

Vocabulary layout:
    0           = blank
    1 .. 16     = symbols (n_tokens = 16; both keys and values draw from these)
    17          = query marker
Total vocab = n_tokens + 2 = 18.

Sequence (T = 2*n_pairs + 3 = 19):
    [k0 v0 k1 v1 ... k7 v7]  marker  query_key  answer
     0 ............... 15      16        17        18
Keys are distinct (well-defined mapping); values sampled with replacement. The
answer slot input is blank; its target is the queried key's value. All other
positions are IGNORE_INDEX. Token-level accuracy is over the single answer slot.

Deterministic per seed; train/val/test split by seed (same namespaces as
selective_copy: train < 1e6, val 2e6+, test 3e6+).
"""

from __future__ import annotations

import torch

N_TOKENS = 16
N_PAIRS = 8

BLANK = 0
QUERY_MARKER = N_TOKENS + 1            # = 17
VOCAB_SIZE = N_TOKENS + 2              # = 18
IGNORE_INDEX = -100

SEQ_LEN = 2 * N_PAIRS + 3             # = 19
_MARKER_POS = 2 * N_PAIRS            # 16
_QUERY_POS = _MARKER_POS + 1         # 17
_ANSWER_POS = _MARKER_POS + 2        # 18

_VAL_SEED_BASE = 2_000_000
_N_VAL_BATCHES = 8
_TEST_SEED_BASE = 3_000_000
_N_TEST_BATCHES = 8


def make_batch(batch_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic batch for ``seed``: ``(tokens [B, T], targets [B, T])``.

    Same ``seed`` -> identical batch. Targets are IGNORE_INDEX except at the answer
    slot, where the target is the value bound to the queried key.
    """
    g = torch.Generator().manual_seed(int(seed))
    tokens = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    targets = torch.full((batch_size, SEQ_LEN), IGNORE_INDEX, dtype=torch.long)

    for i in range(batch_size):
        keys = torch.randperm(N_TOKENS, generator=g)[:N_PAIRS] + 1          # distinct, 1..16
        values = torch.randint(1, N_TOKENS + 1, (N_PAIRS,), generator=g)    # with replacement
        tokens[i, 0:2 * N_PAIRS:2] = keys
        tokens[i, 1:2 * N_PAIRS:2] = values
        tokens[i, _MARKER_POS] = QUERY_MARKER
        q = int(torch.randint(0, N_PAIRS, (1,), generator=g))               # which key is queried
        tokens[i, _QUERY_POS] = keys[q]
        targets[i, _ANSWER_POS] = values[q]

    return tokens, targets


def val_seeds() -> list[int]:
    """Fixed validation seeds, disjoint from training seeds (SPEC §3.2)."""
    return [_VAL_SEED_BASE + i for i in range(_N_VAL_BATCHES)]


def test_seeds() -> list[int]:
    """Fixed held-out test seeds, disjoint from train and val (honest reporting)."""
    return [_TEST_SEED_BASE + i for i in range(_N_TEST_BATCHES)]
