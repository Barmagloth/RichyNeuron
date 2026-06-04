"""Generators are deterministic per seed (SPEC §7/§8)."""

import torch

from rich_unit.tasks.selective_copy_v0_2_0 import (
    make_batch, val_seeds, K_SIGNIFICANT, SEQ_LEN, MARKER, IGNORE_INDEX,
    _ANSWER_START,
)


def test_deterministic_same_seed():
    a_tok, a_tgt = make_batch(8, 123)
    b_tok, b_tgt = make_batch(8, 123)
    assert torch.equal(a_tok, b_tok)
    assert torch.equal(a_tgt, b_tgt)


def test_different_seeds_differ():
    a_tok, _ = make_batch(8, 1)
    b_tok, _ = make_batch(8, 2)
    assert not torch.equal(a_tok, b_tok)


def test_val_seeds_disjoint_from_train():
    # Training seeds live below 1e6 (train_v0_2_0 stream); val seeds above 2e6.
    assert min(val_seeds()) >= 2_000_000


def test_answer_region_well_formed():
    tok, tgt = make_batch(16, 7)
    mask = tgt != IGNORE_INDEX
    # exactly k supervised positions per sample, all in the answer region
    assert torch.all(mask.sum(dim=1) == K_SIGNIFICANT)
    assert torch.all(mask[:, :_ANSWER_START] == False)
    # marker sits at the fixed marker position
    assert torch.all(tok[:, _ANSWER_START - 1] == MARKER)
    # targets equal the data symbols in order of appearance
    for i in range(tok.shape[0]):
        appeared = tok[i, : _ANSWER_START - 1]
        order = appeared[appeared != 0]
        assert torch.equal(order, tgt[i, mask[i]])
