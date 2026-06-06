"""AblationUnit correctness (FULL §1/§7): 4 points differ ONLY by two flags.

Key proofs:
- gate (linear=off, gate=on) reproduces RichUnitLayer EXACTLY at shared weights
  (the §5 reproducibility anchor with pilot 2).
- neither output is independent of the state; linear adds exactly C.s.
- parameter integrity: C only with linear_readout, W_h only with gate_readout.
"""

import torch

from rich_unit.models.ablation_unit_v0_4_1 import AblationUnit
from rich_unit.models.rich_unit_v0_2_0 import RichUnitLayer


def _share(dst, src, names):
    with torch.no_grad():
        for n in names:
            getattr(dst, n).weight.copy_(getattr(src, n).weight)
            if getattr(dst, n).bias is not None:
                getattr(dst, n).bias.copy_(getattr(src, n).bias)


def test_gate_point_equals_rich():
    torch.manual_seed(0)
    d_model, d_state = 6, 4
    unit = AblationUnit(d_model, d_state, linear_readout=False, gate_readout=True)
    rich = RichUnitLayer(d_model, d_state)
    with torch.no_grad():
        rich.alpha_raw.copy_(unit.alpha_raw)
        _share(rich, unit, ("W_s", "W_g", "W_v"))
        rich.W_h.weight.copy_(unit.W_h.weight)
    x = torch.randn(3, 9, d_model)
    assert torch.allclose(unit(x), rich(x), atol=1e-6), (unit(x) - rich(x)).abs().max()


def test_neither_independent_of_state():
    torch.manual_seed(1)
    unit = AblationUnit(5, 3, linear_readout=False, gate_readout=False)
    x = torch.randn(2, 7, 5)
    # perturbing W_s (state write) must NOT change the output of `neither`
    y0 = unit(x)
    with torch.no_grad():
        unit.W_s.weight.add_(10.0)
    assert torch.allclose(unit(x), y0, atol=1e-6)


def test_linear_adds_exactly_Cs():
    torch.manual_seed(2)
    d_model, d_state = 5, 3
    lin = AblationUnit(d_model, d_state, linear_readout=True, gate_readout=False)
    nei = AblationUnit(d_model, d_state, linear_readout=False, gate_readout=False)
    with torch.no_grad():
        nei.alpha_raw.copy_(lin.alpha_raw)
        _share(nei, lin, ("W_s", "W_g", "W_v"))
    x = torch.randn(2, 6, d_model)
    # recompute C.s independently
    alpha = torch.sigmoid(lin.alpha_raw)
    s = torch.zeros(2, d_state)
    cs = []
    Ws = lin.W_s(x)
    for t in range(x.shape[1]):
        s = alpha * s + (1 - alpha) * Ws[:, t]
        cs.append(lin.C(s))
    cs = torch.stack(cs, dim=1)
    assert torch.allclose(lin(x), nei(x) + cs, atol=1e-6)


def test_parameter_integrity():
    nei = AblationUnit(8, 4, linear_readout=False, gate_readout=False)
    assert nei.C is None and nei.W_h is None
    lin = AblationUnit(8, 4, linear_readout=True, gate_readout=False)
    assert lin.C is not None and lin.W_h is None
    gate = AblationUnit(8, 4, linear_readout=False, gate_readout=True)
    assert gate.C is None and gate.W_h is not None
    both = AblationUnit(8, 4, linear_readout=True, gate_readout=True)
    assert both.C is not None and both.W_h is not None


def test_ablate_freezes_W_h():
    both = AblationUnit(8, 4, linear_readout=True, gate_readout=True, ablate_state=True)
    assert torch.count_nonzero(both.W_h.weight) == 0
    assert both.W_h.weight.requires_grad is False
    assert both.C.weight.requires_grad is True   # linear path untouched


def test_assoc_recall_deterministic_and_wellformed():
    from rich_unit.tasks.assoc_recall_v0_2_0 import (
        make_batch, val_seeds, test_seeds, IGNORE_INDEX, _ANSWER_POS, _QUERY_POS,
        QUERY_MARKER, _MARKER_POS, N_PAIRS,
    )
    a, at = make_batch(8, 5)
    b, bt = make_batch(8, 5)
    assert torch.equal(a, b) and torch.equal(at, bt)
    assert not torch.equal(make_batch(8, 5)[0], make_batch(8, 6)[0])
    assert set(val_seeds()).isdisjoint(test_seeds())
    tok, tgt = make_batch(16, 7)
    mask = tgt != IGNORE_INDEX
    assert torch.all(mask.sum(dim=1) == 1)                 # exactly one answer slot
    assert torch.all(mask[:, _ANSWER_POS])
    assert torch.all(tok[:, _MARKER_POS] == QUERY_MARKER)
    # the queried key really appears among the keys, and its value is the target
    for i in range(tok.shape[0]):
        keys = tok[i, 0:2 * N_PAIRS:2]
        vals = tok[i, 1:2 * N_PAIRS:2]
        q = tok[i, _QUERY_POS]
        j = (keys == q).nonzero()[0, 0]
        assert tgt[i, _ANSWER_POS] == vals[j]
