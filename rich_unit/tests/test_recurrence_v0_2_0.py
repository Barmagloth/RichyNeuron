"""Recurrence correctness — s_t against a hand-written numeric reference (SPEC §7/§8).

Recomputes the state independently from the SPEC §2 equations (using the layer's
own weights) and asserts the layer's output matches, so a refactor of the scan can
never silently drift.
"""

import torch

from rich_unit.models.rich_unit_v0_2_0 import RichUnitLayer


def _reference_forward(layer: RichUnitLayer, x: torch.Tensor) -> torch.Tensor:
    """Independent re-implementation of SPEC §2, kept deliberately naive."""
    B, T, _ = x.shape
    alpha = torch.sigmoid(layer.alpha_raw)
    s = torch.zeros(B, layer.d_state)
    ys = []
    for t in range(T):
        s = alpha * s + (1.0 - alpha) * (x[:, t] @ layer.W_s.weight.t())
        g_pre = x[:, t] @ layer.W_g.weight.t() + layer.W_g.bias
        g_pre = g_pre + s @ layer.W_h.weight.t()
        g = torch.sigmoid(g_pre)
        v = x[:, t] @ layer.W_v.weight.t() + layer.W_v.bias
        ys.append(v * g)
    return torch.stack(ys, dim=1)


def test_state_matches_manual_reference():
    torch.manual_seed(0)
    layer = RichUnitLayer(d_model=6, d_state=3)
    x = torch.randn(4, 10, 6)
    got = layer(x)
    ref = _reference_forward(layer, x)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max()


def test_initial_state_is_zero():
    """With s_0 = 0, the first step's gate must not include any state history.

    At t=0, s_1 = (1-alpha) * (x_0 @ W_s); zeroing W_h must change ONLY via the
    gate, never via a non-zero initial state. We check the first output equals the
    reference computed with s starting at exactly zero.
    """
    torch.manual_seed(1)
    layer = RichUnitLayer(d_model=5, d_state=2)
    x = torch.randn(2, 4, 5)
    got = layer(x)[:, 0]
    ref = _reference_forward(layer, x)[:, 0]
    assert torch.allclose(got, ref, atol=1e-6)


def test_alpha_in_unit_interval():
    layer = RichUnitLayer(d_model=8, d_state=8)
    a = layer.alpha
    assert torch.all(a > 0) and torch.all(a < 1)
    # initialised spread across timescales, not collapsed to a constant
    assert a.max() - a.min() > 0.05
