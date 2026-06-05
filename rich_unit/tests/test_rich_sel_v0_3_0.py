"""RichSel correctness (PILOT3 §7): with a constant forget rate it == old rich.

If ``W_alpha`` is made constant (weight=0, bias=logit(c)), RichSel's alpha_t is the
constant c at every step, so it must reproduce ``RichUnitLayer`` with alpha=c
exactly — given the other projections are shared. This proves the diff is the
intended single change and nothing else.
"""

import torch

from rich_unit.models.rich_sel_v0_3_0 import RichSelLayer
from rich_unit.models.rich_unit_v0_2_0 import RichUnitLayer


def test_constant_alpha_reduces_to_rich():
    torch.manual_seed(0)
    d_model, d_state = 6, 4
    sel = RichSelLayer(d_model, d_state)
    rich = RichUnitLayer(d_model, d_state)

    # constant per-channel forget rate
    c = torch.linspace(0.6, 0.9, d_state)
    logit_c = torch.log(c / (1.0 - c))
    with torch.no_grad():
        sel.W_alpha.weight.zero_()       # alpha_t = sigmoid(0 + bias) = const
        sel.W_alpha.bias.copy_(logit_c)
        rich.alpha_raw.copy_(logit_c)
        # share the remaining projections
        for name in ("W_s", "W_g", "W_h", "W_v"):
            getattr(rich, name).weight.copy_(getattr(sel, name).weight)
            if getattr(sel, name).bias is not None:
                getattr(rich, name).bias.copy_(getattr(sel, name).bias)

    x = torch.randn(3, 9, d_model)
    assert torch.allclose(sel(x), rich(x), atol=1e-6), (sel(x) - rich(x)).abs().max()


def test_alpha_for_varies_with_input():
    """A freshly-initialised RichSel must produce input-dependent (non-constant) alpha."""
    torch.manual_seed(0)
    sel = RichSelLayer(8, 4)
    x = torch.randn(5, 10, 8)
    a = sel.alpha_for(x)                  # [B, T, d_state]
    assert a.shape == (5, 10, 4)
    assert torch.all((a > 0) & (a < 1))
    # varies across positions/inputs (not a degenerate constant)
    assert a.std(dim=(0, 1)).mean() > 1e-3


def test_ablate_freezes_W_h_only():
    sel = RichSelLayer(8, 4, ablate_state=True)
    assert torch.count_nonzero(sel.W_h.weight) == 0
    assert sel.W_h.weight.requires_grad is False
    # selectivity path stays trainable (axis 1 alive)
    assert sel.W_alpha.weight.requires_grad is True
