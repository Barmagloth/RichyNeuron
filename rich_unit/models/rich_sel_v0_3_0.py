"""RichSel — rich unit with input-dependent forget rate (PILOT3 §2, v0.3.0).

Minimal diff from ``RichUnitLayer``: the forget rate ``alpha`` stops being a
time-fixed parameter and becomes a thin, input-dependent function (axis 1,
selectivity, Mamba-Δ form). Everything else — the W_s/W_g/W_h/W_v projections, the
state-in-gate readout (axis 2), the y = v·g output — is IDENTICAL.

Current rich (axis 2 only):
    alpha   = sigmoid(alpha_raw)                       # [d_state], a parameter
    s_t     = alpha * s_{t-1} + (1-alpha) * (x_t·W_s)
    g       = sigmoid(x_t·W_g + W_h·s_t)
    y_t     = (x_t·W_v) * g

RichSel (axis 1 added, thin/linear form):
    alpha_t = sigmoid(x_t·W_alpha)                     # [B, d_state], NEW
    s_t     = alpha_t * s_{t-1} + (1-alpha_t) * (x_t·W_s)
    g       = sigmoid(x_t·W_g + W_h·s_t)               # axis 2, unchanged
    y_t     = (x_t·W_v) * g

If ``W_alpha`` is replaced by a constant (weight=0, bias=logit(c)) this reduces
EXACTLY to ``RichUnitLayer`` with alpha=c (see tests).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RichSelLayer(nn.Module):
    """Rich unit + selectivity: ``[B, T, d_model] -> [B, T, d_model]`` (PILOT3 §2).

    Identical to ``RichUnitLayer`` except ``alpha`` is now ``sigmoid(x_t·W_alpha)``
    (``W_alpha: Linear(d_model, d_state, bias=True)``); the old ``alpha_raw`` is
    removed. ``W_alpha.bias`` is initialised so the starting alpha averages ~0.7-0.9
    (unsaturated), spread across state channels like rich's alpha init.

    Parameters
    ----------
    ablate_state:
        Ablation A1. If True, the state->gate readout ``W_h`` is zeroed and frozen.
        The selectivity path ``W_alpha`` is NOT touched (axis 1 stays alive), so the
        ablation isolates whether the readout (axis 2) still does work.
    """

    def __init__(self, d_model: int, d_state: int, ablate_state: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ablate_state = ablate_state

        self.W_s = nn.Linear(d_model, d_state, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=True)
        self.W_h = nn.Linear(d_state, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=True)

        # Axis 1: input-dependent forget rate. Thin/linear (NOT an MLP) by spec.
        self.W_alpha = nn.Linear(d_model, d_state, bias=True)
        # Bias spread so the start-of-training mean alpha sits in [0.7, 0.9] per
        # channel (different base timescales), unsaturated away from 0/1.
        base = torch.linspace(0.7, 0.9, d_state)
        with torch.no_grad():
            self.W_alpha.bias.copy_(torch.log(base / (1.0 - base)))   # logit

        if ablate_state:
            nn.init.zeros_(self.W_h.weight)
            self.W_h.weight.requires_grad_(False)

    def alpha_for(self, x: torch.Tensor) -> torch.Tensor:
        """Input-dependent forget rates ``sigmoid(x·W_alpha)`` -> [B, T, d_state].

        Exposed for the trainability/selectivity logging (PILOT3 §4): does alpha
        actually vary over positions/inputs, or has it degenerated to a constant?
        """
        return torch.sigmoid(self.W_alpha(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Naive per-t scan; alpha_t is time-pointwise so W_alpha(x) is batched out."""
        B, T, _ = x.shape
        s = x.new_zeros(B, self.d_state)         # s_0 = 0

        Ws_x = self.W_s(x)                        # [B, T, d_state]
        Wg_x = self.W_g(x)                        # [B, T, d_model]
        Wv_x = self.W_v(x)                        # [B, T, d_model]
        alpha = torch.sigmoid(self.W_alpha(x))   # [B, T, d_state]   axis 1

        ys = []
        for t in range(T):
            a = alpha[:, t]                                  # [B, d_state]
            s = a * s + (1.0 - a) * Ws_x[:, t]               # [B, d_state]
            g = torch.sigmoid(Wg_x[:, t] + self.W_h(s))      # [B, d_model]
            ys.append(Wv_x[:, t] * g)                        # [B, d_model]
        return torch.stack(ys, dim=1)            # [B, T, d_model]
