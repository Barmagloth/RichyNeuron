"""AblationUnit — 2x2 readout ablation, fixed alpha (FULL_ABLATION_SPEC v0.4.1).

A-fix-off run: the forget rate alpha is a FIXED learnable parameter in all points
(selectivity / W_alpha is NOT used here — it is a separate write-state factor for a
later A-fix-on run). The 2x2 toggles two independent ways of READING the state:

    alpha   = sigmoid(alpha_raw)                              # fixed, all points
    s_t     = alpha * s_{t-1} + (1 - alpha) * (x_t . W_s)
    gate_in = x_t . W_g + (W_h . s_t  if gate_readout else 0)     # axis 2 (gate)
    g       = sigmoid(gate_in)
    y_t     = (x_t . W_v) * g + (C . s_t  if linear_readout else 0)   # axis 1 (linear)

Four points (differ ONLY by the two flags):
    neither  linear=off gate=off   y = (x.W_v).sig(x.W_g)
    linear   linear=on  gate=off   + C.s              (Mamba-style linear readout)
    gate     linear=off gate=on    gate modulated by s   == rich (pilots 1-2)
    both     linear=on  gate=on    both readouts of the same state

Parameter integrity (so params@Q comparisons are honest): C exists only when
linear_readout, W_h exists only when gate_readout — no dangling unused tensors.
``ablate_state`` (A1) zeros+freezes W_h (only meaningful when gate_readout).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AblationUnit(nn.Module):
    """One ablation unit: ``[B, T, d_model] -> [B, T, d_model]`` (FULL §1, v0.4.1).

    Parameters
    ----------
    linear_readout: axis 1 — add ``C . s_t`` to the output.
    gate_readout:   axis 2 — add ``W_h . s_t`` inside the gate argument.
    ablate_state:   A1 — zero and freeze ``W_h`` (requires gate_readout).
    """

    def __init__(self, d_model: int, d_state: int,
                 linear_readout: bool, gate_readout: bool,
                 ablate_state: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.linear_readout = linear_readout
        self.gate_readout = gate_readout
        self.ablate_state = ablate_state

        # Always present (state write + input-driven gate/value).
        self.W_s = nn.Linear(d_model, d_state, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        target = torch.linspace(0.5, 0.95, d_state)
        self.alpha_raw = nn.Parameter(torch.log(target / (1.0 - target)))   # fixed alpha

        # Created ONLY when the corresponding axis is on (no dangling params).
        self.W_h = nn.Linear(d_state, d_model, bias=False) if gate_readout else None
        self.C = nn.Linear(d_state, d_model, bias=False) if linear_readout else None

        if ablate_state and self.W_h is not None:
            nn.init.zeros_(self.W_h.weight)
            self.W_h.weight.requires_grad_(False)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        alpha = self.alpha
        s = x.new_zeros(B, self.d_state)

        Ws_x = self.W_s(x)
        Wg_x = self.W_g(x)
        Wv_x = self.W_v(x)

        ys = []
        for t in range(T):
            s = alpha * s + (1.0 - alpha) * Ws_x[:, t]          # [B, d_state]
            gate_in = Wg_x[:, t]
            if self.gate_readout:
                gate_in = gate_in + self.W_h(s)                 # axis 2
            y = Wv_x[:, t] * torch.sigmoid(gate_in)
            if self.linear_readout:
                y = y + self.C(s)                               # axis 1
            ys.append(y)
        return torch.stack(ys, dim=1)
