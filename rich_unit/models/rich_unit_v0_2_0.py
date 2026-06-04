"""RichUnit — variant B (state-in-gate), the unit under test (SPEC §2).

STATUS: STRUCTURE STUB. The equations below are the FROZEN spec; the builder
fills the body after PREREG is committed (SPEC §0, §5.0).

Per-unit recurrence over time (reference impl = naive ``for t in range(T)`` loop,
T is small — correctness over speed, SPEC §2):

    s_t = alpha * s_{t-1} + (1 - alpha) * (x_t @ W_s)      # [B, d_state]
    g   = sigmoid(x_t @ W_g + s_t @ W_h)                   # [B, d_model]
    v   = x_t @ W_v                                         # [B, d_model]
    y_t = v * g                                             # elementwise

The coupling that makes this "rich" (PROBLEM.md): the temporal state ``s`` enters
the ARGUMENT of the unit's own multiplicative gate, fusing temporal-mixing and
channel-mixing into one op. This is exactly what stacking separate temporal /
channel layers does NOT give per-unit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RichUnitLayer(nn.Module):
    """One rich-unit layer: ``[B, T, d_model] -> [B, T, d_model]`` (SPEC §2).

    Parameters / weights (SPEC §2):
        W_s: [d_model, d_state]   W_g: [d_model, d_model]  (bias)
        W_h: [d_state, d_model]   W_v: [d_model, d_model]  (bias)
        alpha = sigmoid(alpha_raw), a learnable vector [d_state] guaranteed in
        (0, 1). Init alpha_raw so starting alpha in [0.5, 0.95], spread across
        rows (different timescales) — NOT a constant. Initial state s_0 = 0.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        raise NotImplementedError("STRUCTURE STUB — implement per SPEC §2.")

    @property
    def alpha(self) -> torch.Tensor:
        """Effective decay ``sigmoid(alpha_raw)`` in (0, 1), shape [d_state].

        Exposed for ablation A2 (alpha-collapse logging, SPEC §6).
        """
        raise NotImplementedError("STRUCTURE STUB — implement per SPEC §2.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Naive per-t scan (SPEC §2): collect y_t into a list, ``torch.stack``."""
        raise NotImplementedError("STRUCTURE STUB — implement per SPEC §2.")
