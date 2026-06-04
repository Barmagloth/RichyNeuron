"""RichUnit — variant B (state-in-gate), the unit under test (SPEC §2).

Per-unit recurrence over time (reference impl = naive ``for t in range(T)`` loop,
T is small — correctness over speed, SPEC §2). Step t:

    s_t = alpha * s_{t-1} + (1 - alpha) * (x_t @ W_s)      # [B, d_state]
    g   = sigmoid(x_t @ W_g + s_t @ W_h)                   # [B, d_model]
    v   = x_t @ W_v                                         # [B, d_model]
    y_t = v * g                                             # elementwise

The coupling that makes this "rich" (PROBLEM.md): the temporal state ``s`` enters
the ARGUMENT of the unit's own multiplicative gate, fusing temporal-mixing and
channel-mixing into one op.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RichUnitLayer(nn.Module):
    """One rich-unit layer: ``[B, T, d_model] -> [B, T, d_model]`` (SPEC §2).

    Weights (SPEC §2): ``W_s:[d_model,d_state]``, ``W_g:[d_model,d_model]`` (bias),
    ``W_h:[d_state,d_model]``, ``W_v:[d_model,d_model]`` (bias). ``alpha =
    sigmoid(alpha_raw)``, learnable ``[d_state]`` in (0,1), initialised spread over
    [0.5, 0.95] (different timescales, not a constant). Initial state ``s_0 = 0``.

    Parameters
    ----------
    ablate_state:
        Ablation A1 (SPEC §6). If True, the state path into the gate is zeroed and
        frozen (``W_h`` fixed at 0). The state is still computed but cannot reach
        the output, so any accuracy drop measures how much the gate relied on it.
    """

    def __init__(self, d_model: int, d_state: int, ablate_state: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.ablate_state = ablate_state

        # nn.Linear(in, out) computes x @ W.T; this realises the SPEC's x_t @ W_*
        # with the matching shapes. Bias on g and v only (SPEC §2).
        self.W_s = nn.Linear(d_model, d_state, bias=False)
        self.W_g = nn.Linear(d_model, d_model, bias=True)
        self.W_h = nn.Linear(d_state, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=True)

        # alpha_raw such that sigmoid(alpha_raw) is spread across [0.5, 0.95].
        target = torch.linspace(0.5, 0.95, d_state)
        alpha_raw = torch.log(target / (1.0 - target))     # logit
        self.alpha_raw = nn.Parameter(alpha_raw)

        if ablate_state:
            nn.init.zeros_(self.W_h.weight)
            self.W_h.weight.requires_grad_(False)

    @property
    def alpha(self) -> torch.Tensor:
        """Effective decay ``sigmoid(alpha_raw)`` in (0, 1), shape [d_state].

        Exposed for ablation A2 (alpha-collapse logging, SPEC §6).
        """
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Naive per-t scan (SPEC §2): collect y_t into a list, ``torch.stack``."""
        B, T, _ = x.shape
        alpha = self.alpha                       # [d_state]
        s = x.new_zeros(B, self.d_state)         # s_0 = 0

        # Input-driven projections are time-pointwise -> batch them outside the loop.
        Ws_x = self.W_s(x)                       # [B, T, d_state]
        Wg_x = self.W_g(x)                       # [B, T, d_model]
        Wv_x = self.W_v(x)                       # [B, T, d_model]

        ys = []
        for t in range(T):
            s = alpha * s + (1.0 - alpha) * Ws_x[:, t]      # [B, d_state]
            g = torch.sigmoid(Wg_x[:, t] + self.W_h(s))     # [B, d_model]
            ys.append(Wv_x[:, t] * g)                       # [B, d_model]
        return torch.stack(ys, dim=1)            # [B, T, d_model]
