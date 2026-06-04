"""Baselines — mandatory; without them the project is meaningless (SPEC §4).

B1 (stacked): ``[TemporalMixer] -> [ChannelMixer]`` — the honest split stack that
RichUnit is supposed to beat at equal quality.
  * TemporalMixer: minimal linear recurrent ``h_t = a*h_{t-1} + b*u_t`` in a
    d_state space (NO gate, NO link to the channel part), linear output. Same scan
    form as RichUnit's state, but it cannot see the channel mixer.
  * ChannelMixer: SwiGLU ``(x W1) * silu(x W2) @ W3``, stateless.

B2 (GRU ref): a single ``nn.GRU`` layer — sanity baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMixer(nn.Module):
    """Diagonal linear recurrent, no gate, linear output (SPEC §4 B1 first half).

    ``u_t = x_t @ B_in`` ([d_state]); ``h_t = a*h_{t-1} + b*u_t``; ``y_t = h_t @ C_out``.
    ``a = sigmoid(a_raw)`` keeps the recurrence stable; spread like RichUnit's alpha.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_state = d_state
        self.B_in = nn.Linear(d_model, d_state, bias=False)
        self.C_out = nn.Linear(d_state, d_model, bias=True)
        self.b = nn.Parameter(torch.ones(d_state))
        target = torch.linspace(0.5, 0.95, d_state)
        self.a_raw = nn.Parameter(torch.log(target / (1.0 - target)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        a = torch.sigmoid(self.a_raw)            # [d_state]
        u = self.B_in(x)                         # [B, T, d_state]
        h = x.new_zeros(B, self.d_state)
        hs = []
        for t in range(T):
            h = a * h + self.b * u[:, t]         # [B, d_state]
            hs.append(h)
        h_seq = torch.stack(hs, dim=1)           # [B, T, d_state]
        return self.C_out(h_seq)                 # [B, T, d_model]


class ChannelMixer(nn.Module):
    """SwiGLU ``(x W1) * silu(x W2) @ W3``, stateless (SPEC §4 B1 second half).

    Hidden width = d_model (minimal, no extra hyperparameter; swept via d_model).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_model, d_model, bias=False)
        self.W2 = nn.Linear(d_model, d_model, bias=False)
        self.W3 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(self.W1(x) * F.silu(self.W2(x)))


class StackedTemporalChannel(nn.Module):
    """B1 core: TemporalMixer -> ChannelMixer (SPEC §4). ``[B,T,d_model]->[B,T,d_model]``.

    The two stages are sequential and do NOT see each other per-unit — exactly the
    split that RichUnit fuses.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.temporal = TemporalMixer(d_model, d_state)
        self.channel = ChannelMixer(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel(self.temporal(x))


class GRUCore(nn.Module):
    """B2 core: a single ``nn.GRU`` layer (SPEC §4). ``[B,T,d_model]->[B,T,d_model]``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return y
