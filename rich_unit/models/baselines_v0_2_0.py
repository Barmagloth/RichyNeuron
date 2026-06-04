"""Baselines — mandatory; without them the project is meaningless (SPEC §4).

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

B1 (stacked): ``[TemporalMixer] -> [ChannelMixer]`` — the honest split stack that
RichUnit is supposed to beat at equal quality.
  * TemporalMixer: minimal linear recurrent ``h_t = a*h_{t-1} + b*x_t``, NO gate,
    NO link to the channel part, linear output.
  * ChannelMixer: SwiGLU ``(x W1) * silu(x W2) @ W3``, stateless.
  temporal and channel are separated — they do NOT see each other per-unit, which
  is exactly what RichUnit fuses.

B2 (GRU ref): a single ``nn.GRU`` layer — sanity baseline.

ACC_TARGET is calibrated on B1 only, BEFORE evaluating RichUnit (SPEC §5.1).
B1/B2 must train with the SAME budget/optimizer/lr-sweep as RichUnit (SPEC §6 A3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StackedTemporalChannel(nn.Module):
    """B1 core: linear recurrent temporal mixer -> SwiGLU channel mixer (SPEC §4).

    ``[B, T, d_model] -> [B, T, d_model]``.
    """

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        raise NotImplementedError("STRUCTURE STUB — implement B1 per SPEC §4.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("STRUCTURE STUB — implement B1 per SPEC §4.")


class GRUCore(nn.Module):
    """B2 core: a single ``nn.GRU`` layer (SPEC §4).

    ``[B, T, d_model] -> [B, T, d_model]``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        raise NotImplementedError("STRUCTURE STUB — implement B2 per SPEC §4.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("STRUCTURE STUB — implement B2 per SPEC §4.")
