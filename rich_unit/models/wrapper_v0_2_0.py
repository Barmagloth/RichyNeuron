"""Shared model scaffold — ``embed -> [core] -> linear head`` (SPEC §4).

SPEC §4 requires RichUnit, B1 and B2 to share an IDENTICAL scaffold so only the
core differs. Centralising it here keeps that guarantee and lets future variants
(A / C / B+C+A) plug a new core in without touching embed/head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    """``embed -> core -> linear head`` wrapper shared by all models (SPEC §4).

    Parameters
    ----------
    core:
        Any module mapping ``[B, T, d_model] -> [B, T, d_model]``.
    vocab_size:
        Vocabulary size for the embedding and the classification head.
    d_model:
        Width of the core's input/output (the swept hyperparameter, SPEC §5.2).
    """

    def __init__(self, core: nn.Module, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.core = core
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens: [B, T] (long) -> logits: [B, T, vocab_size]``."""
        x = self.embed(tokens)          # [B, T, d_model]
        x = self.core(x)                # [B, T, d_model]
        return self.head(x)             # [B, T, vocab_size]
