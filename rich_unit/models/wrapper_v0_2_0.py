"""Shared model scaffold — ``embed -> [core] -> linear head`` (SPEC §4).

STATUS: STRUCTURE STUB. Signatures and contracts are fixed by the SPEC;
implementation is deferred to the builder phase (after PREREG is frozen, SPEC §0).

Rationale for a dedicated wrapper module: SPEC §4 requires that RichUnit, B1 and
B2 share an IDENTICAL ca  scaffold so that only the core differs. Centralising the
scaffold here keeps that guarantee and lets future variants (A / C / B+C+A) plug
a new core in without touching embed/head — supporting the RESEARCH_MAP
evolution without re-implementing boilerplate per variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceModel(nn.Module):
    """``embed -> core -> linear head`` wrapper shared by all models (SPEC §4).

    Parameters
    ----------
    core:
        Any module mapping ``[B, T, d_model] -> [B, T, d_model]`` (RichUnitLayer,
        the stacked B1 core, or an ``nn.GRU``-based core).
    n_tokens:
        Vocabulary size for the embedding and the classification head.
    d_model:
        Width of the core's input/output (the swept hyperparameter, SPEC §5.2).

    Notes
    -----
    The embedding may be a learned ``nn.Embedding`` or a one-hot projection
    (SPEC §3.1). The head is a single ``nn.Linear`` to ``n_tokens`` logits.
    Token-level cross-entropy is computed by the training loop, not here.
    """

    def __init__(self, core: nn.Module, n_tokens: int, d_model: int) -> None:
        super().__init__()
        raise NotImplementedError("STRUCTURE STUB — implement per SPEC §4.")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens: [B, T] (long) -> logits: [B, T, n_tokens]``."""
        raise NotImplementedError("STRUCTURE STUB — implement per SPEC §4.")
