# PRIOR_ART_SUMMARY.md — consolidated novelty map of the "rich unit" program

> One-page map across all investigated branches. Detail + sources: PRIOR_ART_FINDINGS.md.
> Status legend: **CLOSED** = mechanism + a matching ablation/result are published ·
> **COVERED** = the idea is published, but the param/quality benefit is modest/unclear ·
> **OPEN(thin)** = no exact prior-art match found (candidate for novelty, needs a focused check).

## A. Core "rich unit" mechanisms (the FULL-ablation axes)
| # | Mechanism (ours) | Prior art | Status |
|---|---|---|---|
| 1 | **Selectivity** — input-dependent decay `α=σ(x·W_α)` | Mamba-1 selective Δ(x) (2023) | **CLOSED** |
| 2 | **Linear readout** — `y += C·s` | standard SSM output (S4/S5/Mamba) | **CLOSED** |
| 3 | **Gate readout** — state in the output gate `g=σ(W_g·x + W_h·s)` | LSTM output gate (1997); xLSTM **sLSTM** `o_t=σ(Wx+r·h+b)` (2024); **GateLORD** `p(x,h)⊙o(x,h)` on a diagonal RNN with the exact ablation, gate helps (NeurIPS 2024) | **CLOSED (decisive)** |

## B. Phase / complex channel branches (A-E)
The verdict splits by interpretation — and both readings are occupied:

| Branch | Idea | Prior art | Status |
|---|---|---|---|
| A | complex/phase recurrent, unitary | Unitary RNN (Arjovsky 2016); Deep Complex Nets (2017) | **CLOSED** |
| B | oscillatory SSM | **LinOSS** Oscillatory SSM (2024) — oscillation as a long-memory *dynamics* device | **CLOSED** (as dynamics) |
| — | complex state as a *memory tool* (interp. 1) | S4 (complex eigenvalues), Mamba-3 (complex state) | **CLOSED (hard)** |
| D | phase as a *separate channel* / rate+phase multiplexing (interp. 2) | Deep Complex Nets (2017: amplitude=rate, phase=timing); Phase-Associative Memory (2026, phase=semantics, "competitive with real-valued") | **COVERED** (benefit modest) |
| E | neuroscience phase coding (origin) | theta-gamma phase code; computational working-memory models (multiplex item+order) | **COVERED** (established) |
| C | **`β⊙|z|`** — amplitude of complex state drives/gates the recurrence | **no exact named match found** (adjacent to magnitude-gating / complex-RNN dynamics, but the specific "amplitude-envelope modulates the state update" recurrence not located) | **OPEN(thin)** |

## Bottom line
- **Almost everything is prior art.** Selectivity = Mamba; linear readout = SSM; state-in-
  gate = LSTM/sLSTM/GateLORD (with the ablation + positive result already published);
  complex/oscillatory dynamics = unitary RNN/LinOSS/S4/Mamba-3; phase-as-channel =
  Deep Complex Nets/theta-gamma/PAM. None is brand-new.
- **The single item with no exact prior-art match is branch C** — the `β⊙|z|`
  amplitude-driven-state mechanism (amplitude of the complex/oscillatory state modulating
  the recurrence itself, not just a readout). It is the only candidate left that could be
  non-trivial; it needs a focused search (terms: "magnitude-gated linear recurrence",
  "amplitude-dependent decay", "modulus gating complex RNN") before any novelty claim.
- **Everything else: re-deriving published results.** Running FULL on axes 1-3, or a
  phase-channel experiment under interpretation 1 or 2, would reproduce known work.

## What this buys
A clean, honest novelty boundary for the whole program, established by reading (no CPU).
Lasting assets: the disciplined reproducible CPU harness + honest pilot findings (situated
against the literature) + this prior-art map. Any continuation should target branch C
(if that is the real idea) or a genuinely uncovered regime, defined against this map first.
