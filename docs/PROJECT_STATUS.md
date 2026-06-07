# PROJECT_STATUS.md — "rich unit" line: status & honest closure (2026-06)

> Synthesis of the whole arc. Verdict-bearing FULL was NOT run: a prior-art check
> (cheap, before CPU) established the design's novelty boundary. This is a valid,
> disciplined outcome, not a failure (PROBLEM.md / RESEARCH_MAP §0 / PRIOR_ART_CHECK §5).

## The question
Can returning complexity *into the unit* (temporal state that (a) is read out linearly
and/or (b) modulates the unit's own output gate, possibly with input-dependent
forgetting) make the architecture more **parameter-efficient at equal quality** than a
split temporal+channel stack — on the mass-market stack, CPU, plain backprop? H0
(default, against us): no. Goal = optimality, not "smarter" (HARKing forbidden).

## What was actually done (reproducible, CPU, committed)
- **Harness:** seeded synthetic tasks (selective copy; associative recall; counting/
  majority probes), model-agnostic AdamW loop with honest early-stop (patience vs
  best-so-far + min_delta) and a held-out test split (test@best-val), checkpoint/resume,
  23 unit tests (incl. recurrence-vs-hand-reference and "ablation reduces to baseline").
- **Pilot 1** (selective copy, undertrained, single lr): rich (gate-readout) looked
  ~2.5–11× more param-efficient than the split stack B1 — but in an under-trained,
  weak-baseline, single-lr regime (untrustworthy).
- **Pilot 2** (converged budget, per-cell lr, 5 seeds, held-out test): the gap shrank
  but survived — rich ceiling ~0.793 vs B1 ~0.738; rich matches B1's ceiling at **~2.5×
  fewer params**; **A1: zeroing the state→gate path drops test 0.80→0.13 (≈chance), Δ≈0.67**
  → the gate-readout is genuinely load-bearing, not a param artifact. Caveat: low-accuracy
  regime (GRU solves at 1.0 while rich/B1 cap ~0.74–0.79).
- **Pilot 3** (RichSel = + input-dependent forgetting): with selectivity ON, the
  gate-readout stays load-bearing (**A1 Δ≈0.84**, ablated→chance); alpha non-degenerate;
  selectivity lifts achievable test (~0.97). Green-light gate for a full ablation.
- **FULL design (v0.5.0):** redesigned twice for honesty — (i) the `sel` point was
  degenerate (state had no output path with gate off) → reframed to two independent
  readout axes (linear `C·s` vs gate `W_h·s`), alpha fixed (A-fix-off); (ii) the strict
  2×2 super-additivity was uncomputable (the `neither` cell is memoryless → params@Q
  undefined; empirically stuck at chance) → primary criterion switched to a well-posed
  **Pareto-dominance margin** of `both` over the best single readout; (iii) associative
  recall proved unsolvable by this tiny fixed-alpha class (caps at the trivial ~0.5 floor;
  GRU needs d128) and counting/majority saturated at d16 → no viable d_model-bound second
  task was found for the class.

## Why FULL was NOT run — the novelty boundary (PRIOR_ART_FINDINGS.md)
A targeted code+literature check (no CPU) established that the one component that is NOT
in the Mamba line — **axis 2, the state-dependent multiplicative output gate** — is a
**known mechanism with a published, matching ablation**:
- LSTM (1997) and **xLSTM sLSTM** (2024): output gate `o_t=σ(W_o x + r_o h_{t-1} + b_o)`
  depends on the recurrent state.
- **GateLORD** (Zucchet & Orvieto, NeurIPS 2024, arXiv 2405.21064): output `p(x,h)⊙o(x,h)`
  on a **diagonal linear RNN** (our regime), with an ablation of the multiplicative
  state-dependent gate vs without — and it **helps**. This is essentially our FULL,
  already published.
- HGRN-2 / Gated DeltaNet corroborate the value of state-dependent gating for recall.
- The other two ingredients were never ours: selectivity = Mamba-1; linear readout = the
  standard SSM output.

**Conclusion:** all three building blocks of the "rich unit" are prior art, and the one
controlled ablation worth running has been done. Running FULL would re-derive a published
result. Per PRIOR_ART_CHECK §5 this is the intended, valuable, cheap outcome (~20h CPU
saved; novelty boundary mapped honestly).

## What has lasting value here
- A small, **disciplined, reproducible CPU research harness** with strong anti-self-
  deception hygiene (pre-registration, seed-paired ablations, held-out test, identical
  budgets, "inconclusive ≠ fitting").
- **Honest empirical findings**: state-dependent gating is load-bearing and modestly
  param-efficient at equal quality in this minimal regime (Pilots 2–3) — consistent with
  the literature (GateLORD/sLSTM), now situated against it.
- A clean **prior-art map** of where the "rich unit" sits relative to Mamba-1/2/3,
  Griffin/Hawk, xLSTM, GateLORD, HGRN-2, GLA, Gated DeltaNet.

## Status: line closed pending a re-scoped, genuinely-open question
FULL is not run. If the project continues, it needs a question NOT already answered by
the above (e.g. a regime/measurement the cited works did not cover) — to be defined
against PRIOR_ART_FINDINGS.md before any new CPU. Pilots 1–3 + the prior-art map stand
as the honest deliverable.
