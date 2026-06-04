"""Synthetic tasks (seeded, deterministic generators).

SPEC §3: both tasks demand BOTH memory (temporal) and nonlinear processing
(channel) — that is where the fused unit is supposed to pay off.

* ``selective_copy_v0_2_0`` — SPEC §3.1.
* ``assoc_recall_v0_2_0``   — SPEC §3.2.

Evolution: variant C (RESEARCH_MAP step 3) requires a task where timing /
synchrony PROVABLY carries information; such a task would be added here as a new
versioned generator before C is evaluated, never by retrofitting these two.
"""
