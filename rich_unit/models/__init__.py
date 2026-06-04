"""Model cores and the shared wrapper.

SPEC §4: every model is wrapped in the SAME scaffold ``embed -> [core] -> linear
head``; only ``[core]`` differs. This package holds:

* ``wrapper_v0_2_0``      — the shared ``embed -> core -> head`` scaffold.
* ``rich_unit_v0_2_0``    — variant B (state-in-gate), the unit under test.
* ``baselines_v0_2_0``    — B1 (stacked temporal+channel) and B2 (GRU ref).

Evolution (RESEARCH_MAP "Порядок исследования"): later falsification steps add
NEW versioned files here (e.g. a dendritic branch for variant A, a complex /
phase channel for variant C). They reuse ``wrapper_v0_2_0`` unchanged so the
core stays the only moving part. Do NOT mutate frozen v0.2.0 files in place —
bump the version instead.
"""
