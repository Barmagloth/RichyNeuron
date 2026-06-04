# ROADMAP — research order ↔ repository layout

**This file is NOT a source of truth for the research plan.** The variants, their
profits/risks, the falsification order and its rationale live in
[`docs/RESEARCH_MAP_v0_2_0.md`](docs/RESEARCH_MAP_v0_2_0.md) ("Порядок
исследования") and the executable design in [`docs/SPEC_v0_2_0.md`](docs/SPEC_v0_2_0.md).
To avoid two diverging sources of truth, this file only **maps** those documents
onto the repo's file layout. If they ever disagree, the docs win — fix this file.

## Where each research step lives in the tree

Steps, ordering and "only proceed if H0 refuted" gating are defined in
RESEARCH_MAP — not restated here. Their code locations:

| RESEARCH_MAP step | New code goes in | Reuses unchanged |
|---|---|---|
| variant **B** (current, v0.2.0) | `models/rich_unit_v0_2_0.py`, `models/baselines_v0_2_0.py`, `tasks/*_v0_2_0.py` | `models/wrapper_*` |
| variant **A** on top of B | new `models/rich_unit_dendritic_vX_Y_Z.py` | wrapper, tasks |
| variant **C** | new `tasks/` generator (timing-carries-info) **first**, then new `models/` core | wrapper |
| **B+C+A** | new combined `models/` core | wrapper, tasks |

## Repo conventions that hold across all steps

- **Versioned files** (`vX_Y_Z`); bump on first logic change — never mutate a
  frozen file in place (SPEC §7).
- **Shared scaffold** `embed → [core] → head` (`models/wrapper_*`) reused by every
  variant; only the core changes (SPEC §4).
- **Pre-registration before every sweep** (`rich_unit/PREREG.md`, SPEC §5.0) and
  the RESEARCH_MAP §0 anti-bias discipline (no HARKing, no seed-dropping, equal
  baseline budget). Each claimed profit is gated by the A1/A2/A3 ablations (SPEC §6).
