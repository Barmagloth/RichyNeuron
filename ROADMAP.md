# ROADMAP — evolution of the repository

This repo is scaffolded for the **v0.2.0 prototype** (variant **B**, state-in-gate)
but laid out so later falsification steps slot in without reshuffling. The order
below is the **falsification order** from `docs/RESEARCH_MAP_v0_2_0.md` — sorted by
**cost of refuting H0**, NOT by belief in success. Each step's default expected
outcome is H0 (no win), which is a normal, publishable result.

> A step is attempted **only if the previous step refuted H0** with the
> pre-registered `DELTA_MIN`. Otherwise: stop, record the negative result with
> numbers. (RESEARCH_MAP "Порядок исследования".)

## Step 1 — variant B (this version, v0.2.0)
`RichUnitLayer` vs the split stack B1 (`temporal recurrent → SwiGLU`) + GRU ref B2,
on Selective Copy and Associative Recall. Files: `models/rich_unit_v0_2_0.py`,
`models/baselines_v0_2_0.py`, `tasks/*_v0_2_0.py`. Verdict by SPEC §5.4.

## Step 2 — variant A on top of B (only if Step 1 refutes H0)
Add a dendritic branch (grouped-linear) to the rich core; test whether it adds a
`≥ DELTA_MIN` gain over pure B at equal params. New file, e.g.
`models/rich_unit_dendritic_vX_Y_Z.py`, reusing `models/wrapper_*`. No edits to
frozen B files.

## Step 3 — variant C (only with a task where timing carries information)
Complex / phase channel (rate + phase multiplexing). Requires FIRST building a
task where synchrony/timing **provably** carries information (else the phase
channel is empty) — added under `tasks/`. New core file under `models/`.

## Step 4 — B+C+A combined
Only after each piece individually refuted H0. Highest density, highest risk of
the loss landscape collapsing into a trivial regime.

## Invariants that hold across all steps
- **Versioned files** (`vX_Y_Z`); bump on first logic change — never mutate a
  frozen file in place (SPEC §7).
- **Shared scaffold** `embed → [core] → head` (`models/wrapper_*`) reused by every
  variant; only the core changes (SPEC §4).
- **Pre-registration before every sweep** (`rich_unit/PREREG.md`, SPEC §5.0) and
  the §0 anti-bias discipline (no HARKing, no seed-dropping, equal baseline budget).
- Every claimed profit is gated by the A1/A2/A3 ablations (SPEC §6).
