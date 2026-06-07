# EMERGENCE_BRANCH.md — strict interaction/emergence question (DEFERRED stub)

> **Status: STUB. Not implemented now. Records the question so it is not lost.**
> Deferred until A-fix-off (FULL v0.5.0) and A-fix-on complete. Does NOT block the
> current run. No code here — problem + idea only.

## Why this exists
The project's original central question was **super-additivity / emergence**: does
combining two mechanisms in one unit give *more than the sum of each alone* — a 2×2
**interaction** effect, not just a practical margin. FULL v0.5.0 deliberately
demotes this to a **Pareto-dominance margin** (`both` vs the best single) because the
strict interaction is **structurally unmeasurable by that design** — see below. The
strict question is parked here, not abandoned.

## The problem (why strict interaction is unmeasurable in A-fix-off)
A 2×2 interaction `SYN = [save(both) − save(gate)] − [save(linear) − save(neither)]`
requires the `neither` cell (both axes off) to be a **functional baseline** that
reaches the quality band. But in the A-fix-off design the two axes are the *only*
paths from state to output, so `neither` is **memoryless** and sits at chance on
memory tasks (empirical: `neither/sc` test ∈ [0.118, 0.129] at all widths). Then
`params@Q[neither]` is undefined and the interaction is uncomputable. There is no
"neutral memory" outside the two axes to give `neither` capability without adding a
third mechanism.

## Branch idea (one option, to be formalised later)
Change the baseline from `neither` to a **minimal always-on memory**: e.g. a narrow,
non-switchable linear readout `C0·s` present in ALL points. The two axes then become
*additions* on top of a capable base:
- base (both off) = minimal `C0·s` → has memory → reaches Q → finite `params@Q`.
- axis 1 = *widen* the linear readout; axis 2 = *add* the gate readout.
- the 2×2 interaction is now computable (base reaches the band).

## Cost / risk
- Redefines the factors: axes become "additions to a memory base", not "the only two
  readouts". Blurs the clean "two independent readouts of one state" framing.
- The minimal base's width/rank is a new knob that must be fixed and justified.
- Needs a **separate PREREG** and an explicit definition of what counts as the
  interaction term (and a guard that the base isn't so strong it masks the axes).

## Sequencing
Run order: A-fix-off (margin, this FULL) → A-fix-on (selectivity flag, margin) →
THEN, with that picture, decide whether the emergence branch is worth its added
complexity and design it properly. Revisit this stub at that point.
