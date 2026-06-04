"""Report — params@target table, ablations A1-A3, and the §5.4 verdict.

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

Pre-registered thresholds (SPEC §5.0) are NOT hardcoded here. They live, and are
consciously frozen, in rich_unit/PREREG.md, and must be loaded from there once
filled — copying the SPEC's *proposed* numbers (0.95 / 0.20 / median) into code
before they are deliberately approved would defeat pre-registration. They are left
as None until the builder wires up the loader. Do not change after a sweep has
been looked at (SPEC §5.4, §0): a forced change is logged in PREREG.md as a
deviation with a reason.
"""

from __future__ import annotations

# --- Pre-registered thresholds: load from the FROZEN rich_unit/PREREG.md --------
# Left as None on purpose. The builder reads the frozen values from PREREG.md
# (single source of truth); they are NOT pre-filled from the SPEC's proposals.
ACC_TARGET = None          # token-acc target; calibrated on B1 (§5.1), then frozen.
DELTA_MIN = None           # min relative param reduction counted as significant.
N_SEEDS = None             # seeds per point (SPEC §5.0: >=3).
AGGREGATION = None         # rule by which a point is judged to hit ACC_TARGET.


def decide_verdict() -> str:
    """Apply the §5.4 decision rule, no interpretive freedom.

    Returns one of:
      * "H0 refuted on task X"  — params(RichUnit) <= (1 - DELTA_MIN) * params(B1)
        with BOTH meeting ACC_TARGET by the median rule.
      * "H1 fully supported"    — H0 refuted on BOTH tasks.
      * "partial"               — exactly one task (weak/unstable signal, NOT a win).
      * "H0 not refuted"        — incl. win < DELTA_MIN; a full, publishable
                                  negative result, NOT a project failure.

    A1/A2 ablation failures ANNUL any claimed RichUnit profit even if §5.4 formally
    shows a win (SPEC §6).
    """
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §5.3/§5.4/§6.")


def run_ablations() -> None:
    """A1 (W_h=0 frozen state path), A2 (alpha distribution), A3 (baseline fairness).

    All results go into the report — even inconvenient ones (SPEC §6, §8).
    """
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §6.")


if __name__ == "__main__":
    run_ablations()
    print(decide_verdict())
