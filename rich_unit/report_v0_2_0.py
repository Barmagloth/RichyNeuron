"""Report — params@target table, ablations A1-A3, and the §5.4 verdict.

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

Pre-registered constants (SPEC §5.0) — proposals from the SPEC, to be FROZEN in
rich_unit/PREREG.md and committed BEFORE the first sweep. Do not change after a
sweep has been looked at (SPEC §5.4, §0): a forced change is logged as a deviation
with a reason.
"""

from __future__ import annotations

# --- Pre-registered thresholds (mirror rich_unit/PREREG.md; SPEC §5.0) ---------
ACC_TARGET = 0.95          # SPEC §5.0 proposal; calibrated on B1 (§5.1), then frozen.
DELTA_MIN = 0.20           # min relative param reduction counted as significant.
N_SEEDS = 5                # >=3 required, 5 proposed.
AGGREGATION = "median"     # a point hits ACC_TARGET only if MEDIAN over seed >=.


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
