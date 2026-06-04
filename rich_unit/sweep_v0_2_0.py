"""Minimal-width sweep — the §5 protocol (attempt to falsify H0).

STATUS: STRUCTURE STUB. Implementation deferred to the builder phase.

Protocol (SPEC §5):
  * §5.1 Calibrate ACC_TARGET on B1 ONCE, before evaluating RichUnit; then freeze.
  * §5.2 For each model find the SMALLEST width whose median-over-seed val acc
    >= ACC_TARGET within the fixed budget.
      grid: d_model in {16,32,48,64,96,128} (extend up if none reach target);
            d_state in {4,8,16} for stateful models.
  * §5.3 Record: #params, #layers, steps-to-target, median/min/max acc by seed.
  * Aggregation rule (SPEC §5.0): a point counts as hitting ACC_TARGET only if
    MEDIAN over seed >= ACC_TARGET — never max, never "at least one seed".
  * Run ALL N_SEEDS, drop NO seed (SPEC §0 selection bias, §8).

Thresholds (ACC_TARGET, DELTA_MIN, N_SEEDS, aggregation) are read from the FROZEN
PREREG (rich_unit/PREREG.md, SPEC §5.0) and must be committed before any sweep.
"""

from __future__ import annotations


def run_sweep() -> None:
    """Run the full §5 protocol on both tasks for RichUnit, B1, B2.

    Writes raw per-(model, task, width, seed) records under results/ for
    report_v0_2_0 to aggregate. Performs NO interpretation — the verdict (§5.4)
    lives in the report.
    """
    raise NotImplementedError("STRUCTURE STUB — implement per SPEC §5.")


if __name__ == "__main__":
    run_sweep()
