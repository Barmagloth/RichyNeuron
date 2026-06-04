"""Recurrence correctness — s_t against a hand-written numeric reference (SPEC §7/§8).

STATUS: STRUCTURE STUB — skipped until the builder implements RichUnitLayer.
Computes s_t independently from the SPEC §2 equations and asserts the layer's
internal state matches, so a refactor of the scan can never silently drift.
"""

import pytest

pytestmark = pytest.mark.skip(reason="STRUCTURE STUB — implement per SPEC §2.")


def test_state_matches_manual_reference():
    raise NotImplementedError


def test_initial_state_is_zero():
    raise NotImplementedError
