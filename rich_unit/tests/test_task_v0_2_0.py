"""Generators are deterministic per seed (SPEC §7/§8).

STATUS: STRUCTURE STUB — skipped until the builder implements the generators.
Asserts: same seed -> identical batch; different seeds -> different batches; the
val split is seed-disjoint from training (SPEC §3).
"""

import pytest

pytestmark = pytest.mark.skip(reason="STRUCTURE STUB — implement per SPEC §3.")


def test_selective_copy_deterministic():
    raise NotImplementedError


def test_assoc_recall_deterministic():
    raise NotImplementedError


def test_val_seeds_disjoint_from_train():
    raise NotImplementedError
