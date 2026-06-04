"""Shape + forward/backward smoke tests (SPEC §7).

STATUS: STRUCTURE STUB — skipped until the builder implements the models.
Verifies: RichUnit / B1 / B2 map [B, T, d_model] -> [B, T, d_model], the wrapped
model maps [B, T] -> [B, T, n_tokens], and a backward pass produces grads. CPU-only.
"""

import pytest

pytestmark = pytest.mark.skip(reason="STRUCTURE STUB — implement per SPEC §2/§4.")


def test_rich_unit_shapes():
    raise NotImplementedError


def test_baseline_shapes():
    raise NotImplementedError


def test_forward_backward_cpu():
    raise NotImplementedError
