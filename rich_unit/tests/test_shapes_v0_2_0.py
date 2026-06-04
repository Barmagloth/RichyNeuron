"""Shape + forward/backward smoke tests (SPEC §7). CPU-only (SPEC §8)."""

import torch

from rich_unit.models.rich_unit_v0_2_0 import RichUnitLayer
from rich_unit.models.baselines_v0_2_0 import StackedTemporalChannel, GRUCore
from rich_unit.models.wrapper_v0_2_0 import SequenceModel
from rich_unit.tasks.selective_copy_v0_2_0 import VOCAB_SIZE


def _cores(d_model=16, d_state=4):
    return {
        "rich": RichUnitLayer(d_model, d_state),
        "B1": StackedTemporalChannel(d_model, d_state),
        "B2": GRUCore(d_model),
    }


def test_core_shapes():
    x = torch.randn(3, 7, 16)
    for name, core in _cores().items():
        y = core(x)
        assert y.shape == x.shape, (name, y.shape)


def test_wrapped_model_shapes():
    tokens = torch.randint(0, VOCAB_SIZE, (3, 7))
    for name, core in _cores().items():
        model = SequenceModel(core, VOCAB_SIZE, 16)
        logits = model(tokens)
        assert logits.shape == (3, 7, VOCAB_SIZE), (name, logits.shape)


def test_forward_backward_cpu():
    tokens = torch.randint(0, VOCAB_SIZE, (3, 7))
    for name, core in _cores().items():
        model = SequenceModel(core, VOCAB_SIZE, 16)
        loss = model(tokens).float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() for g in grads), name


def test_ablate_state_freezes_W_h():
    layer = RichUnitLayer(8, 4, ablate_state=True)
    assert torch.count_nonzero(layer.W_h.weight) == 0
    assert layer.W_h.weight.requires_grad is False
