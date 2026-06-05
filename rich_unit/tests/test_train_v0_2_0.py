"""Early-stopping semantics (pilot 2): patience vs best-so-far + min_delta.

The criterion must compare each eval against BEST-SO-FAR (not a neighbouring eval
or an accumulating reset level), so a slow monotone creep below min_delta still
trips patience — the anti-slow-creep guard the budget relies on.
"""

import rich_unit.train_v0_2_0 as T
from rich_unit.models.rich_unit_v0_2_0 import RichUnitLayer
from rich_unit.models.wrapper_v0_2_0 import SequenceModel
from rich_unit.tasks.selective_copy_v0_2_0 import make_batch, VOCAB_SIZE


def _model():
    return SequenceModel(RichUnitLayer(8, 4), VOCAB_SIZE, 8)


def _run_with_val_sequence(monkeypatch, seq, patience, min_delta, max_steps):
    it = iter(seq)
    monkeypatch.setattr(T, "evaluate", lambda *a, **k: next(it))
    cfg = T.TrainConfig(max_steps=max_steps, lr=1e-3, batch_size=4, eval_every=1,
                        seed=0, patience=patience, min_delta=min_delta)
    return T.train_one(_model(), make_batch, [0], cfg, test_seeds=None)


def test_slow_creep_below_min_delta_stops(monkeypatch):
    # +0.003 per eval, min_delta 0.005 -> never an "improvement" -> stop on patience
    seq = [0.50, 0.503, 0.506, 0.509, 0.512, 0.515, 0.518, 0.521]
    res = _run_with_val_sequence(monkeypatch, seq, patience=3, min_delta=0.005, max_steps=20)
    # eval1 best=0.50(reset); 2,3,4 each <min_delta over best -> counter 1,2,3 -> stop@4
    assert res.stopped_step == 4
    assert abs(res.best_val_acc - 0.509) < 1e-9


def test_real_improvement_does_not_stop(monkeypatch):
    # +0.05 per eval, well above min_delta -> counter keeps resetting -> runs to cap
    seq = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    res = _run_with_val_sequence(monkeypatch, seq, patience=3, min_delta=0.005, max_steps=6)
    assert res.stopped_step == 6
    assert abs(res.best_val_acc - 0.55) < 1e-9


def test_plateau_saw_stops(monkeypatch):
    # noisy plateau around 0.40, no >min_delta gain over best -> stops on patience
    seq = [0.40, 0.398, 0.401, 0.399, 0.402, 0.40, 0.401]
    res = _run_with_val_sequence(monkeypatch, seq, patience=3, min_delta=0.005, max_steps=20)
    assert res.stopped_step == 4   # best=0.40@1; 2,3,4 never beat 0.405 -> stop@4
