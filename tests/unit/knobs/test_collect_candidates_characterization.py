"""Characterization of knobs.runtime._collect_candidates (sampling/committee)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from traigent.knobs import runtime as rt
from traigent.knobs.composites import CompositeArm, StageArm
from traigent.knobs.runtime import ResultKind, StageRunner


def _call(body, stages=None, registry=None, config=None, calibrated=None):
    return rt._collect_candidates(
        body, stages or {}, registry or {}, config or {}, calibrated or {}, {}, {}
    )


def _view(cands):
    return [(c.output, c.key) for c in cands]


def _body(arms, cardinality=None):
    return NS(arms=tuple(arms), cardinality=cardinality)


def test_sampling_stage_returns_k_samples_keyed_by_identity():
    runner = StageRunner(run=lambda item: ["a", "b", "a"])
    out, excl = _call(
        _body([StageArm("s")], "k"), {"s": runner}, config={"k": 3}
    )
    assert _view(out) == [("a", "a"), ("b", "b"), ("a", "a")] and excl == 0


def test_sampling_stage_uses_key_fn():
    runner = StageRunner(run=lambda item: ["Ab", "aB"], key_fn=str.lower)
    out, _ = _call(_body([StageArm("s")], "k"), {"s": runner}, calibrated={"k": 2})
    assert _view(out) == [("Ab", "ab"), ("aB", "ab")]


def test_sampling_stage_sample_count_mismatch_raises():
    runner = StageRunner(run=lambda item: ["a"])
    with pytest.raises(ValueError, match="declared k=2 but produced 1 samples"):
        _call(_body([StageArm("s")], "k"), {"s": runner}, config={"k": 2})


@pytest.mark.parametrize("cfg", [{}, {"k": 0}, {"k": True}, {"k": "2"}])
def test_sampling_bad_cardinality_raises_before_running_stage(cfg):
    def boom(item):
        raise AssertionError("stage must not run")

    with pytest.raises(ValueError, match="cardinality"):
        _call(_body([StageArm("s")], "k"), {"s": boom}, config=cfg)


def test_sampling_bare_callable_is_single_sample():
    out, _ = _call(_body([StageArm("s")], "k"), {"s": lambda c: "x"}, config={"k": 1})
    assert _view(out) == [("x", "x")]


def _node_runner(monkeypatch, results):
    calls = []
    it = iter(results)

    def fake(node, stages, registry, config, calibrated, signals, predicates):
        calls.append(node)
        return next(it)

    monkeypatch.setattr(rt, "_execute_node", fake)
    return calls


def _res(kind, output=None, error=None):
    return NS(result_kind=kind, output=output, error=error)


def test_sampling_composite_runs_k_times_excluding_no_accept(monkeypatch):
    calls = _node_runner(
        monkeypatch,
        [_res(ResultKind.OUTPUT, 1), _res(ResultKind.NO_ACCEPT), _res(ResultKind.OUTPUT, 2)],
    )
    out, excl = _call(
        _body([CompositeArm("c")], "k"), registry={"c": "NODE"}, config={"k": 3}
    )
    assert _view(out) == [(1, 1), (2, 2)] and excl == 1
    assert calls == ["NODE"] * 3


def test_sampling_composite_error_is_absorbing_and_stops(monkeypatch):
    calls = _node_runner(
        monkeypatch, [_res(ResultKind.OUTPUT, 1), _res(ResultKind.ERROR, error="bad")]
    )
    with pytest.raises(RuntimeError, match="sampling ensemble arm 'c' failed: bad"):
        _call(_body([CompositeArm("c")], "k"), registry={"c": 0}, config={"k": 3})
    assert len(calls) == 2


def test_committee_stage_arm_contributes_one_winner_candidate():
    runner = StageRunner(run=lambda item: ["x", "y", "y"])
    out, excl = _call(_body([StageArm("s")]), {"s": runner})
    assert _view(out) == [("y", "y")] and excl == 0


def test_committee_stage_arm_with_no_samples_is_skipped():
    out, excl = _call(
        _body([StageArm("e"), StageArm("s")]),
        {"e": StageRunner(run=lambda i: []), "s": lambda c: "z"},
    )
    assert _view(out) == [("z", "z")] and excl == 0


def test_committee_representative_is_first_sample_with_winning_key():
    runner = StageRunner(run=lambda item: ["B", "a", "b", "A"], key_fn=str.lower)
    out, _ = _call(_body([StageArm("s")]), {"s": runner})
    assert len(out) == 1 and out[0].key in {"a", "b"}
    first_with_key = next(s for s in ["B", "a", "b", "A"] if s.lower() == out[0].key)
    assert out[0].output == first_with_key


def test_committee_mixed_arms_in_order_with_exclusion(monkeypatch):
    _node_runner(
        monkeypatch,
        [_res(ResultKind.NO_ACCEPT), _res(ResultKind.OUTPUT, "n")],
    )
    out, excl = _call(
        _body([CompositeArm("c1"), StageArm("s"), CompositeArm("c2")]),
        {"s": lambda c: "st"},
        {"c1": 1, "c2": 2},
    )
    assert _view(out) == [("st", "st"), ("n", "n")] and excl == 1


def test_committee_composite_error_propagates(monkeypatch):
    _node_runner(monkeypatch, [_res(ResultKind.ERROR, error="nope")])
    with pytest.raises(RuntimeError, match="committee arm 'c' failed: nope"):
        _call(_body([CompositeArm("c"), StageArm("s")]), {"s": lambda c: 1}, {"c": 0})


def test_committee_missing_stage_raises_keyerror():
    with pytest.raises(KeyError):
        _call(_body([StageArm("missing")]), {})
