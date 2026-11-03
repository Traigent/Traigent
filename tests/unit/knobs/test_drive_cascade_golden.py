"""Golden characterization tests for the manual gated-cascade driver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from traigent.knobs import runtime as rt
from traigent.knobs.cascade import VoteStats
from traigent.knobs.composites import GateKind as IRGateKind
from traigent.knobs.composites import StageArm
from traigent.knobs.runtime import ResultKind, _drive_cascade


def _vote(margin: float) -> VoteStats:
    return VoteStats(
        top_key="a", margin=margin, valid_rate=1.0, n_distinct=1, tie=False
    )


def _out(
    kind: ResultKind, output: Any = None, vote: Any = None, error: Any = None
) -> Any:
    return rt._ArmOutcome(output=output, vote=vote, kind=kind, error=error)


def _gate(threshold: str = "t", kind: Any = IRGateKind.MARGIN_BELOW) -> Any:
    return SimpleNamespace(kind=kind, threshold=threshold)


def _nested(ref: str) -> Any:
    return SimpleNamespace(ref=ref)


class _Harness:
    """Patches the three arm executors and records every call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> None:
        self.calls: list[Any] = []
        self.script = list(script)

        def _next(tag: str, *args: Any, **kwargs: Any) -> Any:
            self.calls.append((tag, args[0].name if tag == "stage" else args[0].ref))
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(
            rt, "_evaluate_stage_arm", lambda arm, *a, **k: _next("stage", arm, **k)
        )
        monkeypatch.setattr(
            rt,
            "_execute_node",
            lambda node, *a, **k: _next("exec", node),
        )
        monkeypatch.setattr(
            rt,
            "_run_nested_majority_vote",
            lambda node, *a, **k: _next("vote", node),
        )


def _run(
    body: Any, calibrated: dict[str, float] | None = None, registry: Any = None
) -> Any:
    reg = registry or {
        "n0": SimpleNamespace(ref="n0"),
        "n1": SimpleNamespace(ref="n1"),
    }
    return _drive_cascade(
        body, {}, reg, {}, calibrated if calibrated is not None else {"t": 0.5}, {}, {}
    )


def test_non_margin_gate_is_error_before_any_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _Harness(monkeypatch, [])
    body = SimpleNamespace(
        arms=(StageArm("a"), StageArm("b")),
        gates=(_gate(kind=IRGateKind.SIGNAL_BELOW),),
    )
    res = _run(body)
    assert res.result_kind is ResultKind.ERROR
    assert res.error == (
        "composite-runtime: post-cascade gate kind 'signal_below' "
        "is not executable (margin_below only)"
    )
    assert h.calls == []


def test_first_arm_confident_returns_output_with_step_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _Harness(monkeypatch, [_out(ResultKind.OUTPUT, "A", _vote(0.9))])
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    res = _run(body)
    assert (res.result_kind, res.output, res.error) == (ResultKind.OUTPUT, "A", None)
    assert res.measures["stage_selected"] == 0
    assert res.measures["escalation_rate"] == 0.0
    assert h.calls == [("stage", "a")]


def test_low_margin_escalates_to_terminal_nested_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _Harness(
        monkeypatch,
        [
            _out(ResultKind.OUTPUT, "A", _vote(0.1)),
            SimpleNamespace(output="Z", result_kind=ResultKind.OUTPUT, error=None),
        ],
    )
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    res = _run(body)
    assert (res.result_kind, res.output) == (ResultKind.OUTPUT, "Z")
    assert res.measures["stage_selected"] == 1
    assert res.measures["escalation_rate"] == 1.0
    assert h.calls == [("stage", "a"), ("exec", "n1")]


def test_gated_no_accept_escalates_and_terminal_no_accept_is_cascade_no_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _Harness(
        monkeypatch,
        [
            _out(ResultKind.NO_ACCEPT),
            SimpleNamespace(output=None, result_kind=ResultKind.NO_ACCEPT, error=None),
        ],
    )
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    res = _run(body)
    assert res.result_kind is ResultKind.NO_ACCEPT
    assert res.output is None and res.error is None
    assert res.measures["escalation_rate"] == 1.0
    assert h.calls == [("stage", "a"), ("exec", "n1")]


def test_gated_nested_arm_uses_vote_path(monkeypatch: pytest.MonkeyPatch) -> None:
    h = _Harness(monkeypatch, [_out(ResultKind.OUTPUT, "N", _vote(0.8))])
    body = SimpleNamespace(arms=(_nested("n0"), StageArm("b")), gates=(_gate(),))
    res = _run(body)
    assert (res.result_kind, res.output) == (ResultKind.OUTPUT, "N")
    assert h.calls == [("vote", "n0")]


def test_arm_exception_is_error_with_type_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Harness(monkeypatch, [KeyError("boom")])
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    res = _run(body)
    assert res.result_kind is ResultKind.ERROR
    assert res.error == "composite-runtime: KeyError: 'boom'"


def test_arm_error_outcome_uses_detail_or_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Harness(monkeypatch, [_out(ResultKind.ERROR, error="inner bad")])
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    assert _run(body).error == "inner bad"
    _Harness(monkeypatch, [_out(ResultKind.ERROR)])
    assert _run(body).error == "composite-runtime: cascade arm 0 failed (no detail)"


def test_missing_vote_on_gated_arm_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _Harness(monkeypatch, [_out(ResultKind.OUTPUT, "A", None)])
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    assert _run(body).error == (
        "composite-runtime: gated cascade arm 0 has no vote to feed the gate "
        "(margin-bearing arm required)"
    )


@pytest.mark.parametrize(
    ("calibrated", "message"),
    [
        ({}, "ValueError: Gate threshold is unset; calibrate the CVAR before routing."),
        ({"t": float("nan")}, "ValueError: Gate threshold is non-finite"),
    ],
)
def test_unset_or_nonfinite_threshold_fails_closed(
    monkeypatch: pytest.MonkeyPatch, calibrated: dict[str, float], message: str
) -> None:
    h = _Harness(monkeypatch, [_out(ResultKind.OUTPUT, "A", _vote(0.99))])
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    res = _run(body, calibrated)
    assert res.result_kind is ResultKind.ERROR
    assert res.error is not None and res.error.startswith(
        f"composite-runtime: {message}"
    )
    assert h.calls == [("stage", "a")]


def test_three_arms_double_escalation_counts_and_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _Harness(
        monkeypatch,
        [
            _out(ResultKind.NO_ACCEPT),
            _out(ResultKind.OUTPUT, "B", _vote(0.2)),
            SimpleNamespace(output="C", result_kind=ResultKind.OUTPUT, error=None),
        ],
    )
    body = SimpleNamespace(
        arms=(StageArm("a"), StageArm("b"), _nested("n1")),
        gates=(_gate("t"), _gate("u")),
    )
    res = _run(body, {"t": 0.5, "u": 0.5})
    assert (res.result_kind, res.output) == (ResultKind.OUTPUT, "C")
    assert res.measures["stage_selected"] == 2
    assert res.measures["gate_margin_pass_rate"] == {0: 0.0, 1: 0.0}
    assert h.calls == [("stage", "a"), ("stage", "b"), ("exec", "n1")]


def test_terminal_arm_error_outcome_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _Harness(
        monkeypatch,
        [
            _out(ResultKind.NO_ACCEPT),
            SimpleNamespace(output=None, result_kind=ResultKind.ERROR, error="deep"),
        ],
    )
    body = SimpleNamespace(arms=(StageArm("a"), _nested("n1")), gates=(_gate(),))
    assert _run(body).error == "deep"
