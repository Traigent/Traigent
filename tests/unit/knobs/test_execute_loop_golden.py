"""Golden characterization tests for ``_execute_loop``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from traigent.knobs import runtime as rt
from traigent.knobs.composites import StopKind
from traigent.knobs.runtime import (
    CompositeRunResult,
    LoopBodyResult,
    ResultKind,
    _execute_loop,
)

NODE = SimpleNamespace(name="loopy")


def _body(
    stop_kind: StopKind, max_iters: int = 3, keys: tuple[str, ...] = ("k",)
) -> Any:
    return SimpleNamespace(
        stop=SimpleNamespace(kind=stop_kind),
        body=object(),
        max_iters=max_iters,
        state_keys=keys,
    )


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    results: list[Any],
    accepts: list[Any] | None = None,
    runner_error: Exception | None = None,
) -> dict[str, list[Any]]:
    log: dict[str, list[Any]] = {"states": [], "stops": []}
    pending = list(results)
    acc = list(accepts or [])

    def _runner(*_a: Any) -> Any:
        if runner_error is not None:
            raise runner_error
        return "runner"

    def _invoke(runner: Any, config: Any, state: Any, keys: Any) -> Any:
        log["states"].append(dict(state))
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _stop(stop: Any, state: Any, *_a: Any) -> Any:
        log["stops"].append(dict(state))
        item = acc.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(rt, "_loop_body_runner", _runner)
    monkeypatch.setattr(rt, "_invoke_loop_body", _invoke)
    monkeypatch.setattr(rt, "_stop_accepts", _stop)
    return log


def _run(body: Any) -> CompositeRunResult:
    return _execute_loop(NODE, body, {}, {}, {}, {}, {}, {})


def test_runner_resolution_failure_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [], runner_error=KeyError("missing"))
    res = _run(_body(StopKind.EXHAUSTED))
    assert res.result_kind is ResultKind.ERROR
    assert res.error == "composite-runtime: KeyError: 'missing'"


def test_accept_on_second_iteration_threads_restricted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _patch(
        monkeypatch,
        [
            LoopBodyResult(output="o1", state={"k": 1, "drop": 2}),
            LoopBodyResult(output="o2", state={"k": 3}),
        ],
        accepts=[False, True],
    )
    res = _run(_body(StopKind.SIGNAL_ACCEPT))
    assert (res.result_kind, res.output, res.error) == (ResultKind.OUTPUT, "o2", None)
    assert res.measures == {"iterations_used": 2, "stop_reason": "signal_accept"}
    assert log["states"] == [{}, {"k": 1}]
    assert log["stops"] == [{"k": 1}, {"k": 3}]


def test_external_accept_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [LoopBodyResult(output="o")], accepts=[True])
    res = _run(_body(StopKind.EXTERNAL_ACCEPT))
    assert res.measures == {"iterations_used": 1, "stop_reason": "external_accept"}


def test_no_accept_iteration_skips_stop_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = _patch(
        monkeypatch,
        [
            LoopBodyResult(
                output=None, result_kind=ResultKind.NO_ACCEPT, state={"k": 5}
            ),
            LoopBodyResult(output="o", state={"k": 6}),
        ],
        accepts=[True],
    )
    res = _run(_body(StopKind.SIGNAL_ACCEPT))
    assert res.output == "o"
    assert res.measures["iterations_used"] == 2
    assert log["stops"] == [{"k": 6}]
    assert log["states"] == [{}, {"k": 5}]


def test_unaccepted_signal_loop_is_no_accept_with_exhausted_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(
        monkeypatch,
        [LoopBodyResult(output="a"), LoopBodyResult(output="b")],
        accepts=[False, False],
    )
    res = _run(_body(StopKind.SIGNAL_ACCEPT, max_iters=2))
    assert res.result_kind is ResultKind.NO_ACCEPT
    assert res.output is None
    assert res.measures == {"iterations_used": 2, "stop_reason": "exhausted"}


def test_exhausted_loop_returns_final_output(monkeypatch: pytest.MonkeyPatch) -> None:
    log = _patch(
        monkeypatch,
        [LoopBodyResult(output="a"), LoopBodyResult(output="b")],
        accepts=[False, False],
    )
    res = _run(_body(StopKind.EXHAUSTED, max_iters=2))
    assert (res.result_kind, res.output) == (ResultKind.OUTPUT, "b")
    assert res.measures == {"iterations_used": 2, "stop_reason": "exhausted"}
    assert len(log["stops"]) == 2


def test_exhausted_loop_final_no_accept_is_no_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(
        monkeypatch,
        [
            LoopBodyResult(output="a"),
            LoopBodyResult(output=None, result_kind=ResultKind.NO_ACCEPT),
        ],
        accepts=[False],
    )
    res = _run(_body(StopKind.EXHAUSTED, max_iters=2))
    assert res.result_kind is ResultKind.NO_ACCEPT


def test_zero_iterations_is_no_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [])
    res = _run(_body(StopKind.EXHAUSTED, max_iters=0))
    assert res.result_kind is ResultKind.NO_ACCEPT
    assert res.measures == {"iterations_used": 0, "stop_reason": "exhausted"}


def test_body_exception_is_absorbing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [ValueError("bad")])
    res = _run(_body(StopKind.SIGNAL_ACCEPT))
    assert (res.result_kind, res.error) == (
        ResultKind.ERROR,
        "composite-runtime: ValueError: bad",
    )


def test_stop_exception_is_absorbing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, [LoopBodyResult(output="a")], accepts=[RuntimeError("stop")])
    res = _run(_body(StopKind.SIGNAL_ACCEPT))
    assert res.error == "composite-runtime: RuntimeError: stop"


def test_nested_error_result_message_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    nested = CompositeRunResult(
        output=None, result_kind=ResultKind.ERROR, measures={}, error="deep"
    )
    _patch(monkeypatch, [nested])
    assert _run(_body(StopKind.SIGNAL_ACCEPT)).error == (
        "composite-runtime: loop body of 'loopy' failed: deep"
    )
    plain = SimpleNamespace(result_kind=ResultKind.ERROR, error="ignored")
    _patch(monkeypatch, [plain])
    assert _run(_body(StopKind.SIGNAL_ACCEPT)).error == (
        "composite-runtime: loop body of 'loopy' failed: error"
    )
