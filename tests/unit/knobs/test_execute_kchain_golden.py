"""Golden characterization tests for ``execute_kchain``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from traigent.knobs.runtime import (
    CompositeRunResult,
    LoopBodyResult,
    LoopBodyRunner,
    ResultKind,
    execute_kchain,
)


def _chain(n: int = 3, signal: str = "sig", threshold: str = "t") -> Any:
    return SimpleNamespace(
        signal=signal,
        threshold=threshold,
        state_keys=("score",),
        stages=[object()] * n,
    )


def _runner(results: list[Any], seen: list[Any]) -> LoopBodyRunner:
    pending = list(results)

    def _run(item: Any, state: Any) -> Any:
        seen.append(dict(state))
        out = pending.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    return LoopBodyRunner(run=_run)


def _go(chain: Any, runner: Any, signals: Any = None, calibrated: Any = None) -> Any:
    return execute_kchain(
        chain,
        runner,
        config={},
        calibrated_values={"t": 0.5} if calibrated is None else calibrated,
        signals={"sig": lambda s: s["score"]} if signals is None else signals,
    )


def test_accepts_at_first_stage_meeting_threshold_and_threads_state() -> None:
    seen: list[Any] = []
    res = _go(
        _chain(),
        _runner(
            [
                LoopBodyResult("a", {"score": 0.1, "x": 1}),
                LoopBodyResult("b", {"score": 0.5}),
            ],
            seen,
        ),
    )
    assert (res.result_kind, res.output, res.error) == (ResultKind.OUTPUT, "b", None)
    assert res.measures == {"iterations_used": 2, "stop_reason": "signal_accept"}
    assert seen == [{}, {"score": 0.1}]


def test_bare_value_result_is_wrapped_with_empty_state() -> None:
    res = _go(
        _chain(),
        LoopBodyRunner(run=lambda item, state: "raw"),
        signals={"sig": lambda s: 1.0},
    )
    assert (res.output, res.measures["iterations_used"]) == ("raw", 1)


def test_no_accept_body_escalates_without_signal_call() -> None:
    calls: list[Any] = []

    def sig(state: Any) -> float:
        calls.append(dict(state))
        return 1.0

    res = _go(
        _chain(),
        _runner(
            [
                LoopBodyResult(None, {"score": 9}, ResultKind.NO_ACCEPT),
                LoopBodyResult("ok", {"score": 1}),
            ],
            [],
        ),
        signals={"sig": sig},
    )
    assert res.output == "ok"
    assert calls == [{"score": 1}]
    assert res.measures["iterations_used"] == 2


def test_exhausted_chain_is_no_accept_with_stage_count() -> None:
    res = _go(
        _chain(2),
        _runner(
            [LoopBodyResult("a", {"score": 0}), LoopBodyResult("b", {"score": 0})], []
        ),
    )
    assert res.result_kind is ResultKind.NO_ACCEPT
    assert res.output is None
    assert res.measures == {"iterations_used": 2, "stop_reason": "exhausted"}


def test_empty_chain_is_no_accept_zero_iterations() -> None:
    res = _go(_chain(0), _runner([], []))
    assert res.result_kind is ResultKind.NO_ACCEPT
    assert res.measures == {"iterations_used": 0, "stop_reason": "exhausted"}


def test_body_exception_error() -> None:
    res = _go(_chain(), _runner([ValueError("bad")], []))
    assert (res.result_kind, res.error) == (
        ResultKind.ERROR,
        "composite-runtime: ValueError: bad",
    )


def test_body_error_result_kind_message() -> None:
    res = _go(_chain(), _runner([LoopBodyResult(None, {}, ResultKind.ERROR)], []))
    assert res.error == "composite-runtime: K-chain body of 'sig' failed: error"


def test_composite_run_result_return_is_wrapped_as_plain_output() -> None:
    # Odd input: not a LoopBodyResult, so it is wrapped (state empty), and the
    # signal then fails on the missing state key.
    cell = CompositeRunResult(None, ResultKind.ERROR, {}, "deep")
    res = _go(_chain(), _runner([cell], []))
    assert res.error == "composite-runtime: KeyError: 'score'"


def test_missing_signal_and_threshold_fail_closed() -> None:
    res = _go(_chain(), _runner([LoopBodyResult("a", {"score": 1})], []), signals={})
    assert res.result_kind is ResultKind.ERROR
    assert res.error is not None and res.error.startswith(
        "composite-runtime: ValueError: signal 'sig' is not provided"
    )
    res = _go(
        _chain(),
        _runner([LoopBodyResult("a", {"score": 1})], []),
        calibrated={},
    )
    assert res.result_kind is ResultKind.ERROR
    res = _go(
        _chain(),
        _runner([LoopBodyResult("a", {"score": 1})], []),
        calibrated={"t": float("nan")},
    )
    assert res.result_kind is ResultKind.ERROR


def test_signal_exception_is_error() -> None:
    def boom(state: Any) -> float:
        raise RuntimeError("sigfail")

    res = _go(
        _chain(),
        _runner([LoopBodyResult("a", {"score": 1})], []),
        signals={"sig": boom},
    )
    assert res.error == "composite-runtime: RuntimeError: sigfail"
