"""A user interrupt inside an async agent function must not tear down the loop.

Regression test for the defect where ``Ctrl-C`` during an async ``@optimize``d
function escaped ``optimize()`` entirely and destroyed the partial result.

Mechanism, because it is not obvious from the symptom: ``BaseEvaluator``
executes the user's coroutine through ``asyncio.wait_for`` whenever a timeout is
configured, and ``wait_for`` runs it in its OWN Task. CPython's Task step treats
``KeyboardInterrupt``/``SystemExit`` specially — it stores the exception on the
future *and re-raises it out of the step*. That re-raise unwinds the event loop
(``_run_once`` -> ``run_forever`` -> ``run_until_complete``) past every awaiting
frame, so the orchestrator's own interrupt handler finalized a partial result
into a loop that no longer existed to return it.

Only reproducible on Python 3.11 (the SDK's floor, and what CI runs); 3.12+
happens to mask it. So these tests are version-independent by construction: they
assert the observable contract, not the interpreter's internals.
"""

from __future__ import annotations

import pytest

import traigent

_SPACE = {"x": ["a", "b"]}


def _dataset() -> list[dict]:
    return [{"input": {"text": "t"}, "expected_output": "ok"}]


def _isolated_env(monkeypatch, tmp_path) -> None:
    """No key, no network, no LLM spend, private results folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


def _optimized(func):
    """Wrap as a real grid run.

    The function must be ``async``: a sync one is executed in a thread pool,
    which captures its exception and hands back an error string, so nothing
    would cross a Task boundary and the test would prove nothing.
    """
    return traigent.optimize(
        eval_dataset=_dataset(),
        objectives=["accuracy"],
        configuration_space=_SPACE,
        injection_mode="parameter",
    )(func)


@pytest.mark.asyncio
async def test_keyboard_interrupt_returns_partial_result(monkeypatch, tmp_path):
    """The headline contract: interrupt one trial, still get the completed ones.

    Before the fix this did not merely fail — the escaping interrupt unwound the
    event loop and aborted the whole pytest session, which is what made it look
    like a flaky worker crash rather than a product defect.
    """
    _isolated_env(monkeypatch, tmp_path)

    @_optimized
    async def answer(text: str, config=None) -> str:
        if config.custom_params.get("x") == "b":
            raise KeyboardInterrupt
        return "ok"

    result = await answer.optimize(algorithm="grid")

    # The trial that ran before the interrupt is real and reachable. That is the
    # whole point of containing the interrupt rather than letting it escape.
    assert len(result.trials) >= 1


@pytest.mark.asyncio
async def test_system_exit_propagates_catchably(monkeypatch, tmp_path):
    """``SystemExit`` takes the identical CPython Task-step path as Ctrl-C.

    Its *contract* differs on purpose, and this test pins the difference. The
    orchestrator's interrupt handler covers ``KeyboardInterrupt`` and
    ``CancelledError`` — not ``SystemExit``, which means "this program is
    exiting" and should not be quietly downgraded into a partial result.

    So what the fix buys here is not a returned result: it is that the exception
    arrives as an ordinary, catchable ``SystemExit`` at the caller instead of
    unwinding the event loop out from under it. Before the fix this aborted the
    interpreter's run loop and took the pytest session with it.

    Covered separately so a future narrowing of the guard to only
    ``KeyboardInterrupt`` fails here instead of silently reopening half the hole.
    """
    _isolated_env(monkeypatch, tmp_path)

    @_optimized
    async def answer(text: str, config=None) -> str:
        if config.custom_params.get("x") == "b":
            raise SystemExit
        return "ok"

    with pytest.raises(SystemExit):
        await answer.optimize(algorithm="grid")


@pytest.mark.asyncio
async def test_ordinary_exception_still_reaches_the_trial(monkeypatch, tmp_path):
    """Non-BaseException failures must be untouched by the containment.

    The guard captures exactly two exception types. If it ever widened to
    ``BaseException`` — or worse, to ``Exception`` — a normal agent error would
    stop being recorded as a failed trial and start being swallowed. This test is
    what makes that widening visible.
    """
    _isolated_env(monkeypatch, tmp_path)

    @_optimized
    async def answer(text: str, config=None) -> str:
        if config.custom_params.get("x") == "b":
            raise ValueError("agent blew up")
        return "ok"

    result = await answer.optimize(algorithm="grid")

    # Grid still enumerates both configurations; the failing one is recorded as a
    # trial rather than aborting the run or vanishing.
    assert len(result.trials) >= 1
