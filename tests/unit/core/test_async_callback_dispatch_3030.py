"""Async callback dispatch regressions for issue #3030."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.utils.callbacks import CallbackManager, OptimizationCallback

_CALLBACK_CONTEXT = contextvars.ContextVar("callback_context", default="empty")


class _RecordingCallback(OptimizationCallback):
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.first_finished = threading.Event()
        self.calls = 0
        self.context_value: str | None = None
        self.fail = False

    def on_optimization_start(self, config_space, objectives, algorithm) -> None:
        self.events.append(("optimization_start", algorithm))

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        self.calls += 1
        self.events.append(("trial_start", (trial_number, config)))
        self.context_value = _CALLBACK_CONTEXT.get()
        if self.fail:
            raise ValueError("callback failure")
        if self.calls == 1:
            self.first_started.set()
            self.release_first.wait(timeout=2)
            self.first_finished.set()

    def on_trial_complete(self, trial, progress) -> None:
        self.events.append(("trial_complete", trial.trial_id))

    def on_optimization_complete(self, result) -> None:
        self.events.append(("optimization_complete", result.status))


def _orchestrator(callback: OptimizationCallback, *, timeout: float = 1.0):
    orchestrator = object.__new__(OptimizationOrchestrator)
    orchestrator.callback_manager = CallbackManager([callback], timeout=timeout)
    orchestrator._callback_dispatch_lock = asyncio.Lock()
    return orchestrator


@pytest.mark.asyncio
async def test_dispatch_offloads_serially_and_snapshots_before_waiting_for_lock() -> (
    None
):
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback, timeout=1.0)
    first_config = {"value": "first"}
    second_config = {"value": "before-dispatch"}
    ticks = 0
    heartbeat_running = True

    async def heartbeat() -> None:
        nonlocal ticks
        while heartbeat_running:
            ticks += 1
            await asyncio.sleep(0.005)

    pulse = asyncio.create_task(heartbeat())
    first = asyncio.create_task(
        orchestrator._dispatch_callback_manager("on_trial_start", 1, first_config)
    )
    try:
        assert await asyncio.to_thread(callback.first_started.wait, 1)
        second = asyncio.create_task(
            orchestrator._dispatch_callback_manager("on_trial_start", 2, second_config)
        )
        await asyncio.sleep(0)
        second_config["value"] = "mutated-while-queued"
        await asyncio.sleep(0.05)
        assert ticks >= 5
        assert not second.done()
        callback.release_first.set()
        await asyncio.gather(first, second)
        assert callback.events == [
            ("trial_start", (1, {"value": "first"})),
            ("trial_start", (2, {"value": "before-dispatch"})),
        ]
    finally:
        heartbeat_running = False
        callback.release_first.set()
        await pulse
        orchestrator.callback_manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, None])
async def test_repeated_cancellation_keeps_dispatch_lock_until_callback_settles(
    timeout: float | None,
) -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback, timeout=0)
    # Constructor None chooses the default timeout; an explicit None attribute
    # follows the manager's direct-invocation path just like zero.
    orchestrator.callback_manager.timeout = timeout
    first = asyncio.create_task(
        orchestrator._dispatch_callback_manager("on_trial_start", 1, {"n": 1})
    )
    try:
        assert await asyncio.to_thread(callback.first_started.wait, 1)
        first.cancel()
        await asyncio.sleep(0)
        first.cancel()
        second = asyncio.create_task(
            orchestrator._dispatch_callback_manager("on_trial_start", 2, {"n": 2})
        )
        await asyncio.sleep(0.03)
        assert callback.calls == 1
        assert not first.done()
        callback.release_first.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert callback.calls == 2
        assert [event[1][0] for event in callback.events] == [1, 2]
    finally:
        callback.release_first.set()
        orchestrator.callback_manager.close()


@pytest.mark.asyncio
async def test_dispatch_preserves_fresh_context_and_callback_failure_capture() -> None:
    callback = _RecordingCallback()
    callback.release_first.set()
    orchestrator = _orchestrator(callback, timeout=1.0)
    context_restore = _CALLBACK_CONTEXT.set("run-context")
    try:
        await orchestrator._dispatch_callback_manager(
            "on_trial_start", 1, {"value": "x"}
        )
        assert callback.context_value == "empty"

        callback.fail = True
        await orchestrator._dispatch_callback_manager(
            "on_trial_start", 2, {"value": "y"}
        )
        summary = orchestrator.callback_manager.get_failure_summary()
        assert summary["exception_count"] == 1
        assert summary["failures"][0]["message"] == "callback failure"
    finally:
        _CALLBACK_CONTEXT.reset(context_restore)
        orchestrator.callback_manager.close()


@pytest.mark.asyncio
async def test_dispatch_preserves_timeout_and_inflight_suppression() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback, timeout=0.01)
    try:
        await orchestrator._dispatch_callback_manager(
            "on_trial_start", 1, {"value": "slow"}
        )
        assert callback.first_started.is_set()
        assert orchestrator.callback_manager.get_failure_summary()["timeout_count"] == 1

        await orchestrator._dispatch_callback_manager(
            "on_trial_start", 2, {"value": "suppressed"}
        )
        assert callback.calls == 1
    finally:
        callback.release_first.set()
        assert await asyncio.to_thread(callback.first_finished.wait, 1)
        orchestrator.callback_manager.close()


@pytest.mark.asyncio
async def test_batch_cancellation_outside_callback_still_interrupts_trial() -> None:
    orchestrator = object.__new__(OptimizationOrchestrator)
    entered = asyncio.Event()
    interrupted = asyncio.Event()

    async def trial_batch(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            interrupted.set()

    orchestrator._run_parallel_batch_owned = AsyncMock(side_effect=trial_batch)
    task = asyncio.create_task(
        orchestrator._run_parallel_batch(None, None, None, None, 0, 2)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert interrupted.is_set()


@pytest.mark.asyncio
async def test_deadline_during_start_callback_never_dispatches_trial() -> None:
    callback = _RecordingCallback()
    orchestrator = _orchestrator(callback, timeout=0)
    run_trial = AsyncMock()
    orchestrator._trial_lifecycle = SimpleNamespace(run_trial=run_trial)
    orchestrator._trials_prevented = 0
    orchestrator._abandon_optuna_trial = MagicMock()
    orchestrator._get_progress_info = MagicMock(return_value=None)

    async def batch(*args):
        await orchestrator._schedule_and_run_parallel_trials(
            None,
            [
                ({"n": 0}, None, None, 6),
                ({"n": 1}, None, None, 7),
                ({"n": 2}, None, None, 8),
            ],
            [0, 1, 0],
            None,
            0,
        )
        return 0, "continue"

    orchestrator._run_parallel_batch_owned = AsyncMock(side_effect=batch)
    timer = threading.Timer(0.1, callback.release_first.set)
    timer.start()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                orchestrator._run_parallel_batch(None, None, None, None, 0, 2), 0.03
            )
        assert callback.first_started.is_set()
        run_trial.assert_not_called()
        assert [event[0] for event in callback.events] == [
            "trial_start",
            "trial_complete",
        ]
        assert orchestrator._trials_prevented == 3
        assert sorted(
            call.args[0] for call in orchestrator._abandon_optuna_trial.call_args_list
        ) == [6, 7, 8]
    finally:
        callback.release_first.set()
        timer.join(timeout=1)
        orchestrator.callback_manager.close()


@pytest.mark.parametrize("kind", ["keyboard", "system"])
@pytest.mark.parametrize(
    "stage,deadline",
    [
        ("initialize", "none"),
        ("start", "deadline"),
        ("complete", "none"),
        ("complete", "deadline"),
    ],
)
def test_shutdown_interrupt_reaches_real_lifecycle_owner(
    kind, stage, deadline, tmp_path
):
    env = os.environ.copy()
    env["TRAIGENT_OPTIMIZATION_LOG_DIR"] = str(tmp_path / "logs")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("callback_interrupt_probe.py")),
            kind,
            stage,
            deadline,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout.splitlines()[-1])
    expected_status = "cancelled" if kind == "keyboard" else "failed"
    assert outcome["finalized"] == [expected_status]
    if kind == "keyboard":
        assert outcome["result_status"] == expected_status
    else:
        assert outcome["exception"] == "SystemExit"
    expected_trials = 2 if stage == "complete" else 0
    assert outcome["trials"] == expected_trials
    assert outcome["accounted_trials"] == expected_trials
    assert outcome["in_flight"] == 0
    assert outcome["reserved"] == 0
