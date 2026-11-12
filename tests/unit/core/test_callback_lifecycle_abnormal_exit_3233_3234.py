"""Callback lifecycle on abnormal exits (#3233, #3234).

- #3234: a trial interrupted by KeyboardInterrupt still gets on_trial_complete
  (status cancelled), so start/complete pair on the Ctrl+C path.
- #3233: on_optimization_complete fires and the callback worker thread is shut
  down when the run exits through SystemExit or an unexpected error.
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
from typing import Any

import pytest

import traigent
from traigent.api.types import TrialStatus
from traigent.utils.callbacks import OptimizationCallback


class _Rec(OptimizationCallback):
    def __init__(self) -> None:
        self.ev: list[str] = []
        self.statuses: list[Any] = []

    def on_optimization_start(self, *a: Any, **k: Any) -> None:
        self.ev.append("start")

    def on_trial_start(self, n: int, c: dict[str, Any]) -> None:
        self.ev.append(f"tstart:{n}")

    def on_trial_complete(self, t: Any, p: Any) -> None:
        self.ev.append("tcomplete")
        self.statuses.append(t.status)

    def on_optimization_complete(self, r: Any) -> None:
        self.ev.append("done")


def _callback_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("callback_")]


@pytest.fixture(autouse=True)
def _dataset_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))


def _run(tmp_path, exc: BaseException, rec: _Rec) -> None:
    ds = tmp_path / "dataset.jsonl"
    ds.write_text(
        "\n".join(json.dumps({"input": {"x": i}, "output": str(i)}) for i in range(4))
    )

    @traigent.optimize(
        eval_dataset=str(ds),
        objectives=["accuracy"],
        configuration_space={"a": [1, 2, 3]},
        injection_mode="parameter",
        config_param="config",
        offline=True,
        scoring_function=lambda output, expected, **k: float(output == expected),
    )
    def f(x, config):
        if config["a"] == 2:
            raise exc
        return str(x)

    asyncio.run(f.optimize(algorithm="grid", max_trials=3, callbacks=[rec]))


def _assert_no_leaked_threads(before: int) -> None:
    # close() shuts the executor down with wait=False: workers exit promptly
    # but asynchronously, so allow a short grace period.
    deadline = time.monotonic() + 3.0
    while len(_callback_threads()) > before and time.monotonic() < deadline:
        time.sleep(0.02)
    gc.collect()
    assert len(_callback_threads()) <= before


def test_interrupted_trial_gets_complete_callback_3234(tmp_path) -> None:
    before = len(_callback_threads())
    rec = _Rec()
    _run(tmp_path, KeyboardInterrupt(), rec)  # Ctrl+C returns a partial result
    assert rec.ev.count("tstart:0") == 1 and rec.ev.count("tstart:1") == 1
    starts = [e for e in rec.ev if e.startswith("tstart")]
    assert len(starts) == rec.ev.count("tcomplete")
    assert rec.statuses[-1] == TrialStatus.CANCELLED
    assert rec.ev[-1] == "done"
    _assert_no_leaked_threads(before)


def test_systemexit_fires_completion_and_closes_worker_3233(tmp_path) -> None:
    before = len(_callback_threads())
    rec = _Rec()
    with pytest.raises(SystemExit):
        _run(tmp_path, SystemExit(3), rec)
    assert rec.ev[-1] == "done"
    assert rec.ev.count("done") == 1
    _assert_no_leaked_threads(before)


def test_unexpected_error_fires_completion_once_3233(tmp_path, monkeypatch) -> None:
    from traigent.core.orchestrator import OptimizationOrchestrator

    async def boom(self, *a, **k):
        raise RuntimeError("finalize broke")

    monkeypatch.setattr(OptimizationOrchestrator, "_finalize_optimization", boom)
    before = len(_callback_threads())
    rec = _Rec()
    with pytest.raises(Exception, match="finalize broke"):
        _run(tmp_path, ValueError("unused"), rec)
    assert rec.ev.count("done") == 1
    _assert_no_leaked_threads(before)


def test_normal_run_fires_completion_exactly_once(tmp_path) -> None:
    before = len(_callback_threads())
    rec = _Rec()
    _run(tmp_path, ValueError("failed trial"), rec)
    assert rec.ev.count("done") == 1
    assert rec.ev.count("tstart:0") == 1
    _assert_no_leaked_threads(before)


def test_interrupt_after_trial_complete_does_not_double_complete_3234() -> None:
    from unittest.mock import MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = MagicMock(spec=OptimizationOrchestrator)
    orch._trial_complete_dispatched = {0}
    orch.callback_manager = MagicMock()
    OptimizationOrchestrator._notify_interrupted_trial(orch, 0, {"a": 1})
    orch.callback_manager.on_trial_complete.assert_not_called()
    OptimizationOrchestrator._notify_interrupted_trial(orch, 1, {"a": 1})
    orch.callback_manager.on_trial_complete.assert_called_once()


def test_callback_raising_on_abnormal_exit_does_not_mask_original_3233(
    tmp_path,
) -> None:
    class _Bad(_Rec):
        def on_optimization_complete(self, r: Any) -> None:
            raise KeyboardInterrupt("from callback")

    before = len(_callback_threads())
    with pytest.raises(SystemExit):
        _run(tmp_path, SystemExit(3), _Bad())
    _assert_no_leaked_threads(before)


def test_parallel_interrupt_pairs_started_trials_3234() -> None:
    import asyncio as aio
    from unittest.mock import AsyncMock, MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = MagicMock(spec=OptimizationOrchestrator)
    orch._trial_complete_dispatched = set()
    orch.callback_manager = MagicMock()
    orch._notify_interrupted_trial = lambda n, c: (
        OptimizationOrchestrator._notify_interrupted_trial(orch, n, c)
    )
    orch._trials_prevented = 0
    orch.parallel_execution_manager = MagicMock()
    orch._trial_lifecycle = MagicMock()
    orch._run_trial_recording_subset = MagicMock(return_value=None)
    orch.callback_manager.on_trial_complete.reset_mock()
    orch.parallel_execution_manager.run_with_cost_permits = AsyncMock(
        side_effect=aio.CancelledError()
    )
    ds = MagicMock()
    ds.__iter__ = lambda self: iter([])
    descriptors = [({"a": i}, None, ds, None) for i in range(2)]
    with pytest.raises(aio.CancelledError):
        aio.run(
            OptimizationOrchestrator._schedule_and_run_parallel_trials(
                orch, lambda: None, descriptors, [None, None], "s", 5
            )
        )
    assert orch.callback_manager.on_trial_start.call_count == 2
    assert orch.callback_manager.on_trial_complete.call_count == 2


def test_preflight_failure_completion_result_is_not_running_3233(
    tmp_path, monkeypatch
) -> None:
    from traigent.core.orchestrator import OptimizationOrchestrator

    def declined(self, dataset):
        raise RuntimeError("approval declined")

    monkeypatch.setattr(OptimizationOrchestrator, "_check_cost_approval", declined)
    seen: list[Any] = []

    class _Cap(_Rec):
        def on_optimization_complete(self, r: Any) -> None:
            seen.append(r.status)
            super().on_optimization_complete(r)

    before = len(_callback_threads())
    rec = _Cap()
    with pytest.raises(Exception, match="approval declined"):
        _run(tmp_path, ValueError("unused"), rec)
    # #2960: the approval is decided before the run (and its backend session)
    # starts, so a declined run never notifies start; start/complete stay
    # paired (neither fires) and the worker is still closed.
    assert seen == []
    assert rec.ev == []
    _assert_no_leaked_threads(before)


def _bare_orch():
    from unittest.mock import MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = MagicMock(spec=OptimizationOrchestrator)
    orch._trial_complete_dispatched = set()
    orch._trial_real_results = {}
    orch.callback_manager = MagicMock()
    orch._remember_trial_result = lambda n, r: (
        OptimizationOrchestrator._remember_trial_result(orch, n, r)
    )
    orch._notify_interrupted_trial = lambda n, c: (
        OptimizationOrchestrator._notify_interrupted_trial(orch, n, c)
    )
    return orch


def _real_trial(n: int):
    from datetime import UTC, datetime

    from traigent.core.types import TrialResult

    return TrialResult(
        trial_id=f"trial_{n}",
        config={"a": n},
        metrics={"accuracy": 1.0},
        status=TrialStatus.COMPLETED,
        duration=0.1,
        timestamp=datetime.now(UTC),
    )


def test_finished_trial_reports_real_status_not_cancelled_3234() -> None:
    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = _bare_orch()
    real = _real_trial(0)
    orch._remember_trial_result(0, real)
    OptimizationOrchestrator._notify_interrupted_trial(orch, 0, {"a": 0})
    OptimizationOrchestrator._notify_interrupted_trial(orch, 1, {"a": 1})
    calls = orch.callback_manager.on_trial_complete.call_args_list
    assert calls[0].args[0] is real
    assert calls[0].args[0].status == TrialStatus.COMPLETED
    assert calls[1].args[0].status == TrialStatus.CANCELLED


def test_parallel_trial_finished_before_interrupt_keeps_real_result_3234() -> None:
    import asyncio as aio
    from unittest.mock import AsyncMock, MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = _bare_orch()
    orch._trials_prevented = 0
    orch.parallel_execution_manager = MagicMock()
    real = _real_trial(0)
    orch._run_and_remember_trial = lambda n, run, *a, **k: (
        OptimizationOrchestrator._run_and_remember_trial(orch, n, run, *a, **k)
    )

    async def finish_first_then_interrupt(tasks, cancel_sentinel=None):
        await tasks[0]  # first trial finishes (and is remembered)
        tasks[1].close()
        raise aio.CancelledError()

    orch._run_trial_recording_subset = AsyncMock(return_value=real)
    orch._trial_lifecycle = MagicMock()
    orch.parallel_execution_manager.run_with_cost_permits = finish_first_then_interrupt
    ds = MagicMock()
    ds.__iter__ = lambda self: iter([])
    descriptors = [({"a": i}, None, ds, None) for i in range(2)]
    with pytest.raises(aio.CancelledError):
        aio.run(
            OptimizationOrchestrator._schedule_and_run_parallel_trials(
                orch, lambda: None, descriptors, [None, None], "s", 0
            )
        )
    statuses = [
        c.args[0].status for c in orch.callback_manager.on_trial_complete.call_args_list
    ]
    assert statuses == [TrialStatus.COMPLETED, TrialStatus.CANCELLED]


def test_cancel_during_result_processing_pairs_all_started_trials_3234() -> None:
    import asyncio as aio
    from unittest.mock import AsyncMock, MagicMock

    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.core.parallel_execution_manager import PermittedTrialResult

    orch = _bare_orch()
    orch._handle_trial_result = AsyncMock(side_effect=aio.CancelledError())
    orch._trial_result_from_parallel_item = (
        OptimizationOrchestrator._trial_result_from_parallel_item
    )
    orch.cost_enforcer = None
    results = [
        PermittedTrialResult(result=_real_trial(0), permit=MagicMock(active=False)),
        PermittedTrialResult(
            result=RuntimeError("boom"), permit=MagicMock(active=False)
        ),
    ]
    with pytest.raises(aio.CancelledError):
        aio.run(
            OptimizationOrchestrator._process_parallel_results(
                orch, [{"a": 0}, {"a": 1}], results, [None, None], "s", 0
            )
        )
    calls = orch.callback_manager.on_trial_complete.call_args_list
    assert [c.args[0].status for c in calls] == [
        TrialStatus.COMPLETED,
        TrialStatus.FAILED,  # a finished failed trial keeps its failure
    ]
