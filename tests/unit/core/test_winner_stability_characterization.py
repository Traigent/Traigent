"""Characterization of ``OptimizationOrchestrator._maybe_run_winner_stability``.

Pins every branch of the post-selection rerun driver with a bare orchestrator
instance and recorded collaborators (no real evaluator), so behaviour stays
identical across internal restructuring.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.core.orchestrator import OptimizationOrchestrator

LOGGER_PATH = "traigent.core.orchestrator.logger"


class FakePermit:
    def __init__(self, granted: bool = True, active: bool = True) -> None:
        self.is_granted = granted
        self.active = active


def _trial(score: Any = 0.5, successful: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        is_successful=successful,
        get_metric=lambda name: score,
    )


def _make(reps: Any = 2, **over: Any) -> OptimizationOrchestrator:
    o = object.__new__(OptimizationOrchestrator)
    o.config = {"winner_stability_reps": reps}
    o._stop_reason = None
    o._optimization_id = "opt"
    o._trials = []
    o.optimizer = SimpleNamespace(objectives=["accuracy"])
    o.execution_budget = None
    o.cost_enforcer = None
    o._winner_stability_accounted_cost = 0.0
    o._trial_lifecycle = SimpleNamespace(run_trial=AsyncMock(return_value=_trial()))
    o._account_winner_stability_rerun_attempt = AsyncMock(return_value=0.1)
    o._add_winner_stability_cost_to_result = MagicMock()
    for key, value in over.items():
        setattr(o, key, value)
    return o


def _result(**over: Any) -> SimpleNamespace:
    base = {"best_config": {"a": 1}, "stop_reason": None, "metadata": {}}
    base.update(over)
    return SimpleNamespace(**base)


async def _run(o, result=None, func=None, dataset="ds", session_id="sid"):
    result = result if result is not None else _result()
    with patch(LOGGER_PATH) as log:
        await o._maybe_run_winner_stability(
            result, func or (lambda: 1), dataset, session_id
        )
    return result, log


def _msgs(mock_method) -> list[str]:
    return [c.args[0] for c in mock_method.call_args_list]


class TestPreconditions:
    @pytest.mark.asyncio
    async def test_prefill_deactivated_even_when_reps_invalid(self):
        prefill = SimpleNamespace(active=True)
        o = _make(reps="abc", _resume_prefill=prefill)
        await _run(o)
        assert prefill.active is False
        o._trial_lifecycle.run_trial.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reps", ["abc", [], 0, None, -3, "0"])
    async def test_invalid_or_nonpositive_reps_return_silently(self, reps):
        o = _make(reps=reps)
        result, log = await _run(o)
        o._trial_lifecycle.run_trial.assert_not_awaited()
        o._add_winner_stability_cost_to_result.assert_not_called()
        assert "winner_stability" not in result.metadata
        log.info.assert_not_called()
        log.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_reps_key_defaults_to_zero(self):
        o = _make()
        o.config = {}
        await _run(o)
        o._trial_lifecycle.run_trial.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("where", ["self", "result"])
    async def test_no_usage_captured_skips(self, where):
        o = _make(reps=3)
        result = _result()
        if where == "self":
            o._stop_reason = "no_usage_captured"
        else:
            result.stop_reason = "no_usage_captured"
        result, log = await _run(o, result)
        o._trial_lifecycle.run_trial.assert_not_awaited()
        assert "winner_stability" not in result.metadata
        log.info.assert_called_once()
        assert log.info.call_args.args[0].startswith("winner_stability_reps=%d skipped")
        assert log.info.call_args.args[1] == 3

    @pytest.mark.asyncio
    async def test_result_without_stop_reason_attribute_is_tolerated(self):
        o = _make(reps=1)
        result = SimpleNamespace(best_config={"a": 1}, metadata={})
        result, _ = await _run(o, result)
        assert result.metadata["winner_stability"]["reps"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("best", [None, {}])
    async def test_no_winner_skips(self, best):
        o = _make(reps=2)
        result, log = await _run(o, _result(best_config=best))
        o._trial_lifecycle.run_trial.assert_not_awaited()
        log.info.assert_called_once()
        assert "selected no winner" in log.info.call_args.args[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("objectives", [[], None, [""]])
    async def test_no_primary_objective_warns_and_skips(self, objectives):
        o = _make(reps=2)
        o.optimizer = SimpleNamespace(objectives=objectives)
        result, log = await _run(o)
        o._trial_lifecycle.run_trial.assert_not_awaited()
        log.warning.assert_called_once()
        assert "no primary objective" in log.warning.call_args.args[0]
        assert "winner_stability" not in result.metadata


class TestSuccessfulRerun:
    @pytest.mark.asyncio
    async def test_block_contents_and_run_trial_arguments(self):
        scores = iter([0.2, 0.4, 0.6])
        o = _make(reps=3, _trials=[1, 2])
        o._trial_lifecycle.run_trial = AsyncMock(
            side_effect=lambda **kw: _trial(next(scores))
        )
        best = {"a": 1}
        func = object()
        result, log = await _run(
            o, _result(best_config=best), func=func, dataset="DS", session_id="S1"
        )
        block = result.metadata["winner_stability"]
        assert block["reps"] == 3
        assert block["mean"] == pytest.approx(0.4)
        assert block["std"] == pytest.approx(0.2)
        assert block["scores"] == [0.2, 0.4, 0.6]
        assert block["config_hash"] == o._get_config_hash(best)
        datetime.fromisoformat(block["evaluated_at"])
        assert set(block) == {
            "reps",
            "mean",
            "std",
            "scores",
            "config_hash",
            "evaluated_at",
        }
        calls = o._trial_lifecycle.run_trial.await_args_list
        assert [c.kwargs["trial_number"] for c in calls] == [3, 4, 5]
        for c in calls:
            assert c.kwargs["func"] is func
            assert c.kwargs["dataset"] == "DS"
            assert c.kwargs["session_id"] == "S1"
            assert c.kwargs["config"] == best
            assert c.kwargs["config"] is not best
            assert callable(c.kwargs["on_evaluation_dispatch"])
        acc = o._account_winner_stability_rerun_attempt.await_args_list
        assert [c.kwargs["trial_id"] for c in acc] == [
            "winner_stability_opt_1",
            "winner_stability_opt_2",
            "winner_stability_opt_3",
        ]
        assert all(c.kwargs["rerun_dispatched"] is False for c in acc)
        assert all(c.kwargs["permit"] is None for c in acc)
        o._add_winner_stability_cost_to_result.assert_called_once_with(result, 0.0, 0)
        # 3 replicates (<5): the descriptive-only SD note is logged.
        assert any("only %d" in m for m in _msgs(log.info))
        assert result.metadata["winner_stability"] is block

    @pytest.mark.asyncio
    async def test_five_replicates_have_no_small_sample_note(self):
        o = _make(reps=5)
        result, log = await _run(o)
        assert result.metadata["winner_stability"]["reps"] == 5
        log.info.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_replicate_omits_std(self):
        o = _make(reps=1)
        o._trial_lifecycle.run_trial = AsyncMock(return_value=_trial(0.9))
        result, log = await _run(o)
        block = result.metadata["winner_stability"]
        assert "std" not in block
        assert block["scores"] == [0.9]
        assert block["mean"] == 0.9
        log.info.assert_called_once()
        assert "undefined for a single" in log.info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_cost_reconciliation_uses_accounted_state(self):
        o = _make(reps=1, _winner_stability_unmeasured_reruns=2)
        o._winner_stability_accounted_cost = 1.5
        result, _ = await _run(o)
        o._add_winner_stability_cost_to_result.assert_called_once_with(result, 1.5, 2)

    @pytest.mark.asyncio
    async def test_dispatch_callback_is_forwarded_to_accounting(self):
        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            return _trial()

        o = _make(reps=1)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        await _run(o)
        kwargs = o._account_winner_stability_rerun_attempt.await_args.kwargs
        assert kwargs["rerun_dispatched"] is True

    @pytest.mark.asyncio
    async def test_ledger_start_snapshot_taken_before_each_rerun(self):
        o = _make(reps=2, _abandoned_vendor_attempts=[1, 2, 3])
        seen = []

        async def run_trial(**kw):
            seen.append(o._winner_stability_ledger_start)
            o._abandoned_vendor_attempts.append(object())
            return _trial()

        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        await _run(o)
        assert seen == [3, 4]

    @pytest.mark.asyncio
    async def test_ledger_start_defaults_to_zero_without_ledger(self):
        o = _make(reps=1)
        await _run(o)
        assert o._winner_stability_ledger_start == 0


class TestScoreFiltering:
    @pytest.mark.asyncio
    async def test_unsuccessful_and_nonfinite_scores_are_skipped(self):
        trials = iter(
            [
                _trial(0.5, successful=False),
                _trial(float("nan")),
                _trial(True),
                _trial(None),
                _trial("0.8"),
                _trial(math.inf),
            ]
        )
        o = _make(reps=6)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=lambda **kw: next(trials))
        result, log = await _run(o)
        block = result.metadata["winner_stability"]
        assert block["scores"] == [0.8]
        assert block["reps"] == 1
        assert "std" not in block
        skips = [c for c in log.warning.call_args_list if "no finite" in c.args[0]]
        assert len(skips) == 5
        assert skips[0].args[1:] == (1, 6, "accuracy")
        partial = [c for c in log.warning.call_args_list if "of the %d" in c.args[0]]
        assert len(partial) == 1
        assert partial[0].args[1:] == (1, 6)

    @pytest.mark.asyncio
    async def test_no_scores_records_nothing_but_still_reconciles_cost(self):
        o = _make(reps=2)
        o._trial_lifecycle.run_trial = AsyncMock(return_value=_trial(None))
        result, log = await _run(o)
        assert "winner_stability" not in result.metadata
        o._add_winner_stability_cost_to_result.assert_called_once()
        last = log.warning.call_args_list[-1]
        assert "measured no scores" in last.args[0]
        assert last.args[1] == 2

    @pytest.mark.asyncio
    async def test_run_trial_returning_none_is_skipped_silently(self):
        o = _make(reps=2)
        o._trial_lifecycle.run_trial = AsyncMock(return_value=None)
        result, log = await _run(o)
        assert "winner_stability" not in result.metadata
        acc = o._account_winner_stability_rerun_attempt.await_args_list
        assert len(acc) == 2
        assert acc[0].kwargs["rerun_trial"] is None
        assert not any("no finite" in c.args[0] for c in log.warning.call_args_list)


class TestBudgetAndPermits:
    @pytest.mark.asyncio
    async def test_budget_exhausted_dimension_stops_before_first_rerun(self):
        budget = MagicMock()
        budget.snapshot.return_value = SimpleNamespace(
            exhausted_dimension="trials", remaining_cost=99.0
        )
        o = _make(reps=3, execution_budget=budget)
        result, log = await _run(o)
        o._trial_lifecycle.run_trial.assert_not_awaited()
        assert "winner_stability" not in result.metadata
        denied = log.info.call_args_list[0]
        assert "shared execution budget" in denied.args[0]
        assert denied.args[1:] == (1, 3, "trials", 0)

    @pytest.mark.asyncio
    async def test_budget_cost_dimension_derived_from_remaining_cost(self):
        budget = MagicMock()
        budget.snapshot.return_value = SimpleNamespace(
            exhausted_dimension=None, remaining_cost=0.01
        )
        enforcer = MagicMock()
        enforcer.get_status.return_value = SimpleNamespace(estimated_cost_per_trial=0.5)
        o = _make(reps=2, execution_budget=budget, cost_enforcer=enforcer)
        _, log = await _run(o)
        o._trial_lifecycle.run_trial.assert_not_awaited()
        assert log.info.call_args_list[0].args[3] == "cost"
        enforcer.acquire_permit_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_budget_with_headroom_proceeds_and_denies_later(self):
        budget = MagicMock()
        states = iter(
            [
                SimpleNamespace(exhausted_dimension=None, remaining_cost=5.0),
                SimpleNamespace(exhausted_dimension="samples", remaining_cost=5.0),
            ]
        )
        budget.snapshot.side_effect = lambda: next(states)
        o = _make(reps=3, execution_budget=budget)
        result, log = await _run(o)
        assert o._trial_lifecycle.run_trial.await_count == 1
        assert result.metadata["winner_stability"]["scores"] == [0.5]
        denied = log.info.call_args_list[0]
        assert denied.args[1:] == (2, 3, "samples", 1)

    @pytest.mark.asyncio
    async def test_budget_without_enforcer_ignores_cost_check(self):
        budget = MagicMock()
        budget.snapshot.return_value = SimpleNamespace(
            exhausted_dimension=None, remaining_cost=-1.0
        )
        o = _make(reps=1, execution_budget=budget)
        await _run(o)
        o._trial_lifecycle.run_trial.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_permit_denied_stops_and_releases_inactive_permit_not(self):
        permit = FakePermit(granted=False, active=False)
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(return_value=permit)
        o = _make(reps=3, cost_enforcer=enforcer)
        result, log = await _run(o)
        o._trial_lifecycle.run_trial.assert_not_awaited()
        enforcer.release_permit.assert_not_called()
        assert "winner_stability" not in result.metadata
        assert "cost limit" in log.info.call_args.args[0]
        assert log.info.call_args.args[1:] == (1, 3, 0)

    @pytest.mark.asyncio
    async def test_denied_active_permit_is_released(self):
        permit = FakePermit(granted=False, active=True)
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(return_value=permit)
        o = _make(reps=1, cost_enforcer=enforcer)
        await _run(o)
        enforcer.release_permit.assert_called_once_with(permit)

    @pytest.mark.asyncio
    async def test_granted_permit_passed_to_accounting_and_released_each_rerun(self):
        permits = [FakePermit(), FakePermit()]
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(side_effect=permits)
        o = _make(reps=2, cost_enforcer=enforcer)
        await _run(o)
        acc = o._account_winner_stability_rerun_attempt.await_args_list
        assert [c.kwargs["permit"] for c in acc] == permits
        assert [c.args[0] for c in enforcer.release_permit.call_args_list] == permits

    @pytest.mark.asyncio
    async def test_inactive_permit_after_run_is_not_released(self):
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(return_value=FakePermit(active=False))
        o = _make(reps=1, cost_enforcer=enforcer)
        await _run(o)
        enforcer.release_permit.assert_not_called()


class TestFailures:
    @pytest.mark.asyncio
    async def test_exception_before_dispatch_stops_without_accounting(self):
        results = iter([_trial(0.3), RuntimeError("boom")])

        def run_trial(**kw):
            value = next(results)
            if isinstance(value, Exception):
                raise value
            return value

        permits = [FakePermit(), FakePermit()]
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(side_effect=permits)
        o = _make(reps=3, cost_enforcer=enforcer)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        result, log = await _run(o)
        assert result.metadata["winner_stability"]["scores"] == [0.3]
        # Only the first (successful) rerun reached accounting.
        assert o._account_winner_stability_rerun_attempt.await_count == 1
        assert enforcer.release_permit.call_count == 2
        failed = [
            c for c in log.warning.call_args_list if "failed to execute" in c.args[0]
        ]
        assert len(failed) == 1
        assert failed[0].args[1:] == (2, 3, 1)
        assert failed[0].kwargs == {"exc_info": True}

    @pytest.mark.asyncio
    async def test_exception_after_dispatch_is_accounted_as_untracked(self):
        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            raise RuntimeError("late")

        o = _make(reps=2)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        result, log = await _run(o)
        acc = o._account_winner_stability_rerun_attempt.await_args_list
        assert len(acc) == 1
        assert acc[0].kwargs["rerun_trial"] is None
        assert acc[0].kwargs["rerun_dispatched"] is True
        assert acc[0].kwargs["trial_id"] == "winner_stability_opt_1"
        assert "winner_stability" not in result.metadata
        o._add_winner_stability_cost_to_result.assert_called_once()
        assert o._trial_lifecycle.run_trial.await_count == 1

    @pytest.mark.asyncio
    async def test_accounting_failure_after_exception_is_logged_and_swallowed(self):
        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            raise RuntimeError("late")

        o = _make(reps=1)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        o._account_winner_stability_rerun_attempt = AsyncMock(
            side_effect=ValueError("acct")
        )
        _, log = await _run(o)
        msgs = _msgs(log.warning)
        assert any("untracked winner-stability" in m for m in msgs)
        assert any("failed to execute" in m for m in msgs)
        first = log.warning.call_args_list[0]
        assert first.args[1:] == (1, 1)

    @pytest.mark.asyncio
    async def test_accounting_exception_in_normal_path_goes_to_failure_handler(self):
        o = _make(reps=2)
        o._account_winner_stability_rerun_attempt = AsyncMock(
            side_effect=ValueError("acct")
        )
        result, log = await _run(o)
        # run_trial returned a trial, so the handler re-attempts accounting once
        # (the task already exists, so no new task is created) and stops.
        assert o._account_winner_stability_rerun_attempt.await_count == 1
        assert "winner_stability" not in result.metadata
        assert o._trial_lifecycle.run_trial.await_count == 1
        assert any("failed to execute" in m for m in _msgs(log.warning))

    @pytest.mark.asyncio
    async def test_failure_releases_active_permit(self):
        permit = FakePermit()
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(return_value=permit)
        o = _make(reps=1, cost_enforcer=enforcer)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=RuntimeError("x"))
        await _run(o)
        enforcer.release_permit.assert_called_once_with(permit)

    @pytest.mark.asyncio
    async def test_outer_guard_swallows_unexpected_errors(self):
        o = _make(reps=1)
        o._get_config_hash = MagicMock(side_effect=RuntimeError("hash"))
        result, log = await _run(o)
        assert "winner_stability" not in result.metadata
        log.warning.assert_called_once()
        assert "results are unaffected" in log.warning.call_args.args[0]
        assert log.warning.call_args.kwargs == {"exc_info": True}

    @pytest.mark.asyncio
    async def test_cost_reconciliation_failure_is_swallowed(self):
        o = _make(reps=1)
        o._add_winner_stability_cost_to_result = MagicMock(
            side_effect=RuntimeError("x")
        )
        result, log = await _run(o)
        assert "winner_stability" not in result.metadata
        assert "results are unaffected" in log.warning.call_args.args[0]


class TestCancellation:
    @pytest.mark.asyncio
    async def test_pre_dispatch_cancellation_propagates_without_accounting(self):
        permit = FakePermit()
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(return_value=permit)
        o = _make(reps=2, cost_enforcer=enforcer)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await _run(o)
        o._account_winner_stability_rerun_attempt.assert_not_awaited()
        o._add_winner_stability_cost_to_result.assert_not_called()
        enforcer.release_permit.assert_called_once_with(permit)

    @pytest.mark.asyncio
    async def test_post_dispatch_cancellation_accounts_and_reconciles(self):
        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            raise asyncio.CancelledError()

        o = _make(reps=2, _winner_stability_unmeasured_reruns=1)
        o._winner_stability_accounted_cost = 0.7
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        result = _result()
        with pytest.raises(asyncio.CancelledError):
            await _run(o, result)
        acc = o._account_winner_stability_rerun_attempt.await_args_list
        assert len(acc) == 1
        assert acc[0].kwargs["rerun_dispatched"] is True
        assert acc[0].kwargs["rerun_trial"] is None
        o._add_winner_stability_cost_to_result.assert_called_once_with(result, 0.7, 1)

    @pytest.mark.asyncio
    async def test_cancellation_with_unknown_cost_does_not_reconcile(self):
        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            raise asyncio.CancelledError()

        o = _make(reps=1)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        o._account_winner_stability_rerun_attempt = AsyncMock(return_value=None)
        with pytest.raises(asyncio.CancelledError):
            await _run(o)
        o._account_winner_stability_rerun_attempt.assert_awaited_once()
        o._add_winner_stability_cost_to_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_accounting_failure_is_logged_then_cancelled(self):
        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            raise asyncio.CancelledError()

        o = _make(reps=1)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        o._account_winner_stability_rerun_attempt = AsyncMock(
            side_effect=ValueError("acct")
        )
        with patch(LOGGER_PATH) as log:
            with pytest.raises(asyncio.CancelledError):
                await o._maybe_run_winner_stability(_result(), lambda: 1, "ds", "s")
        log.warning.assert_called_once()
        assert "cancelled winner-stability" in log.warning.call_args.args[0]
        assert log.warning.call_args.args[1:] == (1, 1)
        o._add_winner_stability_cost_to_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancellation_with_trial_but_no_dispatch_flag_still_accounts(self):
        # run_trial returned a trial; the accounting await itself is cancelled
        # before its task could be created is impossible here, so cancel inside
        # the shielded accounting and verify it is drained exactly once.
        started = asyncio.Event()
        release = asyncio.Event()

        async def accounting(**kw):
            started.set()
            await release.wait()
            return 0.3

        o = _make(reps=1)
        o._account_winner_stability_rerun_attempt = AsyncMock(side_effect=accounting)
        result = _result()
        with patch(LOGGER_PATH):
            task = asyncio.create_task(
                o._maybe_run_winner_stability(result, lambda: 1, "ds", "s")
            )
            await started.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # repeated cancellation must not strand accounting
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        o._account_winner_stability_rerun_attempt.assert_awaited_once()
        o._add_winner_stability_cost_to_result.assert_called_once_with(result, 0.0, 0)

    @pytest.mark.asyncio
    async def test_cancellation_releases_permit(self):
        permit = FakePermit()
        enforcer = MagicMock()
        enforcer.acquire_permit_async = AsyncMock(return_value=permit)

        async def run_trial(**kw):
            kw["on_evaluation_dispatch"]()
            raise asyncio.CancelledError()

        o = _make(reps=1, cost_enforcer=enforcer)
        o._trial_lifecycle.run_trial = AsyncMock(side_effect=run_trial)
        with pytest.raises(asyncio.CancelledError):
            await _run(o)
        enforcer.release_permit.assert_called_once_with(permit)
