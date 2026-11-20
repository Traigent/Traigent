"""Per-example consistency labels from winner-stability reruns.

The orchestrator reruns the selected winner ``winner_stability_reps`` times and
keeps each rerun's per-example pass/fail. These tests run the real orchestrator
with a scripted evaluator and check the labels in
``result.metadata["example_consistency"]``, that the existing
``winner_stability`` block is unchanged, and that the labels survive save and
reload.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from traigent.api.types import ExampleResult, OptimizationStatus, TrialResult
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.persistence import PersistenceManager

#: example id -> score per evaluation call. Call 0 is the search trial, calls
#: 1..5 are the five winner reruns. ``None`` is an errored evaluation.
_SCRIPT: dict[str, list[float | None]] = {
    "always_wrong": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sometimes_wrong": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "always_right": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "mostly_errors": [1.0, None, None, None, None, 1.0],
}


class _OneConfigOptimizer(BaseOptimizer):
    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kw):
        super().__init__(config_space, objectives, **kw)
        self._count = 0

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        self._count += 1
        return {"param1": 1}

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self._count >= 1

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        return None

    def is_finished(self) -> bool:
        return self._count >= 1


class _ScriptedEvaluator(BaseEvaluator):
    """Returns per-example results whose scores follow ``_SCRIPT`` by call."""

    def __init__(self, script: dict[str, list[float | None]], **kwargs: Any):
        super().__init__(**kwargs)
        self.script = script
        self.calls = 0

    async def evaluate(
        self,
        func: Callable,
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease=None,
        progress_callback=None,
        **_kwargs: Any,
    ) -> EvaluationResult:
        call = self.calls
        self.calls += 1
        rows: list[ExampleResult] = []
        for example_id, scores in self.script.items():
            score = scores[call]
            rows.append(
                ExampleResult(
                    example_id=example_id,
                    input_data={"query": "q"},
                    expected_output="a",
                    actual_output="b",
                    metrics=(
                        {} if score is None else {"score": score, "accuracy": score}
                    ),
                    execution_time=0.01,
                    success=score is not None,
                    error_message="boom" if score is None else None,
                )
            )
        scored = [s for s in (r.metrics.get("score") for r in rows) if s is not None]
        accuracy = sum(scored) / len(scored)
        metrics = {"accuracy": accuracy}
        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=len(rows),
            successful_examples=len(scored),
            duration=0.01,
            metrics=metrics,
            outputs=["b"] * len(rows),
            errors=[r.error_message for r in rows],
            example_results=rows,
        )
        result.sample_budget_exhausted = False
        result.examples_consumed = len(rows)
        return result


@pytest.fixture(autouse=True)
def _isolated_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIGENT_OPTIMIZATION_LOG_DIR", str(tmp_path / "logs"))


async def _run(reps: int, script=None) -> Any:
    evaluator = _ScriptedEvaluator(script or _SCRIPT)
    orchestrator = OptimizationOrchestrator(
        optimizer=_OneConfigOptimizer({"param1": [1]}, ["accuracy"]),
        evaluator=evaluator,
        max_trials=1,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        winner_stability_reps=reps,
    )

    async def func(input_data: dict[str, Any], **config: Any) -> Any:
        return "b"

    dataset = Dataset([EvaluationExample({"query": "q"}, "a")], name="d")
    result = await orchestrator.optimize(func, dataset)
    assert result.status == OptimizationStatus.COMPLETED
    return result


@pytest.mark.asyncio
async def test_five_reruns_label_each_example_by_its_own_outcomes() -> None:
    result = await _run(5)

    stored = result.metadata["example_consistency"]
    examples = stored["examples"]
    assert examples["always_wrong"] == {
        "label": "consistent_fail",
        "passes": 0,
        "fails": 5,
        "unknowns": 0,
    }
    assert examples["sometimes_wrong"] == {
        "label": "flaky",
        "passes": 2,
        "fails": 3,
        "unknowns": 0,
    }
    assert examples["always_right"] == {
        "label": "consistent_pass",
        "passes": 5,
        "fails": 0,
        "unknowns": 0,
    }
    assert stored["reps"] == 5
    assert stored["config_hash"] == result.metadata["winner_stability"]["config_hash"]


@pytest.mark.asyncio
async def test_errored_reruns_are_unknown_and_one_known_evaluation_gets_no_label() -> (
    None
):
    result = await _run(5)

    # Four of five reruns errored: one known evaluation is not enough.
    assert "mostly_errors" not in result.metadata["example_consistency"]["examples"]
    assert "mostly_errors" not in result.example_consistency()["examples"]


@pytest.mark.asyncio
async def test_accessor_reports_counts_and_per_example_labels() -> None:
    result = await _run(5)

    report = result.example_consistency()
    assert report["counts"] == {
        "consistent_pass": 1,
        "consistent_fail": 1,
        "flaky": 1,
    }
    assert report["examples"]["sometimes_wrong"]["label"] == "flaky"
    assert report["reps"] == 5


@pytest.mark.asyncio
async def test_labels_live_outside_the_winner_stability_block() -> None:
    result = await _run(5)

    assert "example_consistency" not in result.metadata["winner_stability"]
    assert set(result.metadata["winner_stability"]) == {
        "reps",
        "mean",
        "std",
        "scores",
        "config_hash",
        "evaluated_at",
    }


@pytest.mark.asyncio
async def test_default_run_records_no_labels() -> None:
    result = await _run(0)

    assert "example_consistency" not in result.metadata
    assert result.example_consistency()["examples"] == {}


@pytest.mark.asyncio
async def test_single_rerun_records_no_labels() -> None:
    result = await _run(1)

    assert "example_consistency" not in result.metadata
    assert result.example_consistency()["counts"] == {
        "consistent_pass": 0,
        "consistent_fail": 0,
        "flaky": 0,
    }


@pytest.mark.asyncio
async def test_labels_survive_save_and_reload(tmp_path) -> None:
    result = await _run(5)
    manager = PersistenceManager(str(tmp_path / "store"))

    manager.save_result(result, name="consistency_run")
    reloaded = manager.load_result("consistency_run")

    assert reloaded.example_consistency() == result.example_consistency()
    assert reloaded.example_consistency()["examples"]["sometimes_wrong"] == {
        "label": "flaky",
        "passes": 2,
        "fails": 3,
        "unknowns": 0,
    }


# --- Upload audit: the labels are local only --------------------------------


def _finalize_with_recording_client(
    result: Any, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Run the real finalize path against a recording backend client."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.core.objectives import create_default_objectives

    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    config = TraigentConfig()
    config.offline = False
    config.no_egress = False
    config.execution_mode = "hybrid"

    class _Mapping:
        experiment_id = "exp"
        experiment_run_id = "run"

    client = Mock()
    client.get_session_mapping = Mock(return_value=_Mapping())
    client.update_trial_weighted_scores = AsyncMock(return_value=True)
    client.finalize_session_sync = Mock(
        return_value={"status": "completed", "metadata": {}}
    )
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"param1": [1]}
    manager = BackendSessionManager(
        backend_client=client,
        traigent_config=config,
        objectives=["accuracy"],
        objective_schema=create_default_objectives(["accuracy"]),
        optimizer=optimizer,
        optimization_id="opt_upload_audit",
        optimization_status=OptimizationStatus.RUNNING,
    )
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orchestrator._trials = []
    orchestrator._stop_reason = None
    orchestrator._optimization_id = "opt_upload_audit"
    orchestrator._status = OptimizationStatus.COMPLETED
    orchestrator.optimizer = Mock(objectives=["accuracy"])
    orchestrator.objective_schema = None
    orchestrator.backend_session_manager = manager
    orchestrator.traigent_config = config
    orchestrator.cost_enforcer = Mock(
        get_status=Mock(
            return_value=SimpleNamespace(
                accumulated_cost_usd=0.0, unmeasured_trial_count=0
            )
        )
    )
    orchestrator._build_certified_selection_report = Mock(return_value=None)
    orchestrator._submit_usage_analytics = AsyncMock()
    orchestrator._submit_workflow_traces = AsyncMock()
    orchestrator.callback_manager = Mock()
    asyncio.run(
        OptimizationOrchestrator._finalize_optimization(
            orchestrator, result, "session-upload-audit", None
        )
    )
    return client


def test_labels_and_example_ids_are_in_no_upload_payload(monkeypatch) -> None:
    """The key and its example ids reach neither session nor trial payloads.

    ``result.metadata`` is never serialized wholesale: the session rollup is
    built from named keys, and trial submissions carry trial metadata only.
    """
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from traigent.api.types import TrialStatus
    from traigent.cloud.client import TraigentCloudClient
    from traigent.core.backend_session_manager import BackendSessionManager

    result = asyncio.run(_run(5))
    example_ids = list(result.metadata["example_consistency"]["examples"])
    assert example_ids  # the run really produced labelled examples
    result.metadata["session_summary"] = {
        "metrics": {"accuracy": 0.5},
        "samples_per_config": {"p": 4},
    }

    # Session / result payloads: the real finalize path, then every call the
    # backend client received (finalize request body included).
    client = _finalize_with_recording_client(result, monkeypatch)
    finalize_kwargs = client.finalize_session_sync.call_args.kwargs
    assert finalize_kwargs.get("session_aggregation")  # a rollup really was sent
    sent = json.dumps(
        [client.finalize_session_sync.call_args_list, client.mock_calls], default=repr
    )

    session_sent = sent

    # Trial payloads: every trial of the run through both trial serializers.
    # (Trial metadata may carry its own per-example rows; what must not appear
    # is this feature's key or labels.)
    sent = ""
    for trial in result.trials:
        submission = SimpleNamespace(
            session_id="s",
            trial_id=trial.trial_id,
            metrics=trial.metrics,
            duration=trial.duration,
            status=TrialStatus.COMPLETED,
            error_message=None,
            metadata=trial.metadata,
        )
        sent += json.dumps(
            TraigentCloudClient._serialize_trial_result(MagicMock(), submission),
            default=repr,
        )
        sent += json.dumps(
            BackendSessionManager._build_trial_metrics_payload(
                trial, trial.metadata, trial.score, trial.score
            ),
            default=repr,
        )

    assert "example_consistency" not in session_sent + sent
    for example_id in example_ids:
        assert example_id not in session_sent
    for label in ("consistent_fail", "consistent_pass", "flaky"):
        assert label not in session_sent + sent
    # Positive control: the same strings do appear in the local result.
    assert "sometimes_wrong" in json.dumps(result.metadata)


@pytest.mark.asyncio
async def test_labels_are_not_written_into_trial_metadata() -> None:
    result = await _run(5)

    assert "example_consistency" in result.metadata
    for trial in result.trials:
        assert "example_consistency" not in trial.metadata
