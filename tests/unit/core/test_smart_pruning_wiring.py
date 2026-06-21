from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from traigent.config.types import TraigentConfig
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.core.types import TrialResult, TrialStatus
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.optimizers.base import BaseOptimizer


class QueueOptimizer(BaseOptimizer):
    def __init__(self, configs: list[dict[str, Any]]) -> None:
        super().__init__({"answer": ["ok"]}, ["accuracy"])
        self._configs = list(configs)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        return dict(self._configs[len(history)])

    def should_stop(self, history: list[TrialResult]) -> bool:
        return len(history) >= len(self._configs)


class SmartPruningOrchestrator:
    def __init__(self, client: Any, configs: list[dict[str, Any]]) -> None:
        self.optimizer = QueueOptimizer(configs)
        self.evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1, detailed=True)
        self.knob_resolver = None
        self._optimization_id = "smart-pruning-test"
        self._sample_budget_manager = None
        self._consumed_examples = 0
        self._examples_capped = 0
        self._stop_reason = None
        self._constraints_pre_eval: list[Any] = []
        self._constraints_post_eval: list[Any] = []
        self._trials: list[TrialResult] = []
        self.max_trials = len(configs)
        self.config = {
            "cache_policy": "allow_repeats",
            "smart_pruning": {"label": "balanced"},
        }
        self.traigent_config = TraigentConfig(
            custom_params={"smart_pruning": {"label": "balanced"}}
        )
        self.cache_policy_handler = MagicMock()
        self.callback_manager = MagicMock()
        self.cost_enforcer = None
        self._default_config = None
        self._default_config_used = False
        self.objective_schema = None
        self.objectives = ["accuracy"]
        self.backend_client = client
        self.backend_session_manager = SimpleNamespace(backend_tracking_enabled=True)
        self._cloud_guidance_client = None
        self._workflow_traces_tracker = None

    def _apply_knob_resolution(self, config: dict[str, Any]) -> dict[str, Any]:
        return config

    def _consume_default_config(self) -> None:
        return None

    def _is_cloud_brain_run(self) -> bool:
        return True

    def _smart_pruning_config(self) -> dict[str, Any]:
        return {"label": "balanced"}

    async def _handle_trial_result(self, **kwargs: Any) -> int:
        trial_result = kwargs["trial_result"]
        self._trials.append(trial_result)
        return int(kwargs.get("current_trial_index", 0)) + 1


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"question": "q1"}, "ok"),
            EvaluationExample({"question": "q2"}, "ok"),
        ],
        name="smart_pruning_eval",
    )


def _candidate(question: str) -> str:
    return "ok"


def test_smart_pruning_true_records_pruned_trial_and_run_continues() -> None:
    asyncio.run(_run_smart_pruning_true_records_pruned_trial_and_continues())


async def _run_smart_pruning_true_records_pruned_trial_and_continues() -> None:
    client = MagicMock()
    client.report_intermediate_trial.side_effect = [
        {"prune": True, "prune_reason": "score_below_band"},
        {"prune": False, "prune_reason": None},
        {"prune": False, "prune_reason": None},
    ]
    orchestrator = SmartPruningOrchestrator(
        client,
        configs=[{"answer": "bad"}, {"answer": "ok"}],
    )
    lifecycle = TrialLifecycle(orchestrator)

    trial_count, action = await lifecycle.run_sequential_trial(
        func=_candidate,
        dataset=_dataset(),
        session_id="session-smart",
        function_name="candidate",
        trial_count=0,
    )

    assert action == "continue"
    assert trial_count == 1
    assert orchestrator._trials[0].status == TrialStatus.PRUNED
    assert orchestrator._trials[0].metadata["stop_reason"] == "smart_prune"
    assert orchestrator._trials[0].metadata["prune_reason"] == "score_below_band"

    trial_count, action = await lifecycle.run_sequential_trial(
        func=_candidate,
        dataset=_dataset(),
        session_id="session-smart",
        function_name="candidate",
        trial_count=trial_count,
    )

    assert action == "continue"
    assert trial_count == 2
    assert [trial.status for trial in orchestrator._trials] == [
        TrialStatus.PRUNED,
        TrialStatus.COMPLETED,
    ]
    assert orchestrator._stop_reason is None


def test_smart_pruning_false_runs_trial_normally() -> None:
    asyncio.run(_run_smart_pruning_false_runs_trial_normally())


async def _run_smart_pruning_false_runs_trial_normally() -> None:
    client = MagicMock()
    client.report_intermediate_trial.return_value = {
        "prune": False,
        "prune_reason": None,
    }
    orchestrator = SmartPruningOrchestrator(client, configs=[{"answer": "ok"}])
    lifecycle = TrialLifecycle(orchestrator)

    trial_count, action = await lifecycle.run_sequential_trial(
        func=_candidate,
        dataset=_dataset(),
        session_id="session-smart",
        function_name="candidate",
        trial_count=0,
    )

    assert action == "continue"
    assert trial_count == 1
    assert orchestrator._trials[0].status == TrialStatus.COMPLETED
    assert client.report_intermediate_trial.call_count == 2
