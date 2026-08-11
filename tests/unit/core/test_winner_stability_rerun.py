"""Winner-stability rerun tests (opt-in ``winner_stability_reps``).

Measured-only contract: with ``winner_stability_reps=3`` a completed run
carries ``result.metadata["winner_stability"]`` with 3 scores measured by
re-executing the already-selected winner through the normal trial execution
path; with the default ``0`` the key is absent and no extra evaluation runs.
The rerun never enters ``result.trials`` and never changes which config wins.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from collections.abc import Callable

import pytest

from traigent.api.types import OptimizationStatus, TrialResult
from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError


class StabilityOptimizer(BaseOptimizer):
    """Deterministic optimizer: suggests param1=0,1,2 then stops."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self._suggest_count = 0
        self._max_suggestions = 3

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        config = {"param1": self._suggest_count}
        self._suggest_count += 1
        return config

    def should_stop(self, history: list[TrialResult]) -> bool:
        return self._suggest_count >= self._max_suggestions

    def tell(self, config: dict[str, Any], result: TrialResult) -> None:
        return None

    def is_finished(self) -> bool:
        return self._suggest_count >= self._max_suggestions


class StabilityEvaluator(BaseEvaluator):
    """Deterministic evaluator: accuracy = 0.5 + param1 * 0.1; counts calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_count = 0
        self.evaluated_configs: list[dict[str, Any]] = []
        self.should_fail = False

    async def evaluate(
        self,
        func: Callable,
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease=None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
        **_kwargs,
    ) -> EvaluationResult:
        self.evaluation_count += 1
        self.evaluated_configs.append(dict(config))
        if self.should_fail:
            raise OptimizationError(f"Evaluation failed for config: {config}")

        processed = 0
        for index, _example in enumerate(dataset.examples):
            if sample_lease and not sample_lease.try_take(1):
                break
            processed += 1
            if progress_callback:
                progress_callback(index, {"success": True})

        accuracy = 0.5 + config.get("param1", 0) * 0.1
        metrics = {"accuracy": accuracy, "examples_attempted": processed}
        result = EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=processed,
            successful_examples=processed,
            duration=0.01,
            metrics=metrics,
            outputs=[f"output_{i}" for i in range(processed)],
            errors=[None for _ in range(processed)],
        )
        result.sample_budget_exhausted = False
        result.examples_consumed = processed
        return result


@pytest.fixture(autouse=True)
def isolated_optimization_logs(monkeypatch, tmp_path):
    """Keep orchestrator logging isolated from developer-local log history."""
    monkeypatch.setenv(
        "TRAIGENT_OPTIMIZATION_LOG_DIR",
        str(tmp_path / "optimization_logs"),
    )


@pytest.fixture
def sample_dataset() -> Dataset:
    examples = [
        EvaluationExample({"query": "Hello"}, "Hi there!"),
        EvaluationExample({"query": "Goodbye"}, "See you later!"),
    ]
    return Dataset(examples, name="test_dataset", description="Test dataset")


@pytest.fixture
def mock_function():
    async def test_function(input_data: dict[str, Any], **config) -> Any:
        return input_data.get("query", "default response")

    return test_function


def _build_orchestrator(
    evaluator: StabilityEvaluator, **extra_kwargs: Any
) -> OptimizationOrchestrator:
    optimizer = StabilityOptimizer({"param1": [0, 1, 2]}, ["accuracy"])
    return OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=3,
        config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        **extra_kwargs,
    )


class TestWinnerStabilityRerun:
    @pytest.mark.asyncio
    async def test_block_appears_with_three_scores(self, sample_dataset, mock_function):
        """reps=3: the block records exactly 3 measured rerun scores."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        block = result.metadata["winner_stability"]
        assert block["reps"] == 3
        assert block["scores"] == [0.7, 0.7, 0.7]
        assert block["mean"] == pytest.approx(0.7)
        assert block["std"] == pytest.approx(0.0)
        assert isinstance(block["config_hash"], str) and block["config_hash"]
        # evaluated_at must be a parseable ISO-8601 timestamp.
        datetime.fromisoformat(block["evaluated_at"])
        # 3 search evaluations + 3 winner reruns, no more, no fewer.
        assert evaluator.evaluation_count == 6
        # Every rerun executed the winning config through the normal path.
        assert [c.get("param1") for c in evaluator.evaluated_configs[3:]] == [2, 2, 2]

    @pytest.mark.asyncio
    async def test_absent_by_default(self, sample_dataset, mock_function):
        """Default (0): no block, no extra evaluations."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert "winner_stability" not in result.metadata
        assert evaluator.evaluation_count == 3

    @pytest.mark.asyncio
    async def test_rerun_stays_out_of_trials_and_selection(
        self, sample_dataset, mock_function
    ):
        """The rerun never enters result.trials and never moves the winner."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=2)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert len(result.trials) == 3
        assert result.best_config is not None
        assert result.best_config.get("param1") == 2
        assert result.best_score == pytest.approx(0.7)
        assert result.metadata["winner_stability"]["reps"] == 2

    @pytest.mark.asyncio
    async def test_no_winner_skips_the_rerun(self, sample_dataset, mock_function):
        """A run with no selected winner records no block and spends nothing."""
        evaluator = StabilityEvaluator()
        evaluator.should_fail = True
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=3)

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.best_config is None
        assert "winner_stability" not in result.metadata
        # Only the 3 (failed) search evaluations ran; zero rerun spend.
        assert evaluator.evaluation_count == 3

    @pytest.mark.asyncio
    async def test_rerun_failure_never_fails_the_run(
        self, sample_dataset, mock_function
    ):
        """A rerun that starts failing keeps the run COMPLETED, block-free."""
        evaluator = StabilityEvaluator()
        orchestrator = _build_orchestrator(evaluator, winner_stability_reps=2)

        original_evaluate = evaluator.evaluate

        async def failing_after_search(*args: Any, **kwargs: Any):
            if evaluator.evaluation_count >= 3:
                evaluator.evaluation_count += 1
                raise OptimizationError("provider went away mid-rerun")
            return await original_evaluate(*args, **kwargs)

        evaluator.evaluate = failing_after_search  # type: ignore[method-assign]

        result = await orchestrator.optimize(mock_function, sample_dataset)

        assert result.status == OptimizationStatus.COMPLETED
        assert result.best_config is not None
        # Nothing was measured, so no block is recorded (measured-only).
        assert "winner_stability" not in result.metadata
