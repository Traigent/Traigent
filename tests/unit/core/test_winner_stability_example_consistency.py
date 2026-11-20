"""Per-example consistency labels from winner-stability reruns.

The orchestrator reruns the selected winner ``winner_stability_reps`` times and
keeps each rerun's per-example pass/fail. These tests run the real orchestrator
with a scripted evaluator and check the labels in
``result.example_consistency_report``, that the existing
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


class _CrashingEvaluator(_ScriptedEvaluator):
    """Raises on the given evaluation calls (0 is the search trial)."""

    def __init__(self, script: dict[Any, list[float | None]], crash: set[int]):
        super().__init__(script)
        self.crash = crash

    async def evaluate(self, *args: Any, **kwargs: Any) -> EvaluationResult:
        if self.calls in self.crash:
            self.calls += 1
            raise RuntimeError("evaluator crashed")
        return await super().evaluate(*args, **kwargs)


async def _run(reps: int, script=None, evaluator=None) -> Any:
    evaluator = evaluator or _ScriptedEvaluator(script or _SCRIPT)
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

    stored = result.example_consistency_report
    examples = stored["examples"]
    assert examples["always_wrong"] == {
        "label": "all_fail",
        "attempted": 5,
        "passes": 0,
        "fails": 5,
        "errored": 0,
        "pass_rate_low": 0.0,
        "pass_rate_high": 0.434,
    }
    assert examples["sometimes_wrong"] == {
        "label": "mixed",
        "attempted": 5,
        "passes": 2,
        "fails": 3,
        "errored": 0,
        "pass_rate_low": 0.118,
        "pass_rate_high": 0.769,
    }
    assert examples["always_right"] == {
        "label": "all_pass",
        "attempted": 5,
        "passes": 5,
        "fails": 0,
        "errored": 0,
        "pass_rate_low": 0.566,
        "pass_rate_high": 1.0,
    }
    assert stored["reps"] == 5
    # What "pass" meant: the per-example score, cut at 0.5.
    assert stored["metric"] == "score"
    assert stored["threshold"] == 0.5
    assert stored["config_hash"] == result.metadata["winner_stability"]["config_hash"]


@pytest.mark.asyncio
async def test_integer_and_string_ids_are_not_merged_in_the_stability_path(
    caplog,
) -> None:
    """Ids ``1`` (always fails) and ``"1"`` (always passes) stay apart.

    Merging them would label one example with the other's outcomes. Nothing is
    recorded for the run instead, the winner-stability block is untouched, and
    the warning names both ids.
    """
    script: dict[Any, list[float | None]] = {
        1: [0.0] * 6,
        "1": [1.0] * 6,
    }
    with caplog.at_level("WARNING"):
        result = await _run(5, script)

    assert result.example_consistency()["examples"] == {}
    assert result.metadata["winner_stability"]["reps"] == 5
    assert "1 (int)" in caplog.text and "'1' (str)" in caplog.text


@pytest.mark.asyncio
async def test_mostly_errored_example_gets_no_label_and_is_counted_as_errored() -> None:
    result = await _run(5)

    # Four of five reruns errored: never a clean pass, and no label at all.
    entry = result.example_consistency()["examples"]["mostly_errors"]
    assert entry == {
        "label": None,
        "attempted": 5,
        "passes": 1,
        "fails": 0,
        "errored": 4,
        "pass_rate_low": 0.207,
        "pass_rate_high": 1.0,
    }
    assert result.example_consistency()["counts"]["errored"] == 1


@pytest.mark.asyncio
async def test_a_crashed_rerun_is_one_errored_attempt_for_every_example() -> None:
    """Reruns that crash are counted, not dropped, so ``reps`` is not undercounted."""
    script = {"a": [1.0] * 6, "b": [0.0] * 6}
    result = await _run(5, evaluator=_CrashingEvaluator(script, crash={2}))

    report = result.example_consistency()
    assert report["reps"] == 5
    for row in report["examples"].values():
        assert row["attempted"] == 5
        assert row["errored"] == 1
        assert row["passes"] + row["fails"] == 4


@pytest.mark.asyncio
async def test_accessor_reports_counts_and_per_example_labels() -> None:
    result = await _run(5)

    report = result.example_consistency()
    assert report["counts"] == {
        "all_pass": 1,
        "all_fail": 1,
        "mixed": 1,
        "errored": 1,
    }
    assert report["examples"]["sometimes_wrong"]["label"] == "mixed"
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

    assert result.example_consistency_report is None
    assert "example_consistency" not in result.metadata
    assert result.example_consistency()["examples"] == {}


@pytest.mark.asyncio
async def test_single_rerun_records_no_labels() -> None:
    result = await _run(1)

    assert result.example_consistency_report is None
    assert "example_consistency" not in result.metadata
    assert result.example_consistency()["counts"] == {
        "all_pass": 0,
        "all_fail": 0,
        "mixed": 0,
        "errored": 0,
    }


@pytest.mark.asyncio
async def test_labels_survive_save_and_reload(tmp_path) -> None:
    result = await _run(5)
    manager = PersistenceManager(str(tmp_path / "store"))

    manager.save_result(result, name="consistency_run")
    reloaded = manager.load_result("consistency_run")

    assert reloaded.example_consistency() == result.example_consistency()
    assert reloaded.example_consistency()["examples"]["sometimes_wrong"] == {
        "label": "mixed",
        "attempted": 5,
        "passes": 2,
        "fails": 3,
        "errored": 0,
        "pass_rate_low": 0.118,
        "pass_rate_high": 0.769,
    }


@pytest.mark.asyncio
async def test_labels_are_not_written_into_trial_metadata() -> None:
    result = await _run(5)

    assert result.example_consistency_report is not None
    assert "example_consistency" not in result.metadata
    for trial in result.trials:
        assert "example_consistency" not in trial.metadata
