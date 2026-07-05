"""Regression tests for orchestrator run config persistence."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from traigent.config.types import TraigentConfig
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.optimizers.base import BaseOptimizer


class TinyOptimizer(BaseOptimizer):
    def suggest_next_trial(self, history: list[Any]) -> dict[str, Any]:
        return {"temperature": 0.0}

    def should_stop(self, history: list[Any]) -> bool:
        return False


class TinyEvaluator(BaseEvaluator):
    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease=None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        raise AssertionError("this regression test only initializes logging")


def test_default_config_run_persists_explicit_config_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "TRAIGENT_OPTIMIZATION_LOG_DIR", str(tmp_path / "optimization_logs")
    )
    config_space = {"temperature": [0.0, 0.7], "model": ["gpt-4o-mini"]}
    optimizer = TinyOptimizer(config_space, ["accuracy"])
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=TinyEvaluator(),
        max_trials=3,
        config=TraigentConfig(execution_mode="local"),
        parallel_trials=2,
        requested_algorithm="grid",
    )
    dataset = Dataset([], name="empty_dataset")

    orchestrator._initialize_logger("session-1234567890", lambda: None, dataset)

    assert orchestrator._logger is not None
    config_path = orchestrator._logger.run_path / "meta" / "config_v2.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_data != {}
    assert config_data["algorithm"] == "grid"
    assert config_data["requested_algorithm"] == "grid"
    assert config_data["resolved_algorithm"] == "TinyOptimizer"
    assert config_data["max_trials"] == 3
    assert config_data["parallel_trials"] == 2
    assert config_data["configuration_space"] == config_space
