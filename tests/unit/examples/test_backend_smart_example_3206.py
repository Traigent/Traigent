"""Regression tests for examples/core/multi-objective-tradeoff (Traigent#3206)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from traigent.testing import _reset_for_tests

ROOT_DIR = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT_DIR
    / "examples"
    / "core"
    / "multi-objective-tradeoff"
    / "run_openai_backend_smart.py"
)
_NAME = "_run_openai_backend_smart_3206"


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _reset_for_tests()
    for key in (
        "TRAIGENT_MOCK_LLM",
        "TRAIGENT_MODEL_SET",
        "TRAIGENT_BUDGET_LIMIT",
        "TRAIGENT_TRIAL_CONCURRENCY",
        "TRAIGENT_CONCURRENCY_PROFILE",
    ):
        monkeypatch.delenv(key, raising=False)
    spec = importlib.util.spec_from_file_location(_NAME, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(_NAME, None)
        _reset_for_tests()


def test_default_models_are_distinct(example: ModuleType) -> None:
    assert len(example.MODEL_CHOICES) == len(set(example.MODEL_CHOICES)) >= 2


def test_budget_does_not_replace_announced_objectives_or_algorithm(
    example: ModuleType,
) -> None:
    kwargs = example._build_runtime_kwargs(20, 0.012)
    assert kwargs == {"max_trials": 20, "cost_limit": 0.012}
    assert "objectives" not in kwargs
    assert "algorithm" not in kwargs
    # The announced accuracy+cost schema and grid algorithm stay on the decorator.
    assert not isinstance(example.OPTIMIZATION_OBJECTIVES, list)
    assert example._OPTIMIZE_KWARGS["algorithm"] == "grid"


def test_no_budget_means_no_cost_limit(example: ModuleType) -> None:
    assert example._build_runtime_kwargs(5, None) == {"max_trials": 5}
    assert example._build_runtime_kwargs(5, 0.0) == {"max_trials": 5}


def test_budgeted_default_runs_trials_one_at_a_time(example: ModuleType) -> None:
    # Parallel trials are cancelled by the cost gate before they start (#2943).
    assert example.BUDGET_LIMIT and example.BUDGET_LIMIT > 0
    assert example.PARALLEL_CONFIG.trial_concurrency == 1
