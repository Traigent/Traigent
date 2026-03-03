"""Tests for hybrid_api evaluator wiring in optimization_pipeline."""

from __future__ import annotations

import pytest

from traigent.core.optimization_pipeline import (
    HybridAPIEvaluatorOptions,
    create_effective_evaluator,
    create_traigent_config,
)
from traigent.evaluators.hybrid_api import HybridAPIEvaluator


def _make_common_kwargs() -> dict[str, object]:
    return {
        "timeout": None,
        "custom_evaluator": None,
        "effective_batch_size": None,
        "effective_thread_workers": None,
        "effective_privacy_enabled": False,
        "objectives": ["accuracy"],
        "js_runtime_config": None,
        "mock_mode_config": None,
        "metric_functions": None,
        "scoring_function": None,
        "decorator_custom_evaluator": None,
    }


def test_create_traigent_config_accepts_hybrid_api_mode() -> None:
    """create_traigent_config supports execution_mode='hybrid_api'."""
    cfg = create_traigent_config(
        execution_mode="hybrid_api",
        local_storage_path=None,
        minimal_logging=True,
        privacy_enabled=False,
    )
    assert cfg.execution_mode == "hybrid_api"


def test_create_effective_evaluator_uses_hybrid_api_evaluator() -> None:
    """Hybrid mode wiring returns HybridAPIEvaluator when requested."""
    evaluator, js_pool = create_effective_evaluator(
        **_make_common_kwargs(),
        execution_mode="hybrid_api",
        hybrid_api_options=HybridAPIEvaluatorOptions(
            endpoint="http://localhost:8080",
            tunable_id="cap-1",
            batch_size=8,
            batch_parallelism=2,
        ),
    )

    assert isinstance(evaluator, HybridAPIEvaluator)
    assert evaluator._api_endpoint == "http://localhost:8080"
    assert evaluator._tunable_id == "cap-1"
    assert evaluator._batch_size == 8
    assert evaluator._batch_parallelism == 2
    assert js_pool is None


def test_create_effective_evaluator_hybrid_api_requires_endpoint_or_transport() -> None:
    """Missing hybrid endpoint/transport raises a clear error."""
    with pytest.raises(ValueError, match="hybrid_api execution mode requires"):
        create_effective_evaluator(
            **_make_common_kwargs(),
            execution_mode="hybrid_api",
        )
