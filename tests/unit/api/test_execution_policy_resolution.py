"""Tests for optimize() execution policy resolution."""

from __future__ import annotations

import warnings

import pytest

from traigent.api.decorators import ExecutionOptions, HybridAPIOptions, optimize
from traigent.config.types import ExecutionIntent
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError


def test_algorithm_auto_default_resolves_cloud_brain_policy() -> None:
    @optimize(configuration_space={"x": [1, 2]})
    def sample(x: int) -> int:
        return x

    assert isinstance(sample, OptimizedFunction)
    assert sample.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert sample.execution_policy.algorithm == "auto"
    assert sample.execution_policy.offline is False
    assert sample.execution_policy.allows_cloud_fallback is True


def test_offline_true_resolves_local_only_policy() -> None:
    @optimize(configuration_space={"x": [1, 2]}, offline=True)
    def sample(x: int) -> int:
        return x

    assert sample.execution_policy.intent is ExecutionIntent.LOCAL_ONLY
    assert sample.execution_policy.offline is True
    assert sample.execution_mode == "edge_analytics"


def test_legacy_edge_analytics_maps_to_offline_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(configuration_space={"x": [1, 2]}, execution_mode="edge_analytics")
        def sample(x: int) -> int:
            return x

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert sample.execution_policy.intent is ExecutionIntent.LOCAL_ONLY
    assert sample.execution_policy.offline is True
    assert any("preserve the legacy no-egress guarantee" in msg for msg in messages)


def test_legacy_cloud_maps_to_cloud_first_with_loud_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(configuration_space={"x": [1, 2]}, execution_mode="cloud")
        def sample(x: int) -> int:
            return x

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert sample.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert sample.execution_policy.offline is False
    assert any("semantic flip" in msg for msg in messages)


def test_privacy_enabled_is_deprecated_noop_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(configuration_space={"x": [1, 2]}, privacy_enabled=True)
        def sample(x: int) -> int:
            return x

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert sample.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert sample.execution_policy.offline is False
    assert any(
        "privacy_enabled is deprecated and has no effect" in msg for msg in messages
    )


def test_legacy_flat_hybrid_api_kwargs_translate_to_options() -> None:
    transport = object()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(
            configuration_space={"x": [1, 2]},
            hybrid_api_endpoint="https://evaluator.example.test",
            tunable_id="tunable-123",
            hybrid_api_transport=transport,
            hybrid_api_batch_size=3,
        )
        def sample(x: int) -> int:
            return x

    assert isinstance(sample.hybrid_api_options, HybridAPIOptions)
    assert sample.hybrid_api_options.endpoint == "https://evaluator.example.test"
    assert sample.hybrid_api_options.tunable_id == "tunable-123"
    assert sample.hybrid_api_options.transport is transport
    assert sample.hybrid_api_options.batch_size == 3
    assert sample.execution_mode == "hybrid_api"
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "Flat hybrid_api_* optimize options are deprecated" in str(w.message)
        for w in caught
    )


def test_execution_options_public_fields_exclude_removed_surface() -> None:
    assert "algorithm" in ExecutionOptions.model_fields
    assert "offline" in ExecutionOptions.model_fields
    assert "execution_mode" not in ExecutionOptions.model_fields
    assert "privacy_enabled" not in ExecutionOptions.model_fields
    assert "cloud_fallback_policy" not in ExecutionOptions.model_fields
    assert "hybrid_api_endpoint" not in ExecutionOptions.model_fields


def test_offline_smart_algorithm_hard_errors() -> None:
    with pytest.raises(ConfigurationError, match="requires cloud execution"):

        @optimize(
            configuration_space={"x": [1, 2]},
            algorithm="tpe",
            offline=True,
        )
        def sample(x: int) -> int:
            return x


def test_unknown_algorithm_name_rejected() -> None:
    # Defect #3 regression: unknown ``optuna_*`` (and any unknown) name must be
    # rejected at the public boundary, not accepted-then-failed later.
    with pytest.raises(ValueError, match="known smart optimizer name"):

        @optimize(configuration_space={"x": [1, 2]}, algorithm="optuna_foo")
        def sample(x: int) -> int:
            return x


def test_known_smart_algorithm_names_accepted() -> None:
    for name in ("tpe", "bayesian", "optuna_tpe", "optuna_cmaes", "nsga2"):

        @optimize(configuration_space={"x": [1, 2]}, algorithm=name)
        def sample(x: int) -> int:
            return x

        assert sample.execution_policy.algorithm == name
