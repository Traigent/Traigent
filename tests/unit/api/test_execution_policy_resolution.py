"""Tests for optimize() execution policy resolution."""

from __future__ import annotations

import inspect
import warnings

import pytest

from traigent.api.decorators import ExecutionOptions, HybridAPIOptions, optimize
from traigent.config.types import (
    ExecutionIntent,
    _reset_deprecation_warning_state_for_tests,
    accepted_algorithm_values,
    resolve_execution_policy,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ConfigurationError

FAIL_CLOSED_MESSAGE_PARTS = (
    "execution_mode=",
    "fails closed",
    "algorithm='auto'",
    "offline=True",
)


@pytest.fixture(autouse=True)
def reset_deprecation_warning_state():
    _reset_deprecation_warning_state_for_tests()
    yield
    _reset_deprecation_warning_state_for_tests()


def _assert_legacy_mode_fails_closed(exc: BaseException, legacy_mode: str) -> None:
    message = str(exc)
    assert f"execution_mode='{legacy_mode}'" in message
    for part in FAIL_CLOSED_MESSAGE_PARTS:
        assert part in message


def _assert_policy(sample: OptimizedFunction):
    policy = sample.execution_policy
    assert policy is not None
    return policy


def test_algorithm_auto_default_resolves_cloud_brain_policy() -> None:
    @optimize(configuration_space={"x": [1, 2]})
    def sample(x: int) -> int:
        return x

    assert isinstance(sample, OptimizedFunction)
    policy = _assert_policy(sample)
    assert policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert policy.algorithm == "auto"
    assert policy.offline is False
    assert policy.allows_cloud_fallback is True


def test_offline_true_resolves_local_only_policy() -> None:
    @optimize(configuration_space={"x": [1, 2]}, offline=True)
    def sample(x: int) -> int:
        return x

    policy = _assert_policy(sample)
    assert policy.intent is ExecutionIntent.LOCAL_ONLY
    assert policy.offline is True
    assert sample.execution_mode == "local"


def test_removed_edge_analytics_raises_via_decorator() -> None:
    """Issue #1684 headline repro: the decorator itself must hard-fail."""
    with pytest.raises(ConfigurationError, match="has been removed") as exc_info:

        @optimize(configuration_space={"x": [1, 2]}, execution_mode="edge_analytics")
        def sample(x: int) -> int:
            return x

    message = str(exc_info.value)
    assert "edge_analytics" in message
    assert "offline=True" in message
    assert "algorithm='grid'" in message
    assert "algorithm='random'" in message
    assert "algorithm='auto'" in message


def test_removed_edge_analytics_raises_in_policy_resolution() -> None:
    """resolve_execution_policy no longer warn-maps the removed selector."""
    with pytest.raises(ConfigurationError, match="has been removed"):
        resolve_execution_policy(execution_mode="edge_analytics")


def test_blank_and_unset_execution_mode_behavior_is_unchanged() -> None:
    """Splitting edge_analytics out of the ""-branch must not change ""/None.

    - execution_mode unset (None) keeps the cloud-first default policy.
    - execution_mode="" keeps its legacy warn-and-map to offline/local.
    """
    unset = resolve_execution_policy(execution_mode=None)
    assert unset.intent is ExecutionIntent.CLOUD_BRAIN
    assert unset.offline is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blank = resolve_execution_policy(execution_mode="")

    assert blank.intent is ExecutionIntent.LOCAL_ONLY
    assert blank.offline is True
    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(messages) == 1
    assert "offline=True" in messages[0]


@pytest.mark.parametrize("algorithm", ("grid", "random"))
def test_preferred_local_policy_emits_no_deprecation_warning(algorithm: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy = resolve_execution_policy(algorithm=algorithm, offline=True)

    assert policy.intent is ExecutionIntent.LOCAL_ONLY
    assert policy.offline is True
    assert policy.algorithm == algorithm
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


def test_local_execution_mode_wire_value_emits_no_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy = resolve_execution_policy(execution_mode="local")

    assert policy.intent is ExecutionIntent.LOCAL_ONLY
    assert policy.offline is True
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


@pytest.mark.parametrize("legacy_mode", ("privacy", "cloud"))
def test_legacy_privacy_cloud_modes_raise_fail_closed(
    monkeypatch: pytest.MonkeyPatch, legacy_mode: str
) -> None:
    monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_execution_policy(execution_mode=legacy_mode)

    _assert_legacy_mode_fails_closed(exc_info.value, legacy_mode)


def test_legacy_cloud_escape_hatch_no_longer_allows_cloud_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_execution_policy(execution_mode="cloud")

    _assert_legacy_mode_fails_closed(exc_info.value, "cloud")


def test_legacy_privacy_mode_with_offline_true_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", raising=False)

    with pytest.raises(ConfigurationError) as exc_info:
        resolve_execution_policy(execution_mode="privacy", offline=True)

    _assert_legacy_mode_fails_closed(exc_info.value, "privacy")


@pytest.mark.parametrize("legacy_mode", ("privacy", "cloud"))
def test_optimize_legacy_privacy_cloud_modes_fail_closed_at_public_surface(
    monkeypatch: pytest.MonkeyPatch, legacy_mode: str
) -> None:
    monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")

    with pytest.raises(ConfigurationError) as exc_info:

        @optimize(configuration_space={"x": [1, 2]}, execution_mode=legacy_mode)
        def sample(x: int) -> int:
            return x

    _assert_legacy_mode_fails_closed(exc_info.value, legacy_mode)


@pytest.mark.parametrize("legacy_mode", ("privacy", "cloud"))
def test_initialize_legacy_privacy_cloud_modes_fail_closed_before_global_config(
    legacy_mode: str,
) -> None:
    from traigent.api.functions import _GLOBAL_CONFIG, initialize

    original_config = _GLOBAL_CONFIG.copy()
    try:
        _GLOBAL_CONFIG.pop("execution_mode", None)
        with pytest.raises(ConfigurationError) as exc_info:
            initialize(execution_mode=legacy_mode)

        assert "execution_mode" not in _GLOBAL_CONFIG
    finally:
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(original_config)

    _assert_legacy_mode_fails_closed(exc_info.value, legacy_mode)


@pytest.mark.parametrize("legacy_mode", ("privacy", "cloud"))
def test_optimize_global_config_legacy_privacy_cloud_modes_fail_closed(
    monkeypatch: pytest.MonkeyPatch, legacy_mode: str
) -> None:
    from traigent.api.functions import _GLOBAL_CONFIG

    monkeypatch.setenv("TRAIGENT_ALLOW_LEGACY_CLOUD_EXECUTION_MODE", "1")
    original_config = _GLOBAL_CONFIG.copy()
    try:
        _GLOBAL_CONFIG["execution_mode"] = legacy_mode
        with pytest.raises(ConfigurationError) as exc_info:

            @optimize(configuration_space={"x": [1, 2]})
            def sample(x: int) -> int:
                return x

    finally:
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(original_config)

    _assert_legacy_mode_fails_closed(exc_info.value, legacy_mode)


def test_initialize_hybrid_global_default_still_matches_decorator_cloud_policy() -> (
    None
):
    from traigent.api.functions import _GLOBAL_CONFIG, initialize

    original_config = _GLOBAL_CONFIG.copy()
    try:
        initialize(execution_mode="hybrid")
        initialized_mode = _GLOBAL_CONFIG.get("execution_mode")

        @optimize(configuration_space={"x": [1, 2]})
        def sample(x: int) -> int:
            return x

    finally:
        _GLOBAL_CONFIG.clear()
        _GLOBAL_CONFIG.update(original_config)

    assert _assert_policy(sample).intent is ExecutionIntent.CLOUD_BRAIN
    assert sample.execution_mode == "hybrid"
    assert initialized_mode == "hybrid"


def test_legacy_auto_execution_mode_is_rejected() -> None:
    with pytest.raises(
        ConfigurationError, match="Unsupported execution selector 'auto'"
    ):

        @optimize(configuration_space={"x": [1, 2]}, execution_mode="auto")
        def sample(x: int) -> int:
            return x


def test_privacy_enabled_is_deprecated_noop_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(configuration_space={"x": [1, 2]}, privacy_enabled=True)
        def sample(x: int) -> int:
            return x

    messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    policy = _assert_policy(sample)
    assert policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert policy.offline is False
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


def test_legacy_hybrid_api_options_evaluator_warns_but_works() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @optimize(
            configuration_space={"x": [1, 2]},
            evaluator=HybridAPIOptions(endpoint="https://evaluator.example.test"),
        )
        def sample(x: int) -> int:
            return x

    assert isinstance(sample.hybrid_api_options, HybridAPIOptions)
    assert sample.hybrid_api_options.endpoint == "https://evaluator.example.test"
    assert sample.execution_mode == "hybrid_api"
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "HybridAPIOptions as an evaluator is deprecated" in str(w.message)
        for w in caught
    )


def test_execution_options_public_fields_exclude_removed_surface() -> None:
    assert "algorithm" in ExecutionOptions.model_fields
    assert "offline" in ExecutionOptions.model_fields
    assert "execution_mode" not in ExecutionOptions.model_fields
    assert "privacy_enabled" not in ExecutionOptions.model_fields
    assert "cloud_fallback_policy" not in ExecutionOptions.model_fields
    assert "hybrid_api_endpoint" not in ExecutionOptions.model_fields


def test_optimize_public_signature_exposes_algorithm_offline_only() -> None:
    public_signature = str(inspect.signature(optimize))

    assert "algorithm" in public_signature
    assert "offline" in public_signature
    for legacy_name in (
        "execution_mode",
        "privacy_enabled",
        "cloud_fallback_policy",
        "HybridAPIOptions",
        "hybrid_api_endpoint",
    ):
        assert legacy_name not in public_signature


def test_offline_smart_algorithm_hard_errors() -> None:
    with pytest.raises(ConfigurationError, match="requires managed optimization"):

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
    with pytest.raises(ValueError) as exc_info:

        @optimize(configuration_space={"x": [1, 2]}, algorithm="optuna_foo")
        def sample(x: int) -> int:
            return x

    message = str(exc_info.value)
    assert message.startswith("algorithm must be one of: ")
    for algorithm_name in accepted_algorithm_values():
        assert algorithm_name in message
    assert "optuna_tpe" in message
    assert "bayesian" in message
    assert "got 'optuna_foo'" in message


def test_known_smart_algorithm_names_accepted() -> None:
    for name in ("tpe", "bayesian", "optuna_tpe", "optuna_cmaes", "nsga2"):

        @optimize(configuration_space={"x": [1, 2]}, algorithm=name)
        def sample(x: int) -> int:
            return x

        assert _assert_policy(sample).algorithm == name


def test_smart_algorithm_resolves_cloud_required_policy() -> None:
    policy = resolve_execution_policy(algorithm="tpe", offline=False)

    assert policy.intent is ExecutionIntent.CLOUD_REQUIRED
    assert policy.algorithm == "tpe"
    assert policy.offline is False
    assert policy.allows_cloud_fallback is False
