"""Regression tests for Traigent#2898 (max_total_examples) and #2899 (numeric env vars)."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import traigent
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.sync_manager import SyncManager
from traigent.core.license import LicenseValidator
from traigent.optimizers.interactive_optimizer import _resolve_optimizer_ready_timeout
from traigent.utils.env_config import get_validation_timeout
from traigent.utils.numeric_validation import (
    parse_positive_finite_float,
    positive_int_env,
    validate_positive_int,
)

BAD_EXAMPLE_BUDGETS = [0, -1, math.nan, math.inf, True, False, "3", 2.5]


@pytest.mark.parametrize("bad", BAD_EXAMPLE_BUDGETS)
def test_validate_positive_int_rejects_naming_field(bad: object) -> None:
    with pytest.raises(ValueError, match="max_total_examples"):
        validate_positive_int("max_total_examples", bad)


def test_validate_positive_int_accepts_ints_and_integral_floats() -> None:
    assert validate_positive_int("x", 3) == 3
    value = validate_positive_int("x", 3.0)
    assert value == 3 and type(value) is int


@pytest.mark.parametrize("bad", BAD_EXAMPLE_BUDGETS)
def test_decorator_rejects_bad_max_total_examples(bad: object) -> None:
    with pytest.raises(ValueError, match="max_total_examples"):

        @traigent.optimize(
            configuration_space={"model": ["a", "b"]}, max_total_examples=bad
        )
        def fn(prompt: str) -> str:
            return prompt


def test_decorator_accepts_positive_max_total_examples() -> None:
    @traigent.optimize(configuration_space={"model": ["a", "b"]}, max_total_examples=5)
    def fn(prompt: str) -> str:
        return prompt

    assert fn.kwargs.get("max_total_examples", 5) == 5


@pytest.mark.parametrize("bad", [0, math.nan, True, "3", 2.5])
def test_override_config_rejects_bad_max_total_examples(bad: object) -> None:
    from traigent.api.functions import override_config

    with pytest.raises(ValueError, match="max_total_examples"):
        override_config(max_total_examples=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, math.nan, True, "3", 2.5])
def test_prepare_dataset_rejects_bad_call_time_max_total_examples(bad: object) -> None:
    from traigent.core.optimization_pipeline import prepare_optimization_dataset
    from traigent.evaluators.base import Dataset, EvaluationExample

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"q": "a"}, expected_output="a")]
    )
    with pytest.raises(ValueError, match="max_total_examples"):
        prepare_optimization_dataset(
            dataset,
            {"max_total_examples": bad},
            max_examples=None,
            max_total_examples=None,
            samples_include_pruned=True,
        )


@pytest.mark.parametrize("bad", [0, math.nan, True, "3", 2.5])
def test_constructor_validation_rejects_bad_budget(bad: object) -> None:
    from traigent.core.orchestrator_helpers import validate_constructor_arguments
    from traigent.evaluators.base import BaseEvaluator
    from traigent.optimizers.base import BaseOptimizer

    with pytest.raises(ValueError, match="max_total_examples"):
        validate_constructor_arguments(
            MagicMock(spec=BaseOptimizer),
            MagicMock(spec=BaseEvaluator),
            max_total_examples=bad,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# #2899 environment variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-1", "abc", ""])
def test_validation_timeout_env_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch, bad: str, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TRAIGENT_VALIDATION_TIMEOUT", bad)
    with caplog.at_level(logging.WARNING):
        assert get_validation_timeout() == 5.0
    assert "TRAIGENT_VALIDATION_TIMEOUT" in caplog.text


def test_validation_timeout_env_accepts_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_VALIDATION_TIMEOUT", "2.5")
    assert get_validation_timeout() == 2.5


@pytest.mark.parametrize("bad", ["nan", "inf", "0", "-1", "abc"])
@pytest.mark.parametrize(
    "env",
    [
        "TRAIGENT_OPTIMIZER_READY_TIMEOUT_SECONDS",
    ],
)
def test_optimizer_ready_timeout_env_rejects_and_names_var(
    monkeypatch: pytest.MonkeyPatch, env: str, bad: str
) -> None:
    monkeypatch.setenv(env, bad)
    with pytest.raises(ValueError, match=env):
        _resolve_optimizer_ready_timeout()


def test_optimizer_ready_timeout_env_accepts_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_OPTIMIZER_READY_TIMEOUT_SECONDS", "30")
    assert _resolve_optimizer_ready_timeout() == 30.0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "0", "-3", "abc"])
def test_sync_http_timeout_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv("TRAIGENT_SYNC_HTTP_TIMEOUT", bad)
    manager = SyncManager.__new__(SyncManager)
    manager.config = SimpleNamespace(custom_params={})
    assert manager._resolve_request_timeout() == 15.0


@pytest.mark.parametrize("bad", ["abc", "1.5", "nan", "0", "-5", ""])
def test_license_validator_env_ttls_fall_back(
    monkeypatch: pytest.MonkeyPatch, bad: str, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TRAIGENT_LICENSE_CACHE_TTL", bad)
    monkeypatch.setenv("TRAIGENT_LICENSE_GRACE_PERIOD", bad)
    with caplog.at_level(logging.WARNING):
        validator = LicenseValidator()
    assert validator._cache_ttl == LicenseValidator.DEFAULT_CACHE_TTL
    assert validator._grace_period == LicenseValidator.DEFAULT_GRACE_PERIOD
    assert "TRAIGENT_LICENSE_CACHE_TTL" in caplog.text
    assert "TRAIGENT_LICENSE_GRACE_PERIOD" in caplog.text


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "nan"])
def test_backend_client_session_env_falls_back(
    monkeypatch: pytest.MonkeyPatch, bad: str, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TRAIGENT_SESSION_TTL", bad)
    monkeypatch.setenv("TRAIGENT_MAX_SESSIONS_PER_USER", bad)
    with caplog.at_level(logging.WARNING):
        BackendIntegratedClient(base_url="http://test.local")
    assert "TRAIGENT_SESSION_TTL" in caplog.text
    assert "TRAIGENT_MAX_SESSIONS_PER_USER" in caplog.text


def test_positive_int_env_accepts_valid_and_defaults_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_X_TEST", raising=False)
    assert positive_int_env("TRAIGENT_X_TEST", 7) == 7
    monkeypatch.setenv("TRAIGENT_X_TEST", " 42 ")
    assert positive_int_env("TRAIGENT_X_TEST", 7) == 42


def test_parse_positive_finite_float_names_variable() -> None:
    with pytest.raises(ValueError, match="TRAIGENT_FOO"):
        parse_positive_finite_float("TRAIGENT_FOO", "nan")


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, True, "3", 2.5])
def test_execution_options_bundle_rejects_bad_max_total_examples(bad: object) -> None:
    from traigent.api.decorators import ExecutionOptions

    with pytest.raises(ValueError, match="max_total_examples"):

        @traigent.optimize(
            configuration_space={"model": ["a", "b"]},
            execution=ExecutionOptions(max_total_examples=bad),  # type: ignore[arg-type]
        )
        def fn(prompt: str) -> str:
            return prompt


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, True, "3", 2.5, [3], {}])
@pytest.mark.parametrize("wrap", [False, True])
def test_tvl_budget_rejects_bad_max_total_examples(bad: object, wrap: bool) -> None:
    from traigent.tvl.spec_loader import _parse_budget

    value = {"value": bad} if wrap else bad
    with pytest.raises(ValueError, match="max_total_examples"):
        _parse_budget({"budget": {"max_total_examples": value}})


def test_tvl_budget_accepts_positive_max_total_examples() -> None:
    from traigent.tvl.spec_loader import _parse_budget

    assert _parse_budget({"budget": {"max_total_examples": 7}}).max_total_examples == 7
    assert _parse_budget({"budget": {}}).max_total_examples is None
