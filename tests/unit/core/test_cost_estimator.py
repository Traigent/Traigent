"""Tests for CostEstimator (traigent.core.cost_estimator)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

import traigent.utils.cost_calculator as cost_calculator
from traigent.core.cost_estimator import (
    ALLOW_UNPRICED_MODELS_ENV,
    CUSTOM_PRICING_FILE_ENV,
    CUSTOM_PRICING_JSON_ENV,
    CostCoverageError,
    CostEstimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeDataset:
    """Minimal dataset for cost estimation tests."""

    examples: list = field(default_factory=lambda: [None] * 50)

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class _FakeTrialResult:
    metrics: dict | None = None
    metadata: dict | None = None


@pytest.fixture(autouse=True)
def _isolate_cost_pricing_state(monkeypatch):
    """Keep custom-pricing env/cache and mock-mode state local to each test."""
    from traigent import testing as traigent_testing

    old_cache = cost_calculator._CUSTOM_PRICING_CACHE
    old_key = cost_calculator._CUSTOM_PRICING_CACHE_KEY
    cost_calculator._CUSTOM_PRICING_CACHE = None
    cost_calculator._CUSTOM_PRICING_CACHE_KEY = None
    traigent_testing._reset_for_tests()
    monkeypatch.delenv(ALLOW_UNPRICED_MODELS_ENV, raising=False)
    monkeypatch.delenv(CUSTOM_PRICING_JSON_ENV, raising=False)
    monkeypatch.delenv(CUSTOM_PRICING_FILE_ENV, raising=False)
    monkeypatch.delenv("TRAIGENT_MOCK_LLM", raising=False)
    try:
        yield
    finally:
        cost_calculator._CUSTOM_PRICING_CACHE = old_cache
        cost_calculator._CUSTOM_PRICING_CACHE_KEY = old_key
        traigent_testing._reset_for_tests()


# ---------------------------------------------------------------------------
# estimate_optimization_cost
# ---------------------------------------------------------------------------


class TestEstimateOptimizationCost:
    def test_per_trial_mode_default_trials(self) -> None:
        """With no max_total_examples and no max_trials, uses 10 default trials."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=None, max_total_examples=None)
        dataset = _FakeDataset()
        cost = estimator.estimate_optimization_cost(dataset)
        # 50 examples * 10 trials * 0.0675 * 1.2 = 40.5
        assert cost == pytest.approx(40.5)

    def test_per_trial_mode_explicit_trials(self) -> None:
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=None)
        dataset = _FakeDataset()
        cost = estimator.estimate_optimization_cost(dataset)
        # 50 * 5 * 0.0675 * 1.2 = 20.25
        assert cost == pytest.approx(20.25)

    def test_total_examples_budget_mode(self) -> None:
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=200)
        dataset = _FakeDataset()
        cost = estimator.estimate_optimization_cost(dataset)
        # 200 * 0.0675 * 1.2 = 16.2
        assert cost == pytest.approx(16.2)

    def test_dataset_without_len(self) -> None:
        """When dataset has no __len__, falls back to 100."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=2, max_total_examples=None)
        dataset = MagicMock(spec=[])  # No __len__
        cost = estimator.estimate_optimization_cost(dataset)
        # 100 * 2 * 0.0675 * 1.2 = 16.2
        assert cost == pytest.approx(16.2)

    def test_model_aware_pricing_path(self) -> None:
        """When model pricing is known, use model-aware base cost."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer, max_trials=10, max_total_examples=None, model_name="gpt-4o-mini"
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            return_value=(0.5e-6, 2.0e-6, "litellm"),
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        # Base = (2000 * 0.5e-6) + (500 * 2.0e-6) = 0.002
        # Total = 50 * 10 * 0.002 * 1.2 = 1.2
        assert cost == pytest.approx(1.2)

    def test_unknown_model_fails_closed_without_override(self) -> None:
        """Unknown real-run model pricing should fail before estimation fallback."""
        from traigent.utils.cost_calculator import UnknownModelError

        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=None,
            max_total_examples=None,
            model_name="unknown-private-model",
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=UnknownModelError("unknown model"),
        ):
            with pytest.raises(CostCoverageError) as exc_info:
                estimator.estimate_optimization_cost(dataset)

        assert "unknown-private-model" in str(exc_info.value)
        assert CUSTOM_PRICING_JSON_ENV in str(exc_info.value)
        assert ALLOW_UNPRICED_MODELS_ENV in str(exc_info.value)

    def test_unknown_model_uses_conservative_pricing_with_override(
        self, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Explicit override keeps the conservative fallback visible."""
        from traigent.utils.cost_calculator import UnknownModelError

        monkeypatch.setenv(ALLOW_UNPRICED_MODELS_ENV, "true")
        caplog.set_level(logging.WARNING, logger="traigent.core.cost_estimator")
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=None,
            max_total_examples=None,
            model_name="unknown-private-model",
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=UnknownModelError("unknown model"),
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        assert cost == pytest.approx(40.5)
        assert ALLOW_UNPRICED_MODELS_ENV in caplog.text
        assert "conservative_fallback" in caplog.text

    def test_unknown_model_uses_custom_pricing_json(self, monkeypatch) -> None:
        """Custom rates satisfy coverage and drive the estimate."""
        monkeypatch.setenv(
            CUSTOM_PRICING_JSON_ENV,
            json.dumps(
                {
                    "unknown-private-model": {
                        "input_cost_per_token": 1e-6,
                        "output_cost_per_token": 2e-6,
                    }
                }
            ),
        )

        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=None,
            max_total_examples=None,
            model_name="unknown-private-model",
        )
        dataset = _FakeDataset()

        cost = estimator.estimate_optimization_cost(dataset)

        # Base = (2000 * 1e-6) + (500 * 2e-6) = 0.003
        # Total = 50 * 10 * 0.003 * 1.2 = 1.8
        assert cost == pytest.approx(1.8)

    def test_model_pricing_exception_fails_closed_without_override(self) -> None:
        """Unexpected pricing lookup failures should fail before fallback."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=None,
            max_total_examples=None,
            model_name="gpt-4o-mini",
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=RuntimeError("pricing backend unavailable"),
        ):
            with pytest.raises(CostCoverageError, match="gpt-4o-mini"):
                estimator.estimate_optimization_cost(dataset)

    def test_unknown_model_in_mock_mode_does_not_fail_coverage(self) -> None:
        """Mock runs can keep conservative estimates because no spend happens."""
        from traigent import testing as traigent_testing
        from traigent.utils.cost_calculator import UnknownModelError

        traigent_testing.enable_mock_mode_for_quickstart()
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=None,
            max_total_examples=None,
            model_name="unknown-private-model",
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=UnknownModelError("unknown model"),
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        assert cost == pytest.approx(40.5)

    def test_candidate_models_use_most_expensive_known_model(self) -> None:
        """When no fixed model is set, use the priciest candidate model."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            candidate_models=["gpt-4o-mini", "gpt-4o"],
        )
        dataset = _FakeDataset()

        def pricing(model: str) -> tuple[float, float, str]:
            if model == "gpt-4o-mini":
                return (0.15e-6, 0.6e-6, "litellm")
            if model == "gpt-4o":
                return (2.5e-6, 10.0e-6, "litellm")
            raise AssertionError(f"unexpected model {model}")

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=pricing,
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        # Base uses gpt-4o = (2000 * 2.5e-6) + (500 * 10e-6) = 0.01
        # Total = 50 * 10 * 0.01 * 1.2 = 6.0
        assert cost == pytest.approx(6.0)

    def test_unknown_candidate_model_fails_closed_without_override(self) -> None:
        """Unknown config-space model pricing should fail closed for real runs."""
        from traigent.utils.cost_calculator import UnknownModelError

        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            candidate_models=["gpt-4o-mini", "unknown-private-model"],
        )
        dataset = _FakeDataset()

        def pricing(model: str) -> tuple[float, float, str]:
            if model == "gpt-4o-mini":
                return (0.15e-6, 0.6e-6, "litellm")
            raise UnknownModelError(model)

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=pricing,
        ):
            with pytest.raises(CostCoverageError) as exc_info:
                estimator.estimate_optimization_cost(dataset)

        assert "unknown-private-model" in str(exc_info.value)

    def test_unknown_candidate_model_uses_conservative_pricing_with_override(
        self, monkeypatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Override-gated candidate fallback keeps the existing conservative math."""
        from traigent.utils.cost_calculator import UnknownModelError

        monkeypatch.setenv(ALLOW_UNPRICED_MODELS_ENV, "true")
        caplog.set_level(logging.WARNING, logger="traigent.core.cost_estimator")
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            candidate_models=["gpt-4o-mini", "unknown-private-model"],
        )
        dataset = _FakeDataset()

        def pricing(model: str) -> tuple[float, float, str]:
            if model == "gpt-4o-mini":
                return (0.15e-6, 0.6e-6, "litellm")
            raise UnknownModelError(model)

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=pricing,
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        assert cost == pytest.approx(40.5)
        assert "unknown-private-model" in caplog.text
        assert "conservative_fallback" in caplog.text

    def test_candidate_model_pricing_exception_fails_closed_without_override(
        self,
    ) -> None:
        """Unexpected candidate pricing failures should fail before fallback."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            candidate_models=["gpt-4o"],
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=RuntimeError("pricing backend unavailable"),
        ):
            with pytest.raises(CostCoverageError, match="gpt-4o"):
                estimator.estimate_optimization_cost(dataset)

    def test_explicit_model_takes_precedence_over_candidate_models(self) -> None:
        """A fixed run model should override config-space candidates."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            model_name="gpt-4o-mini",
            candidate_models=["gpt-4o"],
        )
        dataset = _FakeDataset()

        def pricing(model: str) -> tuple[float, float, str]:
            if model == "gpt-4o-mini":
                return (0.15e-6, 0.6e-6, "litellm")
            if model == "gpt-4o":
                return (2.5e-6, 10.0e-6, "litellm")
            raise AssertionError(f"unexpected model {model}")

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=pricing,
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        # Base uses explicit gpt-4o-mini = (2000 * 0.15e-6) + (500 * 0.6e-6) = 0.0006
        # Total = 50 * 10 * 0.0006 * 1.2 = 0.36
        assert cost == pytest.approx(0.36)

    def test_estimated_tokens_reduce_hybrid_candidate_model_cost(self) -> None:
        """Service-provided token estimates should replace generic 2000/500 defaults."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            candidate_models=["gpt-4o-mini", "gpt-4o"],
            estimated_input_tokens_per_example=100,
            estimated_output_tokens_per_example=50,
        )
        dataset = _FakeDataset()

        def pricing(model: str) -> tuple[float, float, str]:
            if model == "gpt-4o-mini":
                return (0.15e-6, 0.6e-6, "litellm")
            if model == "gpt-4o":
                return (2.5e-6, 10.0e-6, "litellm")
            raise AssertionError(f"unexpected model {model}")

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=pricing,
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        # Base uses gpt-4o with service estimate = (100 * 2.5e-6) + (50 * 10e-6) = 0.00075
        # Total = 50 * 10 * 0.00075 * 1.2 = 0.45
        assert cost == pytest.approx(0.45)

    def test_estimated_tokens_apply_to_override_gated_conservative_fallback_pricing(
        self, monkeypatch
    ) -> None:
        """Token estimate metadata should still improve the conservative fallback."""
        from traigent.utils.cost_calculator import UnknownModelError

        monkeypatch.setenv(ALLOW_UNPRICED_MODELS_ENV, "true")
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            model_name="unknown-private-model",
            estimated_input_tokens_per_example=100,
            estimated_output_tokens_per_example=50,
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            side_effect=UnknownModelError("unknown model"),
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        # Base = (100 * 15e-6) + (50 * 75e-6) = 0.00525
        # Total = 50 * 10 * 0.00525 * 1.2 = 3.15
        assert cost == pytest.approx(3.15)

    def test_zero_estimated_tokens_fall_back_to_default_token_assumptions(self) -> None:
        """Zero token metadata must not produce a zero-cost approval estimate."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(
            enforcer,
            max_trials=10,
            max_total_examples=None,
            candidate_models=["gpt-4o"],
            estimated_input_tokens_per_example=0,
            estimated_output_tokens_per_example=0,
        )
        dataset = _FakeDataset()

        with patch(
            "traigent.core.cost_estimator.get_model_token_pricing",
            return_value=(2.5e-6, 10.0e-6, "litellm"),
        ):
            cost = estimator.estimate_optimization_cost(dataset)

        # Base falls back to default 2000/500 estimate = 0.01
        # Total = 50 * 10 * 0.01 * 1.2 = 6.0
        assert cost == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# check_cost_approval
# ---------------------------------------------------------------------------


class TestCheckCostApproval:
    def test_does_not_skip_in_mock_mode(self) -> None:
        """S2-B Round 3: mock-mode bypass was removed; approval always runs."""
        enforcer = MagicMock()
        # is_mock_mode is no longer consulted; spec it away so attribute access
        # would fail and prove the new code path doesn't read it.
        del enforcer.is_mock_mode
        enforcer.check_and_approve.return_value = True
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=None)
        estimator.check_cost_approval(_FakeDataset())
        enforcer.check_and_approve.assert_called_once()

    def test_passes_when_approved(self) -> None:
        enforcer = MagicMock(is_mock_mode=False)
        enforcer.check_and_approve.return_value = True
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=None)
        estimator.check_cost_approval(_FakeDataset())
        enforcer.check_and_approve.assert_called_once()

    def test_raises_when_declined(self) -> None:
        enforcer = MagicMock(is_mock_mode=False)
        enforcer.check_and_approve.return_value = False
        enforcer.config.limit = 1.0
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=None)

        from traigent.core.cost_enforcement import OptimizationAborted

        with pytest.raises(OptimizationAborted, match="Cost approval declined"):
            estimator.check_cost_approval(_FakeDataset())


# ---------------------------------------------------------------------------
# extract_trial_cost
# ---------------------------------------------------------------------------


class TestExtractTrialCost:
    def test_from_metrics_total_cost(self) -> None:
        result = _FakeTrialResult(metrics={"total_cost": 0.05})
        assert CostEstimator.extract_trial_cost(result) == pytest.approx(0.05)

    def test_from_metrics_cost_key(self) -> None:
        result = _FakeTrialResult(metrics={"cost": 0.03})
        assert CostEstimator.extract_trial_cost(result) == pytest.approx(0.03)

    def test_from_metadata_total_example_cost(self) -> None:
        result = _FakeTrialResult(metrics={}, metadata={"total_example_cost": 0.10})
        assert CostEstimator.extract_trial_cost(result) == pytest.approx(0.10)

    def test_from_metadata_total_cost_key(self) -> None:
        result = _FakeTrialResult(metrics={}, metadata={"total_cost": 0.11})
        assert CostEstimator.extract_trial_cost(result) == pytest.approx(0.11)

    def test_from_metadata_cost_key(self) -> None:
        result = _FakeTrialResult(metrics={}, metadata={"cost": 0.12})
        assert CostEstimator.extract_trial_cost(result) == pytest.approx(0.12)

    def test_returns_none_when_no_cost(self) -> None:
        result = _FakeTrialResult(metrics={}, metadata={})
        assert CostEstimator.extract_trial_cost(result) is None

    def test_returns_none_with_none_metrics(self) -> None:
        result = _FakeTrialResult(metrics=None, metadata=None)
        assert CostEstimator.extract_trial_cost(result) is None

    def test_handles_invalid_cost_type(self) -> None:
        result = _FakeTrialResult(metrics={"total_cost": "not-a-number"})
        assert CostEstimator.extract_trial_cost(result) is None

    def test_handles_invalid_metadata_cost_type(self) -> None:
        result = _FakeTrialResult(metrics={}, metadata={"total_example_cost": "bad"})
        assert CostEstimator.extract_trial_cost(result) is None
