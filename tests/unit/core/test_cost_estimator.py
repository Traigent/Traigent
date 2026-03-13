"""Tests for CostEstimator (traigent.core.cost_estimator)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.cost_estimator import CostEstimator

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

    def test_unknown_model_falls_back_to_conservative_pricing(self) -> None:
        """Unknown model should trigger conservative fallback pricing."""
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

    def test_unknown_candidate_model_falls_back_to_conservative_pricing(self) -> None:
        """Unknown candidate pricing should keep approval conservative."""
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
            cost = estimator.estimate_optimization_cost(dataset)

        assert cost == pytest.approx(40.5)

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

    def test_estimated_tokens_apply_to_conservative_fallback_pricing(self) -> None:
        """Token estimate metadata should still improve the conservative fallback."""
        from traigent.utils.cost_calculator import UnknownModelError

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
    def test_skips_in_mock_mode(self) -> None:
        enforcer = MagicMock(is_mock_mode=True)
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=None)
        # Should not raise
        estimator.check_cost_approval(_FakeDataset())
        enforcer.check_and_approve.assert_not_called()

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
