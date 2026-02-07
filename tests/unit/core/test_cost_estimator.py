"""Tests for CostEstimator (traigent.core.cost_estimator)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

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
        # 50 examples * 10 trials * 0.01 * 1.2 = 6.0
        assert cost == pytest.approx(6.0)

    def test_per_trial_mode_explicit_trials(self) -> None:
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=None)
        dataset = _FakeDataset()
        cost = estimator.estimate_optimization_cost(dataset)
        # 50 * 5 * 0.01 * 1.2 = 3.0
        assert cost == pytest.approx(3.0)

    def test_total_examples_budget_mode(self) -> None:
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=5, max_total_examples=200)
        dataset = _FakeDataset()
        cost = estimator.estimate_optimization_cost(dataset)
        # 200 * 0.01 * 1.2 = 2.4
        assert cost == pytest.approx(2.4)

    def test_dataset_without_len(self) -> None:
        """When dataset has no __len__, falls back to 100."""
        enforcer = MagicMock(is_mock_mode=False)
        estimator = CostEstimator(enforcer, max_trials=2, max_total_examples=None)
        dataset = MagicMock(spec=[])  # No __len__
        cost = estimator.estimate_optimization_cost(dataset)
        # 100 * 2 * 0.01 * 1.2 = 2.4
        assert cost == pytest.approx(2.4)


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
