"""Tests for CostEstimator (traigent.core.cost_estimator)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.core.cost_estimator import CostEstimator, _extract_text

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


@dataclass
class _FakeEvaluationExample:
    """Mimics EvaluationExample with input_data + expected_output."""

    input_data: dict[str, Any] = field(
        default_factory=lambda: {"prompt": "Hello world"}
    )
    expected_output: str | None = "Expected response text here"


class _IndexableDataset:
    """Dataset with both __len__ and __getitem__."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]


class _NonIndexableDataset:
    """Dataset with __len__ but no __getitem__."""

    def __init__(self, n: int = 50) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# estimate_optimization_cost (hardcoded fallback — backward compat)
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


# ---------------------------------------------------------------------------
# _extract_models_from_config_space
# ---------------------------------------------------------------------------


class TestExtractModelsFromConfigSpace:
    """Config-space model extraction for ALL value forms."""

    def _make(self, config_space: dict) -> CostEstimator:
        return CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=5,
            max_total_examples=None,
            configuration_space=config_space,
        )

    def test_list_form(self) -> None:
        est = self._make({"model": ["gpt-4o", "gpt-3.5-turbo"]})
        assert est._extract_models_from_config_space() == ["gpt-4o", "gpt-3.5-turbo"]

    def test_string_form(self) -> None:
        est = self._make({"model": "gpt-4o"})
        assert est._extract_models_from_config_space() == ["gpt-4o"]

    def test_tuple_form(self) -> None:
        est = self._make({"model": ("gpt-4o", "gpt-3.5-turbo")})
        assert est._extract_models_from_config_space() == ["gpt-4o", "gpt-3.5-turbo"]

    def test_dict_choices_form(self) -> None:
        est = self._make({"model": {"choices": ["gpt-4o", "gpt-3.5-turbo"]}})
        assert est._extract_models_from_config_space() == ["gpt-4o", "gpt-3.5-turbo"]

    def test_dict_values_form(self) -> None:
        est = self._make({"model": {"values": ["gpt-4o"]}})
        assert est._extract_models_from_config_space() == ["gpt-4o"]

    def test_dict_value_form(self) -> None:
        est = self._make({"model": {"value": "gpt-4-turbo"}})
        assert est._extract_models_from_config_space() == ["gpt-4-turbo"]

    def test_no_model_key(self) -> None:
        """No model key → empty list (mid-tier default pricing used upstream)."""
        est = self._make({"temperature": [0.1, 0.9]})
        assert est._extract_models_from_config_space() == []

    def test_filters_non_strings(self) -> None:
        """Non-string items (numeric, bool) are filtered out."""
        est = self._make({"model": ["gpt-4o", 0.5, True, "gpt-3.5-turbo"]})
        assert est._extract_models_from_config_space() == ["gpt-4o", "gpt-3.5-turbo"]

    def test_model_name_key(self) -> None:
        est = self._make({"model_name": ["claude-3-sonnet"]})
        assert est._extract_models_from_config_space() == ["claude-3-sonnet"]

    def test_llm_model_key(self) -> None:
        est = self._make({"llm_model": "gpt-4o"})
        assert est._extract_models_from_config_space() == ["gpt-4o"]

    def test_llm_key(self) -> None:
        est = self._make({"llm": ("gpt-4o",)})
        assert est._extract_models_from_config_space() == ["gpt-4o"]

    def test_empty_config_space(self) -> None:
        est = self._make({})
        assert est._extract_models_from_config_space() == []

    def test_none_config_space(self) -> None:
        est = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=5,
            max_total_examples=None,
            configuration_space=None,
        )
        assert est._extract_models_from_config_space() == []


# ---------------------------------------------------------------------------
# _estimate_tokens_per_example
# ---------------------------------------------------------------------------


class TestEstimateTokensPerExample:
    def test_indexable_with_evaluation_examples(self) -> None:
        """EvaluationExample-like objects → extracts input_data + expected_output."""
        examples = [_FakeEvaluationExample() for _ in range(10)]
        dataset = _IndexableDataset(examples)
        inp, out = CostEstimator._estimate_tokens_per_example(dataset)
        assert inp >= 500
        assert out >= 250

    def test_indexable_with_dicts(self) -> None:
        """Plain dicts → falls back to str(example)."""
        examples = [{"prompt": "x" * 4000, "response": "y" * 2000} for _ in range(5)]
        dataset = _IndexableDataset(examples)
        inp, out = CostEstimator._estimate_tokens_per_example(dataset)
        # ~6000 chars -> ~1500 tokens (larger than floor)
        assert inp >= 500
        assert out >= 250

    def test_non_indexable_uses_defaults(self) -> None:
        """Non-indexable dataset → uses defaults, does NOT consume."""
        dataset = _NonIndexableDataset(50)
        inp, out = CostEstimator._estimate_tokens_per_example(dataset)
        assert inp == 2000
        assert out == 1000

    def test_no_len_uses_defaults(self) -> None:
        """Dataset without __len__ → uses defaults."""
        dataset = MagicMock(spec=[])  # No __len__ or __getitem__
        inp, out = CostEstimator._estimate_tokens_per_example(dataset)
        assert inp == 2000
        assert out == 1000

    def test_empty_dataset(self) -> None:
        dataset = _IndexableDataset([])
        inp, out = CostEstimator._estimate_tokens_per_example(dataset)
        assert inp == 2000
        assert out == 1000

    def test_string_examples(self) -> None:
        examples = ["a" * 8000 for _ in range(3)]
        dataset = _IndexableDataset(examples)
        inp, out = CostEstimator._estimate_tokens_per_example(dataset)
        assert inp == 2000  # 8000 / 4 = 2000
        assert out >= 250


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_evaluation_example_with_output(self) -> None:
        ex = _FakeEvaluationExample(input_data={"q": "What?"}, expected_output="Answer")
        text = _extract_text(ex)
        assert "What?" in text
        assert "Answer" in text

    def test_evaluation_example_without_output(self) -> None:
        ex = _FakeEvaluationExample(input_data={"q": "Hello"}, expected_output=None)
        text = _extract_text(ex)
        assert "Hello" in text

    def test_string_example(self) -> None:
        assert _extract_text("raw text") == "raw text"

    def test_dict_example(self) -> None:
        text = _extract_text({"key": "value"})
        assert "key" in text
        assert "value" in text

    def test_other_object(self) -> None:
        text = _extract_text(42)
        assert text == "42"


# ---------------------------------------------------------------------------
# Model-aware estimate_optimization_cost
# ---------------------------------------------------------------------------


class TestModelAwareCostEstimation:
    def test_model_aware_differs_from_hardcoded(self) -> None:
        """Config-space-aware estimate differs from the hardcoded $0.01 fallback."""
        enforcer = MagicMock(is_mock_mode=False)
        # With config space
        est_aware = CostEstimator(
            enforcer,
            max_trials=5,
            max_total_examples=None,
            configuration_space={"model": ["gpt-4o"]},
        )
        # Without config space (hardcoded fallback)
        est_hardcoded = CostEstimator(
            enforcer,
            max_trials=5,
            max_total_examples=None,
        )
        dataset = _FakeDataset()
        cost_aware = est_aware.estimate_optimization_cost(dataset)
        cost_hardcoded = est_hardcoded.estimate_optimization_cost(dataset)
        # They should differ (model-aware uses per-token pricing)
        assert cost_aware != cost_hardcoded

    def test_no_model_key_uses_mid_tier(self) -> None:
        """Config space without model key → mid-tier default."""
        enforcer = MagicMock(is_mock_mode=False)
        est = CostEstimator(
            enforcer,
            max_trials=5,
            max_total_examples=None,
            configuration_space={"temperature": [0.1, 0.5, 0.9]},
        )
        dataset = _FakeDataset()
        cost = est.estimate_optimization_cost(dataset)
        # Should use mid-tier pricing, not hardcoded $0.01
        assert cost > 0

    def test_invocations_multiplier(self) -> None:
        """invocations_per_example=3 produces 3x the cost of 1."""
        enforcer = MagicMock(is_mock_mode=False)
        est_1 = CostEstimator(
            enforcer,
            max_trials=5,
            max_total_examples=None,
            configuration_space={"model": ["gpt-4o"]},
            invocations_per_example=1,
        )
        est_3 = CostEstimator(
            enforcer,
            max_trials=5,
            max_total_examples=None,
            configuration_space={"model": ["gpt-4o"]},
            invocations_per_example=3,
        )
        dataset = _FakeDataset()
        cost_1 = est_1.estimate_optimization_cost(dataset)
        cost_3 = est_3.estimate_optimization_cost(dataset)
        assert cost_3 == pytest.approx(cost_1 * 3)


# ---------------------------------------------------------------------------
# invocations_per_example clamping
# ---------------------------------------------------------------------------


class TestInvocationsPerExampleClamping:
    def test_zero_clamps_to_1(self) -> None:
        est = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=5,
            max_total_examples=None,
            invocations_per_example=0,
        )
        assert est._invocations_per_example == 1

    def test_negative_clamps_to_1(self) -> None:
        est = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=5,
            max_total_examples=None,
            invocations_per_example=-5,
        )
        assert est._invocations_per_example == 1

    def test_positive_passes_through(self) -> None:
        est = CostEstimator(
            MagicMock(is_mock_mode=False),
            max_trials=5,
            max_total_examples=None,
            invocations_per_example=3,
        )
        assert est._invocations_per_example == 3


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_no_config_space_uses_hardcoded(self) -> None:
        """Without configuration_space, old $0.01/example path is used."""
        enforcer = MagicMock(is_mock_mode=False)
        est = CostEstimator(enforcer, max_trials=5, max_total_examples=None)
        dataset = _FakeDataset()
        cost = est.estimate_optimization_cost(dataset)
        # 50 * 5 * 0.01 * 1.2 = 3.0
        assert cost == pytest.approx(3.0)
