"""Pre-run estimate samples the dataset and shows its assumptions (#2498)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.core.cost_estimator import CostEstimator
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import CostLimitExceeded


def _dataset(texts: list[str]) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"q": t}, expected_output="x") for t in texts
        ]
    )


def _estimator(**kw) -> CostEstimator:
    enforcer = MagicMock(is_mock_mode=False)
    return CostEstimator(
        enforcer, max_trials=1, max_total_examples=None, model_name="gpt-4o", **kw
    )


def _tokens(est: CostEstimator, ds: Dataset) -> tuple[int, int]:
    est.estimate_optimization_cost(ds)
    return est._get_estimated_tokens_per_example()


def test_short_classifier_inputs_go_below_old_2000_default() -> None:
    """The motivating #2498 case: 30-char examples are not inflated to 2000."""
    ds = _dataset(["x" * 30] * 5)
    est = _estimator()
    inp, _ = _tokens(est, ds)
    assert 208 <= inp <= 256
    assert inp < 2000
    # A budget covering the small estimate is not refused.
    assert est.estimate_optimization_cost(ds) < 0.05


def test_refusal_not_triggered_for_budget_covering_short_inputs() -> None:
    ds = _dataset(["x" * 30] * 5)
    est = CostEstimator(
        MagicMock(is_mock_mode=False),
        max_trials=1,
        max_total_examples=None,
        model_name="gpt-4o",
    )
    cost = est.estimate_optimization_cost(ds)
    old_cost = 5 * (2000 * 2.5e-6 + 500 * 10e-6) * 1.2
    budget = 0.04  # covers the sampled estimate, below the old 2000-token one
    assert cost < budget < old_cost


def test_tiny_inputs_use_minimum() -> None:
    assert _tokens(_estimator(), _dataset(["hi"] * 5))[0] == 256


def test_long_inputs_raise_estimate_above_floor() -> None:
    inp, _ = _tokens(_estimator(), _dataset(["word " * 4000] * 5))
    assert inp > 2000


def test_non_english_counts_more_than_ascii_of_same_length() -> None:
    ascii_in, _ = _tokens(_estimator(), _dataset(["a" * 20000] * 3))
    cjk_in, _ = _tokens(_estimator(), _dataset(["数" * 20000] * 3))
    assert cjk_in > ascii_in * 2


def test_sample_is_bounded_to_first_20() -> None:
    texts = ["x"] * 20 + ["word " * 100000] * 50
    assert _tokens(_estimator(), _dataset(texts))[0] == 256


def test_backend_override_wins() -> None:
    est = _estimator(
        estimated_input_tokens_per_example=77,
        estimated_output_tokens_per_example=33,
    )
    assert _tokens(est, _dataset(["word " * 4000] * 3)) == (77, 33)


def test_configured_max_tokens_is_output_cap() -> None:
    est = _estimator(configured_max_output_tokens=1024)
    assert _tokens(est, _dataset(["hi"]))[1] == 1024


def test_max_tokens_extraction_from_config_space() -> None:
    f = OptimizationOrchestrator._extract_max_output_tokens_from_config_space
    assert f({"max_tokens": [256, 1024]}) == 1024
    assert f({"max_tokens": {"values": [128, 512]}}) == 512
    assert f({"temperature": [0.1]}) is None


def test_refusal_message_shows_assumptions_without_content() -> None:
    secret = "PATIENT_SECRET_TEXT"
    ds = _dataset([secret + " word" * 5000] * 3)
    enforcer = MagicMock(is_mock_mode=False)
    enforcer.check_and_approve.return_value = False
    enforcer.config.limit = 1.0
    est = CostEstimator(
        enforcer, max_trials=1, max_total_examples=None, model_name="gpt-4o"
    )
    with (
        patch(
            "traigent.core.cost_estimator.cost_preflight_mock_exempt",
            return_value=False,
        ),
        pytest.raises(CostLimitExceeded) as exc,
    ):
        est.check_cost_approval(ds)
    msg = str(exc.value)
    assert "gpt-4o" in msg
    assert "examples in dataset=3" in msg
    assert "input tokens/example=" in msg
    assert "dataset_sample" in msg
    assert "output tokens/example=500" in msg
    assert "safety buffer=1.2x" in msg
    assert "Real billing may differ" in msg
    assert "runtime cost limit still applies" in msg
    assert secret not in msg
