"""Pre-run signature validation for metric_functions / scoring_function.

Issue #1780: a metric/scoring function whose required parameters cannot be
bound was only discovered lazily, per-example, at scoring time -- after the
LLM call that produced ``output`` had already spent money. These tests prove
the failure now surfaces at evaluator construction (run start), before any
LLM call is issued, while every previously-valid signature shape still binds
unchanged.
"""

from __future__ import annotations

import pytest

from traigent.core.optimization_pipeline import (
    create_effective_evaluator,
    validate_metric_function_bindability,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import ValidationError


def _make_common_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "timeout": None,
        "custom_evaluator": None,
        "effective_batch_size": None,
        "effective_thread_workers": None,
        "effective_privacy_enabled": False,
        "objectives": ["accuracy"],
        "execution_mode": "local",
        "mock_mode_config": None,
        "metric_functions": None,
        "scoring_function": None,
        "decorator_custom_evaluator": None,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Unbindable signatures: must raise pre-run, before any LLM call.
# ---------------------------------------------------------------------------


def test_unbindable_required_keyword_only_param_raises_before_construction() -> None:
    """A required, unrecognized keyword-only parameter cannot bind -- raise now."""

    def bad_scorer(output: str, *, retrieval_context: str) -> float:
        raise AssertionError("bad_scorer must never be invoked by this test")

    with pytest.raises(ValidationError, match="retrieval_context") as excinfo:
        create_effective_evaluator(
            **_make_common_kwargs(scoring_function=bad_scorer),
        )
    # Error names the unbindable parameter and points at the recognized set.
    assert "cannot be bound" in str(excinfo.value)
    assert "output" in str(excinfo.value)  # recognized-name set mentioned


def test_unbindable_metric_functions_entry_raises_with_metric_name() -> None:
    def unbindable(output: str, *, foobar: str) -> float:
        raise AssertionError("unbindable must never be invoked by this test")

    with pytest.raises(ValidationError, match="'my_metric'"):
        create_effective_evaluator(
            **_make_common_kwargs(metric_functions={"my_metric": unbindable}),
        )


def test_validate_metric_function_bindability_direct_unit() -> None:
    """Unit-level check of the validator without the full evaluator wiring."""

    def unbindable(output: str, *, foobar: str) -> float:
        return 0.0

    with pytest.raises(ValidationError, match="foobar"):
        validate_metric_function_bindability({"accuracy": unbindable})


# ---------------------------------------------------------------------------
# Valid production signatures: must pass unchanged.
# ---------------------------------------------------------------------------


def test_deepeval_style_names_pass() -> None:
    def deepeval_scorer(
        actual_output: str, expected_output: str, ground_truth: str
    ) -> float:
        return 1.0

    evaluator, _ = create_effective_evaluator(
        **_make_common_kwargs(scoring_function=deepeval_scorer),
    )
    assert isinstance(evaluator, LocalEvaluator)


def test_two_positional_form_passes() -> None:
    def scorer(output: str, expected: str) -> float:
        return 1.0

    evaluator, _ = create_effective_evaluator(
        **_make_common_kwargs(scoring_function=scorer),
    )
    assert isinstance(evaluator, LocalEvaluator)


def test_kwargs_scorer_passes() -> None:
    def scorer(**kwargs: object) -> float:
        return 1.0

    evaluator, _ = create_effective_evaluator(
        **_make_common_kwargs(scoring_function=scorer),
    )
    assert isinstance(evaluator, LocalEvaluator)


def test_no_metric_functions_is_a_noop() -> None:
    """No metric_functions/scoring_function configured -> nothing to validate."""
    validate_metric_function_bindability({})
