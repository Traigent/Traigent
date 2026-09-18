"""Pre-run signature validation for metric_functions / scoring_function.

Issue #1780: a metric/scoring function whose required parameters cannot be
bound was only discovered lazily, per-example, at scoring time -- after the
LLM call that produced ``output`` had already spent money. These tests prove
the failure now surfaces at evaluator construction (run start), before any
LLM call is issued, while every previously-valid signature shape still binds
unchanged.
"""

from __future__ import annotations

import logging

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


def test_unbindable_objective_metric_raises_with_metric_name() -> None:
    def unbindable(output: str, *, foobar: str) -> float:
        raise AssertionError("unbindable must never be invoked by this test")

    with pytest.raises(ValidationError, match="'my_metric'"):
        create_effective_evaluator(
            **_make_common_kwargs(
                metric_functions={"my_metric": unbindable},
                objectives=["my_metric"],
            ),
        )


def test_unbindable_informational_metric_warns_and_still_constructs(caplog) -> None:
    """A non-objective metric must NOT block the run.

    ``LocalEvaluator`` raises only for an objective and refuses to substitute a
    fabricated 0.0 for it (``traigent/evaluators/local.py:851``); an auxiliary
    metric degrades to 0.0 with a ``metric_errors`` record. The no-execution
    contract inspector mirrors that split deliberately
    (``traigent/contract/evaluation.py:930-936``). Hard-failing here would
    refuse runs that complete today, and would be the only place in the
    codebase treating the two alike.
    """

    def unbindable(output: str, *, retrieval_context: str) -> float:
        raise AssertionError("unbindable must never be invoked by this test")

    def ok(output: str, expected: str) -> float:
        return 1.0

    with caplog.at_level(logging.WARNING):
        evaluator, _ = create_effective_evaluator(
            **_make_common_kwargs(
                metric_functions={"ctx_check": unbindable},
                scoring_function=ok,
                objectives=["accuracy"],
            ),
        )

    assert isinstance(evaluator, LocalEvaluator)
    assert "ctx_check" in caplog.text
    assert "not an optimization objective" in caplog.text


def test_unbindable_report_omits_parameters_the_runtime_can_supply() -> None:
    """The message must name only what the user has to fix.

    ``MetricBinding.unmatched_parameters`` is every bindable name when nothing
    bound, so it includes recognized ones like ``output``. Reporting those sends
    the user after a parameter that is already correct.
    """

    def unbindable(output: str, *, foobar: str) -> float:
        return 0.0

    with pytest.raises(ValidationError) as excinfo:
        validate_metric_function_bindability(
            {"accuracy": unbindable}, objectives=["accuracy"]
        )

    message = str(excinfo.value)
    cannot_bind = message.split("cannot be bound:")[1].split(".")[0]
    assert "foobar" in cannot_bind
    assert "output" not in cannot_bind


def test_validate_metric_function_bindability_direct_unit() -> None:
    """Unit-level check of the validator without the full evaluator wiring."""

    def unbindable(output: str, *, foobar: str) -> float:
        return 0.0

    with pytest.raises(ValidationError, match="foobar"):
        validate_metric_function_bindability(
            {"accuracy": unbindable}, objectives=["accuracy"]
        )


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
