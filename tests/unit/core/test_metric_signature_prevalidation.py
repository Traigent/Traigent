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


def test_unbindable_report_names_positional_only_culprit_not_recognized_names() -> None:
    """A required POSITIONAL_ONLY parameter is the real culprit -- name it.

    Follow-up to #1780 (post-merge review, P2): ``resolve_metric_call_binding``
    used to report every visible POSITIONAL_OR_KEYWORD/KEYWORD_ONLY name when
    nothing bound (``unmatched_parameters = bindable_names``), which silently
    excludes required POSITIONAL_ONLY parameters. For
    ``def scorer(x, /, output, expected, *, llm_metrics)`` binding fails (4
    required, 3 positional values), every *visible* name is recognized, so
    the old fallback named ``expected, llm_metrics, output`` -- the exact
    parameters the module's own comment (local.py) says must never be named --
    while never mentioning ``x``, the actual reason nothing could bind.
    """

    def scorer(x: str, /, output: str, expected: str, *, llm_metrics: dict) -> float:
        raise AssertionError("scorer must never be invoked by this test")

    with pytest.raises(ValidationError) as excinfo:
        validate_metric_function_bindability(
            {"accuracy": scorer}, objectives=["accuracy"]
        )

    message = str(excinfo.value)
    cannot_bind = message.split("cannot be bound:")[1].split(".")[0]
    assert "x" in [name.strip() for name in cannot_bind.split(",")]
    assert "output" not in cannot_bind
    assert "expected" not in cannot_bind
    assert "llm_metrics" not in cannot_bind


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
    validate_metric_function_bindability({}, objectives=["accuracy"])


def test_extra_required_positional_param_raises_for_objective() -> None:
    """``(output, expected, threshold)`` must not silently get the usage dict."""

    def accuracy(output: str, expected: str, threshold: float) -> float:
        raise AssertionError("must never be invoked by this test")

    with pytest.raises(ValidationError, match="positional parameter 3") as excinfo:
        validate_metric_function_bindability(
            {"cust_metric_zq": accuracy}, ["cust_metric_zq"]
        )
    # Customer-chosen metric and parameter names are content: never echoed.
    assert "cust_metric_zq" not in str(excinfo.value)
    assert "threshold" not in str(excinfo.value)


def test_extra_required_positional_param_warns_for_non_objective(caplog) -> None:
    def helper(output: str, expected: str, threshold: float) -> float:
        raise AssertionError("must never be invoked by this test")

    with caplog.at_level(logging.WARNING):
        validate_metric_function_bindability({"helper": helper}, ["accuracy"])
    assert "positional parameter 3" in caplog.text
    assert "threshold" not in caplog.text
    assert "helper" not in caplog.text


def test_extra_param_with_default_and_all_positional_names_still_bind() -> None:
    def with_default(output: str, expected: str, threshold: float = 0.5) -> float:
        return 1.0

    def all_positional(a: object, b: object, c: object) -> float:
        return 1.0

    validate_metric_function_bindability(
        {"accuracy": with_default, "other": all_positional}, ["accuracy", "other"]
    )


def test_alias_named_third_param_with_custom_second_is_accepted() -> None:
    """Runtime binds llm_metrics by position; this must not be rejected."""

    def scorer(output: str, gold: str, llm_metrics: dict) -> float:
        return 1.0

    def scorer_kw(output: str, gold: str, llm_metrics: dict, **kwargs: object) -> float:
        return 1.0

    validate_metric_function_bindability(
        {"accuracy": scorer, "other": scorer_kw}, ["accuracy", "other"]
    )


def test_stray_third_param_with_var_kwargs_is_flagged() -> None:
    def scorer(output: str, gold: str, threshold: float, **kwargs: object) -> float:
        return 1.0

    with pytest.raises(ValidationError, match="positional parameter 3"):
        validate_metric_function_bindability({"accuracy": scorer}, ["accuracy"])


def test_positional_only_defaulted_usage_param_is_accepted() -> None:
    def with_default(output, expected, usage=None, /) -> float:  # noqa: ANN001
        return 1.0

    validate_metric_function_bindability({"accuracy": with_default}, ["accuracy"])


def test_positional_only_required_usage_param_is_accepted_or_flagged_consistently() -> (
    None
):
    """Required positional-only third param binds positional_3; with a
    recognized name present it is flagged because it is not an llm alias."""

    def required(output, expected, usage, /) -> float:  # noqa: ANN001
        return 1.0

    # Names output/expected are recognized, `usage` is not an alias: flagged.
    with pytest.raises(ValidationError, match="positional parameter 3"):
        validate_metric_function_bindability({"accuracy": required}, ["accuracy"])


def test_collector_named_like_recognized_param_does_not_count() -> None:
    """A **metadata / *config collector must not make (a, b, c) 'mixed'."""

    def kw_collector(a, b, c, **metadata) -> float:  # noqa: ANN001
        return float(a == b and c["total_tokens"] == 2)

    def star_collector(a, b, c, *config) -> float:  # noqa: ANN001
        return 1.0

    validate_metric_function_bindability(
        {"accuracy": kw_collector, "other": star_collector}, ["accuracy", "other"]
    )


def _s_stray(output, expected, threshold):  # noqa: ANN001
    return 1.0


def _s_alias3(output, gold, llm_metrics):  # noqa: ANN001
    return 1.0


def _s_alias3_kw(output, gold, llm_metrics, **kw):  # noqa: ANN001
    return 1.0


def _s_posonly_default(output, expected, usage=None, /):  # noqa: ANN001
    return 1.0


def _s_posonly_required(output, expected, usage, /):  # noqa: ANN001
    return 1.0


def _s_all_positional(a, b, c):  # noqa: ANN001
    return 1.0


def _s_collector_kw(a, b, c, **metadata):  # noqa: ANN001
    return 1.0


def _s_collector_star(a, b, c, *config):  # noqa: ANN001
    return 1.0


def _s_kwonly_optional(a, b, c, *, config=None):  # noqa: ANN001
    return 1.0


def _s_kwargs_only(**kwargs):  # noqa: ANN003
    return 1.0


def _s_recognized_names(output, expected):  # noqa: ANN001
    return 1.0


def _s_recognized_with_metrics(output, expected, metrics):  # noqa: ANN001
    return 1.0


def _s_stray_with_kwargs(output, expected, threshold, **kw):  # noqa: ANN001
    return 1.0


def _s_stray_default(output, expected, threshold=0.5):  # noqa: ANN001
    return 1.0


def _s_stray_with_optional_recognized(output, expected, threshold, *, config=None):  # noqa: ANN001
    return 1.0


@pytest.mark.parametrize(
    ("scorer", "rejected"),
    [
        (_s_stray, True),
        (_s_stray_with_kwargs, True),
        (_s_posonly_required, True),
        (_s_stray_with_optional_recognized, True),
        (_s_alias3, False),
        (_s_alias3_kw, False),
        (_s_posonly_default, False),
        (_s_all_positional, False),
        (_s_collector_kw, False),
        (_s_collector_star, False),
        (_s_kwonly_optional, False),
        (_s_kwargs_only, False),
        (_s_recognized_names, False),
        (_s_recognized_with_metrics, False),
        (_s_stray_default, False),
    ],
)
def test_prevalidation_rejects_iff_usage_dict_feeds_required_stray(
    scorer, rejected
) -> None:
    """Rejection tracks the real runtime binder, not a re-derived heuristic."""
    if rejected:
        with pytest.raises(ValidationError, match="positional parameter"):
            validate_metric_function_bindability({"accuracy": scorer}, ["accuracy"])
    else:
        validate_metric_function_bindability({"accuracy": scorer}, ["accuracy"])
