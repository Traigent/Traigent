"""Unit tests for the surrogate (pre-screen) evaluator.

The surrogate is a cheap SECOND scorer over the outputs the primary evaluator
already captured. These tests pin the fixed cross-repo contract (per-example
``surrogate_score``, aggregate mean, and the trial-metadata descriptor), the
fail-soft behaviour, and the hard invariant that the surrogate NEVER re-executes
the decorated function.
"""

from __future__ import annotations

import copy
import re

import pytest

from traigent.api.decorators import (
    EvaluationOptions,
    _validate_surrogate_evaluator_signature,
)
from traigent.api.types import ExampleResult
from traigent.core.optimization_pipeline import (
    attach_surrogate_evaluator,
    build_surrogate_descriptor,
    create_effective_evaluator,
    get_surrogate_evaluator,
    get_surrogate_evaluator_name,
    resolve_surrogate_evaluator,
    resolve_surrogate_evaluator_name,
    surrogate_evaluator_id,
)
from traigent.core.trial_lifecycle import (
    TrialLifecycle,
    _coerce_surrogate_score,
    apply_surrogate_scoring,
)
from traigent.evaluators.base import EvaluationResult
from traigent.utils.exceptions import ValidationError

FP_RE = re.compile(r"^fp1:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _example(idx: int, output, expected="gold", metrics=None) -> ExampleResult:
    return ExampleResult(
        example_id=f"example_{idx}",
        input_data={"q": f"q_{idx}"},
        expected_output=expected,
        actual_output=output,
        metrics=dict(metrics or {"accuracy": 1.0}),
        execution_time=0.01,
        success=True,
    )


def _eval_result(example_results, agg=None) -> EvaluationResult:
    aggregated = dict(agg or {"accuracy": 1.0})
    return EvaluationResult(
        config={},
        example_results=list(example_results),
        aggregated_metrics=aggregated,
        total_examples=len(example_results),
        successful_examples=len(example_results),
        metrics=dict(aggregated),
    )


class _Orchestrator:
    def __init__(self, evaluator):
        self.evaluator = evaluator


class _Evaluator:
    """Minimal evaluator stand-in that accepts an attached surrogate."""


def half_scorer(output, expected_output=None, example=None):
    """A trivial deterministic surrogate."""
    return 0.5


def boom_scorer(output, expected_output=None, example=None):
    """A module-level surrogate whose source is fingerprintable but which raises."""
    raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# resolve / attach / get
# ---------------------------------------------------------------------------


def test_resolve_prefers_optimize_arg_over_decorator():
    call_arg = lambda output: 0.1  # noqa: E731
    decorator_arg = lambda output: 0.2  # noqa: E731
    assert (
        resolve_surrogate_evaluator(
            call_arg, decorator_surrogate_evaluator=decorator_arg
        )
        is call_arg
    )
    assert (
        resolve_surrogate_evaluator(None, decorator_surrogate_evaluator=decorator_arg)
        is decorator_arg
    )
    assert resolve_surrogate_evaluator(None, decorator_surrogate_evaluator=None) is None


def test_attach_and_get_roundtrip():
    ev = _Evaluator()
    assert get_surrogate_evaluator(ev) is None
    attach_surrogate_evaluator(ev, half_scorer)
    assert get_surrogate_evaluator(ev) is half_scorer


def test_attach_none_is_noop():
    ev = _Evaluator()
    attach_surrogate_evaluator(ev, None)
    assert get_surrogate_evaluator(ev) is None


# ---------------------------------------------------------------------------
# descriptor + fingerprint
# ---------------------------------------------------------------------------


def test_descriptor_matches_contract_shape():
    desc = build_surrogate_descriptor(half_scorer)
    assert desc["evaluator_id"] == "half_scorer"
    assert desc["metric_name"] == "surrogate_score"
    assert desc["judge_model"] is None
    assert desc["prompt"] is None
    assert set(desc.keys()) == {
        "evaluator_id",
        "metric_name",
        "judge_model",
        "prompt",
        "config",
    }
    assert FP_RE.match(desc["config"]["fingerprint_source"])


def test_evaluator_id_derivation():
    assert surrogate_evaluator_id(half_scorer) == "half_scorer"
    assert surrogate_evaluator_id(lambda output: 0.0) == "surrogate"

    class MyScorer:
        def __call__(self, output, expected_output=None, example=None):
            return 1.0

    assert surrogate_evaluator_id(MyScorer()) == "MyScorer"


def test_fingerprint_is_stable_and_content_free():
    a = build_surrogate_descriptor(half_scorer)["config"]["fingerprint_source"]
    b = build_surrogate_descriptor(half_scorer)["config"]["fingerprint_source"]
    assert a == b and FP_RE.match(a)


def test_descriptor_uses_explicit_name_when_provided():
    # An explicit name overrides the callable-derived id.
    named = build_surrogate_descriptor(half_scorer, name="my_prescreen")
    assert named["evaluator_id"] == "my_prescreen"
    # Omitting the name falls back to the callable derivation.
    unnamed = build_surrogate_descriptor(half_scorer, name=None)
    assert unnamed["evaluator_id"] == "half_scorer"
    # A blank name is ignored (falls back to derivation).
    blank = build_surrogate_descriptor(half_scorer, name="")
    assert blank["evaluator_id"] == "half_scorer"


def test_resolve_name_prefers_optimize_arg_over_decorator():
    assert (
        resolve_surrogate_evaluator_name(
            "call", decorator_surrogate_evaluator_name="dec"
        )
        == "call"
    )
    assert (
        resolve_surrogate_evaluator_name(None, decorator_surrogate_evaluator_name="dec")
        == "dec"
    )
    assert (
        resolve_surrogate_evaluator_name(None, decorator_surrogate_evaluator_name=None)
        is None
    )


def test_attach_and_get_name_roundtrip():
    ev = _Evaluator()
    assert get_surrogate_evaluator_name(ev) is None
    attach_surrogate_evaluator(ev, half_scorer, name="prescreen_a")
    assert get_surrogate_evaluator(ev) is half_scorer
    assert get_surrogate_evaluator_name(ev) == "prescreen_a"


# ---------------------------------------------------------------------------
# per-example + aggregate scoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scores_injected_per_example_and_aggregate():
    result = _eval_result([_example(0, "out0"), _example(1, "out1")])

    def scorer(output, expected_output=None, example=None):
        return 0.25 if output == "out0" else 0.75

    await apply_surrogate_scoring(result, scorer, "trial-1")

    assert result.example_results[0].metrics["surrogate_score"] == 0.25
    assert result.example_results[1].metrics["surrogate_score"] == 0.75
    # Aggregate = mean over scored examples.
    assert result.metrics["surrogate_score"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_out_of_range_scores_are_dropped_not_clipped():
    # The backend evaluator-tensor reader REJECTS out-of-range surrogate scores;
    # the SDK must NOT pre-clip (that would launder garbage past that guard). An
    # out-of-range value drops the surrogate field for that example (fail-soft).
    result = _eval_result([_example(0, "hi"), _example(1, "lo"), _example(2, "ok")])

    def scorer(output, expected_output=None, example=None):
        if output == "hi":
            return 5.0  # > 1 -> dropped
        if output == "lo":
            return -3.0  # < 0 -> dropped
        return 0.5  # in range -> kept

    await apply_surrogate_scoring(result, scorer, "trial-oor")
    assert "surrogate_score" not in result.example_results[0].metrics
    assert "surrogate_score" not in result.example_results[1].metrics
    assert result.example_results[2].metrics["surrogate_score"] == 0.5
    # Aggregate is the mean over ONLY the in-range score.
    assert result.metrics["surrogate_score"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_per_example_error_drops_only_that_example():
    result = _eval_result(
        [_example(0, "good"), _example(1, "boom"), _example(2, "good")]
    )

    def scorer(output, expected_output=None, example=None):
        if output == "boom":
            raise RuntimeError("scorer blew up")
        return 0.4

    await apply_surrogate_scoring(result, scorer, "trial-err")

    assert result.example_results[0].metrics["surrogate_score"] == 0.4
    assert "surrogate_score" not in result.example_results[1].metrics
    assert result.example_results[2].metrics["surrogate_score"] == 0.4
    # Aggregate is mean over the two that scored.
    assert result.metrics["surrogate_score"] == pytest.approx(0.4)
    # Primary metric untouched.
    assert result.example_results[1].metrics["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_failed_example_without_output_is_skipped():
    ok = _example(0, "out")
    failed = _example(1, None)  # actual_output None -> nothing to score
    result = _eval_result([ok, failed])

    await apply_surrogate_scoring(result, half_scorer, "trial-fail")
    assert result.example_results[0].metrics["surrogate_score"] == 0.5
    assert "surrogate_score" not in result.example_results[1].metrics


@pytest.mark.asyncio
async def test_no_scored_examples_leaves_no_aggregate():
    result = _eval_result([_example(0, "x")])

    def always_error(output, expected_output=None, example=None):
        raise ValueError("nope")

    await apply_surrogate_scoring(result, always_error, "trial-none")
    assert "surrogate_score" not in result.metrics
    assert "surrogate_score" not in result.example_results[0].metrics


@pytest.mark.asyncio
async def test_missing_example_results_skips_with_no_crash(caplog):
    result = _eval_result([])
    result.example_results = []
    await apply_surrogate_scoring(result, half_scorer, "trial-empty")
    assert "surrogate_score" not in result.metrics


@pytest.mark.asyncio
async def test_surrogate_never_receives_the_user_function():
    """Hard invariant: the surrogate scores outputs, never re-executes func."""
    seen_args = []

    def recording_scorer(output, expected_output=None, example=None):
        seen_args.append((output, expected_output, example))
        return 0.6

    result = _eval_result([_example(0, "captured-output", expected="gold")])
    await apply_surrogate_scoring(result, recording_scorer, "trial-noexec")

    assert len(seen_args) == 1
    output, expected, example = seen_args[0]
    assert output == "captured-output"
    assert expected == "gold"
    assert isinstance(example, ExampleResult)
    # None of the arguments is a callable — the surrogate cannot re-run anything.
    assert not any(callable(a) for a in seen_args[0])


@pytest.mark.asyncio
async def test_mock_mode_canned_string_outputs_score_fine():
    """Mock mode emits canned constant strings; the surrogate still scores them."""
    canned = "CANNED_MOCK_OUTPUT"
    result = _eval_result([_example(0, canned), _example(1, canned)])

    def length_scorer(output, expected_output=None, example=None):
        return min(1.0, len(output) / 100.0)

    await apply_surrogate_scoring(result, length_scorer, "trial-mock")
    for ex in result.example_results:
        assert ex.metrics["surrogate_score"] == pytest.approx(len(canned) / 100.0)


@pytest.mark.asyncio
async def test_async_surrogate_supported():
    result = _eval_result([_example(0, "a")])

    async def async_scorer(output, expected_output=None, example=None):
        return 0.9

    await apply_surrogate_scoring(result, async_scorer, "trial-async")
    assert result.example_results[0].metrics["surrogate_score"] == 0.9


@pytest.mark.asyncio
async def test_dict_return_coerced_via_known_keys():
    result = _eval_result([_example(0, "a"), _example(1, "b")])

    def dict_scorer(output, expected_output=None, example=None):
        return {"surrogate_score": 0.3, "other": 99}

    await apply_surrogate_scoring(result, dict_scorer, "trial-dict")
    assert result.example_results[0].metrics["surrogate_score"] == 0.3


def test_coerce_rejects_bool_nonfinite_and_out_of_range():
    assert _coerce_surrogate_score(True) is None
    assert _coerce_surrogate_score(float("nan")) is None
    assert _coerce_surrogate_score(float("inf")) is None
    assert _coerce_surrogate_score("not a number") is None
    assert _coerce_surrogate_score({"score": 0.7}) == 0.7
    assert _coerce_surrogate_score(0.42) == 0.42
    # Out-of-range values are DROPPED (None), never clipped.
    assert _coerce_surrogate_score(2.0) is None
    assert _coerce_surrogate_score(-0.5) is None
    assert _coerce_surrogate_score(1.000001) is None
    # Exact interval bounds are kept.
    assert _coerce_surrogate_score(0.0) == 0.0
    assert _coerce_surrogate_score(1.0) == 1.0


# ---------------------------------------------------------------------------
# TrialLifecycle._score_surrogate seam + byte-identical no-surrogate path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_surrogate_returns_none_and_is_byte_identical_when_unconfigured():
    evaluator = _Evaluator()  # no surrogate attached
    lifecycle = TrialLifecycle(_Orchestrator(evaluator))
    result = _eval_result([_example(0, "a"), _example(1, "b")])

    baseline = copy.deepcopy(result.to_dict())
    descriptor = await lifecycle._score_surrogate(result, "trial-x")

    assert descriptor is None
    assert result.to_dict() == baseline  # byte-identical payload


@pytest.mark.asyncio
async def test_score_surrogate_returns_descriptor_and_scores_when_configured():
    evaluator = _Evaluator()
    attach_surrogate_evaluator(evaluator, half_scorer)
    lifecycle = TrialLifecycle(_Orchestrator(evaluator))
    result = _eval_result([_example(0, "a"), _example(1, "b")])

    descriptor = await lifecycle._score_surrogate(result, "trial-y")

    assert descriptor is not None
    assert descriptor["evaluator_id"] == "half_scorer"
    assert result.example_results[0].metrics["surrogate_score"] == 0.5
    assert result.metrics["surrogate_score"] == 0.5


@pytest.mark.asyncio
async def test_score_surrogate_is_fail_soft_on_total_error():
    # ``boom_scorer`` is a module-level function, so its source IS fingerprintable
    # (descriptor survives) — but scoring raises, so the per-example/aggregate
    # scores are dropped fail-soft while the descriptor still rides to the backend.
    evaluator = _Evaluator()
    attach_surrogate_evaluator(evaluator, boom_scorer)
    lifecycle = TrialLifecycle(_Orchestrator(evaluator))
    result = _eval_result([_example(0, "a")])

    descriptor = await lifecycle._score_surrogate(result, "trial-z")
    assert descriptor is not None
    assert descriptor["config"]["fingerprint_source"] is not None
    assert "surrogate_score" not in result.example_results[0].metrics
    assert "surrogate_score" not in result.metrics


@pytest.mark.asyncio
async def test_score_surrogate_dropped_when_fingerprint_unavailable(monkeypatch):
    # Finding 3: a surrogate whose source cannot be fingerprinted is
    # unidentifiable — the descriptor's fingerprint_source would be None,
    # violating ^fp1:[0-9a-f]{64}$. Drop the WHOLE descriptor AND all scores so a
    # scored-but-unidentifiable evaluator can never corrupt the server tensor.
    import traigent.utils.artifact_fingerprints as afp

    monkeypatch.setattr(afp, "compute_surrogate_fingerprint", lambda _s: None)

    evaluator = _Evaluator()
    attach_surrogate_evaluator(evaluator, half_scorer)
    lifecycle = TrialLifecycle(_Orchestrator(evaluator))
    result = _eval_result([_example(0, "a"), _example(1, "b")])

    descriptor = await lifecycle._score_surrogate(result, "trial-nofp")
    assert descriptor is None
    assert "surrogate_score" not in result.example_results[0].metrics
    assert "surrogate_score" not in result.example_results[1].metrics
    assert "surrogate_score" not in result.metrics


# ---------------------------------------------------------------------------
# EvaluationOptions field + signature validation
# ---------------------------------------------------------------------------


def test_evaluation_options_accepts_surrogate():
    opts = EvaluationOptions(surrogate_evaluator=half_scorer)
    assert opts.surrogate_evaluator is half_scorer
    # Backward compatible: still optional.
    assert EvaluationOptions().surrogate_evaluator is None


def test_evaluation_options_still_forbids_extra():
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        EvaluationOptions(not_a_real_field=1)


def test_validate_accepts_output_scorer():
    _validate_surrogate_evaluator_signature(half_scorer)
    _validate_surrogate_evaluator_signature(lambda output: 0.5)

    class Callable_:
        def __call__(self, output, expected_output=None, example=None):
            return 0.5

    _validate_surrogate_evaluator_signature(Callable_())


def test_validate_rejects_func_executor_shape():
    def func_executor(func, config, example):
        return func()

    with pytest.raises(ValidationError):
        _validate_surrogate_evaluator_signature(func_executor)


def test_validate_rejects_non_callable_and_zero_arg():
    with pytest.raises(ValidationError):
        _validate_surrogate_evaluator_signature(42)  # type: ignore[arg-type]

    def zero_arg():
        return 0.5

    with pytest.raises(ValidationError):
        _validate_surrogate_evaluator_signature(zero_arg)


# ---------------------------------------------------------------------------
# create_effective_evaluator attaches the surrogate to the built evaluator
# ---------------------------------------------------------------------------


def test_create_effective_evaluator_attaches_surrogate():
    evaluator, _aux = create_effective_evaluator(
        timeout=None,
        custom_evaluator=None,
        effective_batch_size=1,
        effective_thread_workers=1,
        effective_privacy_enabled=False,
        objectives=["accuracy"],
        execution_mode="edge_analytics",
        mock_mode_config=None,
        metric_functions=None,
        scoring_function=None,
        decorator_custom_evaluator=None,
        decorator_surrogate_evaluator=half_scorer,
    )
    assert get_surrogate_evaluator(evaluator) is half_scorer


def test_create_effective_evaluator_no_surrogate_leaves_attr_absent():
    evaluator, _aux = create_effective_evaluator(
        timeout=None,
        custom_evaluator=None,
        effective_batch_size=1,
        effective_thread_workers=1,
        effective_privacy_enabled=False,
        objectives=["accuracy"],
        execution_mode="edge_analytics",
        mock_mode_config=None,
        metric_functions=None,
        scoring_function=None,
        decorator_custom_evaluator=None,
    )
    assert get_surrogate_evaluator(evaluator) is None


def test_create_effective_evaluator_runtime_surrogate_overrides_decorator():
    # Finding 5: the runtime optimize()-arg surrogate must win over the decorator
    # surrogate, mirroring custom_evaluator's precedence.
    decorator_surrogate = lambda output: 0.2  # noqa: E731
    evaluator, _aux = create_effective_evaluator(
        timeout=None,
        custom_evaluator=None,
        effective_batch_size=1,
        effective_thread_workers=1,
        effective_privacy_enabled=False,
        objectives=["accuracy"],
        execution_mode="edge_analytics",
        mock_mode_config=None,
        metric_functions=None,
        scoring_function=None,
        decorator_custom_evaluator=None,
        surrogate_evaluator=half_scorer,
        decorator_surrogate_evaluator=decorator_surrogate,
        surrogate_evaluator_name="runtime_name",
        decorator_surrogate_evaluator_name="decorator_name",
    )
    assert get_surrogate_evaluator(evaluator) is half_scorer
    # Runtime name wins and rides into the descriptor's evaluator_id.
    assert get_surrogate_evaluator_name(evaluator) == "runtime_name"
    descriptor = build_surrogate_descriptor(
        get_surrogate_evaluator(evaluator),
        name=get_surrogate_evaluator_name(evaluator),
    )
    assert descriptor["evaluator_id"] == "runtime_name"


@pytest.mark.asyncio
async def test_score_surrogate_uses_attached_name_in_descriptor():
    evaluator = _Evaluator()
    attach_surrogate_evaluator(evaluator, half_scorer, name="prescreen_v2")
    lifecycle = TrialLifecycle(_Orchestrator(evaluator))
    result = _eval_result([_example(0, "a")])

    descriptor = await lifecycle._score_surrogate(result, "trial-name")
    assert descriptor is not None
    assert descriptor["evaluator_id"] == "prescreen_v2"
    assert result.example_results[0].metrics["surrogate_score"] == 0.5


# ---------------------------------------------------------------------------
# Public-path runtime override: optimize()-arg surrogate wins over decorator
# ---------------------------------------------------------------------------


def _decorator_scorer(output, expected_output=None, example=None):
    return 0.2


def test_optimized_function_runtime_surrogate_overrides_decorator():
    # Finding 5: on the public OptimizedFunction seam, a runtime surrogate B
    # passed to optimize() (threaded via _create_effective_evaluator) must win
    # over the decorator-configured surrogate A, and the runtime name must
    # propagate into the descriptor's evaluator_id.
    from traigent.core.optimized_function import OptimizedFunction

    def func(text: str) -> str:
        return text

    opt = OptimizedFunction(
        func=func,
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
    )
    # Decorator-level surrogate A + name.
    opt._surrogate_evaluator = _decorator_scorer
    opt._surrogate_evaluator_name = "decorator_A"

    # Runtime optimize()-arg surrogate B + name.
    evaluator = opt._create_effective_evaluator(
        timeout=None,
        custom_evaluator=None,
        effective_batch_size=1,
        effective_thread_workers=1,
        effective_privacy_enabled=False,
        surrogate_evaluator=half_scorer,
        surrogate_evaluator_name="runtime_B",
    )
    assert get_surrogate_evaluator(evaluator) is half_scorer
    assert get_surrogate_evaluator_name(evaluator) == "runtime_B"
    descriptor = build_surrogate_descriptor(
        get_surrogate_evaluator(evaluator),
        name=get_surrogate_evaluator_name(evaluator),
    )
    assert descriptor["evaluator_id"] == "runtime_B"


def test_optimized_function_decorator_surrogate_used_without_runtime_override():
    # Without a runtime override, the decorator surrogate + name are used.
    from traigent.core.optimized_function import OptimizedFunction

    def func(text: str) -> str:
        return text

    opt = OptimizedFunction(
        func=func,
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
    )
    opt._surrogate_evaluator = half_scorer
    opt._surrogate_evaluator_name = "decorator_only"

    evaluator = opt._create_effective_evaluator(
        timeout=None,
        custom_evaluator=None,
        effective_batch_size=1,
        effective_thread_workers=1,
        effective_privacy_enabled=False,
    )
    assert get_surrogate_evaluator(evaluator) is half_scorer
    assert get_surrogate_evaluator_name(evaluator) == "decorator_only"


def test_evaluation_options_accepts_surrogate_name():
    opts = EvaluationOptions(
        surrogate_evaluator=half_scorer, surrogate_evaluator_name="my_id"
    )
    assert opts.surrogate_evaluator_name == "my_id"
    # Backward compatible: still optional.
    assert EvaluationOptions().surrogate_evaluator_name is None
