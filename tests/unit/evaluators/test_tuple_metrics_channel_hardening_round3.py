"""Round-3 hardening probes for the tuple-return metrics channel (captain review).

These pin the two captain-found residuals that survived the G1-G4 round. Written
RED-first: each asserts the *fixed* behavior so it fails against the pre-fix code.

* R1 — the G1 class survives in the LOCAL lane. The local-lane
  ``enforce_user_metric_ceiling`` (``local.py``) and the ``format_for_backend``
  user-metric cap BOTH run, bounding ``eval_result.metrics`` to 50 keys. But
  ``build_success_result`` (``core/trial_result_factory.py``) writes ONE more
  reserved key — ``total_cost`` — into the trial metrics AFTER both caps, so the
  final ``result.metrics`` has 51 keys. The fix re-applies
  ``enforce_user_metric_ceiling`` as the LAST step of ``build_success_result``,
  the SINGLE authoritative final-union cap covering EVERY lane that produces a
  successful trial; only user keys are dropped, the reserved keys all survive.
* R2 — a 2-tuple whose ``[1]`` IS a Mapping but FAILS the strict wire rule (bad
  key syntax, or a bool / non-numeric value) is left as raw output per design
  (absolute back-compat). Pre-fix this is SILENT: the user's accuracy is poisoned
  (the raw tuple becomes the output) with zero signal. The fix logs ONE WARNING
  naming the first offending key/value and stating the tuple was NOT unpacked and
  why — with NO behavior change (raw passthrough stays). Shapes that are silent
  by design (plain return, 3-tuple, tuple whose ``[1]`` is a list) stay silent.
"""

from __future__ import annotations

import logging

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.core.trial_result_factory import build_success_result
from traigent.evaluators.base import Dataset, EvaluationExample, SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.knobs.telemetry import TOTAL_MEASURES_CEILING

#: The reserved (non-user) keys the captain's 60-key local-lane repro produces on
#: the final trial metrics. Every one must SURVIVE the final-union cap — only the
#: user keys may shrink.
EXPECTED_NON_USER_KEYS = {
    "accuracy",
    "cost",
    "duration",
    "examples_attempted",
    "input_tokens",
    "output_tokens",
    "response_time_ms",
    "score",
    "successful_examples",
    "total_cost",
    "total_examples",
    "total_tokens",
}


def _two_example_dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="YES"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="YES"),
        ],
        name="tuple_metrics_hardening_round3",
    )


def _local_evaluator(metrics: list[str] | None = None) -> LocalEvaluator:
    return LocalEvaluator(
        metrics=metrics or ["accuracy"],
        detailed=True,
        execution_mode="local",
    )


# ---------------------------------------------------------------------------
# R1: build_success_result is the authoritative final-union cap (local lane)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_lane_final_trial_metrics_capped_after_total_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1 (captain repro): 60 tuple user keys -> final result.metrics <= ceiling.

    Pre-fix: the local-lane caps bound ``eval_result.metrics`` to 50, then
    ``build_success_result`` writes ``total_cost`` AFTER them -> 51 keys.
    Fixed: ``build_success_result`` re-applies the ceiling as the LAST step, so
    only user keys shrink and ALL reserved keys survive.
    """

    user_keys = {f"composite_metric_{i}": float(i) for i in range(60)}

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", dict(user_keys)

    with caplog.at_level(logging.WARNING):
        eval_result = await _local_evaluator().evaluate(
            func, {}, _two_example_dataset()
        )

    # Build the trial result exactly as the orchestrator does: the lane's
    # reserved keys (examples_attempted, total_cost) are written here, AFTER the
    # local-lane caps already ran on eval_result.metrics.
    trial_result = build_success_result(
        trial_id="r1-local",
        evaluation_config={},
        eval_result=eval_result,
        duration=0.1,
        examples_attempted=2,
        total_cost=0.0,
        optuna_trial_id=None,
    )

    metrics = trial_result.metrics
    # RED before R1: total_cost slips past the local-lane caps -> 51 keys.
    assert len(metrics) <= TOTAL_MEASURES_CEILING
    # ALL 12 non-user (reserved) keys must survive the cap; only user keys shrink.
    present_non_user = {k for k in metrics if not k.startswith("composite_metric_")}
    missing_reserved = EXPECTED_NON_USER_KEYS - present_non_user
    assert not missing_reserved, f"reserved keys dropped: {missing_reserved}"
    # total_cost (the post-cap writer) is present and is a reserved key, not a
    # casualty of the ceiling.
    assert "total_cost" in metrics
    # The final union is wire-valid against the MeasuresDict ceiling.
    assert len(MeasuresDict(dict(metrics))) <= MeasuresDict.MAX_KEYS


@pytest.mark.asyncio
async def test_simple_scoring_60_key_probe_still_holds_after_refactor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1 (regression): the G1 SimpleScoring 60-key probe still holds.

    The authoritative final-union cap must not regress the SimpleScoring lane:
    its own evaluate() still returns <= ceiling and examples_attempted survives.
    """

    def score(output: object, expected: object) -> float:
        return 1.0 if output == expected else 0.0

    user_keys = {f"composite_metric_{i}": float(i) for i in range(60)}

    def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", dict(user_keys)

    evaluator = SimpleScoringEvaluator(
        scoring_function=score,
        metrics=["accuracy"],
        capture_llm_metrics=False,
    )
    with caplog.at_level(logging.WARNING):
        result = await evaluator.evaluate(func, {}, _two_example_dataset())

    assert len(result.metrics) <= TOTAL_MEASURES_CEILING
    assert "examples_attempted" in result.metrics
    assert result.metrics["examples_attempted"] == 2

    # And the same result, assembled through build_success_result, stays capped.
    trial_result = build_success_result(
        trial_id="r1-simple",
        evaluation_config={},
        eval_result=result,
        duration=0.1,
        examples_attempted=2,
        total_cost=0.0,
        optuna_trial_id=None,
    )
    assert len(trial_result.metrics) <= TOTAL_MEASURES_CEILING
    assert "examples_attempted" in trial_result.metrics


def test_build_success_result_caps_reserved_writes_added_after_eval() -> None:
    """R1 (unit): a 50-key eval_result + total_cost write -> capped to ceiling.

    Direct unit on the authoritative site: feed a 50-key metrics dict (49 user +
    accuracy) and let build_success_result add examples_attempted + total_cost.
    The final union must clamp back to the ceiling, dropping only user keys.
    """

    class _EvalResult:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {"accuracy": 1.0}
            self.metrics.update({f"composite_metric_{i}": float(i) for i in range(49)})
            self.summary_stats = None
            self.comparability = None

    eval_result = _EvalResult()
    assert len(eval_result.metrics) == TOTAL_MEASURES_CEILING

    trial_result = build_success_result(
        trial_id="r1-unit",
        evaluation_config={},
        eval_result=eval_result,
        duration=0.1,
        examples_attempted=2,
        total_cost=0.5,
        optuna_trial_id=None,
    )

    metrics = trial_result.metrics
    assert len(metrics) <= TOTAL_MEASURES_CEILING
    # The reserved keys written by build_success_result survive.
    assert metrics["accuracy"] == 1.0
    assert "examples_attempted" in metrics
    assert "total_cost" in metrics


# ---------------------------------------------------------------------------
# R2: silent near-miss unpack now logs ONE warning (no behavior change)
# ---------------------------------------------------------------------------


def test_near_miss_bad_key_logs_warning_no_unpack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R2: ("YES", {"π_metric": 1.0}) -> WARNING + raw passthrough (no unpack)."""
    raw = ("YES", {"π_metric": 1.0})
    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.base"):
        output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)

    # Behavior unchanged: raw passthrough.
    assert output is raw
    assert user_metrics is None

    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "not treated as" in r.getMessage().lower()
    ]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    msg = warnings[0].getMessage()
    assert "π_metric" in msg


def test_near_miss_bad_value_logs_warning_no_unpack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R2: non-numeric / bool value also warns once and passes through raw."""
    raw = ("YES", {"good_key": True})
    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.base"):
        output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)

    assert output is raw
    assert user_metrics is None
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "not treated as" in r.getMessage().lower()
    ]
    assert len(warnings) == 1
    assert "good_key" in warnings[0].getMessage()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_near_miss_non_finite_value_logs_warning_no_unpack(
    bad_value: float,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-finite numeric values are rejected before aggregation."""
    raw = ("YES", {"score": bad_value})
    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.base"):
        output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)

    assert output is raw
    assert user_metrics is None
    warnings = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and "not treated as" in r.getMessage().lower()
    ]
    assert len(warnings) == 1
    assert "score" in warnings[0].getMessage()
    assert "finite number" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_non_finite_tuple_metric_does_not_reach_aggregate() -> None:
    """NaN/Inf tuple metrics must not poison aggregate metrics."""

    async def func(text: str) -> tuple[str, dict[str, float]]:
        return "YES", {"custom_nan_metric": float("nan")}

    result = await _local_evaluator().evaluate(func, {}, _two_example_dataset())

    assert "custom_nan_metric" not in result.metrics
    assert all(
        "custom_nan_metric" not in example.metrics for example in result.example_results
    )


@pytest.mark.parametrize(
    "raw",
    [
        "plain string return",
        ("a", "b", "c"),  # 3-tuple
        ("YES", [1, 2, 3]),  # [1] is a list, not a Mapping
        ("YES",),  # 1-tuple
        42,  # non-tuple scalar
    ],
)
def test_by_design_silent_shapes_emit_no_warning(
    raw: object, caplog: pytest.LogCaptureFixture
) -> None:
    """R2: shapes silent by design stay silent — only the NEAR-MISS shape warns.

    A near-miss is a 2-tuple whose [1] IS a Mapping but fails the strict rule. A
    plain return, a 3-tuple, or a tuple whose [1] is a list are NOT a near-miss:
    the user never signaled intent to ship metrics, so warning would be noise.
    """
    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.base"):
        output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)

    assert output is raw
    assert user_metrics is None
    assert not any(
        "not treated as" in r.getMessage().lower() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_valid_tuple_still_unpacks_and_emits_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R2: a legitimate (output, metrics) tuple unpacks silently (no warning)."""
    raw = ("YES", {"composite_ok": 1.0, "another": 2})
    with caplog.at_level(logging.WARNING, logger="traigent.evaluators.base"):
        output, user_metrics = SimpleScoringEvaluator._unpack_user_metrics(raw)

    assert output == "YES"
    assert user_metrics == {"composite_ok": 1.0, "another": 2.0}
    assert not any("not treated as" in r.getMessage().lower() for r in caplog.records)
