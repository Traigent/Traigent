"""Regression tests for issue #1845 — trial-metrics ``score`` semantics.

Before the fix, a single-objective ("accuracy") run with a custom, non-binary
``scoring_function`` produced a trial whose ``metrics["score"]`` held the SDK's
built-in exact-match scorer (NOT the optimization signal), while
``metrics["accuracy"]`` held the custom scorer mean and ``TrialResult.score``
did not exist. Consumers doing ``trial.score or trial.metrics["score"]`` logged a
number the optimizer never used, disagreeing with ``best_config``.

These tests assert the decided semantics:

* ``metrics["score"]`` == the optimization signal (objective value) — NOT the
  default exact-match scorer, which is relocated to ``metrics["exact_match_default"]``.
* ``TrialResult.score`` is populated with the objective value.
* The no-custom-scorer path and the weighted-run path (#1682) are unchanged.

Each assertion is written so it FAILS on the old behavior and PASSES on the new.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.trial_result_factory import (
    _populate_trial_score,
    build_success_result,
)
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


# --------------------------------------------------------------------------- #
# Scorers                                                                      #
# --------------------------------------------------------------------------- #
def _token_f1(output: str, expected: str) -> float:
    out_tokens = set(output.lower().split())
    exp_tokens = set(expected.lower().split())
    if not out_tokens or not exp_tokens:
        return 0.0
    overlap = len(out_tokens & exp_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(out_tokens)
    recall = overlap / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def _blended_scorer(output: str, expected: str) -> float:
    """(exact-match + token-F1) / 2 — a NON-binary scorer (like the issue)."""
    exact = 1.0 if output.strip().lower() == expected.strip().lower() else 0.0
    return (exact + _token_f1(output, expected)) / 2


# ex1 exact-matches; ex2 is a token-reorder (EM=0, token-F1=1.0 -> blended 0.5).
_EXAMPLES = [
    EvaluationExample({"query": "two"}, "2"),
    EvaluationExample({"query": "capital"}, "paris is the capital"),
]
_EXPECTED_BY_QUERY = {"two": "2", "capital": "the capital is paris"}

# Custom scorer mean: (blended("2","2")=1.0 + blended(reorder)=0.5) / 2 = 0.75
_CUSTOM_ACCURACY_MEAN = pytest.approx(0.75)
# Default exact-match mean: (1.0 + 0.0) / 2 = 0.5
_DEFAULT_EXACT_MATCH_MEAN = pytest.approx(0.5)


def _agent(input_data: dict) -> str:
    return _EXPECTED_BY_QUERY[input_data["query"]]


def _dataset() -> Dataset:
    return Dataset(list(_EXAMPLES), name="qa", description="score-semantics fixture")


# --------------------------------------------------------------------------- #
# Item 2 — metrics["score"] is the optimization signal, not the default scorer #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_custom_scorer_metrics_score_is_objective_not_default_exact_match() -> (
    None
):
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        metric_functions={"accuracy": _blended_scorer},
    )

    result = await evaluator.evaluate(_agent, {}, _dataset())
    metrics = result.metrics

    # The objective is the custom scorer mean (the value best_config argmaxes).
    assert metrics["accuracy"] == _CUSTOM_ACCURACY_MEAN

    # NEW: metrics["score"] is the optimization signal (== objective), NOT the
    # default exact-match scorer. On old behavior score was 0.5 -> this FAILS.
    assert metrics["score"] == _CUSTOM_ACCURACY_MEAN
    assert metrics["score"] == metrics["accuracy"]

    # The default exact-match diagnostic is relocated, not dropped.
    assert metrics["exact_match_default"] == _DEFAULT_EXACT_MATCH_MEAN

    # The two numbers genuinely diverge — proving score is no longer the default.
    assert metrics["score"] != pytest.approx(metrics["exact_match_default"])


@pytest.mark.asyncio
async def test_no_custom_scorer_metrics_score_unchanged_and_no_diagnostic_key() -> None:
    """Control: without a custom scorer the default exact-match IS the objective,
    so metrics["score"] == metrics["accuracy"] and no diagnostic key is added."""
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)

    result = await evaluator.evaluate(_agent, {}, _dataset())
    metrics = result.metrics

    assert metrics["accuracy"] == _DEFAULT_EXACT_MATCH_MEAN
    assert metrics["score"] == _DEFAULT_EXACT_MATCH_MEAN
    assert metrics["score"] == metrics["accuracy"]
    # No custom scorer -> no dual-scorer situation -> no diagnostic key.
    assert "exact_match_default" not in metrics


@pytest.mark.asyncio
async def test_dual_scorer_notice_logged_once_per_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        metric_functions={"accuracy": _blended_scorer},
    )
    import logging

    with caplog.at_level(logging.INFO, logger="traigent.evaluators.local"):
        await evaluator.evaluate(_agent, {}, _dataset())
        await evaluator.evaluate(_agent, {}, _dataset())  # 2nd trial, same instance

    notices = [
        r for r in caplog.records if "carries the optimization signal" in r.getMessage()
    ]
    assert len(notices) == 1  # once per run (evaluator instance), not per trial


# --------------------------------------------------------------------------- #
# Item 1 — TrialResult.score is populated with the optimization signal          #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_trial_score_populated_with_objective_end_to_end() -> None:
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        metric_functions={"accuracy": _blended_scorer},
    )
    eval_result = await evaluator.evaluate(_agent, {}, _dataset())

    trial = build_success_result(
        trial_id="t0",
        evaluation_config={"model": "test"},
        eval_result=eval_result,
        duration=0.1,
        examples_attempted=2,
        total_cost=None,
        optuna_trial_id=None,
        primary_objective="accuracy",
    )

    # NEW: trial.score is the objective value (== metrics["accuracy"]), not None.
    assert trial.score is not None
    assert trial.score == _CUSTOM_ACCURACY_MEAN
    assert trial.score == trial.metrics["accuracy"]
    # ... and equals metrics["score"] (the aligned optimization signal).
    assert trial.score == pytest.approx(trial.metrics["score"])


def test_populate_trial_score_prefers_primary_objective() -> None:
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 0.75, "score": 0.75, "cost": 0.1},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )
    _populate_trial_score(trial, "accuracy")
    assert trial.score == pytest.approx(0.75)


def test_populate_trial_score_falls_back_to_metrics_score() -> None:
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"score": 0.42},  # objective key absent
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )
    _populate_trial_score(trial, "accuracy")
    assert trial.score == pytest.approx(0.42)


def test_populate_trial_score_stays_none_when_signal_absent() -> None:
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"cost": 0.1},  # no objective, no score
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )
    _populate_trial_score(trial, "accuracy")
    assert trial.score is None


def test_trial_result_score_field_defaults_none_and_serializes() -> None:
    trial = TrialResult(
        trial_id="t",
        config={},
        metrics={"accuracy": 0.9},
        status=TrialStatus.COMPLETED,
        duration=0.0,
        timestamp=datetime.now(UTC),
    )
    assert trial.score is None  # the issue's observed "Trial.score = None"
    trial.score = 0.9
    assert trial.to_dict()["score"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Weighted control — #1682 basis still wins and drives trial.score             #
# --------------------------------------------------------------------------- #
def test_weighted_run_metrics_score_is_weighted_basis_and_wins() -> None:
    """#1682: weighted runs overwrite metrics["score"] with the weighted basis
    at terminal selection, and (per #1845) trial.score mirrors it."""
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
    from traigent.core.result_selection import (
        _populate_weighted_scores,
        observed_metric_ranges,
    )

    schema = ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(name="accuracy", orientation="maximize", weight=0.5),
            ObjectiveDefinition(name="cost", orientation="minimize", weight=0.5),
        ]
    )

    trials = [
        TrialResult(
            trial_id="a",
            config={"m": 1},
            metrics={"accuracy": 0.9, "cost": 0.9, "score": 0.123},  # stale score
            status=TrialStatus.COMPLETED,
            duration=0.0,
            timestamp=datetime.now(UTC),
        ),
        TrialResult(
            trial_id="b",
            config={"m": 2},
            metrics={"accuracy": 0.1, "cost": 0.1, "score": 0.999},  # stale score
            status=TrialStatus.COMPLETED,
            duration=0.0,
            timestamp=datetime.now(UTC),
        ),
    ]

    ranges = observed_metric_ranges(trials, ("accuracy", "cost"))
    _populate_weighted_scores(trials, schema, ranges)

    for trial in trials:
        weighted = schema.compute_weighted_score(trial.metrics, ranges=ranges)
        # metrics["score"] is the weighted basis, overwriting the stale value.
        assert trial.metrics["score"] == pytest.approx(weighted)
        # trial.score mirrors it (#1845 consistency).
        assert trial.score == pytest.approx(weighted)
