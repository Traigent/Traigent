"""Tests for the continuous per-item defect score (issue #1881).

Covers the continuous layer that stacks on #1880's binary flags, as a pure
function of the outcome matrix (#1838):
- ordering: never-correct scores > mixed > always-correct;
- ``defect_percentile`` is monotonic in score, handles ties and single-item runs;
- ``contributing_signals`` surfaces the dominant feature (mean_wrong);
- items run in <2 configs are excluded (features undefined);
- the logistic score is always in [0, 1];
- ALL scored items are surfaced (ranked worklist), not only the flagged ones;
- the refit hook (custom intercept/weights) flows through;
- end-to-end through ``OptimizationResult.eval_audit`` and ``to_dict``.

No LLM calls: matrices are built in memory / from real ExampleResult shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from traigent.api.types import (
    EvalAudit,
    ExampleResult,
    ItemDefectScore,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.eval_audit import (
    DEFECT_SCORE_INTERCEPT,
    DEFECT_SCORE_WEIGHTS,
    _item_defect_features,
    _logistic,
    compute_defect_scores,
    compute_eval_audit,
)

# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def _cell(*, correct: bool, tokens: int | None = None) -> dict[str, Any]:
    return {
        "score": 1.0 if correct else 0.0,
        "accuracy": 1.0 if correct else 0.0,
        "metrics": {},
        "success": correct,
        "tokens": {"total": tokens} if tokens is not None else None,
        "cost_usd": None,
        "execution_time": None,
        "latency_ms": None,
        "predicted": None,
        "error": None if correct else "wrong",
    }


def _column(trial_id: str, model: str) -> dict[str, Any]:
    return {
        "index": 0,
        "trial_id": trial_id,
        "config": {"model": model},
        "config_hash": trial_id,
    }


def _matrix(
    trials: list[dict[str, Any]], examples: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "optimization_id": "opt-1881",
        "algorithm": "GridSearchOptimizer",
        "objectives": ["accuracy"],
        "created_at": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        "trial_count": len(trials),
        "example_count": len(examples),
        "trials": trials,
        "examples": examples,
    }


def _row(example_id: str, outcomes: list[bool]) -> dict[str, Any]:
    """An example row with one cell per outcome (trials t0..tN)."""
    return {
        "example_id": example_id,
        "cells": {f"t{i}": _cell(correct=o) for i, o in enumerate(outcomes)},
    }


def _grid(n_cols: int) -> list[dict[str, Any]]:
    return [_column(f"t{i}", "gpt-4o") for i in range(n_cols)]


# ---------------------------------------------------------------------------
# Ordering: never-correct > mixed > always-correct
# ---------------------------------------------------------------------------


def test_score_ordering_never_gt_mixed_gt_always() -> None:
    matrix = _matrix(
        _grid(4),
        [
            _row("ex-never", [False, False, False, False]),
            _row("ex-mixed", [True, False, True, False]),
            _row("ex-always", [True, True, True, True]),
        ],
    )
    scored = compute_defect_scores(matrix)
    by_id = {s.example_id: s.defect_score for s in scored}
    assert by_id["ex-never"] > by_id["ex-mixed"] > by_id["ex-always"]
    # sane calibration: never-correct high, always-correct low
    assert by_id["ex-never"] > 0.9
    assert by_id["ex-always"] < 0.1


def test_scored_sorted_by_score_descending() -> None:
    matrix = _matrix(
        _grid(4),
        [
            _row("ex-always", [True, True, True, True]),
            _row("ex-never", [False, False, False, False]),
            _row("ex-mixed", [True, False, True, False]),
        ],
    )
    scored = compute_defect_scores(matrix)
    ids = [s.example_id for s in scored]
    assert ids == ["ex-never", "ex-mixed", "ex-always"]
    scores = [s.defect_score for s in scored]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Logistic bounds
# ---------------------------------------------------------------------------


def test_logistic_bounded_zero_one() -> None:
    for z in (-1000.0, -10.0, 0.0, 10.0, 1000.0):
        value = _logistic(z)
        assert 0.0 <= value <= 1.0
    assert _logistic(0.0) == 0.5


def test_all_defect_scores_in_unit_interval() -> None:
    matrix = _matrix(
        _grid(3),
        [
            _row("a", [False, False, False]),
            _row("b", [True, False, False]),
            _row("c", [True, True, False]),
            _row("d", [True, True, True]),
        ],
    )
    for s in compute_defect_scores(matrix):
        assert 0.0 <= s.defect_score <= 1.0
        assert 0.0 <= s.defect_percentile <= 1.0


# ---------------------------------------------------------------------------
# Percentile convention: fraction with score <= this; monotonic; ties; single
# ---------------------------------------------------------------------------


def test_percentile_monotonic_in_score() -> None:
    matrix = _matrix(
        _grid(4),
        [
            _row("ex-never", [False, False, False, False]),
            _row("ex-mostly-wrong", [True, False, False, False]),
            _row("ex-mixed", [True, True, False, False]),
            _row("ex-always", [True, True, True, True]),
        ],
    )
    scored = compute_defect_scores(matrix)
    # sorted by score desc; percentile must be non-increasing down that order
    percentiles = [s.defect_percentile for s in scored]
    assert percentiles == sorted(percentiles, reverse=True)
    # most suspicious item is at the top percentile
    assert scored[0].defect_percentile == 1.0


def test_percentile_ties_share_value() -> None:
    """Two items with identical outcomes -> identical score -> identical
    percentile (both counted by the ``<=`` convention)."""
    matrix = _matrix(
        _grid(2),
        [
            _row("tie-a", [True, False]),
            _row("tie-b", [False, True]),
            _row("low", [True, True]),
        ],
    )
    scored = compute_defect_scores(matrix)
    by_id = {s.example_id: s for s in scored}
    assert by_id["tie-a"].defect_score == by_id["tie-b"].defect_score
    assert by_id["tie-a"].defect_percentile == by_id["tie-b"].defect_percentile
    # both tied items sit above the always-correct one, and share the top
    # percentile (fraction with score <= theirs = 3/3 = 1.0)
    assert by_id["tie-a"].defect_percentile == 1.0
    assert by_id["low"].defect_percentile < by_id["tie-a"].defect_percentile


def test_percentile_single_item_run_is_one() -> None:
    matrix = _matrix(_grid(2), [_row("only", [True, False])])
    scored = compute_defect_scores(matrix)
    assert len(scored) == 1
    assert scored[0].defect_percentile == 1.0


# ---------------------------------------------------------------------------
# contributing_signals: dominant feature is mean_wrong
# ---------------------------------------------------------------------------


def test_contributing_signals_dominant_feature_is_mean_wrong() -> None:
    matrix = _matrix(_grid(4), [_row("ex-never", [False, False, False, False])])
    scored = compute_defect_scores(matrix)
    signals = scored[0].contributing_signals
    assert signals, "never-correct item must have contributing signals"
    # sorted by contribution desc -> mean_wrong (weight 6.0 * value 1.0) leads
    assert signals[0].feature == "mean_wrong"
    assert signals[0].contribution == DEFECT_SCORE_WEIGHTS["mean_wrong"]
    features = {s.feature for s in signals}
    assert "never_correct" in features  # also contributed (weight 1.5 * 1.0)
    # every reported contribution equals weight * value
    for s in signals:
        assert s.contribution == s.weight * s.value


def test_contributing_signals_empty_for_always_correct() -> None:
    matrix = _matrix(
        _grid(4),
        [
            _row("ex-always", [True, True, True, True]),
            _row("ex-never", [False, False, False, False]),
        ],
    )
    scored = compute_defect_scores(matrix)
    by_id = {s.example_id: s for s in scored}
    # nothing drove an always-correct item's score up
    assert by_id["ex-always"].contributing_signals == []


def test_instability_surfaces_for_split_item() -> None:
    matrix = _matrix(_grid(2), [_row("ex-split", [True, False])])
    scored = compute_defect_scores(matrix)
    signals = {s.feature for s in scored[0].contributing_signals}
    # a 1/2 split: mean_wrong=0.5 (contrib 3.0) and instability=0.5 (contrib 0.5)
    assert "mean_wrong" in signals
    assert "instability" in signals
    assert "never_correct" not in signals  # not wrong everywhere


# ---------------------------------------------------------------------------
# Degenerate: item run in <2 configs excluded
# ---------------------------------------------------------------------------


def test_sparse_single_config_item_excluded() -> None:
    matrix = _matrix(
        _grid(2),
        [
            {"example_id": "ex-sparse", "cells": {"t0": _cell(correct=False)}},
            _row("ex-real", [False, False]),
        ],
    )
    scored = compute_defect_scores(matrix)
    ids = {s.example_id for s in scored}
    assert ids == {"ex-real"}


def test_no_scorable_items_returns_empty() -> None:
    matrix = _matrix(
        _grid(2),
        [{"example_id": "ex", "cells": {"t0": _cell(correct=True)}}],
    )
    assert compute_defect_scores(matrix) == []
    assert compute_defect_scores(None) == []
    assert compute_defect_scores({}) == []


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_item_features_values() -> None:
    assert _item_defect_features([False, False]) == {
        "mean_wrong": 1.0,
        "never_correct": 1.0,
        "instability": 0.0,
    }
    assert _item_defect_features([True, True]) == {
        "mean_wrong": 0.0,
        "never_correct": 0.0,
        "instability": 0.0,
    }
    assert _item_defect_features([True, False]) == {
        "mean_wrong": 0.5,
        "never_correct": 0.0,
        "instability": 0.5,
    }
    # 1 correct of 4 -> mostly wrong, minority is the single correct one
    assert _item_defect_features([True, False, False, False]) == {
        "mean_wrong": 0.75,
        "never_correct": 0.0,
        "instability": 0.25,
    }


# ---------------------------------------------------------------------------
# Refit hook: custom coefficients flow through
# ---------------------------------------------------------------------------


def test_refit_weights_change_scores() -> None:
    matrix = _matrix(_grid(2), [_row("ex", [True, False])])
    default = compute_defect_scores(matrix)[0].defect_score
    # a refit that zeroes every feature -> score is sigmoid(intercept) for all
    flat = compute_defect_scores(
        matrix,
        intercept=0.0,
        weights={"mean_wrong": 0.0, "never_correct": 0.0, "instability": 0.0},
    )[0].defect_score
    assert flat == 0.5
    assert default != flat


def test_compute_eval_audit_passes_defect_coefficients() -> None:
    matrix = _matrix(_grid(2), [_row("a", [True, False]), _row("b", [False, False])])
    audit = compute_eval_audit(
        matrix,
        defect_score_intercept=0.0,
        defect_score_weights={
            "mean_wrong": 0.0,
            "never_correct": 0.0,
            "instability": 0.0,
        },
    )
    assert audit is not None
    assert all(s.defect_score == 0.5 for s in audit.scored)


# ---------------------------------------------------------------------------
# Integration with compute_eval_audit: scored surfaces ALL items
# ---------------------------------------------------------------------------


def test_eval_audit_scored_covers_all_items_not_just_flagged() -> None:
    matrix = _matrix(
        _grid(3),
        [
            _row("ex-never", [False, False, False]),  # flagged (never-correct)
            _row("ex-mixed", [True, False, True]),  # NOT flagged
            _row("ex-always", [True, True, True]),  # NOT flagged
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    flagged_ids = {f.example_id for f in audit.flagged}
    scored_ids = {s.example_id for s in audit.scored}
    assert flagged_ids == {"ex-never"}
    # the ranked worklist covers everything scorable, not only the flagged item
    assert scored_ids == {"ex-never", "ex-mixed", "ex-always"}
    # binary flag is expressible as score >= threshold: the flagged item ranks top
    assert audit.scored[0].example_id == "ex-never"


def test_default_constants_are_documented_values() -> None:
    assert DEFECT_SCORE_INTERCEPT == -4.0
    assert DEFECT_SCORE_WEIGHTS["mean_wrong"] > DEFECT_SCORE_WEIGHTS["never_correct"]
    assert DEFECT_SCORE_WEIGHTS["never_correct"] > DEFECT_SCORE_WEIGHTS["instability"]


# ---------------------------------------------------------------------------
# End-to-end through OptimizationResult.eval_audit + to_dict
# ---------------------------------------------------------------------------


def _example(example_id: str, *, correct: bool) -> dict[str, Any]:
    return ExampleResult(
        example_id=example_id,
        input_data={},
        expected_output="GOLD",
        actual_output="GOLD" if correct else "WRONG",
        metrics={"accuracy": 1.0 if correct else 0.0, "score": 1.0 if correct else 0.0},
        execution_time=0.1,
        success=correct,
        error_message=None if correct else "wrong",
    ).to_dict()


def _trial(trial_id: str, model: str, examples: list[Any]) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"model": model},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"example_results": examples},
    )


# ---------------------------------------------------------------------------
# Documented calibration example values are pinned (docstring can't drift)
# ---------------------------------------------------------------------------


def test_documented_default_score_values_are_pinned() -> None:
    """The exact scores the module docstring advertises for the default
    (illustrative) coefficients. If someone changes a default weight/intercept
    these must be updated in lock-step, so the doc can't silently overclaim."""
    matrix = _matrix(
        _grid(2),
        [
            _row("never", [False, False]),  # mean_wrong=1, never_correct=1, inst=0
            _row("split", [True, False]),  # mean_wrong=0.5, never=0, inst=0.5
            _row("always", [True, True]),  # mean_wrong=0, never=0, inst=0
        ],
    )
    by_id = {s.example_id: s.defect_score for s in compute_defect_scores(matrix)}
    assert by_id["never"] == pytest.approx(0.971, abs=1e-3)
    assert by_id["always"] == pytest.approx(0.018, abs=1e-3)
    assert by_id["split"] == pytest.approx(0.378, abs=1e-3)


# ---------------------------------------------------------------------------
# Exact percentile fractions for N distinct scores (1/3, 2/3, 3/3)
# ---------------------------------------------------------------------------


def test_percentile_exact_fractions_for_distinct_scores() -> None:
    matrix = _matrix(
        _grid(3),
        [
            _row("never", [False, False, False]),  # mean_wrong=1.0 (highest)
            _row("mostly-wrong", [True, False, False]),  # mean_wrong=2/3 (middle)
            _row("always", [True, True, True]),  # mean_wrong=0.0 (lowest)
        ],
    )
    by_id = {s.example_id: s.defect_percentile for s in compute_defect_scores(matrix)}
    # percentile = fraction of scored items with score <= this one's
    assert by_id["always"] == pytest.approx(1 / 3)
    assert by_id["mostly-wrong"] == pytest.approx(2 / 3)
    assert by_id["never"] == pytest.approx(3 / 3)


# ---------------------------------------------------------------------------
# Refit footgun 1: scale-aware signal selection survives tiny weights
# ---------------------------------------------------------------------------


def test_small_refit_weights_still_yield_signals() -> None:
    """Under a small-weight refit, an absolute contribution threshold would blank
    the explanation. Scale-aware selection keeps the drivers regardless of scale."""
    matrix = _matrix(_grid(2), [_row("never", [False, False])])
    tiny = {"mean_wrong": 0.01, "never_correct": 0.01, "instability": 0.01}
    scored = compute_defect_scores(matrix, intercept=-0.02, weights=tiny)
    signals = scored[0].contributing_signals
    assert signals, "a max-suspicion item must still explain itself under tiny weights"
    features = {s.feature for s in signals}
    # both fired features (mean_wrong=1, never_correct=1) are surfaced; each
    # contribution is 0.01 — far below the old absolute 0.05 floor.
    assert features == {"mean_wrong", "never_correct"}
    for s in signals:
        assert abs(s.contribution) < 0.05
        assert s.contribution == s.weight * s.value


# ---------------------------------------------------------------------------
# Refit footgun 2: a negative refit weight is reported as LOWERING suspicion
# ---------------------------------------------------------------------------


def test_negative_refit_weight_signal_lowers_suspicion() -> None:
    matrix = _matrix(_grid(2), [_row("never", [False, False])])
    # never_correct gets a negative weight -> it should pull suspicion DOWN, and
    # its signal must carry that (negative contribution), not read as a driver up.
    weights = {"mean_wrong": 6.0, "never_correct": -2.0, "instability": 1.0}
    scored = compute_defect_scores(matrix, weights=weights)
    by_feature = {s.feature: s for s in scored[0].contributing_signals}
    assert "never_correct" in by_feature
    nc = by_feature["never_correct"]
    assert nc.weight == -2.0
    assert nc.value == 1.0
    assert nc.contribution == pytest.approx(-2.0)  # negative -> lowered suspicion
    assert nc.contribution < 0.0
    # ranked by magnitude: mean_wrong (|6.0|) still leads, negative signal is not
    # promoted above the true positive driver.
    assert scored[0].contributing_signals[0].feature == "mean_wrong"


def test_eval_audit_property_populates_scored_and_to_dict() -> None:
    t0 = _trial(
        "t0",
        "gpt-4o",
        [_example("ex-never", correct=False), _example("ex-always", correct=True)],
    )
    t1 = _trial(
        "t1",
        "gpt-4o-mini",
        [_example("ex-never", correct=False), _example("ex-always", correct=True)],
    )
    result = OptimizationResult(
        trials=[t0, t1],
        best_config={"model": "gpt-4o"},
        best_score=0.5,
        optimization_id="opt-1881-e2e",
        duration=2.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    audit = result.eval_audit
    assert isinstance(audit, EvalAudit)
    assert [s.example_id for s in audit.scored] == ["ex-never", "ex-always"]
    assert isinstance(audit.scored[0], ItemDefectScore)
    # to_dict round-trips the scored layer
    d = audit.to_dict()
    assert [s["example_id"] for s in d["scored"]] == ["ex-never", "ex-always"]
    top = d["scored"][0]
    assert set(top) == {
        "example_id",
        "defect_score",
        "defect_percentile",
        "features",
        "contributing_signals",
    }
    assert top["contributing_signals"][0]["feature"] == "mean_wrong"
