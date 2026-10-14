"""Tests for the failing-example selector (Traigent#2565).

Covers the pure matrix selector
``traigent.utils.outcome_matrix.select_failing_example_ids`` and its API
wrapper ``OptimizationResult.failing_examples``:

- threshold semantics (score < threshold fails, equality passes) with the
  ``score`` -> ``accuracy`` fallback;
- errors / ``success=False`` count as failures; ``success=True`` alone does not
  prove correctness;
- missing cells and unusable values are *unknown*, never zero;
- ``all_configs`` requires every column to fail; ``best_config`` restricts to
  the winner's columns (``winning_trial_ids`` first, then config equality);
- ``ValueError`` when there is no evidence to classify; ``[]`` when there is
  evidence but nothing qualifies.

No LLM calls: matrices and results are constructed in memory.
"""

from __future__ import annotations

import inspect
import math
from datetime import UTC, datetime
from typing import Any

import pytest

from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.outcome_matrix import (
    build_outcome_matrix,
    select_failing_example_ids,
)

# ---------------------------------------------------------------------------
# Matrix helpers (the persisted outcome-matrix shape, see outcome_matrix.py)
# ---------------------------------------------------------------------------


def _cell(
    *,
    score: Any = None,
    accuracy: Any = None,
    success: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "score": score,
        "accuracy": accuracy,
        "metrics": {},
        "success": success,
        "tokens": None,
        "cost_usd": None,
        "execution_time": None,
        "latency_ms": None,
        "predicted": None,
        "error": error,
    }


def _matrix(
    trials: list[tuple[Any, dict[str, Any]]],
    rows: list[tuple[str, dict[Any, dict[str, Any]]]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "trial_count": len(trials),
        "example_count": len(rows),
        "trials": [
            {"index": i, "trial_id": trial_id, "config": config, "config_hash": ""}
            for i, (trial_id, config) in enumerate(trials)
        ],
        "examples": [
            {"example_id": example_id, "cells": cells} for example_id, cells in rows
        ],
    }


TWO_TRIALS = [("t0", {"model": "a"}), ("t1", {"model": "b"})]


def _both(cell_t0: dict[str, Any], cell_t1: dict[str, Any]) -> dict[str, Any]:
    return {"t0": cell_t0, "t1": cell_t1}


# ---------------------------------------------------------------------------
# Threshold and signal semantics
# ---------------------------------------------------------------------------


class TestThresholdSemantics:
    def test_score_below_threshold_fails_and_equality_passes(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("below", _both(_cell(score=0.49), _cell(score=0.1))),
                ("equal", _both(_cell(score=0.5), _cell(score=0.5))),
            ],
        )
        assert select_failing_example_ids(matrix, threshold=0.5) == ["below"]

    def test_default_threshold_is_half(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("below", _both(_cell(score=0.49), _cell(score=0.49))),
                ("equal", _both(_cell(score=0.5), _cell(score=0.5))),
            ],
        )
        assert select_failing_example_ids(matrix) == ["below"]

    def test_integer_threshold_is_accepted(self) -> None:
        matrix = _matrix(
            TWO_TRIALS, [("ex", _both(_cell(score=0.9), _cell(score=0.9)))]
        )
        assert select_failing_example_ids(matrix, threshold=1) == ["ex"]

    def test_accuracy_is_the_fallback_when_score_is_absent(self) -> None:
        # The local evaluator writes ``accuracy`` but no ``score`` metric, so
        # the matrix cell carries score=None (measured on a real offline run).
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("wrong", _both(_cell(accuracy=0.0), _cell(accuracy=0.0))),
                ("right", _both(_cell(accuracy=1.0), _cell(accuracy=1.0))),
            ],
        )
        assert select_failing_example_ids(matrix) == ["wrong"]

    def test_score_wins_over_accuracy_when_both_present(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                (
                    "score-passes",
                    _both(
                        _cell(score=0.9, accuracy=0.0), _cell(score=0.9, accuracy=0.0)
                    ),
                ),
                (
                    "score-fails",
                    _both(
                        _cell(score=0.1, accuracy=1.0), _cell(score=0.1, accuracy=1.0)
                    ),
                ),
            ],
        )
        assert select_failing_example_ids(matrix) == ["score-fails"]

    def test_success_true_with_a_wrong_answer_still_fails(self) -> None:
        # success=True only means the call executed; it is not correctness.
        matrix = _matrix(
            TWO_TRIALS,
            [("ex", _both(_cell(score=0.0, success=True), _cell(score=0.0)))],
        )
        assert select_failing_example_ids(matrix) == ["ex"]

    def test_error_overrides_a_passing_score(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                (
                    "errored",
                    _both(
                        _cell(score=1.0, success=False, error="boom"),
                        _cell(score=1.0, success=False, error="boom"),
                    ),
                )
            ],
        )
        assert select_failing_example_ids(matrix) == ["errored"]

    def test_error_without_any_score_is_a_failure(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                (
                    "errored",
                    _both(
                        _cell(success=False, error="timeout"),
                        _cell(success=False, error="timeout"),
                    ),
                )
            ],
        )
        assert select_failing_example_ids(matrix) == ["errored"]

    def test_explicit_success_false_overrides_a_passing_score(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                (
                    "failed",
                    _both(
                        _cell(score=1.0, success=False),
                        _cell(score=1.0, success=False),
                    ),
                )
            ],
        )
        assert select_failing_example_ids(matrix) == ["failed"]


class TestUnknownIsNotZero:
    @pytest.mark.parametrize(
        "unknown",
        [
            _cell(),  # successful execution, no quality signal at all
            _cell(score=math.nan),
            _cell(score="not-a-number"),
        ],
        ids=["no-signal", "nan", "non-numeric"],
    )
    def test_unknown_cell_excludes_the_example(self, unknown: dict[str, Any]) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("partly-unknown", _both(_cell(score=0.0), unknown)),
                ("failing", _both(_cell(score=0.0), _cell(score=0.0))),
            ],
        )
        assert select_failing_example_ids(matrix) == ["failing"]

    def test_missing_cell_excludes_the_example(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("missing-t1", {"t0": _cell(score=0.0)}),
                ("failing", _both(_cell(score=0.0), _cell(score=0.0))),
            ],
        )
        assert select_failing_example_ids(matrix) == ["failing"]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


class TestAllConfigsMode:
    def test_requires_every_column_to_fail(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("one-config-right", _both(_cell(score=0.0), _cell(score=1.0))),
                ("never-right", _both(_cell(score=0.0), _cell(score=0.2))),
            ],
        )
        assert select_failing_example_ids(matrix, mode="all_configs") == ["never-right"]

    def test_returns_empty_list_when_evidence_exists_but_nothing_fails(self) -> None:
        matrix = _matrix(
            TWO_TRIALS, [("ok", _both(_cell(score=1.0), _cell(score=1.0)))]
        )
        assert select_failing_example_ids(matrix) == []

    def test_preserves_matrix_row_order_and_returns_each_id_once(self) -> None:
        failing = _both(_cell(score=0.0), _cell(score=0.0))
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("ex-3", failing),
                ("ex-1", failing),
                ("ok", _both(_cell(score=1.0), _cell(score=1.0))),
                ("ex-2", failing),
                ("ex-1", failing),
            ],
        )
        assert select_failing_example_ids(matrix) == ["ex-3", "ex-1", "ex-2"]


class TestBestConfigMode:
    def test_only_the_winner_columns_count(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("winner-wrong", _both(_cell(score=0.0), _cell(score=1.0))),
                ("winner-right", _both(_cell(score=1.0), _cell(score=0.0))),
            ],
        )
        assert select_failing_example_ids(
            matrix, mode="best_config", best_config={"model": "a"}
        ) == ["winner-wrong"]
        # The same matrix selects nothing in all_configs mode.
        assert select_failing_example_ids(matrix, mode="all_configs") == []

    def test_config_equality_includes_every_replicate_of_the_winner(self) -> None:
        trials = [
            ("t0", {"model": "a"}),
            ("t1", {"model": "a"}),
            ("t2", {"model": "b"}),
        ]
        matrix = _matrix(
            trials,
            [
                (
                    "replicate-right",
                    {
                        "t0": _cell(score=0.0),
                        "t1": _cell(score=1.0),
                        "t2": _cell(score=0.0),
                    },
                ),
                (
                    "persistently-wrong",
                    {
                        "t0": _cell(score=0.0),
                        "t1": _cell(score=0.0),
                        "t2": _cell(score=1.0),
                    },
                ),
            ],
        )
        assert select_failing_example_ids(
            matrix, mode="best_config", best_config={"model": "a"}
        ) == ["persistently-wrong"]

    def test_winning_trial_ids_take_precedence_over_config_equality(self) -> None:
        # Replicates whose full config differs in an aux key (e.g. a seed) are
        # dropped by config equality; the stamped winner ids keep them (#1854).
        trials = [("t0", {"model": "a", "seed": 1}), ("t1", {"model": "a", "seed": 2})]
        matrix = _matrix(
            trials,
            [("ex", {"t0": _cell(score=0.0), "t1": _cell(score=1.0)})],
        )
        best = {"model": "a", "seed": 1}

        assert select_failing_example_ids(
            matrix, mode="best_config", best_config=best
        ) == ["ex"]
        assert (
            select_failing_example_ids(
                matrix,
                mode="best_config",
                best_config=best,
                winning_trial_ids=["t0", "t1"],
            )
            == []
        )

    def test_empty_best_config_is_a_real_winner(self) -> None:
        # best_config == {} must be matched by equality, not treated as "no
        # winner" by a truthiness test.
        matrix = _matrix(
            [("t0", {}), ("t1", {"x": 1})],
            [("ex", {"t0": _cell(score=0.0), "t1": _cell(score=1.0)})],
        )
        assert select_failing_example_ids(
            matrix, mode="best_config", best_config={}
        ) == ["ex"]

    def test_integer_trial_ids_match_string_cell_keys(self) -> None:
        # Local-session trial ids are ints, but JSON object keys reload as str.
        matrix = _matrix(
            [(1, {"model": "a"}), (2, {"model": "b"})],
            [("ex", {"1": _cell(score=0.0), "2": _cell(score=1.0)})],
        )
        assert select_failing_example_ids(
            matrix, mode="best_config", best_config=None, winning_trial_ids=[1]
        ) == ["ex"]

    def test_no_winner_raises(self) -> None:
        matrix = _matrix(
            TWO_TRIALS, [("ex", _both(_cell(score=0.0), _cell(score=0.0)))]
        )
        with pytest.raises(ValueError):
            select_failing_example_ids(matrix, mode="best_config", best_config=None)

    def test_winner_matching_no_column_raises(self) -> None:
        matrix = _matrix(
            TWO_TRIALS, [("ex", _both(_cell(score=0.0), _cell(score=0.0)))]
        )
        with pytest.raises(ValueError):
            select_failing_example_ids(
                matrix, mode="best_config", best_config={"model": "zzz"}
            )


# ---------------------------------------------------------------------------
# Evidence and argument validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_no_per_example_detail_raises(self) -> None:
        with pytest.raises(ValueError):
            select_failing_example_ids(_matrix(TWO_TRIALS, []))

    def test_no_usable_signal_anywhere_raises(self) -> None:
        matrix = _matrix(
            TWO_TRIALS,
            [
                ("a", _both(_cell(), _cell())),
                ("b", _both(_cell(score=math.nan), _cell())),
            ],
        )
        with pytest.raises(ValueError):
            select_failing_example_ids(matrix)

    def test_malformed_matrix_raises(self) -> None:
        with pytest.raises(ValueError):
            select_failing_example_ids({"trials": [], "examples": "not-a-list"})

    @pytest.mark.parametrize(
        "threshold",
        [True, math.nan, math.inf, -math.inf, "0.5"],
        ids=["bool", "nan", "inf", "-inf", "str"],
    )
    def test_invalid_threshold_is_rejected(self, threshold: Any) -> None:
        matrix = _matrix(
            TWO_TRIALS, [("ex", _both(_cell(score=0.0), _cell(score=0.0)))]
        )
        with pytest.raises((ValueError, TypeError)):
            select_failing_example_ids(matrix, threshold=threshold)

    def test_invalid_mode_is_rejected(self) -> None:
        matrix = _matrix(
            TWO_TRIALS, [("ex", _both(_cell(score=0.0), _cell(score=0.0)))]
        )
        with pytest.raises(ValueError):
            select_failing_example_ids(matrix, mode="any_config")


# ---------------------------------------------------------------------------
# OptimizationResult.failing_examples
# ---------------------------------------------------------------------------


def _example(example_id: str, accuracy: float) -> dict[str, Any]:
    """A real ``ExampleResult.to_dict()`` payload, as stored in trial metadata."""
    return ExampleResult(
        example_id=example_id,
        input_data={},
        expected_output="e",
        actual_output="a",
        metrics={"accuracy": accuracy},
        execution_time=0.1,
        success=True,
    ).to_dict()


def _trial(trial_id: str, config: dict[str, Any], accuracies: dict[str, float]):
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics={"accuracy": sum(accuracies.values()) / len(accuracies)},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={
            "example_results": [_example(ex, acc) for ex, acc in accuracies.items()]
        },
    )


def _result(
    trials: list[TrialResult],
    best_config: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        trials=trials,
        best_config=best_config,
        best_score=0.5,
        optimization_id="opt-2565",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata=metadata or {},
    )


def _three_example_result(metadata: dict[str, Any] | None = None):
    return _result(
        [
            _trial("t0", {"model": "a"}, {"ex-0": 1.0, "ex-1": 0.0, "ex-2": 0.0}),
            _trial("t1", {"model": "b"}, {"ex-0": 1.0, "ex-1": 0.0, "ex-2": 1.0}),
        ],
        best_config={"model": "a"},
        metadata=metadata,
    )


class TestOptimizationResultFailingExamples:
    def test_signature_defaults(self) -> None:
        params = inspect.signature(OptimizationResult.failing_examples).parameters
        assert params["threshold"].default == 0.5
        assert params["mode"].default == "all_configs"

    def test_defaults_select_examples_no_config_got_right(self) -> None:
        assert _three_example_result().failing_examples() == ["ex-1"]

    def test_best_config_mode_uses_the_result_winner(self) -> None:
        assert _three_example_result().failing_examples(mode="best_config") == [
            "ex-1",
            "ex-2",
        ]

    def test_best_config_mode_honors_stamped_winning_trial_ids(self) -> None:
        result = _three_example_result(
            metadata={"session_summary": {"winning_trial_ids": ["t1"]}}
        )
        assert result.failing_examples(mode="best_config") == ["ex-1"]

    def test_agrees_with_the_pure_selector_on_the_same_matrix(self) -> None:
        result = _three_example_result()
        matrix = build_outcome_matrix(result)
        for mode in ("all_configs", "best_config"):
            assert result.failing_examples(threshold=0.5, mode=mode) == (
                select_failing_example_ids(
                    matrix, threshold=0.5, mode=mode, best_config=result.best_config
                )
            )

    def test_best_config_mode_without_a_winner_raises(self) -> None:
        result = _result(
            [_trial("t0", {"model": "a"}, {"ex-0": 0.0})], best_config=None
        )
        with pytest.raises(ValueError):
            result.failing_examples(mode="best_config")

    def test_run_without_per_example_detail_raises(self) -> None:
        trial = TrialResult(
            trial_id="t0",
            config={"model": "a"},
            metrics={"accuracy": 0.5},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metadata={},
        )
        with pytest.raises(ValueError):
            _result([trial], best_config={"model": "a"}).failing_examples()
