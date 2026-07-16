"""Tests for the eval-dataset defect detectors (issue #1880).

Covers the three detectors as pure functions of the outcome matrix (#1838):
- D1 never-correct: flags an item wrong in every config, not an always-correct one;
- D3 token-leak: flags a token outlier among the always-correct cohort;
- D7 cross-family consensus-on-wrong: flags a same-wrong-answer agreement across
  >=2 families (with ``suggested_answer``), stays inactive for single-family runs;
- the summary counts + lift;
- opt-in behaviour: absent (``None``) with <2 configs or no per-example detail,
  including through the ``OptimizationResult.eval_audit`` property end to end.

No LLM calls: matrices are constructed in memory / from real ExampleResult shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from traigent.api.types import (
    EvalAudit,
    ExampleResult,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.utils.eval_audit import (
    UNKNOWN_FAMILY,
    compute_eval_audit,
    model_family,
)
from traigent.utils.outcome_matrix import build_outcome_matrix

# ---------------------------------------------------------------------------
# Matrix builders (test the pure function directly)
# ---------------------------------------------------------------------------


def _cell(
    *,
    correct: bool,
    tokens: int | None = None,
    predicted: Any = None,
) -> dict[str, Any]:
    """One outcome cell in the shape build_outcome_matrix emits."""
    return {
        "score": 1.0 if correct else 0.0,
        "accuracy": 1.0 if correct else 0.0,
        "metrics": {},
        "success": correct,
        "tokens": {"total": tokens} if tokens is not None else None,
        "cost_usd": None,
        "execution_time": None,
        "latency_ms": None,
        "predicted": predicted,
        "error": None if correct else "wrong",
    }


def _matrix(
    trials: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "optimization_id": "opt-1880",
        "algorithm": "GridSearchOptimizer",
        "objectives": ["accuracy"],
        "created_at": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        "trial_count": len(trials),
        "example_count": len(examples),
        "trials": trials,
        "examples": examples,
    }


def _column(trial_id: str, model: str) -> dict[str, Any]:
    return {
        "index": 0,
        "trial_id": trial_id,
        "config": {"model": model},
        "config_hash": trial_id,
    }


# ---------------------------------------------------------------------------
# D1 never-correct
# ---------------------------------------------------------------------------


def test_d1_flags_never_correct_not_always_correct() -> None:
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "gpt-4o-mini")],
        [
            {
                "example_id": "ex-never",
                "cells": {
                    "t0": _cell(correct=False),
                    "t1": _cell(correct=False),
                },
            },
            {
                "example_id": "ex-always",
                "cells": {
                    "t0": _cell(correct=True),
                    "t1": _cell(correct=True),
                },
            },
            {
                "example_id": "ex-mixed",
                "cells": {
                    "t0": _cell(correct=True),
                    "t1": _cell(correct=False),
                },
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    flagged_by = {
        f.example_id: f.detectors
        for f in audit.flagged
        if "never-correct" in f.detectors
    }
    assert "ex-never" in flagged_by
    assert "ex-always" not in flagged_by
    assert "ex-mixed" not in flagged_by
    d1 = audit.summary.detectors["never-correct"]
    assert d1.active is True
    assert d1.flagged_count == 1
    assert d1.eligible_count == 3
    # 1/3 observed vs p_wrong**2 null; p_wrong = 3 wrong / 6 cells = 0.5 -> lift = (1/3)/0.25
    assert d1.lift is not None and d1.lift > 1.0


def test_d1_single_model_grid_still_works() -> None:
    """Single-model grid (two configs of one model) still yields D1."""
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "gpt-4o")],
        [
            {
                "example_id": "ex-never",
                "cells": {"t0": _cell(correct=False), "t1": _cell(correct=False)},
            },
            {
                "example_id": "ex-ok",
                "cells": {"t0": _cell(correct=True), "t1": _cell(correct=True)},
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    assert audit.summary.detectors["never-correct"].flagged_count == 1
    # single family -> D7 inactive
    assert audit.summary.detectors["cross-family-consensus-on-wrong"].active is False


def test_d1_excludes_sparse_row() -> None:
    """FIX 3 (#1880 rework): the >=2-config gate guarantees >=2 *columns* exist,
    not that every *row* ran in all of them. A sparse row (example run in only one
    trial) has a single determinable cell -> excluded from D1 eligibility, never
    crashes, never flagged."""
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "gpt-4o-mini")],
        [
            {
                # ran in one column only -> sparse row
                "example_id": "ex-sparse",
                "cells": {"t0": _cell(correct=False)},
            },
            {
                "example_id": "ex-never",
                "cells": {"t0": _cell(correct=False), "t1": _cell(correct=False)},
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    flagged = {f.example_id for f in audit.flagged if "never-correct" in f.detectors}
    assert flagged == {"ex-never"}
    assert "ex-sparse" not in flagged
    d1 = audit.summary.detectors["never-correct"]
    # only ex-never is eligible; the sparse single-cell row is excluded
    assert d1.eligible_count == 1
    assert d1.flagged_count == 1


# ---------------------------------------------------------------------------
# D3 token-leak
# ---------------------------------------------------------------------------


def test_d3_flags_token_outlier_among_correct() -> None:
    tokens = {"a": 100, "b": 110, "c": 120, "d": 130, "leak": 1000}
    examples = [
        {
            "example_id": f"ex-{name}",
            "cells": {
                "t0": _cell(correct=True, tokens=tok),
                "t1": _cell(correct=True, tokens=tok),
            },
        }
        for name, tok in tokens.items()
    ]
    matrix = _matrix([_column("t0", "gpt-4o"), _column("t1", "gpt-4o")], examples)
    audit = compute_eval_audit(matrix)
    assert audit is not None
    flagged = {f.example_id for f in audit.flagged if "token-leak" in f.detectors}
    assert flagged == {"ex-leak"}
    d3 = audit.summary.detectors["token-leak"]
    assert d3.active is True
    assert d3.flagged_count == 1
    assert d3.eligible_count == 5
    # token-burn multiple: 1000 / median(120) ~= 8.3x
    assert d3.lift is not None and d3.lift > 5.0


def test_d3_flags_outlier_in_degenerate_zero_iqr_cohort() -> None:
    """FIX 2 (#1880 rework): a clustered always-correct cohort whose IQR == 0
    (Q1 == Q3) still catches a clear token outlier via the median-multiple
    fallback. Before the fix the ``iqr > 0`` guard silently dropped it."""
    tokens = {"a": 100, "b": 100, "c": 100, "d": 100, "leak": 1000}
    examples = [
        {
            "example_id": f"ex-{name}",
            "cells": {
                "t0": _cell(correct=True, tokens=tok),
                "t1": _cell(correct=True, tokens=tok),
            },
        }
        for name, tok in tokens.items()
    ]
    matrix = _matrix([_column("t0", "gpt-4o"), _column("t1", "gpt-4o")], examples)
    audit = compute_eval_audit(matrix)
    assert audit is not None
    flagged = {f.example_id for f in audit.flagged if "token-leak" in f.detectors}
    assert flagged == {"ex-leak"}
    d3 = audit.summary.detectors["token-leak"]
    assert d3.active is True
    assert d3.flagged_count == 1
    assert d3.eligible_count == 5
    # 1000 / median(100) = 10x burn multiple
    assert d3.lift is not None and d3.lift > 5.0


def test_d3_skipped_when_too_few_always_correct() -> None:
    examples = [
        {
            "example_id": f"ex-{i}",
            "cells": {
                "t0": _cell(correct=True, tokens=100 + i),
                "t1": _cell(correct=True, tokens=100 + i),
            },
        }
        for i in range(2)
    ]
    matrix = _matrix([_column("t0", "gpt-4o"), _column("t1", "gpt-4o")], examples)
    audit = compute_eval_audit(matrix)
    assert audit is not None
    d3 = audit.summary.detectors["token-leak"]
    assert d3.active is False
    assert d3.flagged_count == 0
    assert d3.note is not None and "robust IQR" in d3.note


# ---------------------------------------------------------------------------
# D7 cross-family consensus-on-wrong
# ---------------------------------------------------------------------------


def test_d7_flags_cross_family_consensus_on_wrong() -> None:
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "claude-3-opus")],
        [
            {
                "example_id": "ex-consensus",
                "cells": {
                    "t0": _cell(correct=False, predicted="B"),
                    "t1": _cell(correct=False, predicted="B"),
                },
            },
            {
                "example_id": "ex-disagree",
                "cells": {
                    "t0": _cell(correct=False, predicted="A"),
                    "t1": _cell(correct=False, predicted="C"),
                },
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    consensus = {
        f.example_id: f.suggested_answer
        for f in audit.flagged
        if "cross-family-consensus-on-wrong" in f.detectors
    }
    assert consensus == {"ex-consensus": "B"}
    d7 = audit.summary.detectors["cross-family-consensus-on-wrong"]
    assert d7.active is True
    assert d7.flagged_count == 1
    assert audit.summary.family_count == 2


def test_d7_inactive_for_single_family() -> None:
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "gpt-4o-mini")],
        [
            {
                "example_id": "ex-consensus",
                "cells": {
                    "t0": _cell(correct=False, predicted="B"),
                    "t1": _cell(correct=False, predicted="B"),
                },
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    d7 = audit.summary.detectors["cross-family-consensus-on-wrong"]
    assert d7.active is False
    assert d7.flagged_count == 0
    assert audit.summary.family_count == 1
    assert not any(
        "cross-family-consensus-on-wrong" in f.detectors for f in audit.flagged
    )


def test_d7_agreement_needs_two_families_not_two_configs() -> None:
    """Same wrong answer across two configs of the *same* family is not D7."""
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "gpt-3.5-turbo")],
        [
            {
                "example_id": "ex",
                "cells": {
                    "t0": _cell(correct=False, predicted="B"),
                    "t1": _cell(correct=False, predicted="B"),
                },
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    assert audit.summary.family_count == 1
    assert audit.summary.detectors["cross-family-consensus-on-wrong"].active is False


def test_d7_answer_key_normalization_forms_consensus() -> None:
    """#1880 rework: ``_answer_key`` normalization drives D7. Two families whose
    raw answers are ``' B '`` and ``'b'`` normalize to the same key (trim +
    lowercase) and therefore count as the SAME wrong answer -> consensus forms."""
    matrix = _matrix(
        [_column("t0", "gpt-4o"), _column("t1", "claude-3-opus")],
        [
            {
                "example_id": "ex",
                "cells": {
                    "t0": _cell(correct=False, predicted=" B "),
                    "t1": _cell(correct=False, predicted="b"),
                },
            },
        ],
    )
    audit = compute_eval_audit(matrix)
    assert audit is not None
    by_id = {f.example_id: f for f in audit.flagged}
    assert "cross-family-consensus-on-wrong" in by_id["ex"].detectors
    d7 = audit.summary.detectors["cross-family-consensus-on-wrong"]
    assert d7.active is True
    assert d7.flagged_count == 1


# ---------------------------------------------------------------------------
# Model-family mapping
# ---------------------------------------------------------------------------


def test_model_family_mapping() -> None:
    assert model_family("gpt-4o") == "openai"
    assert model_family("openai/gpt-4o-mini") == "openai"
    assert model_family("claude-3-opus") == "anthropic"
    assert model_family("anthropic/claude-3-5-sonnet") == "anthropic"
    assert model_family("gemini-1.5-pro") == "google"
    assert model_family("meta-llama/Llama-3-70b") == "meta"
    assert model_family("mistral-large") == "mistral"
    assert model_family("deepseek-chat") == "deepseek"
    assert model_family("grok-2") == "xai"
    assert model_family("some-unknown-model") == UNKNOWN_FAMILY
    assert model_family("") == UNKNOWN_FAMILY
    assert model_family(None) == UNKNOWN_FAMILY


# ---------------------------------------------------------------------------
# Opt-in / applicability
# ---------------------------------------------------------------------------


def test_audit_none_for_single_config() -> None:
    matrix = _matrix(
        [_column("t0", "gpt-4o")],
        [{"example_id": "ex", "cells": {"t0": _cell(correct=False)}}],
    )
    assert compute_eval_audit(matrix) is None


def test_audit_none_for_no_examples() -> None:
    matrix = _matrix([_column("t0", "gpt-4o"), _column("t1", "gpt-4o")], [])
    assert compute_eval_audit(matrix) is None


def test_audit_none_for_missing_matrix() -> None:
    assert compute_eval_audit(None) is None
    assert compute_eval_audit({}) is None


# ---------------------------------------------------------------------------
# End-to-end through OptimizationResult.eval_audit (real ExampleResult shapes)
# ---------------------------------------------------------------------------


def _example(
    example_id: str,
    *,
    correct: bool,
    predicted: str,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": 1.0 if correct else 0.0,
        "score": 1.0 if correct else 0.0,
    }
    if total_tokens is not None:
        metrics["total_tokens"] = total_tokens
    return ExampleResult(
        example_id=example_id,
        input_data={},
        expected_output="GOLD",
        actual_output=predicted,
        metrics=metrics,
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


def test_eval_audit_property_end_to_end() -> None:
    """The property builds the matrix (incl. predicted) and runs detectors."""
    trial_openai = _trial(
        "t0",
        "gpt-4o",
        [
            _example("ex-consensus", correct=False, predicted="B"),
            _example("ex-never", correct=False, predicted="X"),
        ],
    )
    trial_anthropic = _trial(
        "t1",
        "claude-3-opus",
        [
            _example("ex-consensus", correct=False, predicted="B"),
            _example("ex-never", correct=False, predicted="Y"),
        ],
    )
    result = OptimizationResult(
        trials=[trial_openai, trial_anthropic],
        best_config={"model": "gpt-4o"},
        best_score=0.5,
        optimization_id="opt-e2e",
        duration=2.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    audit = result.eval_audit
    assert isinstance(audit, EvalAudit)
    by_id = {f.example_id: f for f in audit.flagged}
    # ex-consensus: wrong in both, cross-family same answer "B", also never-correct
    assert "never-correct" in by_id["ex-consensus"].detectors
    assert "cross-family-consensus-on-wrong" in by_id["ex-consensus"].detectors
    assert by_id["ex-consensus"].suggested_answer == "B"
    # ex-never: wrong in both but different answers -> never-correct only
    assert by_id["ex-never"].detectors == ["never-correct"]
    assert by_id["ex-never"].suggested_answer is None
    # to_dict round-trips and omits absent suggested_answer
    d = audit.to_dict()
    assert d["summary"]["config_count"] == 2
    assert "suggested_answer" not in by_id["ex-never"].to_dict()


def test_predicted_stored_only_on_non_correct_cells() -> None:
    """FIX 1 (#1880 rework): ``build_outcome_matrix`` records ``predicted`` only on
    non-correct cells (``success is not True``). A clearly-correct cell carries a
    null ``predicted``; a wrong cell keeps its answer verbatim; and D7 still forms
    cross-family consensus off those retained wrong answers (so the pruning does
    not reduce D7's inputs)."""
    trial_openai = _trial(
        "t0",
        "gpt-4o",
        [
            _example("ex-correct", correct=True, predicted="GOLD"),
            _example("ex-wrong", correct=False, predicted="B"),
        ],
    )
    trial_anthropic = _trial(
        "t1",
        "claude-3-opus",
        [
            _example("ex-correct", correct=True, predicted="GOLD"),
            _example("ex-wrong", correct=False, predicted="B"),
        ],
    )
    result = OptimizationResult(
        trials=[trial_openai, trial_anthropic],
        best_config={"model": "gpt-4o"},
        best_score=0.5,
        optimization_id="opt-predicted",
        duration=2.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    matrix = build_outcome_matrix(result)
    cells = {e["example_id"]: e["cells"] for e in matrix["examples"]}
    # clearly-correct cells: predicted pruned to null (no output bloat persisted)
    assert cells["ex-correct"]["t0"]["predicted"] is None
    assert cells["ex-correct"]["t1"]["predicted"] is None
    # wrong cells: predicted retained verbatim (D7's input)
    assert cells["ex-wrong"]["t0"]["predicted"] == "B"
    assert cells["ex-wrong"]["t1"]["predicted"] == "B"
    # D7 still forms consensus off the retained wrong answers
    audit = compute_eval_audit(matrix)
    assert audit is not None
    by_id = {f.example_id: f for f in audit.flagged}
    assert "cross-family-consensus-on-wrong" in by_id["ex-wrong"].detectors
    assert by_id["ex-wrong"].suggested_answer == "B"


def test_eval_audit_property_none_single_config() -> None:
    result = OptimizationResult(
        trials=[_trial("t0", "gpt-4o", [_example("ex", correct=False, predicted="B")])],
        best_config={"model": "gpt-4o"},
        best_score=0.0,
        optimization_id="opt-single",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert result.eval_audit is None
