"""Round-4 hardening probes for the tuple-return metrics channel (captain repro).

These pin the two captain-verified BLOCKERS that survived rounds G1-G4 + R1-R2.
Written RED-first: each asserts the *fixed* behavior so it fails against the
pre-fix code.

* H1 — ``_evaluator_computable_metric_names()`` returned ``_metric_registry`` ∪
  ``_ragas_metric_names`` but NOT ``self.metric_functions`` keys (user-passed
  custom metric functions, never registered into ``_metric_registry``). So a
  user tuple key named e.g. ``"f1"`` (matching a custom metric function) passed
  the merge filter and could overwrite the evaluator-computed value. The fix
  also folds ``metric_functions`` keys into the computable set, which threads
  through every merge site and lane cap that already takes ``extra_reserved``.
* H2 — ``RESERVED_METRIC_KEYS`` omitted the TRANSPORT-reserved keys ``"measures"``
  and ``"summary_stats"``. ``_unpack_user_metrics`` accepted them (valid
  identifier + numeric value), they rode into trial metrics, and then the
  submission path broke: ``_extract_measures_from_metrics`` extracts them by name
  as top-level payload fields and ``_validate_metrics_no_misplaced_fields``
  hard-raises ``ValueError`` for either key in a metrics dict. The fix reserves
  both names so user tuple keys with these names are skipped at merge.
* H3 — the final factory cap (``build_success_result`` →
  ``enforce_user_metric_ceiling``) passed NO ``extra_reserved``, so under ceiling
  pressure evaluator-computed runtime names (``"f1"``, ``"context_precision"``,
  …) were treated as droppable user keys and the primary objective could be
  dropped from ``TrialResult.metrics``. The fix threads the evaluator's runtime
  computable names into the final cap.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from traigent.cloud.validators import (
    _validate_metrics_no_misplaced_fields,
    validate_configuration_run_submission,
)
from traigent.core.trial_result_factory import build_success_result
from traigent.evaluators.base import SimpleScoringEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import RESERVED_METRIC_KEYS


def _custom_f1(output: object, expected: object) -> float:
    return 1.0


# ---------------------------------------------------------------------------
# H1: metric_functions names are evaluator-computable (reserved)
# ---------------------------------------------------------------------------


def test_computable_names_include_metric_functions_simple_scoring() -> None:
    """H1: SimpleScoringEvaluator folds metric_functions keys into computable set.

    Pre-fix: ``_evaluator_computable_metric_names()`` is registry ∪ RAGAS only,
    so a custom metric function named ``f1`` is absent and a user tuple key
    ``f1`` would overwrite the evaluator's computed value.
    """
    evaluator = SimpleScoringEvaluator(metric_functions={"f1": _custom_f1})
    assert "f1" in evaluator._evaluator_computable_metric_names()


def test_computable_names_include_metric_functions_local_evaluator() -> None:
    """H1: LocalEvaluator (inherits the base method) folds metric_functions keys."""
    evaluator = LocalEvaluator(
        metric_functions={"f1": _custom_f1},
        detailed=True,
        execution_mode="local",
    )
    assert "f1" in evaluator._evaluator_computable_metric_names()


# ---------------------------------------------------------------------------
# H1/H2: merge site skips metric_functions name and transport-reserved names
# ---------------------------------------------------------------------------


def test_merge_skips_user_key_colliding_with_metric_functions_name() -> None:
    """H1: a user tuple key colliding with a metric_functions name is SKIPPED.

    ``extra_reserved`` is derived from the evaluator's computable names — pre-fix
    that set excludes ``f1`` (a metric_functions key never registered into
    ``_metric_registry``), so the user's value would merge into an EMPTY target
    and overwrite the slot the evaluator will later fill. With ``target`` empty,
    the only thing that can keep ``f1`` out is the reserved-name skip, so this is
    a true probe of the computable-names fix (not the "already present" guard).
    """
    evaluator = SimpleScoringEvaluator(metric_functions={"f1": _custom_f1})
    target: dict[str, float] = {}
    evaluator._merge_user_metrics(
        target,
        {"f1": 0.99, "composite_ok": 1.0},
        context="h1 merge",
        extra_reserved=evaluator._evaluator_computable_metric_names(),
    )
    # f1 is evaluator-computable → user value skipped (never lands in target).
    assert "f1" not in target
    # A genuine user key still merges.
    assert target["composite_ok"] == pytest.approx(1.0)


def test_merge_skips_transport_reserved_keys_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """H2: ``_merge_user_metrics`` skips ``measures``/``summary_stats`` (reserved).

    Pre-fix: these names are not in RESERVED_METRIC_KEYS and ride into metrics,
    later breaking the submission path. Fixed: they are skipped at merge with a
    WARNING (the per-example merge surfaces genuine user collisions).
    """
    evaluator = SimpleScoringEvaluator()
    target: dict[str, float] = {}
    with caplog.at_level(logging.WARNING):
        evaluator._merge_user_metrics(
            target,
            {"measures": 1.0, "summary_stats": 2.0, "composite_ok": 3.0},
            context="h2 merge",
        )
    assert "measures" not in target
    assert "summary_stats" not in target
    assert target["composite_ok"] == pytest.approx(3.0)
    warnings = " ".join(rec.message for rec in caplog.records)
    assert "measures" in warnings
    assert "summary_stats" in warnings


def test_transport_reserved_merge_keeps_submission_valid() -> None:
    """H2 end-to-end shape: post-merge metrics carry no transport key, so the
    submission validator that hard-rejects ``measures`` in a metrics dict passes.
    """
    evaluator = SimpleScoringEvaluator()
    metrics: dict[str, float] = {"accuracy": 0.9}
    evaluator._merge_user_metrics(
        metrics,
        {"measures": 1.0, "summary_stats": 2.0},
        context="h2 e2e",
    )
    assert "measures" not in metrics
    assert "summary_stats" not in metrics
    # Validation that previously raised ValueError("metrics dict should not
    # contain 'measures'") now passes because the transport keys never landed.
    assert (
        validate_configuration_run_submission(
            {"configuration_id": "cfg-1", "metrics": metrics}
        )
        is True
    )


# ---------------------------------------------------------------------------
# H3: final factory cap protects evaluator-computed runtime names
# ---------------------------------------------------------------------------


def _eval_result_with_metrics(metrics: dict[str, float]) -> Mock:
    result = Mock()
    result.metrics = dict(metrics)
    result.success_rate = 1.0
    result.has_errors = False
    result.outputs = []
    result.example_results = None
    result.successful_examples = 0
    result.summary_stats = None
    return result


def test_final_cap_keeps_evaluator_computed_name_drops_user_key() -> None:
    """H3 (captain repro): 49 composite_metric_* user keys + ``f1`` arrive at the
    ceiling from the lane, then assembly writes ``examples_attempted`` and
    ``total_cost`` AFTER the lane caps ran. Assembly ALSO materializes the reserved
    ``cost`` metric from ``total_cost`` when it is otherwise absent (#1423), so the
    final union is 53. The cap must fire, KEEP ``f1`` (the primary objective, via
    ``extra_reserved``) and the reserved assembly keys, and drop composite user
    keys instead.

    Pre-fix: ``build_success_result`` passed no ``extra_reserved``, so ``f1`` was
    seen as a droppable user key and (sorting after the composites) the cap
    removed it — the objective vanished from ``TrialResult.metrics``. The
    non-``None`` ``examples_attempted``/``total_cost`` are essential: they (plus
    the derived ``cost``) push the union past the ceiling so the cap is exercised.
    """
    metrics = {f"composite_metric_{i:03d}": 1.0 for i in range(49)}
    metrics["f1"] = 0.93
    eval_result = _eval_result_with_metrics(metrics)

    result = build_success_result(
        trial_id="trial_h3",
        evaluation_config={"model": "gpt-4"},
        eval_result=eval_result,
        duration=1.0,
        examples_attempted=2,
        total_cost=0.01,
        optuna_trial_id=None,
        extra_reserved=frozenset({"f1"}),
    )

    # The cap actually fired: 53-key union (49 composites + f1 + examples_attempted
    # + total_cost + the derived ``cost``) clamped back to the ceiling.
    assert len(result.metrics) == 50
    # The evaluator-computed objective survived...
    assert result.metrics["f1"] == pytest.approx(0.93)
    # ...as did the reserved assembly keys written after the lane caps...
    assert "examples_attempted" in result.metrics
    assert "total_cost" in result.metrics
    # ...including the per-config ``cost`` derived from ``total_cost`` (#1423)...
    assert result.metrics["cost"] == pytest.approx(0.01)
    # ...and exactly THREE composite user keys were dropped to make room.
    composite_kept = [k for k in result.metrics if k.startswith("composite_metric_")]
    assert len(composite_kept) == 46


def test_final_cap_without_extra_reserved_would_drop_objective() -> None:
    """H3 counterfactual guard: the same 53-key union WITHOUT ``extra_reserved``
    drops ``f1`` — proving the threaded set is what protects the objective (if a
    refactor silently stops threading it, the positive test above could go
    vacuous; this pins the mechanism). The union is 53 because assembly also
    derives the reserved ``cost`` metric from ``total_cost`` (#1423).
    """
    metrics = {f"composite_metric_{i:03d}": 1.0 for i in range(49)}
    metrics["f1"] = 0.93
    eval_result = _eval_result_with_metrics(metrics)

    result = build_success_result(
        trial_id="trial_h3_cf",
        evaluation_config={"model": "gpt-4"},
        eval_result=eval_result,
        duration=1.0,
        examples_attempted=2,
        total_cost=0.01,
        optuna_trial_id=None,
    )

    assert len(result.metrics) == 50
    assert "f1" not in result.metrics


# ---------------------------------------------------------------------------
# H2 consistency: every misplaced-field name is reserved
# ---------------------------------------------------------------------------


def test_misplaced_submission_fields_are_reserved() -> None:
    """H2 (consistency): every field rejected by
    ``_validate_metrics_no_misplaced_fields`` (``measures``, ``summary_stats``)
    is in RESERVED_METRIC_KEYS, so the merge filter skips them before they can
    ever reach a metrics dict the submission validator rejects.

    Mirrors ``test_user_metric_pattern_matches_measuresdict_pattern``: if the
    transport contract grows a new misplaced field, THIS test fails rather than
    the bypass silently reopening.
    """
    import inspect

    # The validator enumerates the misplaced fields in a literal tuple; pull the
    # names straight from its source so this stays pinned to the validator.
    src = inspect.getsource(_validate_metrics_no_misplaced_fields)
    for field in ("measures", "summary_stats"):
        assert f'"{field}"' in src, (
            f"{field!r} no longer rejected by validator — update this test"
        )
        assert field in RESERVED_METRIC_KEYS, (
            f"{field!r} is rejected in a metrics dict by the submission "
            f"validator but is NOT in RESERVED_METRIC_KEYS"
        )
