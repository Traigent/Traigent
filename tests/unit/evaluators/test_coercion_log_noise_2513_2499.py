"""String-coercion and constraint-prune logging is quiet and content-free
(issues #2513, #2499)."""

import logging

from traigent.core.trial_result_factory import build_pruned_result
from traigent.evaluators.base import (
    _accuracy_values_match,
    coercion_log_scope,
)
from traigent.utils.exceptions import TrialPrunedError

SECRET_ACTUAL = "SYNTH_ACTUAL_MARKER"
SECRET_EXPECTED = "SYNTH_EXPECTED_MARKER"


def _coercion_records(caplog):
    return [r for r in caplog.records if "Coercing string output" in r.getMessage()]


def test_one_warning_per_trial_with_count_and_no_content(caplog):
    caplog.set_level(logging.DEBUG, logger="traigent")
    with coercion_log_scope():
        for _ in range(50):
            assert _accuracy_values_match(
                f'{{"k": "{SECRET_ACTUAL}"}}', {"k": SECRET_ACTUAL}
            )
    warnings = [r for r in _coercion_records(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "50" in warnings[0].getMessage()
    assert SECRET_ACTUAL not in caplog.text
    # per-example detail is DEBUG only
    assert all(
        r.levelno == logging.DEBUG
        for r in _coercion_records(caplog)
        if r is not warnings[0]
    )


def test_numeric_branch_logs_no_content_and_keeps_equality(caplog):
    caplog.set_level(logging.DEBUG, logger="traigent")
    assert _accuracy_values_match("1.0", "1") is True
    assert _accuracy_values_match("1.5", "2") is False
    text = caplog.text
    assert "numeric" in text
    assert "1.5" not in text


def test_marker_content_never_logged_for_string_pairs(caplog):
    caplog.set_level(logging.DEBUG, logger="traigent")
    _accuracy_values_match(SECRET_ACTUAL, SECRET_EXPECTED)
    _accuracy_values_match('{"a": 1}', {"a": 1})
    assert SECRET_ACTUAL not in caplog.text
    assert SECRET_EXPECTED not in caplog.text


def test_constraint_rejected_prune_is_info_not_warning(caplog):
    caplog.set_level(logging.INFO, logger="traigent")
    build_pruned_result(
        trial_id="t_rej1",
        evaluation_config={},
        duration=0.0,
        prune_error=TrialPrunedError("trial_rejected_by_constraint: x", step=0),
        progress_state={"evaluated": 0, "total_examples": 3},
        optuna_trial_id=None,
    )
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "skipped by constraint" in caplog.text


def test_other_prune_without_results_warns_once(caplog):
    caplog.set_level(logging.DEBUG, logger="traigent")
    import traigent.core.trial_result_factory as trf

    trf._PRUNED_WITHOUT_RESULTS_COUNT = 0
    for i in range(3):
        build_pruned_result(
            trial_id=f"t{i}",
            evaluation_config={},
            duration=0.0,
            prune_error=TrialPrunedError("median pruner", step=1),
            progress_state=None,
            optuna_trial_id=None,
        )
    recs = [r for r in caplog.records if "No example_results" in r.getMessage()]
    assert [r.levelno for r in recs] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
