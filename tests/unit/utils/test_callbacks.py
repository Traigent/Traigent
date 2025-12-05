"""Regression tests for callback utilities."""

import logging
from types import SimpleNamespace

import pytest

from traigent.utils.callbacks import DetailedProgressCallback, ProgressInfo


@pytest.mark.parametrize(
    "config_space,expected_total",
    [
        ({"model": []}, 0),
        ({"temperature": [0.1, 0.2]}, 2),
    ],
)
def test_detailed_progress_callback_handles_zero_total(
    config_space, expected_total, capsys, caplog
):
    """DetailedProgressCallback should not divide by zero when total trials is zero."""
    callback = DetailedProgressCallback()
    caplog.set_level(logging.WARNING, logger="traigent.utils.callbacks")

    callback.on_optimization_start(config_space, ["accuracy"], "grid")
    assert callback.total_trials == expected_total

    trial = SimpleNamespace(status="completed", metrics={"accuracy": 0.42})
    progress = ProgressInfo(
        current_trial=0,
        total_trials=0,
        completed_trials=1,
        successful_trials=1,
        failed_trials=0,
        best_score=0.42,
        best_config=None,
        elapsed_time=1.0,
        estimated_remaining=None,
        current_algorithm="grid",
    )

    callback.on_trial_start(0, {"model": "gpt-4"})
    callback.on_trial_complete(trial, progress)

    output = capsys.readouterr().out
    assert "Progress:" in output
    assert "%" in output

    if expected_total == 0:
        assert any(
            "configuration combinations is zero" in record.message
            for record in caplog.records
        )
