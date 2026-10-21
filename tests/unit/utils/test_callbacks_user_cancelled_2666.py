"""User-cancelled runs are announced as cancelled/partial, never timeout/complete (#2666)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from io import StringIO
from unittest.mock import Mock

from rich.console import Console

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.utils.callbacks import (
    DetailedProgressCallback,
    LoggingCallback,
    ProgressBarCallback,
    SimpleProgressCallback,
)


def _result(stop_reason: str | None = "user_cancelled") -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"model": "m"},
        best_score=0.5,
        optimization_id="opt_1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.CANCELLED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(UTC),
        stop_reason=stop_reason,
    )


def test_detailed_header_says_cancelled_not_timeout(capsys) -> None:
    DetailedProgressCallback().on_optimization_complete(_result())
    out = capsys.readouterr().out
    assert "CANCELLED BY USER" in out
    assert "TIMEOUT" not in out.upper()
    assert "OPTIMIZATION COMPLETE" not in out


def test_progress_bar_says_cancelled_and_partial(capsys) -> None:
    cb = ProgressBarCallback()
    cb._last_progress = None
    cb.on_optimization_complete(_result())
    out = capsys.readouterr().out
    assert "CANCELLED BY USER" in out
    assert "partial" in out
    assert "timeout" not in out.lower()
    assert "Optimization complete" not in out


def test_simple_progress_says_cancelled(capsys) -> None:
    SimpleProgressCallback().on_optimization_complete(_result())
    out = capsys.readouterr().out
    assert "CANCELLED BY USER" in out
    assert "complete!" not in out


def test_logging_callback_says_cancelled(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        LoggingCallback(log_level="INFO").on_optimization_complete(_result())
    assert "CANCELLED BY USER" in caplog.text
    assert "Optimization complete" not in caplog.text


def test_timeout_still_labelled_timeout(capsys) -> None:
    DetailedProgressCallback().on_optimization_complete(_result("timeout"))
    assert "TIMEOUT REACHED" in capsys.readouterr().out


def test_cli_output_says_cancelled_and_partial(monkeypatch, tmp_path) -> None:
    import traigent.cli.main as cli_main

    buf = StringIO()
    monkeypatch.setattr(cli_main, "console", Console(file=buf, width=200))
    res = _result()
    persistence = Mock()
    cli_main._display_optimization_result("fn", res, persistence, "grid", 3)
    cli_main._display_optimization_summary([("fn", res)], tmp_path)
    out = buf.getvalue()
    assert "Cancelled by user" in out
    assert "partial" in out
    assert "CANCELLED BY USER" in out
    assert "Optimization Complete" not in out
    assert "✅ Best score" not in out


def test_cli_completed_run_unchanged(monkeypatch, tmp_path) -> None:
    import traigent.cli.main as cli_main

    buf = StringIO()
    monkeypatch.setattr(cli_main, "console", Console(file=buf, width=200))
    res = _result("max_trials_reached")
    cli_main._display_optimization_result("fn", res, Mock(), "grid", 3)
    cli_main._display_optimization_summary([("fn", res)], tmp_path)
    out = buf.getvalue()
    assert "Optimization Complete" in out
    assert "Cancelled by user" not in out
