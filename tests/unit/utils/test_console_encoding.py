"""Tests for safe console output on non-UTF-8 (e.g. Windows cp1252) consoles.

Regression tests for issue #1321: UnicodeEncodeError when printing progress
output on Windows systems with a cp1252 or other narrow-encoding console.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-API-ENTRY

from __future__ import annotations

import io
import sys
from typing import Any
from unittest.mock import patch

import pytest

from traigent.utils.callbacks import (
    DetailedProgressCallback,
    ProgressBarCallback,
    ProgressInfo,
    _safe_print,
)
from traigent.utils.console import _safe_print as console_safe_print
from traigent.utils.console import configure_stdout_encoding
from traigent.utils.diagnostics import DiagnosticReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Cp1252StdOut:
    """Simulates a Windows cp1252 console that rejects characters outside cp1252.

    Characters outside cp1252 (e.g. box-drawing chars, emoji) will raise
    ``UnicodeEncodeError`` on ``write()``, reproducing the Windows bug.
    """

    encoding = "cp1252"
    errors = "strict"

    def __init__(self) -> None:
        self._buf: list[str] = []

    def write(self, s: str) -> int:
        # Encode with strict errors to reproduce the Windows crash
        s.encode(self.encoding, errors=self.errors)
        self._buf.append(s)
        return len(s)

    def flush(self) -> None:
        pass

    @property
    def value(self) -> str:
        return "".join(self._buf)


def _make_progress(
    completed: int = 1,
    total: int = 3,
    best_score: float | None = 0.85,
) -> ProgressInfo:
    return ProgressInfo(
        current_trial=completed,
        total_trials=total,
        completed_trials=completed,
        successful_trials=completed,
        failed_trials=0,
        best_score=best_score,
        best_config={"model": "gpt-4o"},
        elapsed_time=10.0,
        estimated_remaining=20.0,
        current_algorithm="grid",
    )


def _make_trial(status: str = "completed") -> Any:
    """Return a minimal trial-like object."""

    class _Trial:
        def __init__(self) -> None:
            self.status = status
            self.metrics = {"accuracy": 0.87}
            self.config = {"model": "gpt-4o", "temperature": 0.5}

    return _Trial()


def _make_result(
    best_score: float | None = 0.87,
    stop_reason: str | None = None,
) -> Any:
    """Return a minimal OptimizationResult-like object."""

    class _Result:
        def __init__(self) -> None:
            self.best_score = best_score
            self.best_config = {"model": "gpt-4o"}
            self.duration = 30.0
            self.success_rate = 1.0
            self.stop_reason = stop_reason
            self.status = "completed"
            self.run_label = "run-1"
            self.cloud_url = None
            self.metadata = {}
            self.trials = None

    return _Result()


# ---------------------------------------------------------------------------
# Tests for traigent.utils.console._safe_print
# ---------------------------------------------------------------------------


class TestConsoleSafePrint:
    """_safe_print from traigent.utils.console must not raise on cp1252 stdout."""

    def test_plain_ascii_passes_through(self, capsys: pytest.CaptureFixture) -> None:
        """ASCII text is printed unchanged."""
        console_safe_print("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_unicode_on_cp1252_does_not_raise(self) -> None:
        """Unicode emoji/arrows must not raise UnicodeEncodeError on cp1252 console."""
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            # These chars are outside cp1252 and would crash a bare print()
            console_safe_print("🚀 Starting optimization")
            console_safe_print("✅ done ❌ failed ⏱️ time 🏆 best")
            console_safe_print("▶️ trial 1/3 starting...")
            console_safe_print("⚠️ Optimization stopped early")

    def test_box_drawing_on_cp1252_does_not_raise(self) -> None:
        """Box-drawing characters used in results_table must not raise."""
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            console_safe_print("┌─────┐")
            console_safe_print("│ row │")
            console_safe_print("└─────┘")
            console_safe_print("★ best")

    def test_star_char_on_cp1252_does_not_raise(self) -> None:
        """The ★ best-trial marker must not raise on cp1252."""
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            console_safe_print("★ 2  gpt-4o  0.0  87.4%")

    def test_flush_kwarg_accepted(self) -> None:
        """flush=True should not cause an error."""
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            console_safe_print("flushed", flush=True)


# ---------------------------------------------------------------------------
# Tests for traigent.utils.callbacks._safe_print (defined in callbacks module)
# ---------------------------------------------------------------------------


class TestCallbacksSafePrint:
    """The _safe_print in callbacks.py must also be robust."""

    def test_emoji_on_cp1252_does_not_raise(self) -> None:
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            _safe_print("🚀 hello 📊 data ⚙️ config")

    def test_sep_kwarg(self, capsys: pytest.CaptureFixture) -> None:
        _safe_print("a", "b", "c", sep="-")
        assert "a-b-c" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Tests for ProgressBarCallback on cp1252
# ---------------------------------------------------------------------------


class TestProgressBarCallbackEncoding:
    """ProgressBarCallback output must not raise on cp1252 stdout (issue #1321)."""

    def test_on_optimization_start_safe(self) -> None:
        cb = ProgressBarCallback()
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_optimization_start(
                config_space={"model": ["gpt-4o", "gpt-4o-mini"]},
                objectives=["accuracy"],
                algorithm="grid",
            )

    def test_on_trial_complete_progress_bar_safe(self) -> None:
        cb = ProgressBarCallback(update_interval=0.0)
        cb.start_time = 0.0
        cb.last_update = 0.0
        cb._config_space = {"model": ["gpt-4o"]}
        cb._objectives = ["accuracy"]

        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_trial_complete(_make_trial(), _make_progress())

    def test_on_optimization_complete_safe(self) -> None:
        cb = ProgressBarCallback()
        cb._config_space = {"model": ["gpt-4o"]}
        cb._objectives = ["accuracy"]

        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_optimization_complete(_make_result())

    def test_on_optimization_complete_timeout_safe(self) -> None:
        cb = ProgressBarCallback()
        cb._config_space = {}
        cb._objectives = []

        result = _make_result(stop_reason="timeout")
        result.status = "cancelled"

        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_optimization_complete(result)


# ---------------------------------------------------------------------------
# Tests for DetailedProgressCallback on cp1252
# ---------------------------------------------------------------------------


class TestDetailedProgressCallbackEncoding:
    """DetailedProgressCallback output must not raise on cp1252 stdout."""

    def test_on_optimization_start_safe(self) -> None:
        cb = DetailedProgressCallback()
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_optimization_start(
                config_space={"model": ["gpt-4o"], "temperature": [0.0, 0.5]},
                objectives=["accuracy", "cost"],
                algorithm="grid",
            )

    def test_on_trial_start_safe(self) -> None:
        cb = DetailedProgressCallback(show_config_details=True)
        cb.trial_count = 0
        cb.total_trials = 3
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_trial_start(0, {"model": "gpt-4o", "temperature": 0.5})

    def test_on_trial_complete_safe(self) -> None:
        cb = DetailedProgressCallback(show_metrics=True)
        cb.trial_count = 1
        cb.total_trials = 3
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_trial_complete(_make_trial(), _make_progress())

    def test_on_trial_complete_failed_trial_safe(self) -> None:
        cb = DetailedProgressCallback()
        cb.trial_count = 1
        cb.total_trials = 3
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_trial_complete(_make_trial("failed"), _make_progress())

    def test_on_optimization_complete_safe(self) -> None:
        cb = DetailedProgressCallback()
        cb.trial_count = 3
        cb.total_trials = 3
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_optimization_complete(_make_result())

    def test_on_optimization_complete_timeout_safe(self) -> None:
        cb = DetailedProgressCallback()
        result = _make_result(stop_reason="timeout")
        result.status = "cancelled"
        result.metadata = {"timeout": 60}
        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            cb.on_optimization_complete(result)


# ---------------------------------------------------------------------------
# Tests for DiagnosticReport on cp1252
# ---------------------------------------------------------------------------


class TestDiagnosticReportEncoding:
    """DiagnosticReport.print_report() must not raise on cp1252 stdout."""

    def test_print_report_safe_on_cp1252(self) -> None:
        report = DiagnosticReport()
        report.add_issue("SDK", "Backend unreachable", "Check your API key")
        report.warnings.append({"category": "Config", "message": "Missing timeout"})
        report.successes.append({"category": "Auth", "message": "Token valid"})
        report.recommendations.append("Run traigent info for diagnostics")

        fake_stdout = _Cp1252StdOut()
        with patch("sys.stdout", fake_stdout):
            report.print_report()


# ---------------------------------------------------------------------------
# Tests for configure_stdout_encoding()
# ---------------------------------------------------------------------------


class TestConfigureStdoutEncoding:
    """configure_stdout_encoding() should wrap stdout when encoding is narrow."""

    def test_noop_when_already_utf8(self) -> None:
        """If stdout is already UTF-8, configure_stdout_encoding is a no-op."""
        original = sys.stdout
        utf8_stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with patch("sys.stdout", utf8_stream):
            configure_stdout_encoding()
            # Should not have replaced the stream
            assert sys.stdout is utf8_stream
        assert sys.stdout is original

    def test_noop_when_no_buffer(self) -> None:
        """If stdout has no .buffer (e.g. StringIO), configure_stdout_encoding is a no-op."""
        original = sys.stdout
        string_io = io.StringIO()
        with patch("sys.stdout", string_io):
            configure_stdout_encoding()
            assert sys.stdout is string_io
        assert sys.stdout is original

    def test_wraps_narrow_encoding(self) -> None:
        """If stdout uses a narrow encoding, it should be replaced with UTF-8 wrapper."""
        buf = io.BytesIO()
        narrow_stream = io.TextIOWrapper(buf, encoding="cp1252")

        with patch("sys.stdout", narrow_stream):
            configure_stdout_encoding()
            assert sys.stdout is not narrow_stream
            new_encoding = getattr(sys.stdout, "encoding", "")
            assert new_encoding.lower().replace("-", "") in ("utf8", "utf-8")
