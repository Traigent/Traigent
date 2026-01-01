"""Pytest plugin for streaming test results to the viewer.

This plugin writes test results to a JSONL file as each test completes,
enabling real-time updates in the test viewer UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line option for streaming results file."""
    parser.addoption(
        "--streaming-results-file",
        action="store",
        default=None,
        help="Path to write streaming test results (JSONL format)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the streaming plugin if output file is specified."""
    streaming_file = config.getoption("--streaming-results-file")
    if streaming_file:
        config.pluginmanager.register(
            StreamingResultsPlugin(Path(streaming_file)), "streaming_results"
        )


class StreamingResultsPlugin:
    """Plugin that writes test results as they complete."""

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        # Clear/create the file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text("")

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Called after each test phase (setup/call/teardown)."""
        # Only report on the 'call' phase (actual test execution)
        # Also report failures in setup/teardown
        if report.when == "call" or (report.when != "call" and report.failed):
            result = self._format_result(report)
            self._write_result(result)

    def _format_result(self, report: pytest.TestReport) -> dict[str, Any]:
        """Format a test report as a JSON-serializable dict."""
        # Map pytest outcomes to our status format
        if report.passed:
            outcome = "passed"
        elif report.failed:
            outcome = "failed"
        elif report.skipped:
            outcome = "skipped"
        else:
            outcome = "unknown"

        result: dict[str, Any] = {
            "nodeid": report.nodeid,
            "outcome": outcome,
            "duration": report.duration,
            "when": report.when,
        }

        # Include failure details
        if report.failed and report.longrepr:
            result["longrepr"] = str(report.longrepr)[:1000]  # Truncate long errors

        # Include skip reason
        if report.skipped and hasattr(report, "wasxfail"):
            result["wasxfail"] = report.wasxfail

        return result

    def _write_result(self, result: dict[str, Any]) -> None:
        """Append a result to the streaming file."""
        try:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure it's written immediately
        except OSError:
            pass  # Don't fail tests if we can't write streaming results
