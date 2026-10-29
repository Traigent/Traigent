"""#2823: ``traigent optimize`` must exit non-zero and not announce success when
every trial of a function failed; cancelled and partially successful runs keep
their exit code and output."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from click.testing import CliRunner

from traigent.cli.main import cli

_MODULES_DIR = Path(__file__).resolve().parent / "temp_modules"

_RESULT_CLASS = (
    "class FakeTrial:\n"
    "    def __init__(self, ok, error=None):\n"
    "        self.is_successful = ok\n"
    "        self.error_message = error\n"
    "\n"
    "\n"
    "class FakeResult:\n"
    "    def __init__(self, status, trials, best_score=None, stop_reason=None):\n"
    "        self.status = status\n"
    "        self.trials = trials\n"
    "        self.successful_trials = [t for t in trials if t.is_successful]\n"
    "        self.best_config = None if best_score is None else {'p': 1}\n"
    "        self.best_score = best_score\n"
    "        self.stop_reason = stop_reason\n"
    "\n"
    "\n"
)


def _module(result_expr: str) -> Path:
    _MODULES_DIR.mkdir(exist_ok=True)
    path = _MODULES_DIR / f"optimize_failed_{uuid4().hex}.py"
    path.write_text(
        _RESULT_CLASS
        + "async def _optimize_stub(*args, **kwargs):\n"
        + f"    return {result_expr}\n"
        + "\n\n"
        + "def fake_optimized_function():\n"
        + "    return 'ok'\n"
        + "\n\n"
        + "fake_optimized_function.optimize = _optimize_stub\n"
    )
    return path


def _cleanup(path: Path) -> None:
    sys.modules.pop("user_module", None)
    if path.exists():
        path.unlink()
    if _MODULES_DIR.exists() and not any(_MODULES_DIR.iterdir()):
        _MODULES_DIR.rmdir()


def _run(result_expr: str, plain: Callable[[str], str]):
    path = _module(result_expr)
    try:
        with patch("traigent.cli.main.PersistenceManager") as persistence:
            persistence.return_value = Mock()
            result = CliRunner().invoke(cli, ["optimize", str(path)])
        return result, plain(result.output)
    finally:
        _cleanup(path)


def test_all_trials_failed_exits_nonzero_without_success_banner(
    plain: Callable[[str], str],
) -> None:
    result, output = _run(
        "FakeResult('failed', [FakeTrial(False, 'provider said no'), "
        "FakeTrial(False, 'second error')])",
        plain,
    )

    assert result.exit_code == 1, output
    assert "Optimization Complete!" not in output
    assert "Best score" not in output
    assert "fake_optimized_function" in output
    assert "provider said no" in output
    assert "function(s) failed to optimize" in output


def test_no_successful_trial_with_completed_status_is_still_a_failure(
    plain: Callable[[str], str],
) -> None:
    result, output = _run("FakeResult('completed', [FakeTrial(False, 'x')])", plain)

    assert result.exit_code == 1, output
    assert "Optimization Complete!" not in output


def test_partially_successful_run_keeps_exit_zero_and_success_banner(
    plain: Callable[[str], str],
) -> None:
    result, output = _run(
        "FakeResult('completed', [FakeTrial(True), FakeTrial(False, 'one bad')], 0.5)",
        plain,
    )

    assert result.exit_code == 0, output
    assert "Optimization Complete!" in output
    assert "Best score: 0.500" in output


def test_cancelled_run_keeps_its_exit_code_and_output(
    plain: Callable[[str], str],
) -> None:
    result, output = _run(
        "FakeResult('cancelled', [FakeTrial(False, 'stopped')], None, 'user_cancelled')",
        plain,
    )

    assert result.exit_code == 0, output
    assert "Cancelled by user" in output or "CANCELLED BY USER" in output


def test_trial_error_with_rich_markup_does_not_crash_the_report(
    plain: Callable[[str], str],
) -> None:
    result, output = _run(
        "FakeResult('failed', [FakeTrial(False, 'bad [/oops] tag')])", plain
    )

    assert result.exit_code == 1, output
    assert "MarkupError" not in output
    assert "bad [/oops] tag" in output
