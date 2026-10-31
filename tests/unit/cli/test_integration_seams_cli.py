"""Seams between fix groups in the CLI.

* an all-trials-failed ``optimize`` run shows the run's own warnings
  (e.g. ``MOCK_TRANSPORT_BLOCKED``) and says when it replaces a saved result;
* ``check`` prints validator errors redacted and markup-escaped, like every
  other CLI error path (``_safe_error_text``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from click.testing import CliRunner

from traigent.cli.main import cli

_MODULES_DIR = Path(__file__).resolve().parent / "temp_modules"

_FAILED_MODULE = (
    "class FakeTrial:\n"
    "    is_successful = False\n"
    "    error_message = 'Connection error.'\n"
    "\n\n"
    "class FakeResult:\n"
    "    status = 'failed'\n"
    "    trials = [FakeTrial()]\n"
    "    successful_trials = []\n"
    "    best_config = None\n"
    "    best_score = None\n"
    "    stop_reason = None\n"
    "    warnings = ['MOCK_TRANSPORT_BLOCKED: a raw provider call was blocked']\n"
    "\n\n"
    "async def _optimize_stub(*args, **kwargs):\n"
    "    return FakeResult()\n"
    "\n\n"
    "def fake_fn():\n"
    "    return 'ok'\n"
    "\n\n"
    "fake_fn.optimize = _optimize_stub\n"
)


def _write(body: str) -> Path:
    _MODULES_DIR.mkdir(exist_ok=True)
    path = _MODULES_DIR / f"seam_{uuid4().hex}.py"
    path.write_text(body)
    return path


def _cleanup(path: Path) -> None:
    sys.modules.pop("user_module", None)
    if path.exists():
        path.unlink()
    if _MODULES_DIR.exists() and not any(_MODULES_DIR.iterdir()):
        _MODULES_DIR.rmdir()


def _optimize_failed(exists: bool, plain: Callable[[str], str]):
    path = _write(_FAILED_MODULE)
    try:
        with patch("traigent.cli.main.PersistenceManager") as persistence:
            manager = Mock()
            manager.result_exists.return_value = exists
            persistence.return_value = manager
            result = CliRunner().invoke(cli, ["optimize", str(path)])
        return result, plain(result.output)
    finally:
        _cleanup(path)


def test_all_failed_run_prints_the_run_warnings(plain: Callable[[str], str]) -> None:
    result, output = _optimize_failed(False, plain)

    assert result.exit_code == 1, output
    assert "MOCK_TRANSPORT_BLOCKED" in output


def test_all_failed_run_announces_replacing_an_earlier_result(
    plain: Callable[[str], str],
) -> None:
    _, output = _optimize_failed(True, plain)
    assert "Replacing the earlier saved result" in output

    _, output = _optimize_failed(False, plain)
    assert "Replacing the earlier saved result" not in output


def test_check_prints_validator_errors_redacted_and_escaped(
    plain: Callable[[str], str],
) -> None:
    path = _write(
        "import traigent\n\n"
        "@traigent.optimize(\n"
        "    eval_dataset='x.jsonl',\n"
        "    objectives=['accuracy'],\n"
        "    configuration_space={'t': [0.1, 0.2]},\n"
        ")\n"
        "def agent(q):\n"
        "    return q\n"
    )
    fake_secret = (
        "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"  # pragma: allowlist secret (fake)
    )
    try:
        with patch(
            "traigent.cli.optimization_validator.OptimizationValidator.validate_optimization",
            side_effect=RuntimeError(f"boom [/bad] key={fake_secret}"),
        ):
            result = CliRunner().invoke(cli, ["check", str(path)])
        output = plain(result.output)
    finally:
        _cleanup(path)

    assert "MarkupError" not in output, output
    assert fake_secret not in output, output
    assert "boom" in output, output
