"""Regression tests for Traigent#1721: ``traigent optimize`` must exit nonzero
on every real failure path (module load failure, no optimizable functions,
and per-function optimize() failures), while leaving the happy path exit
code (0) unchanged."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from click.testing import CliRunner

from traigent.cli.main import cli

_MODULES_DIR = Path(__file__).resolve().parent / "temp_modules"


def _write_module(content: str) -> Path:
    _MODULES_DIR.mkdir(exist_ok=True)
    path = _MODULES_DIR / f"optimize_exit_{uuid4().hex}.py"
    path.write_text(content)
    return path


def _cleanup(path: Path) -> None:
    sys.modules.pop("user_module", None)
    if path.exists():
        path.unlink()
    if _MODULES_DIR.exists() and not any(_MODULES_DIR.iterdir()):
        _MODULES_DIR.rmdir()


def test_optimize_exits_nonzero_when_no_optimizable_functions() -> None:
    module_path = _write_module("def plain_function(x: int) -> int:\n    return x\n")
    try:
        result = CliRunner().invoke(cli, ["optimize", str(module_path)])
        assert result.exit_code != 0, result.output
        assert "No functions with @traigent.optimize decorator found" in result.output
    finally:
        _cleanup(module_path)


def test_optimize_exits_nonzero_when_module_fails_to_load() -> None:
    module_path = _write_module("raise RuntimeError('boom at import time')\n")
    try:
        result = CliRunner().invoke(cli, ["optimize", str(module_path)])
        assert result.exit_code != 0, result.output
        assert "Error loading file" in result.output
    finally:
        _cleanup(module_path)


def test_optimize_exits_nonzero_when_a_function_optimize_raises() -> None:
    module_path = _write_module(
        "async def _optimize_stub(*args, **kwargs):\n"
        "    raise RuntimeError('cost gate declined the run')\n"
        "\n"
        "\n"
        "def fake_optimized_function():\n"
        "    return 'ok'\n"
        "\n"
        "\n"
        "fake_optimized_function.optimize = _optimize_stub\n"
    )
    try:
        result = CliRunner().invoke(cli, ["optimize", str(module_path)])
        assert result.exit_code != 0, result.output
        assert "Error optimizing fake_optimized_function" in result.output
        assert "function(s) failed to optimize" in result.output
    finally:
        _cleanup(module_path)


def test_optimize_exits_zero_on_happy_path() -> None:
    module_path = _write_module(
        "class DummyResult:\n"
        "    def __init__(self):\n"
        "        self.status = 'COMPLETED'\n"
        "        self.trials = [object(), object()]\n"
        "        self.successful_trials = [object()]\n"
        "        self.best_config = {'param': 1}\n"
        "        self.best_score = 0.75\n"
        "\n"
        "\n"
        "async def _optimize_stub(*args, **kwargs):\n"
        "    return DummyResult()\n"
        "\n"
        "\n"
        "def fake_optimized_function():\n"
        "    return 'ok'\n"
        "\n"
        "\n"
        "fake_optimized_function.optimize = _optimize_stub\n"
    )
    try:
        # DummyResult intentionally omits the metadata/algorithm/objectives
        # surface PersistenceManager.save_result needs; stub persistence so
        # this test isolates the happy-path exit-code contract.
        with patch("traigent.cli.main.PersistenceManager") as mock_persistence_class:
            mock_persistence_class.return_value = Mock()
            result = CliRunner().invoke(cli, ["optimize", str(module_path)])

        assert result.exit_code == 0, result.output
        assert "Optimization Complete!" in result.output
    finally:
        _cleanup(module_path)
