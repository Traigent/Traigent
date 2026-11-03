"""Regression tests for Traigent#1721: ``traigent check`` must not exit 0
when zero optimizable functions are discovered, unless ``--allow-empty`` is
passed."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli


# Module fixtures must live inside the workspace root (secure_path enforces
# this), so use a scratch dir under this test package rather than tmp_path.
class _FixtureNotApplied:
    """Placeholder that fails loudly if used before the fixture rebinds it."""

    def _fail(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("cli_scratch_dir fixture not applied")

    __truediv__ = mkdir = _fail


_MODULES_DIR: Path | _FixtureNotApplied = _FixtureNotApplied()  # set per test


@pytest.fixture()
def module_dir(cli_scratch_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys.modules[__name__], "_MODULES_DIR", cli_scratch_dir)
    return []  # files live in cli_scratch_dir, removed with it


def _write_module(created: list[Path], content: str) -> Path:
    path = _MODULES_DIR / f"check_zero_{uuid4().hex}.py"
    path.write_text(content)
    created.append(path)
    return path


_NO_OPTIMIZE_SOURCE = "def plain_function(x: int) -> int:\n    return x + 1\n"

_ONE_OPTIMIZED_SOURCE = (
    "import os\n"
    "os.environ['TRAIGENT_MOCK_LLM'] = 'true'\n"
    "import traigent\n"
    "\n"
    "@traigent.optimize(\n"
    "    eval_dataset='data.jsonl',\n"
    "    objectives=['accuracy'],\n"
    "    configuration_space={'temperature': [0.0, 0.5]},\n"
    ")\n"
    "def sample_func(x: int, temperature: float = 0.0) -> int:\n"
    "    return x\n"
)


def test_check_exits_nonzero_when_no_functions_discovered(module_dir) -> None:
    module_path = _write_module(module_dir, _NO_OPTIMIZE_SOURCE)

    result = CliRunner().invoke(cli, ["check", str(module_path)])

    assert result.exit_code == 1, result.output
    assert "No optimizable functions found" in result.output


def test_check_allow_empty_exits_zero_when_no_functions_discovered(
    module_dir,
) -> None:
    module_path = _write_module(module_dir, _NO_OPTIMIZE_SOURCE)

    result = CliRunner().invoke(cli, ["check", str(module_path), "--allow-empty"])

    assert result.exit_code == 0, result.output
    assert "No optimizable functions found" in result.output


def test_check_exits_nonzero_when_objectives_filter_matches_nothing(
    module_dir,
) -> None:
    module_path = _write_module(module_dir, _ONE_OPTIMIZED_SOURCE)

    result = CliRunner().invoke(
        cli,
        ["check", str(module_path), "--objectives", "cost", "--dry-run"],
    )

    assert result.exit_code == 1, result.output
    assert "No functions found with objectives" in result.output


def test_check_allow_empty_exits_zero_when_objectives_filter_matches_nothing(
    module_dir,
) -> None:
    module_path = _write_module(module_dir, _ONE_OPTIMIZED_SOURCE)

    result = CliRunner().invoke(
        cli,
        [
            "check",
            str(module_path),
            "--objectives",
            "cost",
            "--dry-run",
            "--allow-empty",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "No functions found with objectives" in result.output
