"""Regression tests for #2828/#2831: no tracebacks for refused Python output,
and --apply validates --function before any generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from traigent.cli.generate_config_command import generate_config

_AGENT = """
import litellm

def predict(q):
    return litellm.completion(
        model="gpt-4o-mini", messages=[], max_tokens=1800, temperature=0.2
    )
"""


def _agent(tmp_path: Path) -> Path:
    path = tmp_path / "a2.py"
    path.write_text(_AGENT)
    return path


def _assert_clean_failure(result) -> None:
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert not isinstance(result.exception, ValueError)
    assert "-o tvl" in result.output
    assert "--only tvars,objectives" in result.output


def test_output_python_no_traceback(tmp_path: Path) -> None:
    path = _agent(tmp_path)
    result = CliRunner().invoke(
        generate_config, [str(path), "-f", "predict", "-o", "python"]
    )
    _assert_clean_failure(result)


def test_apply_no_traceback_and_file_untouched(tmp_path: Path) -> None:
    path = _agent(tmp_path)
    result = CliRunner().invoke(
        generate_config, [str(path), "-f", "predict", "--apply"]
    )
    _assert_clean_failure(result)
    assert path.read_text() == _AGENT


def test_only_tvars_objectives_python_still_works(tmp_path: Path) -> None:
    path = _agent(tmp_path)
    result = CliRunner().invoke(
        generate_config,
        [str(path), "-f", "predict", "-o", "python", "--only", "tvars,objectives"],
    )
    assert result.exit_code == 0, result.output
    assert "@traigent.optimize(" in result.output


def test_apply_without_function_fails_before_generation(tmp_path: Path) -> None:
    path = _agent(tmp_path)
    with patch("traigent.config_generator.generate_config") as gen:
        result = CliRunner().invoke(generate_config, [str(path), "--apply", "--enrich"])
    assert result.exit_code == 1
    assert "--apply requires --function" in result.output
    gen.assert_not_called()
