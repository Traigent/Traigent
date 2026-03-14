"""Typing tests for parameter range config value helpers."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_parameter_range_config_value_typing_with_mypy(tmp_path: Path) -> None:
    """Mypy should narrow range config unions via the exported type guards."""
    pytest.importorskip("mypy")

    snippet = tmp_path / "typing_check.py"
    snippet.write_text(
        textwrap.dedent("""
            from typing import Literal, assert_type

            from traigent.api.parameter_ranges import (
                Choices,
                ChoicesConfigValue,
                FloatRangeConfigDict,
                FloatRangeConfigValue,
                IntRange,
                IntRangeConfigDict,
                IntRangeConfigValue,
                LogRange,
                Range,
                is_float_range_config_dict,
                is_int_range_config_dict,
            )

            float_value = Range(0.0, 1.0, step=0.1).to_config_value()
            assert_type(float_value, FloatRangeConfigValue)
            if is_float_range_config_dict(float_value):
                assert_type(float_value, FloatRangeConfigDict)
                assert_type(float_value["low"], float)

            int_value = IntRange(1, 10, step=1).to_config_value()
            assert_type(int_value, IntRangeConfigValue)
            if is_int_range_config_dict(int_value):
                assert_type(int_value, IntRangeConfigDict)
                assert_type(int_value["high"], int)

            log_value = LogRange(0.001, 1.0).to_config_value()
            assert_type(log_value, FloatRangeConfigDict)
            assert_type(log_value["type"], Literal["float"])

            choices: Choices[str] = Choices(["gpt-4", "gpt-4.1"])
            choice_value = choices.to_config_value()
            assert_type(choice_value, ChoicesConfigValue[str])
            """),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            "pyproject.toml",
            str(snippet),
            "traigent/api/parameter_ranges.py",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
