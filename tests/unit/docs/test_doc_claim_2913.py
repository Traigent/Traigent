"""Doc claim matches code (#2913)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_tvl_tutorial_python_floor_matches_pyproject_2913() -> None:
    pyproject = _read("pyproject.toml")
    floor = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', pyproject).group(1)
    readme = _read("examples/tvl/tutorials/README.md")
    stated = re.search(r"Python (\d+\.\d+)\+", readme).group(1)
    assert stated == floor
