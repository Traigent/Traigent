"""Doc claim matches code (#2912)."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_evaluation_guide_imports_resolve_2912() -> None:
    """Every ``from traigent... import X`` in the evaluation guide must resolve."""
    text = _read("docs/guides/evaluation.md")
    pattern = re.compile(r"^\s*from (traigent[\w.]*) import ([\w, ]+)$", re.M)
    missing = []
    for module, names in pattern.findall(text):
        mod = importlib.import_module(module)
        for name in (n.strip() for n in names.split(",")):
            if not hasattr(mod, name):
                missing.append(f"{module}.{name}")
    assert not missing, missing
