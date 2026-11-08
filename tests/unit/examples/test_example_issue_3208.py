"""Regression tests for example-script defect #3208."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples"


def _load(path: Path, name: str, monkeypatch: pytest.MonkeyPatch):
    """Import an example file as a module with mock mode off (no import side effects)."""
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "false")
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


# --- #3208: export_results on a missing results dir --------------------------


def test_export_results_missing_directory_gives_documented_message(
    tmp_path, monkeypatch
):
    module = _load(EXAMPLES / "ci-pipeline/export_results.py", "ex_export", monkeypatch)
    with pytest.raises(SystemExit) as exc:
        module._newest_run(tmp_path / "optimization_results")
    assert "no persisted optimization result found" in str(exc.value)
