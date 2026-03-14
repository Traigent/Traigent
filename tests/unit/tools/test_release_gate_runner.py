"""Tests for release gate runner path guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_release_gate_runner_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    automation_dir = repo_root / ".release_review" / "automation"
    module_path = automation_dir / "release_gate_runner.py"
    sys.path.insert(0, str(automation_dir))
    try:
        spec = importlib.util.spec_from_file_location("release_gate_runner", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_resolve_child_path_rejects_directory_components(tmp_path: Path) -> None:
    module = _load_release_gate_runner_module()

    with pytest.raises(ValueError, match="child filename"):
        module.resolve_child_path(tmp_path, "../escape.log")


def test_run_check_rejects_invalid_check_key(tmp_path: Path) -> None:
    module = _load_release_gate_runner_module()
    check = module.GateCheck(
        key="../escape",
        description="bad",
        command=["python3", "-c", "print('x')"],
        required=True,
        timeout_seconds=1,
    )

    with pytest.raises(ValueError, match="check key"):
        module.run_check(check, tmp_path, strict=True)


def test_ensure_run_workspace_writes_guarded_manifest(tmp_path: Path) -> None:
    module = _load_release_gate_runner_module()
    run_dir = tmp_path / "run-1"

    module.ensure_run_workspace(run_dir, "run-1", "main")

    manifest = module.resolve_child_path(run_dir, "run_manifest.json")
    assert manifest.exists()
    assert manifest.is_relative_to(run_dir.resolve())


def test_write_summary_markdown_writes_guarded_summary(tmp_path: Path) -> None:
    module = _load_release_gate_runner_module()
    gate_results = tmp_path / "gate_results"
    gate_results.mkdir()

    module.write_summary_markdown(
        output_dir=gate_results,
        release_id="release-1",
        mode="ci",
        strict=True,
        started_at="2026-03-14T00:00:00Z",
        finished_at="2026-03-14T00:01:00Z",
        results=[],
    )

    summary = module.resolve_child_path(gate_results, "summary.md")
    assert summary.exists()
    assert summary.is_relative_to(gate_results.resolve())
