"""Tests for release-review tracking generation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_generate_tracking_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / ".release_review" / "automation" / "generate_tracking.py"
    spec = importlib.util.spec_from_file_location("generate_tracking", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_inventory_path_rejects_directory_components(tmp_path: Path) -> None:
    module = _load_generate_tracking_module()
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="inventory filename"):
        module.resolve_inventory_path(run_dir, "../escape.txt")


def test_write_inventories_uses_guarded_inventory_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_generate_tracking_module()
    monkeypatch.chdir(tmp_path)

    source_file = tmp_path / "traigent" / "sample.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("print('ok')\n")

    test_file = tmp_path / "tests" / "test_sample.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_placeholder():\n    assert True\n")

    monkeypatch.setattr(module, "git_changed_files", lambda _: ["traigent/sample.py"])
    monkeypatch.setattr(module, "filter_review_scope_files", lambda files: files)

    run_dir = tmp_path / "runs" / "release-1"
    review_scope_files = module.write_inventories(run_dir, "develop")
    inventories_dir = (run_dir / "inventories").resolve()

    assert review_scope_files == ["traigent/sample.py"]

    expected = {
        "src_files.txt": "traigent/sample.py\n",
        "tests_files.txt": "tests/test_sample.py\n",
        "changed_files.txt": "traigent/sample.py\n",
        "review_scope_files.txt": "traigent/sample.py\n",
    }
    for filename, content in expected.items():
        path = module.resolve_inventory_path(run_dir, filename)
        assert path.exists()
        assert path.read_text() == content
        assert path.is_relative_to(inventories_dir)
