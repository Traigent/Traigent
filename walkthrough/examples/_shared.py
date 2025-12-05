#!/usr/bin/env python3
"""Shared utilities for TraiGent walkthrough examples.

Each example script imports these helpers to avoid duplicating setup logic.
The helpers provide:

* `add_repo_root_to_sys_path` — guarantees imports resolve when running files directly.
* `dataset_path`/`ensure_dataset` — opinionated helpers for generating JSONL datasets.
* `init_mock_mode` — standard way to place TraiGent in mock mode when the env flag is set.
* `check_dependencies` — optional guard to ensure required SDKs are installed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_TRUTHY = {"1", "true", "yes"}
__version__ = "1.0.0"
__all__ = [
    "add_repo_root_to_sys_path",
    "dataset_path",
    "ensure_dataset",
    "init_mock_mode",
    "check_dependencies",
]


def add_repo_root_to_sys_path(module_file: str) -> Path:
    """Ensure the repository root is on ``sys.path`` so relative imports succeed."""
    module_path = Path(module_file).resolve()
    repo_root = module_path.parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def dataset_path(module_file: str, filename: str) -> Path:
    """Return the datasets/<filename> path adjacent to the module."""
    return Path(module_file).resolve().parent / "datasets" / filename


def ensure_dataset(path: Path, rows: Sequence[Mapping]) -> Path:
    """Create a JSONL dataset at ``path`` unless it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    return path


def init_mock_mode(*, execution_mode: str = "edge_analytics") -> bool:
    """Initialize TraiGent mock execution mode if the ``TRAIGENT_MOCK_MODE`` flag is set."""
    mock_enabled = str(os.getenv("TRAIGENT_MOCK_MODE", "")).lower() in _TRUTHY
    if mock_enabled:
        import traigent

        print("🧪 Running in MOCK mode (no API calls)\n")
        traigent.initialize(execution_mode=execution_mode)
    return mock_enabled


def check_dependencies(packages: Iterable[str] | None = None) -> None:
    """Raise ``ImportError`` if any package in ``packages`` is missing."""
    required = tuple(packages or ("traigent",))
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        install_hint = " ".join(missing)
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install them with: pip install {install_hint}"
        )
