"""Shared sample callables (they need real source files for inspect.getsource)."""

from __future__ import annotations

import functools
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests._git_isolation import scrubbed_git_env

FIXED_CLOCK_ARGS = (2026, 10, 27, 12, 30, 45)


def metric_a(output: Any, expected: Any) -> float:
    return 1.0 if output == expected else 0.0


def metric_b(output: Any, expected: Any) -> float:
    return float(len(str(output)) == len(str(expected)))


def scorer(output: Any, expected: Any) -> float:
    return 0.5


def custom(*args: Any, **kwargs: Any) -> dict[str, float]:
    return {"accuracy": 1.0}


def sample_agent(question: str) -> str:
    return question.upper()


def git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrubbed_git_env(),
    )
    return completed.stdout


def make_git_agent(tmp_path: Path, name: str = "snapagent") -> tuple[Path, Any]:
    """A committed one-file agent in a throwaway git repo, imported as a module."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / f"{name}.py").write_text(
        "def run(question):\n    return question.upper()\n", encoding="utf-8"
    )
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.invalid")
    git(repo, "config", "user.name", "t")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "init")
    spec = importlib.util.spec_from_file_location(name, repo / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return repo, module.run


def bound_agent() -> Any:
    return functools.partial(sample_agent)
