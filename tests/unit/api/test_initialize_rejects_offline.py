"""Regression for #2837: ``initialize(offline=True)`` was accepted but never read.

The global configuration stored the value and nothing consumed it, so a
decorated function still resolved to the cloud-brain execution policy.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

import traigent
from traigent.api.functions import get_global_config

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("value", [True, False])
def test_initialize_offline_raises_type_error_naming_the_real_option(
    value: bool,
) -> None:
    before = get_global_config()
    with pytest.raises(TypeError) as exc:
        traigent.initialize(offline=value, minimal_logging=True)
    message = str(exc.value)
    assert "offline" in message
    assert "@traigent.optimize" in message and "TraigentConfig" in message
    # Fails before touching any global state.
    assert get_global_config() == before


def test_initialize_still_accepts_other_overrides() -> None:
    assert traigent.initialize(minimal_logging=True) is True


def _python_files_under(*roots: str) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--", *[f"{r}/**/*.py" for r in roots]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [REPO_ROOT / name for name in listed]


def test_shipped_scripts_do_not_pass_offline_to_initialize() -> None:
    offenders: list[str] = []
    for path in _python_files_under("examples", "walkthrough", "plugins", "scripts"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # not part of this guard
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                == "initialize"
                and any(kw.arg == "offline" for kw in node.keywords)
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, offenders
