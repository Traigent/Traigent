"""Guard scripts must fail loudly instead of passing when they cannot search."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS = REPO_ROOT / "scripts" / "hooks"
SCRIPTS = ["check-cost-strictness.sh", "check-test-assertion-contracts.sh"]


def _run(script: str, cwd: Path, path: str | None = None):
    env = dict(os.environ)
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        ["bash", str(HOOKS / script)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_passes_from_repo_root(script: str) -> None:
    proc = _run(script, REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_fails_when_run_from_wrong_directory(script: str, tmp_path: Path) -> None:
    proc = _run(script, tmp_path)
    assert proc.returncode != 0, "guard passed although its target dirs are missing"
    assert "not found" in (proc.stdout + proc.stderr).lower()


def test_cost_strictness_fails_when_a_search_target_is_missing(
    tmp_path: Path,
) -> None:
    # traigent/ present but the allowed file is not: must not pass silently.
    (tmp_path / "traigent").mkdir()
    proc = _run("check-cost-strictness.sh", tmp_path)
    assert proc.returncode != 0


def test_assertion_contracts_fails_when_one_target_dir_is_missing(
    tmp_path: Path,
) -> None:
    for d in ("tests/integration", "tests/unit/api", "tests/unit/integrations"):
        (tmp_path / d).mkdir(parents=True)
    # tests/unit/optimizers and tests/unit/utils are absent (renamed)
    proc = _run("check-test-assertion-contracts.sh", tmp_path)
    assert proc.returncode != 0
    assert "tests/unit/optimizers" in (proc.stdout + proc.stderr)


@pytest.mark.parametrize("script", SCRIPTS)
def test_search_tool_error_is_not_swallowed(script: str, tmp_path: Path) -> None:
    """A search tool exiting >=2 (error) must fail the guard, not read as 'no match'."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("bash", "awk", "dirname"):
        found = shutil.which(tool)
        if found:
            (bindir / tool).symlink_to(found)
    broken = bindir / "rg"
    broken.write_text("#!/bin/sh\necho 'rg: simulated failure' >&2\nexit 2\n")
    broken.chmod(0o755)
    proc = _run(script, REPO_ROOT, path=str(bindir))
    assert proc.returncode != 0, "search-tool error was swallowed"
