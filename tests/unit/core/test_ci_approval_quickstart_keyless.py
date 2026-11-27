"""The keyless quickstart inside a detected CI needs the uniform CI approval.

Since 0.34.0 the quickstart sets ``TRAIGENT_OFFLINE_MODE`` itself, which makes
its config local, so ``check_ci_approval`` no longer skips it. Mock and offline
runs are deliberately inside the same CI-approval rule as every other run.
Each case runs the packaged quickstart in a fresh interpreter with a minimal,
mock-LLM environment and no API key.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_quickstart(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="traigent-ci-approval-") as home:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": home,
            "PYTHONPATH": str(_REPO_ROOT),
            "PYTHONUNBUFFERED": "1",
            "TRAIGENT_MOCK_LLM": "true",
            "CI": "true",
            "GITHUB_ACTIONS": "true",
            **extra_env,
        }
        return subprocess.run(  # noqa: S603 - fixed argv, our own interpreter
            [sys.executable, "-m", "traigent.examples.quickstart"],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )


def test_keyless_quickstart_in_ci_without_approval_is_refused() -> None:
    proc = _run_quickstart({})
    assert "CI/CD Approval Required" in proc.stdout + proc.stderr


def test_keyless_quickstart_in_ci_with_approval_passes() -> None:
    proc = _run_quickstart(
        {"TRAIGENT_RUN_APPROVED": "1", "TRAIGENT_APPROVED_BY": "unit-test"}
    )
    assert "CI/CD Approval Required" not in proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr[-2000:]
