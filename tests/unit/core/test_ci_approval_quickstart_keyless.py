"""The keyless quickstart inside a detected CI needs the uniform CI approval.

The quickstart has always set ``TRAIGENT_OFFLINE_MODE`` itself. What 0.34.0
changed is that the config now honours it, so the quickstart's config is local
and ``check_ci_approval`` no longer skips it. Mock and offline runs are
deliberately inside the same CI-approval rule as every other run.
Each case runs the packaged quickstart in a fresh interpreter with a minimal,
mock-LLM environment, no API key, no ``.env`` auto-load and network blocked at
interpreter start (same sitecustomize pattern as the quickstart smoke test).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.integration.test_quickstart_smoke import _SITECUSTOMIZE_SRC

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_quickstart(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="traigent-ci-approval-") as home:
        site = Path(home) / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(_SITECUSTOMIZE_SRC, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": home,
            "PYTHONPATH": f"{site}:{_REPO_ROOT}",
            "TRAIGENT_SKIP_DOTENV": "1",
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
            timeout=120,
            check=False,
        )


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    return f"stderr: {proc.stderr[-2000:]}\nstdout: {proc.stdout[-2000:]}"


def test_keyless_quickstart_in_ci_without_approval_is_refused() -> None:
    proc = _run_quickstart({})
    assert proc.returncode != 0, _tail(proc)
    assert "CI/CD Approval Required" in proc.stdout + proc.stderr, _tail(proc)


def test_keyless_quickstart_in_ci_with_approval_passes() -> None:
    proc = _run_quickstart(
        {"TRAIGENT_RUN_APPROVED": "1", "TRAIGENT_APPROVED_BY": "unit-test"}
    )
    assert "CI/CD Approval Required" not in proc.stdout + proc.stderr, _tail(proc)
    assert proc.returncode == 0, _tail(proc)
