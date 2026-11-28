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

# The blocker prints this marker to stderr only after a child-side self-check
# proves the guard works: IPv4 and IPv6 socket creation must both raise the
# guard's NETWORK_BLOCKED error. If the guard is absent, no marker is printed.
_BLOCKER_MARKER = "CI_APPROVAL_NETWORK_BLOCKER_LOADED"
_BLOCKER_SELF_CHECK = f"""
import socket as _probe_socket
import sys as _probe_sys

_probe_ok = True
for _probe_family in (_probe_socket.AF_INET, _probe_socket.AF_INET6):
    try:
        _probe_sock = _probe_socket.socket(_probe_family)
    except RuntimeError as _probe_exc:
        if "NETWORK_BLOCKED" not in str(_probe_exc):
            _probe_ok = False
    except OSError:
        _probe_ok = False
    else:
        _probe_sock.close()
        _probe_ok = False
if _probe_ok:
    _probe_sys.stderr.write("{_BLOCKER_MARKER}\\n")
"""
_BLOCKER_SRC = _SITECUSTOMIZE_SRC + _BLOCKER_SELF_CHECK


def _run_quickstart(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="traigent-ci-approval-") as home:
        site = Path(home) / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(_BLOCKER_SRC, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": home,
            "PYTHONPATH": os.pathsep.join([str(site), str(_REPO_ROOT)]),
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


def _assert_blocker_loaded(proc: subprocess.CompletedProcess[str]) -> None:
    assert _BLOCKER_MARKER in proc.stderr, (
        "network blocker sitecustomize did not load in the child interpreter\n"
        + _tail(proc)
    )


def test_keyless_quickstart_in_ci_without_approval_is_refused() -> None:
    proc = _run_quickstart({})
    _assert_blocker_loaded(proc)
    assert proc.returncode != 0, _tail(proc)
    assert "CI/CD Approval Required" in proc.stdout + proc.stderr, _tail(proc)


def test_keyless_quickstart_in_ci_with_approval_passes() -> None:
    proc = _run_quickstart(
        {"TRAIGENT_RUN_APPROVED": "1", "TRAIGENT_APPROVED_BY": "unit-test"}
    )
    _assert_blocker_loaded(proc)
    assert "CI/CD Approval Required" not in proc.stdout + proc.stderr, _tail(proc)
    assert proc.returncode == 0, _tail(proc)
