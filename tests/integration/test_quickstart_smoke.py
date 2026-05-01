"""Smoke test for the bundled quickstart.

Asserts the load-bearing contract advertised on the website: ``traigent
quickstart`` (and ``python -m traigent.examples.quickstart``) must
complete cleanly with **no API keys** and **no successful LLM provider
calls**, in a true subprocess where any network attempt is blocked
before any traigent module is imported.

What this test does NOT promise: zero outbound network *attempts*.
LiteLLM's pricing-map fetch from raw.githubusercontent.com is a
known import-time attempt that gracefully falls back to bundled
pricing data when blocked. The contract this file enforces is "no
real LLM provider call succeeds" + "the user sees a working demo
with non-zero results" — i.e., no spend and no broken funnel —
which is what actually matters for the website experience.

Codex's review of the first draft of this file flagged that an
in-process ``CliRunner`` test cannot prove the no-spend claim if
heavy SDK modules in ``traigent.__init__.py`` execute network code
during their own import. So this file runs the bundled quickstart in
true subprocesses, with a ``sitecustomize.py`` injected via
``PYTHONPATH`` that monkeypatches ``socket`` BEFORE the interpreter
processes the first ``import traigent``. Failures here mean the
website funnel is broken before a user has any chance to see value.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# A sitecustomize.py the subprocess will pick up via PYTHONPATH. It
# runs at interpreter start (BEFORE traigent imports) and replaces
# socket.socket / create_connection / getaddrinfo with versions that
# raise on AF_INET / AF_INET6 traffic. AF_UNIX is allowed so asyncio's
# self-pipe (and other local IPC) keeps working.
_SITECUSTOMIZE_SRC = '''
"""Block all outbound network for this interpreter."""
import socket as _socket

_NETWORK_FAMILIES = {_socket.AF_INET, _socket.AF_INET6}
_real_socket = _socket.socket


class _BlockingSocket(_real_socket):
    def __init__(self, family=_socket.AF_INET, *args, **kwargs):
        if family in _NETWORK_FAMILIES:
            raise RuntimeError(
                f"NETWORK_BLOCKED: socket.socket(family={family!r}) — "
                "smoke test expected no network access"
            )
        super().__init__(family, *args, **kwargs)


def _blocked_create_connection(address, *args, **kwargs):
    raise RuntimeError(
        f"NETWORK_BLOCKED: socket.create_connection({address!r})"
    )


def _blocked_getaddrinfo(host, port, *args, **kwargs):
    raise RuntimeError(
        f"NETWORK_BLOCKED: socket.getaddrinfo({host!r}, {port!r})"
    )


_socket.socket = _BlockingSocket
_socket.create_connection = _blocked_create_connection
_socket.getaddrinfo = _blocked_getaddrinfo
'''


@pytest.fixture
def hermetic_subprocess_env(tmp_path: Path) -> dict[str, str]:
    """Build a clean subprocess env: no provider keys, no Traigent state,
    network blocked at interpreter start via injected sitecustomize."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE_SRC, encoding="utf-8")

    # Build env from scratch — explicitly DROP any provider keys and any
    # mock-mode toggles the parent shell or the autouse conftest fixture
    # might have set. The bundled quickstart sets its own env vars at
    # the top of its module, which is exactly the path we're verifying.
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        # PYTHONPATH: site (for sitecustomize) + worktree (so `import
        # traigent` finds my changes, not the editable install).
        "PYTHONPATH": f"{site}:{Path(__file__).resolve().parents[2]}",
        # Skip the SDK's .env auto-load — the main checkout's .env has
        # TRAIGENT_BACKEND_URL=localhost:5000 etc. that would muddy the
        # smoke contract.
        "TRAIGENT_SKIP_DOTENV": "1",
        # Ensure dev semantics so the in-code API can activate mock mode.
        "ENVIRONMENT": "development",
    }
    return env


def _run_in_subprocess(
    venv_python: Path, args: list[str], env: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Spawn a fresh interpreter and capture both streams."""
    return subprocess.run(
        [str(venv_python), *args],
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture
def venv_python() -> Path:
    """Path to the venv's python — same one running these tests."""
    return Path(sys.executable)


def _assert_no_provider_call_succeeded(combined: str) -> None:
    """The contract is "no real LLM API call succeeded" — not "no
    network attempt was ever made." LiteLLM still tries a model-cost
    fetch from raw.githubusercontent.com at import time and falls back
    gracefully when blocked; that's a different (lower-stakes) network
    path than the LLM provider endpoints whose leak would bill a real
    account. Assert that the LLM-provider domains specifically do not
    appear in any successful response context."""
    forbidden = (
        "api.openai.com",
        "api.anthropic.com",
        "api.cohere.ai",
        "generativelanguage.googleapis.com",
    )
    for host in forbidden:
        # NETWORK_BLOCKED appearing alongside the host means the
        # interceptor/blocker caught the attempt — that is the contract
        # working, not a failure.
        if host in combined and "NETWORK_BLOCKED" not in combined:
            raise AssertionError(
                f"Provider host {host!r} appeared in subprocess output "
                f"WITHOUT a NETWORK_BLOCKED marker — a real LLM call may "
                f"have succeeded:\n{combined}"
            )


def test_python_dash_m_runs_with_no_keys(
    hermetic_subprocess_env: dict[str, str],
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """``python -m traigent.examples.quickstart`` must complete in a
    fresh subprocess with no provider keys and no successful network
    call to any LLM provider, with the network block installed BEFORE
    any traigent import.

    A regression here means the website funnel is broken: the user
    copy-pastes the install command, the demo crashes, they bounce.
    """
    result = _run_in_subprocess(
        venv_python,
        ["-m", "traigent.examples.quickstart"],
        env=hermetic_subprocess_env,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"quickstart exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    _assert_no_provider_call_succeeded(combined)
    # Mandatory once-per-process WARN must appear when mock mode flips
    # on (proves the in-code API was actually called by the script).
    assert "mock mode is now ACTIVE" in combined, (
        "Expected the in-code activation WARN — got:\n" + combined
    )
    # Demo scoring must produce non-zero, model-correlated accuracy so
    # the user actually sees a meaningful "best config" hit (Codex
    # review #4 — exit-code-only is not enough).
    assert "85.0%" in combined or "85%" in combined, (
        "Demo scoring should produce non-zero, model-correlated scores "
        "(gpt-4o ~ 85%). Got:\n" + combined
    )
    # And the optimization actually ranked a winner.
    assert "Trial Results" in combined or "Best" in combined, (
        "Expected a results table or best-config line. Got:\n" + combined
    )


def test_traigent_quickstart_cli_runs_with_no_keys(
    hermetic_subprocess_env: dict[str, str],
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """The ``traigent quickstart`` CLI subcommand must behave exactly
    the same as ``python -m traigent.examples.quickstart`` — same
    hermetic guarantees, same exit, same activation WARN. The CLI is
    the canonical install instruction on the website."""
    # Run via `python -c` so PYTHONPATH (worktree-first) takes precedence
    # over the venv's editable finder for the main checkout. Same
    # observable behavior as `traigent quickstart` for the user.
    result = _run_in_subprocess(
        venv_python,
        ["-c", "from traigent.cli.main import cli; cli(['quickstart'])"],
        env=hermetic_subprocess_env,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"traigent quickstart exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    _assert_no_provider_call_succeeded(combined)
    assert "mock mode is now ACTIVE" in combined


def test_subprocess_with_real_looking_key_does_not_spend_it(
    hermetic_subprocess_env: dict[str, str],
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Codex's last point: even if a real-looking ``OPENAI_API_KEY`` is
    present in the env, the bundled quickstart must NOT spend it. The
    script overrides the key with a placeholder before any LLM client
    instantiation, so a mock-regression cannot accidentally bill a real
    account."""
    env = dict(hermetic_subprocess_env)
    env["OPENAI_API_KEY"] = (
        "sk-fake-real-looking-key-DO-NOT-USE"  # pragma: allowlist secret
    )

    result = _run_in_subprocess(
        venv_python,
        ["-m", "traigent.examples.quickstart"],
        env=env,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    combined = result.stdout + result.stderr
    # The provided "real" key must NOT appear anywhere — the script
    # overwrites it before any client is built.
    assert "sk-fake-real-looking-key-DO-NOT-USE" not in combined, (
        "User's OPENAI_API_KEY leaked into output — a mock-regression "
        "could spend it for real:\n" + combined
    )
    _assert_no_provider_call_succeeded(combined)
