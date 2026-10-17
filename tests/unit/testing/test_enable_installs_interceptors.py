"""enable_mock_mode_for_quickstart() installs the mock interceptors (#2416).

Before the fix the LiteLLM/LangChain interceptors were installed lazily, at
the first ``LocalEvaluator`` construction inside an optimization run. A
``litellm.completion`` call made after enabling mock mode but before the
first optimize run went to the real provider.

Each case runs in a fresh interpreter: the interceptors patch the litellm
module globally, so an in-process test would see whatever an earlier test
already installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Refuse every outbound connection and DNS lookup, and record the attempt.
_SPY = """
import socket
SEEN = []
def _gai(host, *a, **k):
    SEEN.append(("dns", host))
    raise OSError("blocked by test spy")
def _conn(self, addr):
    SEEN.append(("connect", addr))
    raise OSError("blocked by test spy")
socket.getaddrinfo = _gai
socket.socket.connect = _conn
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and k != "TRAIGENT_MOCK_LLM"
    }
    env.update(
        PYTHONPATH=str(_REPO_ROOT),
        TRAIGENT_OFFLINE_MODE="true",
        TRAIGENT_SKIP_DOTENV="1",
        LITELLM_MODE="PRODUCTION",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        ENVIRONMENT="development",
    )
    return subprocess.run(
        [sys.executable, "-c", _SPY + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=180,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> str:
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    return result.stdout


def test_litellm_calls_are_mocked_right_after_enable() -> None:
    _mock_script = """
            import asyncio
            import litellm
            from traigent.testing import enable_mock_mode_for_quickstart
            enable_mock_mode_for_quickstart()
            print("PATCHED", getattr(litellm, "_traigent_patched_completion", False),
                  getattr(litellm, "_traigent_patched_acompletion", False))
            msgs = [{"role": "user", "content": "hi"}]
            r = litellm.completion(model="gpt-4o-mini", messages=msgs)
            print("SYNC", r.choices[0].message.content)
            r = asyncio.run(litellm.acompletion(model="gpt-4o-mini", messages=msgs))
            print("ASYNC", r.choices[0].message.content)
            print("SEEN", SEEN)
            """
    out = _assert_ok(_run(_mock_script))
    assert "PATCHED True True" in out
    assert "SYNC This is a mock response for testing." in out
    assert "ASYNC This is a mock response for testing." in out
    assert "SEEN []" in out


def test_later_optimize_path_does_not_patch_twice() -> None:
    _mock_script = """
            import litellm
            from traigent.testing import enable_mock_mode_for_quickstart
            enable_mock_mode_for_quickstart()
            first = litellm.completion
            # The optimize path installs through the same guarded helper, and
            # the patch function itself is idempotent.
            from traigent.evaluators.local import _ensure_metadata_capture_patches
            from traigent.utils.litellm_interceptor import (
                patch_litellm_for_metadata_capture,
            )
            _ensure_metadata_capture_patches()
            print("REPATCHED", patch_litellm_for_metadata_capture())
            print("SAME", litellm.completion is first)
            """
    out = _assert_ok(_run(_mock_script))
    assert "REPATCHED False" in out
    assert "SAME True" in out


def test_import_traigent_alone_does_not_patch_litellm() -> None:
    _mock_script = """
            import sys
            import traigent
            print("LOADED", "litellm" in sys.modules)
            import litellm
            print("PATCHED", getattr(litellm, "_traigent_patched_completion", False))
            """
    out = _assert_ok(_run(_mock_script))
    assert "LOADED False" in out
    assert "PATCHED False" in out
