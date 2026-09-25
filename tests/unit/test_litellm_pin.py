"""Tests for the LITELLM_LOCAL_* import-time pin in ``traigent/__init__.py``.

LiteLLM fetches its model-cost table (and, for Anthropic, its beta-header
config) from raw.githubusercontent.com the first time it is imported unless
``LITELLM_LOCAL_MODEL_COST_MAP`` / ``LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS``
are already set. Traigent pins both to the bundled tables by default so
``import traigent`` never makes that outbound call; the pin can be opted
out of with ``TRAIGENT_LITELLM_LIVE_PRICES=1``.

Every case here runs `import traigent` in a **fresh subprocess** because the
behavior under test only happens once, at first import, in a given process.
``tests/conftest.py`` already sets ``LITELLM_LOCAL_MODEL_COST_MAP`` for the
whole suite (to keep the suite itself offline), so that variable -- and its
sibling -- must be explicitly stripped from the subprocess's environment
before each run, or the fixture's own setdefault would mask the thing being
tested.
"""

# Traceability: CONC-Layer-API CONC-Quality-Reliability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_PIN_VARS = ("LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS")


def _clean_env(**overrides: str) -> dict[str, str]:
    """Base environment for a subprocess, with the pin vars removed.

    Starting from ``os.environ`` (not an empty dict) keeps PATH, the venv,
    and everything else the interpreter needs to actually start; only the
    variables this test cares about are stripped so the subprocess sees a
    real "nothing has set this yet" state.
    """
    env = dict(os.environ)
    for name in (*_PIN_VARS, "TRAIGENT_LITELLM_LIVE_PRICES"):
        env.pop(name, None)
    env.update(overrides)
    return env


def _run(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


class TestLiteLLMPinDefault:
    def test_default_import_sets_both_vars(self) -> None:
        """Plain `import traigent`, no opt-out: both pin vars land as True."""
        result = _run(
            "import os, traigent\n"
            "print(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'))\n"
            "print(os.environ.get('LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS'))\n",
            env=_clean_env(),
        )
        lines = result.stdout.strip().splitlines()
        assert lines == ["True", "True"], result.stdout

    def test_explicit_user_value_is_preserved(self) -> None:
        """setdefault must not clobber a value the user already set."""
        result = _run(
            "import os, traigent\n"
            "print(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'))\n"
            "print(os.environ.get('LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS'))\n",
            env=_clean_env(
                LITELLM_LOCAL_MODEL_COST_MAP="False",
                LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS="False",
            ),
        )
        lines = result.stdout.strip().splitlines()
        assert lines == ["False", "False"], (
            "an explicit user value must win over the SDK's default pin"
        )

    def test_opt_out_leaves_both_unset(self) -> None:
        """TRAIGENT_LITELLM_LIVE_PRICES=1 disables the pin entirely."""
        result = _run(
            "import os, traigent\n"
            "print(os.environ.get('LITELLM_LOCAL_MODEL_COST_MAP'))\n"
            "print(os.environ.get('LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS'))\n",
            env=_clean_env(TRAIGENT_LITELLM_LIVE_PRICES="1"),
        )
        lines = result.stdout.strip().splitlines()
        assert lines == ["None", "None"], (
            "the opt-out must leave both vars unset, not just unset-by-default"
        )

    def test_litellm_imported_first_logs_notice_and_does_not_reload(self) -> None:
        """If litellm is imported before traigent, the pin cannot retroactively
        apply to litellm's own already-completed import-time fetch. Traigent
        must not reload litellm to force it -- reloading an arbitrary
        third-party module is unsafe -- it must only log why the pin missed.
        """
        result = _run(
            "import logging, os, sys\n"
            "logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)\n"
            "import litellm\n"
            "mod_id_before = id(sys.modules['litellm'])\n"
            "import traigent\n"
            "mod_id_after = id(sys.modules['litellm'])\n"
            "print(mod_id_before == mod_id_after)\n",
            env=_clean_env(),
        )
        assert result.stdout.strip().splitlines()[-1] == "True", (
            "litellm must not be reloaded once already imported"
        )
        assert "could not prevent" in result.stderr, (
            "expected the import-order notice on stderr (debug log); got:\n"
            f"{result.stderr}"
        )
