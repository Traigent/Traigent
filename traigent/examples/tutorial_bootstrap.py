"""Bootstrap helpers for bundled tutorial examples.

These helpers are intentionally scoped to examples shipped with the repo.
The SDK should not silently turn arbitrary user code into a mock run just
because a provider key is missing.
"""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from traigent.testing import enable_mock_mode_for_quickstart

_FALSEY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}
_PORTAL_SIGNUP_URL = "https://app.traigent.ai"


def _normalized_env_flag(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSEY:
        return False
    return None


def should_auto_enable_mock_mode(
    provider_env_keys: Sequence[str],
    *,
    env: MutableMapping[str, str] | None = None,
) -> bool:
    """Return whether a bundled tutorial should run with mocked LLM calls."""
    active_env = os.environ if env is None else env
    explicit_mock_mode = _normalized_env_flag(active_env.get("TRAIGENT_MOCK_LLM"))
    if explicit_mock_mode is not None:
        return explicit_mock_mode

    return not any(active_env.get(key, "").strip() for key in provider_env_keys)


def configure_tutorial_mock_mode(
    *,
    provider_env_keys: Sequence[str],
    tutorial_name: str,
    results_base: Path | None = None,
    env: MutableMapping[str, str] | None = None,
) -> bool:
    """Enable safe mock/offline defaults for bundled tutorial code.

    Returns ``True`` when mock mode is active for this tutorial. If a user
    explicitly sets ``TRAIGENT_MOCK_LLM=false``, this helper leaves real-provider
    execution alone so missing API keys still fail loudly.
    """
    active_env = os.environ if env is None else env
    if not should_auto_enable_mock_mode(provider_env_keys, env=active_env):
        return False

    enable_mock_mode_for_quickstart()

    if results_base is not None:
        results_dir = results_base / ".traigent_local"
        active_env.setdefault("HOME", str(results_base))
        results_dir.mkdir(parents=True, exist_ok=True)
        active_env.setdefault("TRAIGENT_RESULTS_FOLDER", str(results_dir))

    offline_mode = _normalized_env_flag(active_env.get("TRAIGENT_OFFLINE_MODE"))
    if offline_mode is not False:
        print(
            f"[traigent] {tutorial_name} is running in local mock mode. "
            "Set TRAIGENT_OFFLINE_MODE=false and TRAIGENT_API_KEY "
            f"(get one at {_PORTAL_SIGNUP_URL}) to sync results to the portal.",
            file=sys.stderr,
        )
        active_env["TRAIGENT_OFFLINE_MODE"] = "true"
        active_env.pop("TRAIGENT_API_KEY", None)

    return True
