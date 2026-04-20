"""Env-var seeding helper for the quickstart example.

Kept in a small module so tests can exercise the logic without importing the
full ``__main__`` (which pulls in LangChain and registers an ``@optimize``
function at import time).
"""

from __future__ import annotations

import sys
from collections.abc import MutableMapping

from traigent.utils.env_config import is_truthy

_PORTAL_SIGNUP_URL = "https://app.traigent.ai"


def configure_quickstart_env(env: MutableMapping[str, str]) -> None:
    """Seed mock-mode env vars and warn when the portal won't receive results.

    - Defaults ``TRAIGENT_MOCK_LLM=true`` so the script works without API keys.
    - In mock mode, seeds a placeholder ``OPENAI_API_KEY`` so LangChain is happy.
    - If no ``TRAIGENT_API_KEY`` is set, prints a clear stderr notice *before*
      forcing ``TRAIGENT_OFFLINE_MODE=true`` — otherwise the downstream
      "No API key found" warning is suppressed by the offline short-circuit.
    """
    env.setdefault("TRAIGENT_MOCK_LLM", "true")
    if not is_truthy(env.get("TRAIGENT_MOCK_LLM")):
        return

    env.setdefault("OPENAI_API_KEY", "mock-key-for-demos")

    if not env.get("TRAIGENT_API_KEY"):
        print(
            "[traigent] No TRAIGENT_API_KEY set — running fully offline. "
            f"Set TRAIGENT_API_KEY (get one at {_PORTAL_SIGNUP_URL}) to sync "
            "results to the portal while keeping LLM calls mocked.",
            file=sys.stderr,
        )
        env.setdefault("TRAIGENT_OFFLINE_MODE", "true")
