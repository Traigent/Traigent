"""Env-var seeding helper for the quickstart example.

Kept in a small module so tests can exercise the logic without importing the
full ``__main__`` (which pulls in LangChain and registers an ``@optimize``
function at import time).

Mock mode itself is enabled in code via
:func:`traigent.testing.enable_mock_mode_for_quickstart`; this helper only
seeds the surrounding env so LangChain can instantiate and the user gets a
clear notice when results won't sync to the portal.
"""

from __future__ import annotations

import sys
from collections.abc import MutableMapping

_PORTAL_SIGNUP_URL = "https://app.traigent.ai"


def configure_quickstart_env(env: MutableMapping[str, str]) -> None:
    """Seed env vars LangChain expects and force offline-mode without an
    API key.

    - Seeds a placeholder ``OPENAI_API_KEY`` (via ``setdefault``) so
      :class:`ChatOpenAI` can instantiate. In the canonical invocation
      path the bootstrap in :mod:`traigent.__init__` has already
      OVERWRITTEN ``OPENAI_API_KEY`` with the placeholder before this
      function runs, so the ``setdefault`` here is a no-op in that
      flow — intentional. If a future entry point ever calls this
      helper WITHOUT going through the package bootstrap, the
      placeholder is seeded here as a fallback. The real LLM call is
      intercepted by the mock-mode flag set via
      :func:`traigent.testing.enable_mock_mode_for_quickstart`.
    - If no ``TRAIGENT_API_KEY`` is set, prints a clear stderr notice
      and **forces** ``TRAIGENT_OFFLINE_MODE=true`` (override, not
      ``setdefault``). Without a portal key the SDK can't sync results
      anyway — leaving offline-mode unset would just produce noisy
      auth errors instead of useful output.
    - If ``TRAIGENT_API_KEY`` IS set, this function does NOT touch
      ``TRAIGENT_OFFLINE_MODE`` — the user wants results synced and
      should control offline-mode themselves.
    """
    env.setdefault("OPENAI_API_KEY", "mock-key-for-demos")  # pragma: allowlist secret

    if not env.get("TRAIGENT_API_KEY"):
        print(
            "[traigent] No TRAIGENT_API_KEY set — running fully offline. "
            f"Set TRAIGENT_API_KEY (get one at {_PORTAL_SIGNUP_URL}) to sync "
            "results to the portal while keeping LLM calls mocked.",
            file=sys.stderr,
        )
        env["TRAIGENT_OFFLINE_MODE"] = "true"
