"""Explicit mock-mode controls for the SDK.

Mock mode intercepts LLM calls and returns canned responses so demos
and tests can run without API keys or network. It is the load-bearing
path for ``traigent quickstart`` and the bundled walkthroughs.

There are TWO ways mock mode can be enabled, with different safety
properties:

1. **Recommended path — in-code API** (this module):
   :func:`enable_mock_mode_for_quickstart` flips a process-local flag
   that interceptors and SDK code consult. Hard-blocked in production:
   if ``ENVIRONMENT=production`` the function raises ``RuntimeError``
   instead of activating.

2. **Legacy path — env var** (``TRAIGENT_MOCK_LLM=true``): kept for
   backward compatibility with existing test fixtures and example
   scripts that set the env var as a convention. Honored ONLY outside
   production; the import-time guard in
   :mod:`traigent.utils.env_config` raises ``OSError`` if the env var
   is set together with ``ENVIRONMENT=production``. Provider-specific
   ``*_MOCK`` env vars (the worst offenders in the original prod
   incident — one var per provider, easy to miss) are completely
   ignored.

Both paths converge on :func:`is_mock_mode_enabled` and
:func:`traigent.utils.env_config.is_mock_llm`, which are the single
sources of truth for interceptors and SDK behavior.

Public surface:

* :func:`enable_mock_mode_for_quickstart` — opt in (warns once on
  activation, ``RuntimeError`` in prod).
* :func:`disable_mock_mode` — opt back out within the same process
  (info-logs on deactivation; always allowed, no prod guard needed
  since turning mock mode OFF can never substitute a fake result for
  a real one).
* :func:`is_mock_mode_enabled` — runtime check (in-code flag only;
  use :func:`is_mock_llm` if you also want the env-var fallback).

The module-level flag is process-local; importing the SDK in a fresh
interpreter starts with mock mode off. Within a single process, a
mock test-run followed by a "real" run must call
:func:`disable_mock_mode` in between or the "real" run silently keeps
using mocked responses — call :func:`disable_mock_mode` explicitly to
avoid that.
"""

from __future__ import annotations

import logging
import threading

__all__ = [
    "enable_mock_mode_for_quickstart",
    "disable_mock_mode",
    "is_mock_mode_enabled",
]

_logger = logging.getLogger(__name__)

_lock = threading.Lock()
_enabled = False
_activation_logged = False


def enable_mock_mode_for_quickstart() -> None:
    """Activate process-local mock mode for LLM calls.

    Idempotent: subsequent calls are no-ops. The first activation emits
    a single mandatory ``WARNING`` log line so anyone tailing logs in a
    production-shaped environment sees that mock mode just turned on.

    Hard-blocked in production: if ``ENVIRONMENT=production`` (or
    ``prod``), this function raises ``RuntimeError`` rather than
    enabling mock mode. The same guarantee applies to the legacy
    ``TRAIGENT_MOCK_LLM=true`` env-var path. Mock mode in production
    is a safety bug, not a feature — the policy is "no mock mode in
    prod, period."

    Intended for: bundled demos, walkthroughs, examples, and tests.
    """
    import os

    global _enabled, _activation_logged
    with _lock:
        if _enabled:
            return
        # Re-read ENVIRONMENT inside the lock (Greptile review #4 —
        # close the TOCTOU window where another thread could mutate
        # ENVIRONMENT to "production" between an unlocked check and
        # ``_enabled = True``). A correctness re-check elsewhere
        # (``is_mock_llm()`` recomputes ``is_production_strict_env()`` live) makes
        # exploitability low, but tightening the invariant here is
        # cheap.
        env_name = os.environ.get("ENVIRONMENT", "development").strip().lower()
        if env_name in {"prod", "production"}:
            raise RuntimeError(
                "traigent.testing.enable_mock_mode_for_quickstart() was "
                "called with ENVIRONMENT=production. Mock mode is "
                "hard-blocked in production to prevent silent substitution "
                "of real LLM calls. If you want a keyless demo, run with "
                "ENVIRONMENT!=production."
            )
        _enabled = True
        if not _activation_logged:
            _activation_logged = True
            _logger.warning(
                "[traigent.testing] mock mode is now ACTIVE — LLM calls will "
                "be intercepted and return canned responses. This must NEVER "
                "run in production. Enabled via "
                "traigent.testing.enable_mock_mode_for_quickstart()."
            )
            _logger.info(
                "[traigent.testing] mock interception scope: LiteLLM and "
                "LangChain calls are mocked. Raw openai.chat.completions.create(...) "
                "and anthropic.messages.create(...) calls are not intercepted and "
                "can hit the real provider API."
            )


def disable_mock_mode() -> None:
    """Deactivate process-local mock mode for LLM calls.

    Public counterpart to :func:`enable_mock_mode_for_quickstart`.
    Idempotent: calling this when mock mode is already off is a no-op.
    The first deactivation after an activation emits a single mandatory
    ``INFO`` log line, mirroring the ``WARNING`` emitted on activation, so
    the transition is visible to anyone tailing logs.

    Unlike :func:`enable_mock_mode_for_quickstart`, this has no
    ``ENVIRONMENT=production`` guard: turning mock mode OFF can never
    substitute a fake result for a real one, so there is nothing to
    hard-block.

    Intended for: same-process flows that do a free mock test-run and
    then a "real" run and need mock mode fully off before that real run,
    so it does not silently keep returning canned responses. After this
    call, the next :func:`enable_mock_mode_for_quickstart` will log its
    activation warning again (the "already logged" state resets too).

    Note: this only resets the in-code flag consulted by
    :func:`is_mock_mode_enabled`. The legacy ``TRAIGENT_MOCK_LLM`` env var
    (if set) is a separate mechanism — unset it yourself if your process
    also relies on that path.
    """
    global _enabled, _activation_logged
    with _lock:
        if not _enabled:
            return
        _enabled = False
        _activation_logged = False
        _logger.info(
            "[traigent.testing] mock mode is now INACTIVE — LLM calls will "
            "no longer be intercepted and will hit real providers again. "
            "Disabled via traigent.testing.disable_mock_mode()."
        )


def is_mock_mode_enabled() -> bool:
    """Return ``True`` iff :func:`enable_mock_mode_for_quickstart` was called.

    Read by ``MockAdapter.is_mock_enabled`` (LLM interceptors) and
    ``env_config.is_mock_llm`` (SDK-level mock-aware behavior). Both
    consult this single flag so the SDK never ends up in a "real LLM,
    mock SDK behavior" inconsistent state.
    """
    return _enabled


def _reset_for_tests() -> None:
    """Reset the flag. Tests only — not part of the public API."""
    global _enabled, _activation_logged
    with _lock:
        _enabled = False
        _activation_logged = False
