"""B4 ROUND 4 regression tests: ``BackendIntegratedClient`` must fail closed.

Codex's external review identified a parallel fail-open path in
``traigent/cloud/backend_client.py`` that defeated the round-3 fix to
``AuthManager.get_auth_headers()``:

* ``_ensure_session()`` caught ``Exception`` from ``get_headers()`` and
  fell back to ``_build_session_fallback_headers()`` -- which in turn
  used ``_get_api_key_headers()`` and ultimately the raw stored
  ``_api_key_fallback`` to mint ``X-API-Key`` / ``Authorization``
  headers, even after the backend rejected the key.

After the round-4 fix:

* ``AuthenticationError`` (and its ``InvalidCredentialsError`` subclass)
  raised by ``get_headers()`` must propagate from ``_ensure_session``
  and ``__aenter__``.
* No raw-key fallback path may build auth headers.
* ``_build_session_fallback_headers()`` is removed entirely so it cannot
  be reached by any code path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from traigent.cloud.auth import (
    AuthenticationError,
    AuthManager,
    InvalidCredentialsError,
)
from traigent.cloud.backend_client import BackendIntegratedClient

_VALID_LOOKING_KEY = "tg_" + "x" * 61


def _force_backend_reject():
    """Force ``AuthManager._validate_api_key_with_backend`` to fail."""

    async def _fail(self, api_key):  # noqa: ARG001
        return "backend-rejected"

    return patch.object(
        AuthManager,
        "_validate_api_key_with_backend",
        new=_fail,
    )


@pytest.mark.asyncio
async def test_ensure_session_raises_when_backend_rejects_key():
    """``_ensure_session`` must surface auth failure, not silently mint headers.

    This is the core regression. Prior to round 4, ``_ensure_session``
    caught the ``InvalidCredentialsError`` raised by round-3
    ``get_auth_headers()`` and rebuilt the session with raw-key headers
    via ``_build_session_fallback_headers``.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        with _force_backend_reject():
            with pytest.raises(AuthenticationError):
                await client._ensure_session()
        # The session must NOT have been created with rejected-key headers.
        assert client._session is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ensure_session_raises_invalid_credentials_subclass():
    """The specific ``InvalidCredentialsError`` raised by auth must propagate."""
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        with _force_backend_reject():
            with pytest.raises(InvalidCredentialsError):
                await client._ensure_session()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_aenter_raises_when_backend_rejects_key():
    """The async context manager entry must also fail closed.

    Previously ``__aenter__`` swallowed every exception from
    ``get_headers()`` and built a session with empty auth headers. While
    that did not directly leak the raw key, it produced a usable session
    whose later requests could be paired with the rejected key elsewhere
    in the stack. After round 4, ``AuthenticationError`` must propagate.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        with _force_backend_reject():
            with pytest.raises(AuthenticationError):
                await client.__aenter__()
        assert client._session is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_no_session_fallback_helper_remains():
    """The unsafe helper must not exist on the class any more.

    ``_build_session_fallback_headers`` was the entry point that turned a
    rejected key into ``X-API-Key`` / ``Authorization`` headers. Round 4
    removes it outright so no future regression can re-introduce a fail-
    open call site.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        assert not hasattr(client, "_build_session_fallback_headers")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ensure_session_does_not_emit_raw_key_headers():
    """Direct guard: even on partial failure paths, no raw-key headers leak.

    Inspect ``client._session`` after a forced rejection: it must not
    exist, and (defensively) if it somehow does, its default headers must
    not contain the raw API key.
    """
    client = BackendIntegratedClient(api_key=_VALID_LOOKING_KEY)

    try:
        with _force_backend_reject():
            try:
                await client._ensure_session()
            except AuthenticationError:
                pass

        session = client._session
        if session is not None:  # pragma: no cover - defensive
            default_headers = dict(getattr(session, "_default_headers", {}) or {})
            assert _VALID_LOOKING_KEY not in default_headers.get("X-API-Key", "")
            assert _VALID_LOOKING_KEY not in default_headers.get("Authorization", "")
    finally:
        await client.close()
