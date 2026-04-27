"""B4 ROUND 3 + ROUND 7 regression tests: get_auth_headers() must fail closed.

Codex's external review identified that ``AuthManager.get_auth_headers()``
defeated the purpose of B4 backend validation: when ``authenticate()``
returned an unsuccessful ``AuthResult``, the method silently fell back to
``_get_api_key_headers()`` -- emitting ``X-API-Key`` / ``Authorization``
headers built from the rejected raw key.

After the fix, ``get_auth_headers()`` must raise ``AuthenticationError``
(``InvalidCredentialsError``) instead of returning fallback headers when
the underlying authentication fails.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from traigent.cloud.auth import (
    AuthenticationError,
    AuthManager,
    InvalidCredentialsError,
)


def _force_backend_reject():
    """Force ``_validate_api_key_with_backend`` to report a failure."""

    async def _fail(self, api_key):  # noqa: ARG001
        return "backend-rejected"

    return patch.object(
        AuthManager,
        "_validate_api_key_with_backend",
        new=_fail,
    )


@pytest.mark.asyncio
async def test_get_auth_headers_raises_when_backend_rejects_key():
    """Backend-rejected keys must NOT yield headers, even silently.

    This is the core regression: prior to the fix, the code would fall back
    to ``_get_api_key_headers()`` and ship the raw rejected key as
    ``X-API-Key`` / ``Authorization``.
    """
    manager = AuthManager(api_key="tg_" + "x" * 61)

    with _force_backend_reject():
        with pytest.raises(AuthenticationError):
            await manager.get_auth_headers()


@pytest.mark.asyncio
async def test_get_auth_headers_raises_invalid_credentials_subclass():
    """Confirm we raise the more specific ``InvalidCredentialsError``."""
    manager = AuthManager(api_key="tg_" + "x" * 61)

    with _force_backend_reject():
        with pytest.raises(InvalidCredentialsError):
            await manager.get_auth_headers()


@pytest.mark.asyncio
async def test_get_auth_headers_does_not_emit_api_key_on_failure():
    """Even via try/except, the raw key must never reach the headers dict."""
    manager = AuthManager(api_key="tg_" + "x" * 61)

    captured: dict[str, str] | None = None
    with _force_backend_reject():
        try:
            captured = await manager.get_auth_headers()
        except AuthenticationError:
            captured = None

    # If anything was returned, ensure no auth header carries the raw key.
    if captured is not None:  # pragma: no cover - defensive
        assert "X-API-Key" not in captured
        assert "Authorization" not in captured


@pytest.mark.asyncio
async def test_get_auth_headers_raises_when_credentials_none_after_auth():
    """B4 ROUND 7: _credentials=None while is_authenticated()=True must fail closed.

    Codex found that the post-authentication branch (line 898) returned {}
    when _credentials was None but _current_token existed -- yielding empty
    headers without an exception. That path is an internal inconsistency
    and must now raise InvalidCredentialsError.
    """
    manager = AuthManager(api_key="tg_" + "x" * 61)

    # Force is_authenticated() to return True while _credentials is None
    with patch.object(manager, "is_authenticated", return_value=True):
        manager._credentials = None  # simulate inconsistent state
        with pytest.raises(InvalidCredentialsError):
            await manager.get_auth_headers()
