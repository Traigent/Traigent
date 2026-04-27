"""Strict-bool regression test for ``_validate_api_key_with_backend``.

The backend response body must be parsed strictly: only the literal boolean
``True`` for the ``valid`` key indicates a successful validation. Truthy
non-boolean values (e.g. the string ``"false"``, the int ``1``) MUST be
treated as a rejection so a misconfigured or hostile backend cannot smuggle
a fake-pass through the SDK.

Repro for the prior weakness:

    if isinstance(data, dict) and data.get("valid"):  # truthy check
        return None

A response of ``{"valid": "false"}`` would silently authenticate because the
non-empty string ``"false"`` is truthy in Python. The fix tightens this to
``data.get("valid") is True``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.auth import AuthManager


class _FakeResponse:
    """Minimal aiohttp response stub for the success-status path."""

    def __init__(self, payload):
        self.status = 200
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self, content_type=None):  # noqa: ARG002
        return self._payload


class _FakeSession:
    """Stub aiohttp.ClientSession returning a configurable JSON payload."""

    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None):  # noqa: ARG002
        return _FakeResponse(self._payload)


def _patch_aiohttp(payload):
    """Patch the aiohttp surface used by ``_validate_api_key_with_backend``."""
    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=_FakeSession(payload))
    fake_aiohttp.ClientTimeout = MagicMock()
    return patch("traigent.cloud.auth.aiohttp", fake_aiohttp)


@pytest.mark.asyncio
async def test_validate_rejects_string_false_payload():
    """A backend returning ``{"valid": "false"}`` must NOT authenticate."""
    manager = AuthManager()

    with _patch_aiohttp({"valid": "false"}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is not None, (
        "String 'false' was treated as truthy — strict ``is True`` check missing"
    )
    assert "invalid" in reason.lower() or "reported" in reason.lower()


@pytest.mark.asyncio
async def test_validate_rejects_truthy_int_payload():
    """A backend returning ``{"valid": 1}`` must NOT authenticate (strict bool)."""
    manager = AuthManager()

    with _patch_aiohttp({"valid": 1}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is not None
    assert "invalid" in reason.lower() or "reported" in reason.lower()


@pytest.mark.asyncio
async def test_validate_accepts_strict_true_payload():
    """A backend returning the literal ``{"valid": True}`` is the only success."""
    manager = AuthManager()

    with _patch_aiohttp({"valid": True}):
        reason = await manager._validate_api_key_with_backend("tg_" + "x" * 61)

    assert reason is None
