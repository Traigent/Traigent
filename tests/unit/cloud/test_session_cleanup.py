"""Tests for aiohttp session cleanup — no ResourceWarning on shutdown.

Validates that BackendIntegratedClient and TraigentCloudClient properly
drain the event loop after session.close(), preventing the
"Unclosed client session" warning when asyncio.run() tears down.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBackendClientSessionCleanup:
    """BackendIntegratedClient should not produce ResourceWarning on close."""

    @pytest.mark.asyncio
    async def test_no_unclosed_warning_with_real_session(self):
        """close() with a real aiohttp session should not warn on loop teardown.

        Runs a subprocess that creates a real aiohttp.ClientSession, closes it
        via BackendIntegratedClient.close(), and exits. If the drain sleep is
        missing, the subprocess stderr will contain 'Unclosed client session'.
        """
        script = """\
import asyncio
import warnings
warnings.simplefilter("always")

async def main():
    import aiohttp
    from traigent.cloud.backend_client import BackendIntegratedClient

    client = object.__new__(BackendIntegratedClient)
    client._session = aiohttp.ClientSession()
    client._session_lock = asyncio.Lock()
    await client.close()

asyncio.run(main())
"""
        result = subprocess.run(
            [sys.executable, "-Walways", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"Subprocess crashed (rc={result.returncode}):\n{result.stderr}"
        )
        assert "Unclosed client session" not in result.stderr, (
            f"Got unclosed session warning:\n{result.stderr}"
        )

    @pytest.mark.asyncio
    async def test_aexit_delegates_to_close(self):
        """__aexit__ should delegate to close() to preserve subclass extensibility."""
        from traigent.cloud.backend_client import BackendIntegratedClient

        client = object.__new__(BackendIntegratedClient)
        client._session = AsyncMock()
        client._session.closed = False
        client._session_lock = asyncio.Lock()

        with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
            await client.__aexit__(None, None, None)

        mock_close.assert_awaited_once_with(_reason="context-exit")

    @pytest.mark.asyncio
    async def test_reset_skips_drain_on_retry_path(self):
        """_reset_http_session should NOT sleep on retry/error paths."""
        from traigent.cloud.backend_client import BackendIntegratedClient

        mock_session = AsyncMock()
        mock_session.closed = False

        client = object.__new__(BackendIntegratedClient)
        client._session = mock_session
        client._session_lock = asyncio.Lock()

        with patch(
            "traigent.cloud.backend_client.asyncio.sleep", new_callable=AsyncMock
        ):
            await client._reset_http_session("retry")

    @pytest.mark.asyncio
    async def test_reset_drains_on_shutdown_path(self):
        """_reset_http_session should sleep(0.25) on shutdown path."""
        from traigent.cloud.backend_client import BackendIntegratedClient

        mock_session = AsyncMock()
        mock_session.closed = False

        client = object.__new__(BackendIntegratedClient)
        client._session = mock_session
        client._session_lock = asyncio.Lock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._reset_http_session("shutdown")

        mock_session.close.assert_awaited_once()
        mock_sleep.assert_awaited_once_with(0.25)


class TestCloudClientSessionCleanup:
    """TraigentCloudClient should not produce ResourceWarning on close."""

    @pytest.mark.asyncio
    async def test_close_passes_shutdown_reason(self):
        """close() should pass reason='shutdown' to _close_http_session."""
        from traigent.cloud.client import TraigentCloudClient

        client = object.__new__(TraigentCloudClient)

        with patch.object(
            client, "_close_http_session", new_callable=AsyncMock
        ) as mock_close:
            await client.close()

        mock_close.assert_awaited_once_with(reason="shutdown")

    @pytest.mark.asyncio
    async def test_aexit_delegates_to_close(self):
        """__aexit__ should delegate to close() to preserve subclass extensibility."""
        from traigent.cloud.client import TraigentCloudClient

        client = object.__new__(TraigentCloudClient)

        with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
            await client.__aexit__(None, None, None)

        mock_close.assert_awaited_once_with(_reason="context-exit")

    @pytest.mark.asyncio
    async def test_close_http_session_drains_on_shutdown(self):
        """_close_http_session should sleep(0.25) on shutdown path."""
        from traigent.cloud.client import TraigentCloudClient

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        client = object.__new__(TraigentCloudClient)
        client._aio_session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._close_http_session(reason="shutdown")

        mock_session.close.assert_awaited_once()
        mock_sleep.assert_awaited_once_with(0.25)

    @pytest.mark.asyncio
    async def test_close_http_session_skips_drain_on_retry(self):
        """_close_http_session should NOT sleep on retry/error paths."""
        from traigent.cloud.client import TraigentCloudClient

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        client = object.__new__(TraigentCloudClient)
        client._aio_session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._close_http_session(reason="retry")

        mock_session.close.assert_awaited_once()
        mock_sleep.assert_not_awaited()
