"""Tests for aiohttp session cleanup — no ResourceWarning on shutdown.

Validates that BackendIntegratedClient and TraigentCloudClient properly
drain the event loop after session.close(), preventing the
"Unclosed client session" warning when asyncio.run() tears down.
"""

from __future__ import annotations

import asyncio
import gc
import socket
import subprocess
import sys
import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_TEST_API_KEY = "tg_test_" + "x" * 56


def _resource_warning_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, ResourceWarning)
    ]


def _unclosed_resource_warning_messages(
    caught: list[warnings.WarningMessage],
) -> list[str]:
    return [
        message
        for message in _resource_warning_messages(caught)
        if "Unclosed" in message
    ]


class _LoopClosedTransport:
    """Socket-backed transport that exposes whether close/abort was attempted."""

    def __init__(self) -> None:
        self._sock, self._peer = socket.socketpair()
        self._closing = False

    def close(self) -> None:
        self._closing = True
        raise RuntimeError("Event loop is closed")

    def abort(self) -> None:
        self._closing = True
        raise RuntimeError("Event loop is closed")

    def is_closing(self) -> bool:
        return self._closing

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        if name == "socket":
            return self._sock
        return default

    def fileno(self) -> int:
        return self._sock.fileno()

    def cleanup(self) -> None:
        self._peer.close()
        if self._sock.fileno() != -1:
            self._sock.close()


class _TrackedProtocol:
    def __init__(self, transport: _LoopClosedTransport) -> None:
        self.transport = transport

    def abort(self) -> None:
        self.transport.abort()

    def close(self) -> None:
        self.transport.close()


async def _build_backend_client_with_lazy_session() -> Any:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    client = BackendIntegratedClient(
        api_key=_TEST_API_KEY,
        base_url="http://localhost:8000",
        backend_config=BackendClientConfig(backend_base_url="http://localhost:8000"),
        enable_fallback=False,
    )
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"X-API-Key": _TEST_API_KEY}
    )
    await client._ensure_session()
    return client


async def _build_cloud_client_with_lazy_session() -> Any:
    from traigent.cloud.client import TraigentCloudClient

    client = TraigentCloudClient(
        api_key=_TEST_API_KEY,
        base_url="http://localhost:8000",
    )
    client.auth.get_headers = AsyncMock(return_value={"X-API-Key": _TEST_API_KEY})
    await client._ensure_session()
    return client


def _build_client_on_private_loop(client_kind: str) -> tuple[Any, Any, Any]:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if client_kind == "backend":
        client = loop.run_until_complete(_build_backend_client_with_lazy_session())
    elif client_kind == "cloud":
        client = loop.run_until_complete(_build_cloud_client_with_lazy_session())
    else:  # pragma: no cover - test authoring guard
        raise AssertionError(f"unknown client kind: {client_kind}")

    session = client._session
    assert session is not None
    connector = session._connector
    assert connector is not None
    return loop, client, connector


@pytest.mark.parametrize("client_kind", ["backend", "cloud"])
def test_finalizer_closes_transports_when_session_loop_is_already_closed(
    client_kind: str,
) -> None:
    """Finalizer must close connector transports after the aiohttp loop is closed."""

    loop, client, connector = _build_client_on_private_loop(client_kind)
    pooled_transport = _LoopClosedTransport()
    acquired_transport = _LoopClosedTransport()
    cleanup_transport = _LoopClosedTransport()

    try:
        connector._conns[object()].append((_TrackedProtocol(pooled_transport), 0.0))
        connector._acquired.add(_TrackedProtocol(acquired_transport))
        connector._cleanup_closed_transports.append(cleanup_transport)

        loop.close()
        asyncio.set_event_loop(None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            doomed_client = client
            client = None
            del doomed_client
            gc.collect()
            gc.collect()

        assert pooled_transport.is_closing()
        assert pooled_transport.fileno() == -1
        assert acquired_transport.is_closing()
        assert acquired_transport.fileno() == -1
        assert cleanup_transport.is_closing()
        assert cleanup_transport.fileno() == -1
        assert connector.closed
        assert not connector._conns
        assert not connector._acquired
        assert not _unclosed_resource_warning_messages(caught)
    finally:
        if client is not None and not loop.is_closed():
            loop.run_until_complete(client.close())
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)
        pooled_transport.cleanup()
        acquired_transport.cleanup()
        cleanup_transport.cleanup()


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
    async def test_finalizer_closes_lazy_session_without_explicit_close(self):
        """GC finalizer should close a lazily-created session if close() is skipped."""
        from traigent.cloud.backend_client import (
            BackendClientConfig,
            BackendIntegratedClient,
        )

        client = BackendIntegratedClient(
            api_key="tg_test_" + "x" * 56,
            base_url="http://localhost:8000",
            backend_config=BackendClientConfig(
                backend_base_url="http://localhost:8000"
            ),
            enable_fallback=False,
        )
        client.auth_manager.auth.get_headers = AsyncMock(
            return_value={"X-API-Key": "tg_test_" + "x" * 56}
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            await client._ensure_session()
            assert client._session is not None
            del client
            gc.collect()
            await asyncio.sleep(0)
            gc.collect()

        assert not _unclosed_resource_warning_messages(caught)

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
    async def test_finalizer_closes_lazy_session_without_explicit_close(self):
        """GC finalizer should close a lazily-created session if close() is skipped."""
        from traigent.cloud.client import TraigentCloudClient

        client = TraigentCloudClient(
            api_key="tg_test_" + "x" * 56,
            base_url="http://localhost:8000",
        )
        client.auth.get_headers = AsyncMock(
            return_value={"X-API-Key": "tg_test_" + "x" * 56}
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            await client._ensure_session()
            assert client._session is not None
            del client
            gc.collect()
            await asyncio.sleep(0)
            gc.collect()

        assert not _unclosed_resource_warning_messages(caught)

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
