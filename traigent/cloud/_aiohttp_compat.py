"""aiohttp compatibility module for graceful degradation.

This module provides mock aiohttp classes when aiohttp is not installed,
allowing cloud modules to import and have consistent type checking while
raising appropriate errors at runtime if aiohttp features are used.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - minimal environments
    AIOHTTP_AVAILABLE = False

    class _MockClientError(Exception):
        """Replacement for aiohttp.ClientError when aiohttp is unavailable."""

    class _MockClientTimeout:
        """Replacement for aiohttp.ClientTimeout."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _MockClientSession:
        """ClientSession stub that raises ImportError on use."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("aiohttp not available") from None

        async def __aenter__(self) -> _MockClientSession:
            raise ImportError("aiohttp not available") from None

        async def __aexit__(self, *exc: Any) -> None:
            raise ImportError("aiohttp not available") from None

    class _MockTCPConnector:
        """Replacement for aiohttp.TCPConnector."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _MockAioHttp:
        """Mock aiohttp module with stub classes."""

        ClientSession = _MockClientSession
        ClientTimeout = _MockClientTimeout
        ClientError = _MockClientError
        ClientConnectorError = _MockClientError
        TCPConnector = _MockTCPConnector

    aiohttp = cast(Any, _MockAioHttp())

if TYPE_CHECKING:
    import aiohttp  # pragma: no cover

__all__ = ["aiohttp", "AIOHTTP_AVAILABLE"]
