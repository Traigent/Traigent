"""Integration-test fixtures.

Currently provides ``block_network`` — a deny-list socket monkeypatch used
by the quickstart smoke test to prove the bundled demo runs without any
outbound network traffic.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest


class NetworkBlockedError(RuntimeError):
    """Raised when test code tries to open a network socket while
    :func:`block_network` is in effect.

    This is a separate exception class so smoke tests can catch (or not)
    network attempts unambiguously, without colliding with Python's
    built-in ``OSError`` hierarchy that real network failures use.
    """


_NETWORK_FAMILIES = {socket.AF_INET, socket.AF_INET6}


@pytest.fixture
def block_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Block outbound network socket creation for the duration of the test.

    Wraps ``socket.socket`` so any AF_INET / AF_INET6 construction raises
    :class:`NetworkBlockedError`. AF_UNIX (and ``socket.socketpair`` which
    uses it) is allowed — asyncio creates AF_UNIX pipes for its internal
    self-pipe mechanism on Linux, and those are not network access.
    Also wraps :func:`socket.create_connection` and
    :func:`socket.getaddrinfo` to catch DNS resolution attempts a layer
    above raw sockets.
    """
    real_socket = socket.socket

    class _BlockingSocket(real_socket):  # type: ignore[misc, valid-type]
        def __init__(  # noqa: D401 — wrapper, not a separate API
            self, family: int = socket.AF_INET, *args: object, **kwargs: object
        ) -> None:
            if family in _NETWORK_FAMILIES:
                raise NetworkBlockedError(
                    f"socket.socket(family={family!r}) — "
                    "test marked block_network expected no network access"
                )
            super().__init__(family, *args, **kwargs)  # type: ignore[arg-type]

    def _blocked_create_connection(
        address: tuple[str, int], *args: object, **kwargs: object
    ) -> None:
        raise NetworkBlockedError(
            f"socket.create_connection({address!r}) — "
            "test marked block_network expected no network access"
        )

    def _blocked_getaddrinfo(
        host: str, port: object, *args: object, **kwargs: object
    ) -> None:
        raise NetworkBlockedError(
            f"socket.getaddrinfo({host!r}, {port!r}) — "
            "test marked block_network expected no network access"
        )

    monkeypatch.setattr(socket, "socket", _BlockingSocket)
    monkeypatch.setattr(socket, "create_connection", _blocked_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked_getaddrinfo)
    yield
