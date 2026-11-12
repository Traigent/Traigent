"""Raise a wrapped transport error with no exception chain attached (#3264).

``raise X from None`` inside an ``except`` block only sets
``__suppress_context__``; the transport exception stays reachable through
``X.__context__``. aiohttp ``ClientResponseError`` and requests
``RequestException`` carry the request (``RequestInfo`` / ``PreparedRequest``)
whose headers include the live API key, and their ``repr`` prints it.
``raise_detached`` clears ``__cause__`` and ``__context__`` after the raise has
chained, so nothing in the raised error's chain can leak request headers.
"""

from __future__ import annotations

from typing import NoReturn


def raise_detached(exc: BaseException) -> NoReturn:
    """Raise ``exc`` with ``__cause__`` and ``__context__`` both cleared."""
    try:
        raise exc from None
    except BaseException as raised:
        # The interpreter attaches the handled exception as __context__ during the
        # raise; clear it afterwards, then re-raise bare (a bare raise does not
        # re-chain).
        raised.__cause__ = None
        raised.__context__ = None
        raised.__suppress_context__ = True
        raise
