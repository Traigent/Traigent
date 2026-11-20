"""The agent code version an acceptance binds (RFC 0012 section 2.2, 2.3).

A code identifier, never customer content: an explicit string, or a digest of the
decorated function's module source.  Resolution order (first hit wins):

1. the ``TRAIGENT_AGENT_CODE_VERSION`` environment variable (the deployment states
   which code it really runs, with no source edit),
2. ``Tuning(code_version=...)``,
3. ``"src:" + sha256`` of the decorated function's module source (``inspect``).

When none can be determined there is NO default: promotion evidence is refused
(``E_NO_CODE_VERSION``), because an acceptance that binds no code binds nothing.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import sys
from typing import Any

from traigent.tuning.errors import TuningError

E_NO_CODE_VERSION = "E_NO_CODE_VERSION"
E_INVALID_CODE_VERSION = "E_INVALID_CODE_VERSION"
CODE_VERSION_ENV = "TRAIGENT_AGENT_CODE_VERSION"
# Same shape as the CodeVersion definition of tvl-release-acceptance.schema.json.
_CODE_VERSION = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")


def check_code_version(value: Any, origin: str) -> str:
    """``value`` if it is a legal code identifier, else a closed error (never echoed)."""
    if not isinstance(value, str) or not _CODE_VERSION.match(value):
        raise TuningError(
            E_INVALID_CODE_VERSION,
            f"{origin} must be 1-128 characters of [A-Za-z0-9_.:+-]",
        )
    return value


_MAX_UNWRAP = 32


def _inner_callable(source: Any) -> Any:
    """The agent callable under any SDK or ``functools`` wrapper layers.

    Follows ``OptimizedFunction.func``, ``functools`` ``__wrapped__`` and bound
    ``__func__`` until none is left; the innermost callable is the agent.
    """
    seen: set[int] = set()
    current = source
    for _ in range(_MAX_UNWRAP):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        nxt = None
        for attr in ("__wrapped__", "func", "__func__"):
            candidate = getattr(current, attr, None)
            if candidate is not None and callable(candidate):
                nxt = candidate
                break
        if nxt is None:
            break
        current = nxt
    return current


def source_digest(source: Any) -> str | None:
    """``src:<sha256>`` of the module source of the AGENT under ``source``.

    Wrapper layers (``@traigent.optimize``, ``@observe``, ``functools.wraps``) are
    unwrapped first, so the digest follows the agent's own code and never the SDK
    wrapper's module.  None when the agent's source is unreadable: there is no
    fallback to an outer layer, because that would bind code that is not the agent.
    """
    current = _inner_callable(source)
    if current is None:
        return None
    module = sys.modules.get(getattr(current, "__module__", None) or "")
    if module is None:
        module = inspect.getmodule(current)
    if module is None:
        return None
    try:
        text = inspect.getsource(module)
    except (OSError, TypeError):
        return None
    return "src:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_code_version(explicit: str | None, source: Any) -> str:
    """The code version now in effect; ``TuningError`` when it cannot be determined."""
    env = os.environ.get(CODE_VERSION_ENV, "").strip()
    if env:
        return check_code_version(env, CODE_VERSION_ENV)
    if explicit is not None:
        return check_code_version(explicit, "code_version")
    digest = source_digest(source)
    if digest is None:
        raise TuningError(
            E_NO_CODE_VERSION,
            "the agent code version cannot be determined: set Tuning(code_version=...) "
            f"or {CODE_VERSION_ENV}; there is no default",
        )
    return digest


__all__ = [
    "CODE_VERSION_ENV",
    "E_INVALID_CODE_VERSION",
    "E_NO_CODE_VERSION",
    "check_code_version",
    "resolve_code_version",
    "source_digest",
]
