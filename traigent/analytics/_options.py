"""Guard for analytics options that are accepted but not implemented."""

from __future__ import annotations

import warnings
from typing import Any


def warn_unsupported(
    func: str, name: str, value: Any, allowed: tuple = (None,)
) -> None:
    """Warn if a caller supplies an option this release does not honour.

    The call still proceeds, but the caller is told the option has no effect
    instead of silently receiving results that look tuned by it. Passing a
    value listed in ``allowed`` (the former default) changes nothing and does
    not warn.
    """
    if any(value is a or (type(value) is type(a) and value == a) for a in allowed):
        return
    warnings.warn(
        f"{func}() option {name!r} is not implemented and has no effect.",
        UserWarning,
        stacklevel=3,
    )
