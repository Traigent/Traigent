"""Guard for analytics options that are accepted but not implemented."""

from __future__ import annotations

from typing import Any


def reject_unsupported(
    func: str, name: str, value: Any, allowed: tuple = (None,)
) -> None:
    """Raise if a caller supplies an option this release does not honour.

    Silently ignoring a supplied option would return results that look tuned
    by it, so unsupported values fail with a clear error instead.
    """
    if any(value is a or (type(value) is type(a) and value == a) for a in allowed):
        return
    raise ValueError(
        f"{func}() does not support the {name!r} option; it is not implemented "
        f"and would be ignored. Omit it."
    )
