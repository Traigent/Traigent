"""Safe construction of URL path segments from caller-supplied identifiers."""

from __future__ import annotations

from urllib.parse import quote


def quote_path_segment(value: object, *, field: str = "id") -> str:
    """Percent-encode ``value`` so it stays exactly one URL path segment.

    ``/``, ``?``, ``#``, ``%`` and whitespace are all encoded (``safe=""``), so an
    identifier can never add path components, start a query string or a fragment.
    Empty, whitespace-only, ``.`` and ``..`` values are rejected without echoing
    the value, since URL normalisation would drop or resolve them.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a non-empty string.")
    text = str(value)
    if not text.strip() or text in {".", ".."}:
        raise ValueError(f"{field} is not a valid identifier.")
    return quote(text, safe="")
