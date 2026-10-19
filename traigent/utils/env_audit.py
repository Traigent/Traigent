"""Warn about unrecognised ``TRAIGENT_*`` environment variables (#2669).

A misspelt variable (a dropped letter in a cost-limit name) is otherwise ignored without
a trace and the default silently applies. The set of recognised names is
derived from the package's own source (every ``TRAIGENT_*`` name the code
mentions), so it cannot drift from the real reads. The check is advisory:
it only emits a warning, lists variable NAMES (never values) and never raises.
"""

from __future__ import annotations

import difflib
import os
import re
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path


_NAME_RE = re.compile(r"\bTRAIGENT_[A-Z0-9_]*")
_PREFIX = "TRAIGENT_"


@lru_cache(maxsize=1)
def known_traigent_env_names() -> frozenset[str]:
    """Return every ``TRAIGENT_*`` name mentioned in the package source.

    Tokens ending in ``_`` are dynamic prefixes (``f"TRAIGENT_{key}"``); they
    are kept so :func:`find_unknown_traigent_env_vars` can treat any variable
    starting with one as recognised.
    """
    root = Path(__file__).resolve().parent.parent
    names: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        names.update(_NAME_RE.findall(text))
    return frozenset(names)


def find_unknown_traigent_env_vars(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Map each unrecognised ``TRAIGENT_*`` name to its nearest known name."""
    env = os.environ if environ is None else environ
    candidates = [n for n in env if n.startswith(_PREFIX)]
    if not candidates:
        return {}
    known = known_traigent_env_names()
    prefixes = tuple(n for n in known if n.endswith("_") and n != _PREFIX)
    exact = sorted(n for n in known if not n.endswith("_"))
    unknown: dict[str, str | None] = {}
    for name in sorted(candidates):
        if name in known or (prefixes and name.startswith(prefixes)):
            continue
        match = difflib.get_close_matches(name, exact, n=1, cutoff=0.8)
        unknown[name] = match[0] if match else None
    return unknown


def warn_unknown_traigent_env_vars(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Emit one warning naming unrecognised ``TRAIGENT_*`` variables (no values)."""
    try:
        unknown = find_unknown_traigent_env_vars(environ)
    except Exception:  # pragma: no cover - advisory check must never raise
        return
    if not unknown:
        return
    parts = [
        f"{name} (did you mean {hint}?)" if hint else name
        for name, hint in unknown.items()
    ]
    from traigent.utils.exceptions import TraigentWarning

    warnings.warn(
        "Unrecognised TRAIGENT_* environment variable(s) ignored: " + ", ".join(parts),
        TraigentWarning,
        stacklevel=2,
    )
