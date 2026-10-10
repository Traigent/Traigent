"""Connector discovery through the installed package entry-point group."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, cast

ENTRY_POINT_GROUP = "traigent.connectors"


@dataclass(frozen=True, slots=True)
class ConnectorLoadFailure:
    """Closed diagnostic for an entry point that could not be imported."""

    reason: str = "entry_point_load_failed"


def discover_connectors() -> dict[str, Any]:
    selected = entry_points()
    entries = (
        selected.select(group=ENTRY_POINT_GROUP)
        if hasattr(selected, "select")
        else cast(Any, selected).get(ENTRY_POINT_GROUP, ())
    )
    found: dict[str, Any] = {}
    for entry in entries:
        if entry.name in found:
            raise ValueError(f"duplicate connector entry point: {entry.name}")
        try:
            found[entry.name] = entry.load()
        except Exception:
            # Do not expose implementation exception text, which can contain
            # customer paths, while allowing independent entries to load.
            found[entry.name] = ConnectorLoadFailure()
    return found
