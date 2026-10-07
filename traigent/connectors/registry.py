"""Connector discovery through the installed package entry-point group."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, cast


ENTRY_POINT_GROUP = "traigent.connectors"


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
        found[entry.name] = entry.load()
    return found
