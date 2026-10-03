"""Shared helpers for parameters that were removed from the public API."""

from __future__ import annotations

from collections.abc import Iterable

# Inert parameters removed from the public API (Traigent#1370 Item 6). They are
# refused loudly instead of being absorbed into runtime overrides.
REMOVED_MOCK_PARAMETERS: frozenset[str] = frozenset(("mock_mode_config", "mock"))


def removed_mock_parameter_message(parameter_name: str) -> str:
    return (
        f"{parameter_name} parameter has been removed (it was inert). For "
        "tutorial or test code, call "
        "traigent.testing.enable_mock_mode_for_quickstart() instead."
    )


def reject_removed_mock_parameters(names: Iterable[str]) -> None:
    """Raise ``TypeError`` naming the first removed mock parameter in ``names``."""
    for name in sorted(set(names) & REMOVED_MOCK_PARAMETERS):
        raise TypeError(removed_mock_parameter_message(name))
