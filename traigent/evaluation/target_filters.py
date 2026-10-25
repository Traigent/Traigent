"""Closed ``target_filters`` contract for observability-trace evaluators.

Mirrors TraigentSchema ``evaluator_definition_schema.json#/definitions/TargetFilters``
(TraigentSchema 36de73e9). Keys are optional and combine with AND; ``{}`` means
no filtering. ``environment``, ``release`` and ``name`` match any listed value;
``tags`` matches only traces carrying all listed tags. The filter is only
meaningful for ``observability_trace`` evaluators.

Validation errors name only fixed contract terms (allowed keys, limits), never
the caller's keys or values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict

__all__ = ["TargetFilters", "validate_target_filters"]


class TargetFilters(TypedDict, total=False):
    """Closed evaluator trace filter; every key is optional."""

    environment: list[str]
    release: list[str]
    name: list[str]
    tags: list[str]


_MAX_ITEMS = 20
_MAX_LENGTH: dict[str, int] = {
    "environment": 64,
    "release": 128,
    "name": 255,
    "tags": 128,
}


def validate_target_filters(target_filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate against the closed schema and return a wire-format copy.

    ``None`` yields ``{}``. Raises ``ValueError`` without echoing user values.
    """
    if target_filters is None:
        return {}
    if not isinstance(target_filters, Mapping):
        raise ValueError("target_filters must be an object (mapping)")
    allowed = ", ".join(_MAX_LENGTH)
    result: dict[str, Any] = {}
    for key, value in target_filters.items():
        if key not in _MAX_LENGTH:
            raise ValueError(
                f"target_filters contains an unsupported key; allowed keys: {allowed}"
            )
        if not isinstance(value, list):
            raise ValueError(f"target_filters.{key} must be a list of strings")
        if not 1 <= len(value) <= _MAX_ITEMS:
            raise ValueError(
                f"target_filters.{key} must contain 1 to {_MAX_ITEMS} items"
            )
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"target_filters.{key} items must be strings")
            if not 1 <= len(item) <= _MAX_LENGTH[key]:
                raise ValueError(
                    f"target_filters.{key} items must be 1 to "
                    f"{_MAX_LENGTH[key]} characters"
                )
        if len(set(value)) != len(value):
            raise ValueError(f"target_filters.{key} items must be unique")
        result[key] = list(value)
    return result
