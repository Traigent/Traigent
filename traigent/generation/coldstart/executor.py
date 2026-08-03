"""Minimal opaque cold-start client."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .models import ColdStartResult, DiscoveryGap, Receipt

_Transport = Callable[[Mapping[str, Any]], Mapping[str, Any]]
_RESPONSE_KEYS = frozenset({"handle", "status", "gaps", "receipts"})
_ITEM_KEYS = frozenset({"handle", "status"})


def build_cold_start_eval_set(
    payload: Mapping[str, Any], *, transport: _Transport
) -> ColdStartResult:
    """Forward ``payload`` to ``transport`` and parse its opaque response."""

    return _parse_response(transport(payload))


def _parse_response(response: Mapping[str, Any]) -> ColdStartResult:
    if not isinstance(response, Mapping) or set(response) != _RESPONSE_KEYS:
        raise ValueError("malformed cold-start response")
    handle = _string(response["handle"])
    status = _string(response["status"])
    return ColdStartResult(
        handle=handle,
        status=status,
        gaps=_parse_items(response["gaps"], DiscoveryGap),
        receipts=_parse_items(response["receipts"], Receipt),
    )


def _parse_items(
    items: object, item_type: type[DiscoveryGap] | type[Receipt]
) -> tuple[DiscoveryGap, ...] | tuple[Receipt, ...]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("malformed cold-start response")
    parsed = []
    for item in items:
        if not isinstance(item, Mapping) or set(item) != _ITEM_KEYS:
            raise ValueError("malformed cold-start response")
        parsed.append(
            item_type(handle=_string(item["handle"]), status=_string(item["status"]))
        )
    return tuple(parsed)


def _string(value: object) -> str:
    if type(value) is not str:
        raise ValueError("malformed cold-start response")
    return value
