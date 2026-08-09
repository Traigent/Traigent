"""Focused contracts for the opaque cold-start SDK boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import traigent
from traigent import generation
from traigent.generation import coldstart
from traigent.generation.coldstart import build_cold_start_eval_set


def test_builder_forwards_payload_to_injected_transport() -> None:
    payload = {"request": "opaque"}
    received: list[Mapping[str, Any]] = []

    def transport(request: Mapping[str, Any]) -> Mapping[str, Any]:
        received.append(request)
        return {"handle": "result-1", "status": "complete", "gaps": [], "receipts": []}

    build_cold_start_eval_set(payload, transport=transport)

    assert received == [payload]
    assert received[0] is payload


def test_builder_parses_opaque_response() -> None:
    result = build_cold_start_eval_set(
        {"request": "opaque"},
        transport=lambda payload: {
            "handle": "result-1",
            "status": "complete",
            "gaps": [{"handle": "gap-1", "status": "pending"}],
            "receipts": [{"handle": "receipt-1", "status": "issued"}],
        },
    )

    assert result.handle == "result-1"
    assert result.status == "complete"
    assert [(item.handle, item.status) for item in result.gaps] == [
        ("gap-1", "pending")
    ]
    assert [(item.handle, item.status) for item in result.receipts] == [
        ("receipt-1", "issued")
    ]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"handle": "result-1", "status": "complete", "gaps": [], "receipts": "bad"},
        {
            "handle": "result-1",
            "status": "complete",
            "gaps": [{"handle": 1, "status": "pending"}],
            "receipts": [],
        },
    ],
)
def test_builder_rejects_malformed_response(response: Mapping[str, Any]) -> None:
    with pytest.raises(ValueError, match="malformed cold-start response"):
        build_cold_start_eval_set({}, transport=lambda payload: response)


def test_coldstart_survives_generation_package_edits() -> None:
    """coldstart must stay exported from traigent.generation.

    Regression guard: restoring guided generation (revert of 114d9386) rewrote
    traigent/generation/__init__.py wholesale and dropped coldstart from both the
    imports and __all__. The subpackage stayed importable by path, so nothing
    failed loudly -- it simply vanished from the package's public surface.
    """
    assert "coldstart" in generation.__all__
    assert generation.coldstart is coldstart


def test_exports_are_closed() -> None:
    assert coldstart.__all__ == [
        "ColdStartResult",
        "DiscoveryGap",
        "Receipt",
        "build_cold_start_eval_set",
    ]
    assert not hasattr(traigent, "build_cold_start_eval_set")


def test_implementation_has_no_forbidden_markers() -> None:
    package = Path(coldstart.__file__).parent
    source = "\n".join(
        path.read_text(encoding="utf-8").lower() for path in package.glob("*.py")
    )
    forbidden = [
        "inspect.",
        "verifier",
        "candidate",
        "plan_",
        "score",
        "prompt",
        "threshold",
        "urllib",
        "http",
        "api_key",
        "write_text",
        "jsonl",
    ]
    assert not {marker for marker in forbidden if marker in source}
