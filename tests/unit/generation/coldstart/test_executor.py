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


def test_exports_are_closed() -> None:
    assert generation.__all__ == ["coldstart"]
    assert coldstart.__all__ == [
        "ColdStartResult",
        "DiscoveryGap",
        "Receipt",
        "build_cold_start_eval_set",
    ]
    assert not hasattr(traigent, "build_cold_start_eval_set")


def test_only_coldstart_source_remains() -> None:
    package = Path(generation.__file__).parent
    source_files = {
        path.relative_to(package).as_posix() for path in package.rglob("*.py")
    }

    assert source_files == {
        "__init__.py",
        "coldstart/__init__.py",
        "coldstart/executor.py",
        "coldstart/models.py",
    }

    forbidden_exports = {
        "BackendGuidanceProvider",
        "ExampleSynthesizer",
        "GuidanceLoop",
        "PromptRewriter",
        "SkillTrainOptions",
        "SkillTrainer",
    }
    assert not {name for name in forbidden_exports if hasattr(generation, name)}


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
