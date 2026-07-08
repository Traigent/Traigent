from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from traigent.observability.dtos import (
    ObservationRecord,
    PromptLinkRecord,
    SessionRecord,
    TraceRecord,
)


RecordFactory = Callable[[dict[str, Any]], Any]


def _prompt_link_payload() -> dict[str, Any]:
    return {
        "id": "prompt_link_1",
        "trace_id": "trace_1",
    }


def _trace_payload() -> dict[str, Any]:
    return {
        "id": "trace_1",
        "name": "trace",
    }


def _session_payload() -> dict[str, Any]:
    return {
        "id": "session_1",
    }


def _observation_payload() -> dict[str, Any]:
    return {
        "id": "observation_1",
        "trace_id": "trace_1",
        "type": "generation",
        "name": "llm-call",
    }


_RECORD_CASES = [
    pytest.param(
        PromptLinkRecord.from_dict,
        _prompt_link_payload,
        ("input_tokens", "output_tokens", "total_tokens", "cost_usd", "latency_ms"),
        id="prompt-link",
    ),
    pytest.param(
        TraceRecord.from_dict,
        _trace_payload,
        (
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_cost_usd",
            "total_latency_ms",
        ),
        id="trace",
    ),
    pytest.param(
        SessionRecord.from_dict,
        _session_payload,
        (
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_cost_usd",
            "total_latency_ms",
        ),
        id="session",
    ),
    pytest.param(
        ObservationRecord.from_dict,
        _observation_payload,
        ("latency_ms", "input_tokens", "output_tokens", "total_tokens", "cost_usd"),
        id="observation",
    ),
]


@pytest.mark.parametrize(("factory", "base_payload", "usage_fields"), _RECORD_CASES)
@pytest.mark.parametrize("unknown_shape", ["null", "absent"])
def test_observability_query_usage_fields_parse_unknown_as_none(
    factory: RecordFactory,
    base_payload: Callable[[], dict[str, Any]],
    usage_fields: tuple[str, ...],
    unknown_shape: str,
):
    payload = base_payload()
    if unknown_shape == "null":
        payload.update(dict.fromkeys(usage_fields))

    record = factory(payload)

    for field in usage_fields:
        assert getattr(record, field) is None


@pytest.mark.parametrize(("factory", "base_payload", "usage_fields"), _RECORD_CASES)
def test_observability_query_usage_fields_keep_explicit_zero(
    factory: RecordFactory,
    base_payload: Callable[[], dict[str, Any]],
    usage_fields: tuple[str, ...],
):
    payload = base_payload()
    payload.update(dict.fromkeys(usage_fields, 0))

    record = factory(payload)

    for field in usage_fields:
        value = getattr(record, field)
        assert value == 0
        assert value is not None
