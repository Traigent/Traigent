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


# ---------------------------------------------------------------------------
# Cost/usage fields (Schema cost-usage-fields): shared fixtures prove that null,
# zero, partial cost and provenance survive parsing in observation, trace and
# session records.
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_COST_FIXTURES = _json.loads(
    (
        _Path(__file__).parents[2] / "fixtures/observability/cost_usage_records_v1.json"
    ).read_text()
)


@pytest.mark.parametrize("case", _COST_FIXTURES["observations"], ids=lambda c: c["id"])
def test_observation_record_keeps_cost_and_usage_fields(case):
    from traigent.observability.dtos import ObservationRecord

    record = ObservationRecord.from_dict(case["payload"])
    for name, expected in case["expected"].items():
        got = getattr(record, name)
        assert got == expected, (case["id"], name)
        # null must stay None (unknown) and a reported zero must not become None
        assert (got is None) == (expected is None), (case["id"], name)
        assert isinstance(got, bool) == isinstance(expected, bool), (case["id"], name)


@pytest.mark.parametrize("case", _COST_FIXTURES["rollups"], ids=lambda c: c["id"])
def test_trace_and_session_records_keep_completeness_and_lower_bounds(case):
    from traigent.observability.dtos import SessionRecord, TraceRecord

    trace = TraceRecord.from_dict(
        {"id": "t", "name": "n", "status": "completed", **case["payload"]}
    )
    session = SessionRecord.from_dict({"id": "s", **case["payload"]})
    for record in (trace, session):
        for name, expected in case["expected"].items():
            got = getattr(record, name)
            assert got == expected, (case["id"], type(record).__name__, name)
            assert (got is None) == (expected is None)


def test_negative_control_the_fixtures_detect_dropped_fields():
    """The fixtures can fail: a parser that forgets cost_status/priced_cost_usd
    leaves them None and the partial-cost fixture notices."""
    from traigent.observability.dtos import ObservationRecord

    (partial,) = [
        c
        for c in _COST_FIXTURES["observations"]
        if c["id"] == "partial_cost_lower_bound"
    ]
    stripped = {
        k: v
        for k, v in partial["payload"].items()
        if k not in {"cost_status", "priced_cost_usd"}
    }
    record = ObservationRecord.from_dict(stripped)
    assert record.cost_status != partial["expected"]["cost_status"]
    assert record.priced_cost_usd != partial["expected"]["priced_cost_usd"]


# #2605: the backend's trace/session read responses carry the scope the data
# was stored under; the records must keep it so a caller can confirm which
# project its traces landed in.
def test_trace_record_keeps_backend_tenant_and_project_ids() -> None:
    payload = {
        **_trace_payload(),
        "tenant_id": "tenant_a",
        "project_id": "project_b",
        "session": {
            **_session_payload(),
            "tenant_id": "tenant_a",
            "project_id": "project_b",
        },
    }

    record = TraceRecord.from_dict(payload)

    assert record.tenant_id == "tenant_a"
    assert record.project_id == "project_b"
    assert record.session is not None
    assert record.session.tenant_id == "tenant_a"
    assert record.session.project_id == "project_b"


def test_session_record_keeps_ids_on_session_and_nested_traces() -> None:
    payload = {
        **_session_payload(),
        "tenant_id": "tenant_a",
        "project_id": "project_b",
        "traces": [
            {**_trace_payload(), "tenant_id": "tenant_a", "project_id": "project_b"}
        ],
    }

    record = SessionRecord.from_dict(payload)

    assert (record.tenant_id, record.project_id) == ("tenant_a", "project_b")
    assert (record.traces[0].tenant_id, record.traces[0].project_id) == (
        "tenant_a",
        "project_b",
    )


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("factory", [TraceRecord.from_dict, SessionRecord.from_dict])
def test_records_default_scope_ids_to_none_when_absent_or_blank(
    factory: RecordFactory, value: str | None
) -> None:
    base = _trace_payload() if factory == TraceRecord.from_dict else _session_payload()

    absent = factory(base)
    blank = factory({**base, "tenant_id": value, "project_id": value})

    for record in (absent, blank):
        assert record.tenant_id is None
        assert record.project_id is None
