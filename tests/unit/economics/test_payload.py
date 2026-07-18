"""Payload construction, idempotency, and the typed eligible-funnel helper (WI-B)."""

from __future__ import annotations

import logging
import re

import pytest

from traigent.economics.contract import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    IDEMPOTENCY_KEY_PATTERN,
    MAX_BATCH_EVENTS,
    SOURCE_KIND,
)
from traigent.economics.errors import EconomicsTelemetryContractError, EgressPolicyError
from traigent.economics.payload import (
    build_telemetry_request,
    funnel_eligible_event,
    utc_now_z,
)

_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?Z$")


def test_utc_now_z_matches_contract_form() -> None:
    assert _UTC_RE.match(utc_now_z())


def test_funnel_eligible_event_shape() -> None:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    assert event == {
        "event_type": "funnel_event",
        "event_id": "evt-1",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "stage": "eligible",
        "outcome": "entered",
    }


def test_funnel_eligible_event_generates_opaque_event_id() -> None:
    event = funnel_eligible_event("proj-1")
    assert event["event_id"].startswith("evt-")
    assert _UTC_RE.match(event["occurred_at"])


def test_funnel_eligible_event_requires_project_ref() -> None:
    with pytest.raises(EconomicsTelemetryContractError, match="project_ref"):
        funnel_eligible_event("  ")


def test_build_request_envelope_and_source() -> None:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    body = build_telemetry_request(
        [event], batch_id="batch-1", sent_at="2026-07-18T10:00:01.000Z"
    )
    assert body["contract"] == CONTRACT_ID
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["batch_id"] == "batch-1"
    assert body["sent_at"] == "2026-07-18T10:00:01.000Z"
    assert body["source"]["kind"] == SOURCE_KIND
    assert body["events"][0] == event
    assert re.match(IDEMPOTENCY_KEY_PATTERN, body["idempotency_key"])


def test_idempotency_key_is_stable_across_identical_builds() -> None:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    a = build_telemetry_request(
        [event], batch_id="b", sent_at="2026-07-18T10:00:01.000Z"
    )
    b = build_telemetry_request(
        [event], batch_id="b", sent_at="2026-07-18T10:00:01.000Z"
    )
    assert a["idempotency_key"] == b["idempotency_key"]
    assert a == b


def test_idempotency_key_changes_with_content() -> None:
    e1 = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    e2 = funnel_eligible_event(
        "proj-2", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    k1 = build_telemetry_request(
        [e1], batch_id="b", sent_at="2026-07-18T10:00:01.000Z"
    )["idempotency_key"]
    k2 = build_telemetry_request(
        [e2], batch_id="b", sent_at="2026-07-18T10:00:01.000Z"
    )["idempotency_key"]
    assert k1 != k2


def test_explicit_idempotency_key_is_honoured() -> None:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    body = build_telemetry_request([event], idempotency_key="my-stable-key-123")
    assert body["idempotency_key"] == "my-stable-key-123"


def test_invalid_explicit_idempotency_key_rejected() -> None:
    event = funnel_eligible_event("proj-1")
    with pytest.raises(EconomicsTelemetryContractError, match="IdempotencyKey"):
        build_telemetry_request([event], idempotency_key="short")


def test_default_batch_id_is_content_stable() -> None:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    b1 = build_telemetry_request([event], sent_at="2026-07-18T10:00:01.000Z")
    b2 = build_telemetry_request([event], sent_at="2026-07-18T10:00:09.000Z")
    assert b1["batch_id"] == b2["batch_id"]
    assert b1["batch_id"].startswith("batch-")


def test_empty_batch_rejected() -> None:
    with pytest.raises(EconomicsTelemetryContractError, match="at least one event"):
        build_telemetry_request([])


def test_oversized_batch_rejected() -> None:
    event = funnel_eligible_event("proj-1")
    with pytest.raises(EconomicsTelemetryContractError, match="cannot exceed"):
        build_telemetry_request([event] * (MAX_BATCH_EVENTS + 1))


def test_non_object_event_rejected() -> None:
    with pytest.raises(EconomicsTelemetryContractError, match="must be an object"):
        build_telemetry_request(["not-an-event"])  # type: ignore[list-item]


def test_events_must_be_a_sequence() -> None:
    with pytest.raises(EconomicsTelemetryContractError, match="sequence"):
        build_telemetry_request("evt")  # type: ignore[arg-type]


def test_run_economics_event_runs_egress_enforcement() -> None:
    # A run_economics event carrying a withheld+present characterization is
    # refused at build time, before any transport.
    run_event = {
        "event_type": "run_economics",
        "event_id": "evt-2",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "run_id": "run-1",
        "characterization": {
            "bands": {"error_cost_band": "not_measured"},
            "field_reports": [
                {
                    "field": "error_cost_band",
                    "provenance": "asked",
                    "confidence": 1.0,
                    "sharing_outcome": "withheld_by_policy",
                }
            ],
        },
    }
    with pytest.raises(EgressPolicyError):
        build_telemetry_request([run_event])


def test_build_deep_copies_events() -> None:
    event = funnel_eligible_event(
        "proj-1", event_id="evt-1", occurred_at="2026-07-18T10:00:00.000Z"
    )
    body = build_telemetry_request(
        [event], batch_id="b", sent_at="2026-07-18T10:00:01.000Z"
    )
    body["events"][0]["stage"] = "mutated"
    assert event["stage"] == "eligible"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected_locally(bad: float) -> None:
    # A non-finite number cannot be represented in the bounded contract types and
    # must fail locally rather than travel as an invalid JSON token.
    event = {
        "event_type": "run_economics",
        "event_id": "evt-9",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "run_id": "run-1",
        "actual_spend_usd": bad,
    }
    with pytest.raises(EconomicsTelemetryContractError):
        build_telemetry_request([event])


def test_serialization_error_is_payload_free() -> None:
    # A non-serializable value must fail as a typed contract error whose
    # cause/context chain does not carry the offending payload.
    from traigent.economics.payload import canonical_json

    class _Sensitive:
        def __repr__(self) -> str:  # pragma: no cover - must never be chained
            return "SENSITIVE-REPR-55521"

    with pytest.raises(EconomicsTelemetryContractError) as excinfo:
        canonical_json({"x": _Sensitive()})
    chain = _exception_chain(excinfo.value)
    assert all("SENSITIVE-REPR-55521" not in str(link) for link in chain)
    assert not any(isinstance(link, TypeError) for link in chain)


# --- HIGH: lone-surrogate paths fail closed, payload-free -----------------------
#
# `canonical_json_bytes` is the single UTF-8 chokepoint shared by batch-id
# derivation, idempotency-key derivation, and wire serialization. A lone
# surrogate is a valid `str` but not valid UTF-8; `.encode("utf-8")` would raise
# a UnicodeEncodeError whose `.object` carries the ENTIRE canonical payload. The
# helper catches it and raises a fresh, payload-free contract error outside the
# handler (cause/context both None), so no path can leak the payload.

_LONE_SURROGATE = "\ud800"  # valid Python str, invalid UTF-8


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and not any(c is current for c in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _assert_surrogate_error_is_payload_free(
    excinfo: pytest.ExceptionInfo[EconomicsTelemetryContractError], marker: str
) -> None:
    chain = _exception_chain(excinfo.value)
    assert all(marker not in str(link) for link in chain)
    assert not any(isinstance(link, UnicodeError) for link in chain)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_canonical_json_bytes_lone_surrogate_is_payload_free() -> None:
    from traigent.economics.payload import canonical_json_bytes

    marker = f"SENSITIVE-BYTES-{_LONE_SURROGATE}-9931"
    with pytest.raises(EconomicsTelemetryContractError) as excinfo:
        canonical_json_bytes({"secret_field": marker})
    _assert_surrogate_error_is_payload_free(excinfo, "SENSITIVE-BYTES")


def _event_with_marker(marker: str) -> dict[str, object]:
    return {
        "event_type": "funnel_event",
        "event_id": "evt-1",
        "occurred_at": "2026-07-18T10:00:00.000Z",
        "project_ref": "proj-1",
        "stage": "eligible",
        "outcome": "entered",
        "note": marker,
    }


def test_batch_id_derivation_lone_surrogate_is_payload_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No batch_id supplied -> the batch-id derivation routes through the chokepoint.
    marker = f"SENSITIVE-BATCHID-{_LONE_SURROGATE}-7777"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EconomicsTelemetryContractError) as excinfo:
            build_telemetry_request([_event_with_marker(marker)])
    _assert_surrogate_error_is_payload_free(excinfo, "SENSITIVE-BATCHID")
    assert "SENSITIVE-BATCHID" not in caplog.text


def test_idempotency_key_derivation_lone_surrogate_is_payload_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An explicit batch_id skips batch-id derivation, so the failure surfaces from
    # the idempotency-key derivation path (also the chokepoint).
    marker = f"SENSITIVE-IDKEY-{_LONE_SURROGATE}-8888"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EconomicsTelemetryContractError) as excinfo:
            build_telemetry_request(
                [_event_with_marker(marker)],
                batch_id="batch-fixed-1",
                sent_at="2026-07-18T10:00:01.000Z",
            )
    _assert_surrogate_error_is_payload_free(excinfo, "SENSITIVE-IDKEY")
    assert "SENSITIVE-IDKEY" not in caplog.text
