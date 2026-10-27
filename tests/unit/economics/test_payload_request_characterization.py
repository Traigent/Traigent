"""Characterization of telemetry batch assembly: order, defaults and error messages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from traigent.economics import payload as payload_module
from traigent.economics.contract import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    MAX_BATCH_EVENTS,
)
from traigent.economics.errors import EconomicsTelemetryContractError, EgressPolicyError
from traigent.economics.payload import build_telemetry_request

Trace = list[tuple[Any, ...]]

_SRC = {"kind": "sdk", "name": "src-name", "version": "1.2.3"}


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> Trace:
    """Record calls of every collaborator the batch builder touches."""
    events: Trace = []
    real_roundtrip = payload_module._json_roundtrip
    real_enforce = payload_module.enforce_characterization_egress
    real_batch = payload_module._derive_batch_id
    real_key = payload_module._derive_idempotency_key

    def roundtrip(event: Mapping[str, Any], index: int) -> dict[str, Any]:
        events.append(("roundtrip", index))
        return real_roundtrip(event, index)

    def enforce(characterization: Mapping[str, Any]) -> None:
        events.append(("enforce", dict(characterization)))
        real_enforce(characterization)

    def batch(evs: Sequence[Mapping[str, Any]]) -> str:
        events.append(("batch_id", len(evs)))
        return real_batch(evs)

    def key(body: Mapping[str, Any]) -> str:
        events.append(("key", sorted(body)))
        return real_key(body)

    def now() -> str:
        events.append(("now",))
        return "2026-01-02T03:04:05.006Z"

    def source() -> dict[str, str]:
        events.append(("source",))
        return dict(_SRC)

    monkeypatch.setattr(payload_module, "_json_roundtrip", roundtrip)
    monkeypatch.setattr(payload_module, "enforce_characterization_egress", enforce)
    monkeypatch.setattr(payload_module, "_derive_batch_id", batch)
    monkeypatch.setattr(payload_module, "_derive_idempotency_key", key)
    monkeypatch.setattr(payload_module, "utc_now_z", now)
    monkeypatch.setattr(payload_module, "build_source", source)
    return events


def _funnel(n: int = 0) -> dict[str, Any]:
    return {"event_type": "funnel_event", "event_id": f"evt-{n}"}


def _canon(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _err(*args: Any, **kwargs: Any) -> str:
    with pytest.raises(EconomicsTelemetryContractError) as info:
        build_telemetry_request(*args, **kwargs)
    return str(info.value)


def test_full_result_with_all_explicit_arguments(trace: Trace) -> None:
    events = [_funnel(1), _funnel(2)]
    result = build_telemetry_request(
        events,
        batch_id="batch-x",
        sent_at="2026-05-06T07:08:09.000Z",
        idempotency_key="key-12345",
        source={"name": "n"},
    )
    assert result == {
        "contract": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "batch_id": "batch-x",
        "sent_at": "2026-05-06T07:08:09.000Z",
        "source": {"name": "n"},
        "events": [_funnel(1), _funnel(2)],
        "idempotency_key": "key-12345",
    }
    assert list(result) == [
        "contract",
        "contract_version",
        "batch_id",
        "sent_at",
        "source",
        "events",
        "idempotency_key",
    ]
    assert trace == [("roundtrip", 0), ("roundtrip", 1)]
    assert result["events"][0] is not events[0]


def test_full_result_with_all_defaults_and_trace(trace: Trace) -> None:
    result = build_telemetry_request([_funnel(1)])
    body = {
        "contract": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "batch_id": result["batch_id"],
        "sent_at": "2026-01-02T03:04:05.006Z",
        "source": dict(_SRC),
        "events": [_funnel(1)],
    }
    digest = hashlib.sha256(_canon([_funnel(1)])).hexdigest()
    assert result["batch_id"] == "batch-" + digest[:32]
    key_digest = hashlib.sha256(_canon(body)).hexdigest()
    assert result == {**body, "idempotency_key": "econ-tel-" + key_digest[:48]}
    assert trace == [
        ("roundtrip", 0),
        ("batch_id", 1),
        ("now",),
        ("source",),
        ("key", sorted(body)),
    ]


def test_run_economics_events_enforce_after_roundtrip_in_order(trace: Trace) -> None:
    ok = {
        "event_type": "run_economics",
        "characterization": {
            "field_reports": [
                {
                    "field": "value_channel",
                    "provenance": "asked",
                    "sharing_outcome": "withheld_by_policy",
                }
            ]
        },
    }
    build_telemetry_request(
        [_funnel(1), ok, _funnel(2), ok],
        batch_id="b",
        sent_at="s",
        idempotency_key="key-12345",
        source={},
    )
    assert trace == [
        ("roundtrip", 0),
        ("roundtrip", 1),
        ("enforce", ok["characterization"]),
        ("roundtrip", 2),
        ("roundtrip", 3),
        ("enforce", ok["characterization"]),
    ]


@pytest.mark.parametrize("missing", [None, {}, [], 0, ""])
def test_falsy_characterization_is_enforced_as_empty_object(
    trace: Trace, missing: Any
) -> None:
    event = {"event_type": "run_economics", "characterization": missing}
    with pytest.raises(EgressPolicyError) as info:
        build_telemetry_request([event])
    assert str(info.value) == "characterization.field_reports must be a non-empty list"
    assert trace == [("roundtrip", 0), ("enforce", {})]


def test_absent_characterization_is_enforced_as_empty_object(trace: Trace) -> None:
    with pytest.raises(EgressPolicyError):
        build_telemetry_request([{"event_type": "run_economics"}])
    assert trace == [("roundtrip", 0), ("enforce", {})]


def test_egress_error_stops_before_later_events_and_key(trace: Trace) -> None:
    bad = {"event_type": "run_economics", "characterization": {}}
    with pytest.raises(EgressPolicyError):
        build_telemetry_request([bad, _funnel(1)], idempotency_key="")
    assert trace == [("roundtrip", 0), ("enforce", {})]


@pytest.mark.parametrize("bad", ["abc", b"abc", 5, None, {"a": 1}, {1, 2}])
def test_non_sequence_events_rejected_first(trace: Trace, bad: Any) -> None:
    assert _err(bad) == "events must be a sequence"
    assert trace == []


def test_generator_is_not_a_sequence(trace: Trace) -> None:
    assert _err(_funnel(i) for i in range(1)) == "events must be a sequence"


def test_tuple_is_accepted(trace: Trace) -> None:
    result = build_telemetry_request((_funnel(1),), idempotency_key="key-12345")
    assert result["events"] == [_funnel(1)]


def test_empty_batch_message(trace: Trace) -> None:
    assert _err([]) == "a telemetry batch requires at least one event"
    assert _err(()) == "a telemetry batch requires at least one event"
    assert trace == []


def test_batch_size_boundary(trace: Trace) -> None:
    events = [_funnel(i) for i in range(MAX_BATCH_EVENTS)]
    result = build_telemetry_request(
        events, batch_id="b", sent_at="s", idempotency_key="key-12345", source={}
    )
    assert len(result["events"]) == 500
    assert MAX_BATCH_EVENTS == 500
    trace.clear()
    events.append(_funnel(999))
    assert _err(events) == "a telemetry batch cannot exceed 500 events"
    assert trace == []


def test_size_checked_before_event_shapes(trace: Trace) -> None:
    assert _err(["x"] * (MAX_BATCH_EVENTS + 1)) == (
        "a telemetry batch cannot exceed 500 events"
    )


def test_events_are_listed_exactly_once() -> None:
    class Counted(list[Any]):
        iters = 0

        def __iter__(self) -> Iterator[Any]:
            type(self).iters += 1
            return super().__iter__()

    build_telemetry_request(
        Counted([_funnel(1)]),
        batch_id="b",
        sent_at="s",
        idempotency_key="key-12345",
        source={},
    )
    assert Counted.iters == 1


def test_raise_once_listing_error_propagates_without_retry() -> None:
    class Flaky(list[Any]):
        calls = 0

        def __iter__(self) -> Iterator[Any]:
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("once")
            return super().__iter__()

    events = Flaky([_funnel(1)])
    with pytest.raises(RuntimeError, match="once"):
        build_telemetry_request(events)
    assert Flaky.calls == 1


def test_non_mapping_event_message_uses_index(trace: Trace) -> None:
    assert _err([_funnel(1), "x"]) == "event 1 must be an object"
    assert trace == [("roundtrip", 0)]
    assert _err([5]) == "event 0 must be an object"


def test_non_serializable_event_message_and_no_chaining(trace: Trace) -> None:
    with pytest.raises(EconomicsTelemetryContractError) as info:
        build_telemetry_request([_funnel(1), {"x": float("nan")}])
    assert str(info.value) == (
        "event 1 is not JSON-serializable or contains a non-finite value"
    )
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is False
    assert info.value.__context__ is None
    assert trace == [("roundtrip", 0), ("roundtrip", 1)]


def test_event_type_is_read_from_the_copy_not_the_original(trace: Trace) -> None:
    class Tricky(dict[str, Any]):
        pass

    event = Tricky({"event_type": "run_economics", "characterization": {}})
    with pytest.raises(EgressPolicyError):
        build_telemetry_request([event])


@pytest.mark.parametrize("falsy", ["", None])
def test_falsy_batch_id_and_sent_at_are_derived(trace: Trace, falsy: Any) -> None:
    result = build_telemetry_request(
        [_funnel(1)], batch_id=falsy, sent_at=falsy, idempotency_key="key-12345"
    )
    assert result["batch_id"].startswith("batch-")
    assert result["sent_at"] == "2026-01-02T03:04:05.006Z"
    assert trace == [("roundtrip", 0), ("batch_id", 1), ("now",), ("source",)]


def test_explicit_batch_id_and_sent_at_skip_derivation(trace: Trace) -> None:
    build_telemetry_request(
        [_funnel(1)], batch_id="given", sent_at="when", idempotency_key="key-12345"
    )
    assert trace == [("roundtrip", 0), ("source",)]


def test_empty_source_is_explicit_and_copied(trace: Trace) -> None:
    given: dict[str, str] = {}
    result = build_telemetry_request(
        [_funnel(1)],
        batch_id="b",
        sent_at="s",
        idempotency_key="key-12345",
        source=given,
    )
    assert result["source"] == {}
    assert result["source"] is not given
    assert ("source",) not in trace


def test_source_mapping_is_converted_to_dict(trace: Trace) -> None:
    class Ro(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            return {"a": "b"}[key]

        def __iter__(self) -> Iterator[str]:
            return iter(["a"])

        def __len__(self) -> int:
            return 1

    result = build_telemetry_request(
        [_funnel(1)],
        batch_id="b",
        sent_at="s",
        idempotency_key="key-12345",
        source=Ro(),
    )
    assert type(result["source"]) is dict
    assert result["source"] == {"a": "b"}


def test_explicit_key_skips_derivation(trace: Trace) -> None:
    result = build_telemetry_request(
        [_funnel(1)], batch_id="b", sent_at="s", idempotency_key="abcdefgh", source={}
    )
    assert result["idempotency_key"] == "abcdefgh"
    assert not any(e[0] == "key" for e in trace)


@pytest.mark.parametrize(
    "key,ok",
    [
        ("", False),
        ("a" * 7, False),
        ("a" * 8, True),
        ("a" * 128, True),
        ("a" * 129, False),
        ("abcdefgh\n", False),
        ("abcdefg!", False),
        ("a.b_c:d-e1", True),
        (12345678, False),
        (b"abcdefgh", False),
    ],
)
def test_idempotency_key_validation_boundaries(
    trace: Trace, key: Any, ok: bool
) -> None:
    kwargs: dict[str, Any] = dict(
        batch_id="b", sent_at="s", idempotency_key=key, source={}
    )
    if ok:
        assert build_telemetry_request([_funnel(1)], **kwargs)["idempotency_key"] == key
    else:
        assert _err([_funnel(1)], **kwargs) == (
            "idempotency_key is not a valid IdempotencyKey"
        )


def test_invalid_key_is_reported_after_body_collaborators_ran(trace: Trace) -> None:
    assert _err([_funnel(1)], idempotency_key="bad") == (
        "idempotency_key is not a valid IdempotencyKey"
    )
    assert trace == [("roundtrip", 0), ("batch_id", 1), ("now",), ("source",)]


def test_derived_key_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, trace: Trace
) -> None:
    def boom(body: Mapping[str, Any]) -> str:
        raise RuntimeError("derive failed")

    monkeypatch.setattr(payload_module, "_derive_idempotency_key", boom)
    with pytest.raises(RuntimeError, match="derive failed"):
        build_telemetry_request([_funnel(1)])


@pytest.mark.parametrize("name", ["_derive_batch_id", "build_source", "utc_now_z"])
def test_collaborator_errors_propagate_in_body_order(
    monkeypatch: pytest.MonkeyPatch, trace: Trace, name: str
) -> None:
    def boom(*args: Any) -> Any:
        trace.append(("boom", name))
        raise RuntimeError(name)

    monkeypatch.setattr(payload_module, name, boom)
    with pytest.raises(RuntimeError, match=name):
        build_telemetry_request([_funnel(1)])
    expected_prefix = {
        "_derive_batch_id": [("roundtrip", 0), ("boom", name)],
        "utc_now_z": [("roundtrip", 0), ("batch_id", 1), ("boom", name)],
        "build_source": [("roundtrip", 0), ("batch_id", 1), ("now",), ("boom", name)],
    }[name]
    assert trace == expected_prefix


def test_input_events_are_not_mutated(trace: Trace) -> None:
    ev = {"event_type": "funnel_event", "nested": {"a": [1, 2]}}
    snapshot = json.dumps(ev)
    result = build_telemetry_request(
        [ev], batch_id="b", sent_at="s", idempotency_key="key-12345", source={}
    )
    assert json.dumps(ev) == snapshot
    assert result["events"][0]["nested"] is not ev["nested"]
