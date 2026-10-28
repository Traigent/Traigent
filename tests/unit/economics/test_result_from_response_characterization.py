"""Characterization of TelemetryIngestResult.from_response: order, reads, boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast, overload

import pytest

from traigent.economics import result as result_module
from traigent.economics.contract import CONTRACT_ID, CONTRACT_VERSION
from traigent.economics.errors import EconomicsResponseError
from traigent.economics.result import Rejection, TelemetryIngestResult

Trace = list[tuple[Any, ...]]

_KEY = "key-12345678"
_BATCH = "batch-abc"
_EVENT_IDS = ("e0", "e1", "e2")


class TracedBody(dict[str, Any]):
    """A dict body recording every ``get`` and optionally raising on a key."""

    def __init__(
        self,
        trace: Trace,
        data: dict[str, Any],
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        super().__init__(data)
        self._trace = trace
        self._raise_on = raise_on or {}

    @overload
    def get(self, key: str, default: None = None, /) -> Any | None: ...

    @overload
    def get(self, key: str, default: Any, /) -> Any: ...

    def get(self, key: str, default: Any = None, /) -> Any:
        self._trace.append(("get", key))
        if key in self._raise_on:
            raise self._raise_on[key]
        return super().get(key, default)


def _body(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "contract": "economics_telemetry",
        "contract_version": "1.0.0",
        "replayed": False,
        "batch_id": _BATCH,
        "idempotency_key": _KEY,
        "received_at": "2026-01-02T03:04:05.006Z",
        "counts": {"submitted": 3, "accepted": 2, "duplicate": 1, "rejected": 0},
        "rejections": [],
    }
    base.update(over)
    return base


@pytest.fixture
def trace(monkeypatch: pytest.MonkeyPatch) -> Trace:
    events: Trace = []
    real_counts: Callable[[Any], tuple[int, int, int, int]] = (
        result_module._parse_counts
    )
    real_rejections: Callable[..., tuple[Rejection, ...]] = (
        result_module._parse_rejections
    )

    def validate(body: Any, *, http_status: int) -> None:
        events.append(("validate", http_status))

    def counts(raw: Any) -> tuple[int, int, int, int]:
        events.append(("counts",))
        return real_counts(raw)

    def rejections(raw: Any, **kwargs: Any) -> tuple[Rejection, ...]:
        events.append(("rejections", kwargs))
        return real_rejections(raw, **kwargs)

    monkeypatch.setattr(result_module, "validate_response_or_fail", validate)
    monkeypatch.setattr(result_module, "_parse_counts", counts)
    monkeypatch.setattr(result_module, "_parse_rejections", rejections)
    return events


def _parse(body: Any, status: int = 201, **over: Any) -> TelemetryIngestResult:
    kwargs: dict[str, Any] = {
        "http_status": status,
        "body": body,
        "expected_idempotency_key": _KEY,
        "expected_batch_id": _BATCH,
        "expected_submitted": 3,
        "expected_event_ids": _EVENT_IDS,
    }
    kwargs.update(over)
    return TelemetryIngestResult.from_response(**kwargs)


def _err(body: Any, status: int = 201, **over: Any) -> str:
    with pytest.raises(EconomicsResponseError) as info:
        _parse(body, status, **over)
    return str(info.value)


def test_fresh_201_result_is_exact(trace: Trace) -> None:
    assert _parse(_body()) == TelemetryIngestResult(
        http_status=201,
        replayed=False,
        batch_id=_BATCH,
        idempotency_key=_KEY,
        contract="economics_telemetry",
        contract_version="1.0.0",
        received_at="2026-01-02T03:04:05.006Z",
        submitted=3,
        accepted=2,
        duplicate=1,
        rejected=0,
        rejections=(),
    )


def test_replayed_200_result_is_exact(trace: Trace) -> None:
    result = _parse(_body(replayed=True), 200)
    assert result.http_status == 200
    assert result.replayed is True


def test_all_rejected_422_result_is_exact(trace: Trace) -> None:
    body = _body(
        counts={"submitted": 3, "accepted": 0, "duplicate": 0, "rejected": 3},
        rejections=[
            {"event_index": 0, "reason": "schema_violation", "event_id": "e0"},
            {"event_index": 2, "reason": "schema_violation", "detail": "x"},
            {"event_index": 1, "reason": "schema_violation"},
        ],
    )
    result = _parse(body, 422)
    assert result.rejections == (
        Rejection(event_index=0, reason="schema_violation", event_id="e0"),
        Rejection(event_index=2, reason="schema_violation", event_id=None),
        Rejection(event_index=1, reason="schema_violation", event_id=None),
    )
    assert (result.submitted, result.accepted, result.duplicate, result.rejected) == (
        3,
        0,
        0,
        3,
    )


def test_ids_are_coerced_to_plain_str(trace: Trace) -> None:
    class S(str):
        pass

    result = _parse(_body(batch_id=S(_BATCH), idempotency_key=S(_KEY)))
    assert type(result.batch_id) is str
    assert type(result.idempotency_key) is str


def test_read_trace_on_success(trace: Trace) -> None:
    reads: Trace = []
    body = TracedBody(reads, _body())
    _parse(body)
    assert reads == [
        ("get", "contract"),
        ("get", "contract_version"),
        ("get", "replayed"),
        ("get", "batch_id"),
        ("get", "idempotency_key"),
        ("get", "received_at"),
        ("get", "counts"),
        ("get", "rejections"),
    ]
    assert trace == [
        ("validate", 201),
        ("counts",),
        (
            "rejections",
            {"submitted": 3, "rejected": 0, "expected_event_ids": _EVENT_IDS},
        ),
    ]


@pytest.mark.parametrize(
    ("key", "prefix"),
    [
        ("contract", ["contract"]),
        ("contract_version", ["contract", "contract_version"]),
        ("replayed", ["contract", "contract_version", "replayed"]),
        ("batch_id", ["contract", "contract_version", "replayed", "batch_id"]),
        (
            "idempotency_key",
            ["contract", "contract_version", "replayed", "batch_id", "idempotency_key"],
        ),
        (
            "received_at",
            [
                "contract",
                "contract_version",
                "replayed",
                "batch_id",
                "idempotency_key",
                "received_at",
            ],
        ),
        (
            "counts",
            [
                "contract",
                "contract_version",
                "replayed",
                "batch_id",
                "idempotency_key",
                "received_at",
                "counts",
            ],
        ),
        (
            "rejections",
            [
                "contract",
                "contract_version",
                "replayed",
                "batch_id",
                "idempotency_key",
                "received_at",
                "counts",
                "rejections",
            ],
        ),
    ],
)
def test_raising_accessor_propagates_with_partial_reads(
    trace: Trace, key: str, prefix: list[str]
) -> None:
    reads: Trace = []
    body = TracedBody(reads, _body(), raise_on={key: RuntimeError("read " + key)})
    with pytest.raises(RuntimeError, match="read " + key):
        _parse(body)
    assert [r[1] for r in reads] == prefix


def test_raise_once_accessor_is_read_exactly_once(trace: Trace) -> None:
    state = {"n": 0}

    class Flaky(TracedBody):
        @overload
        def get(self, key: str, default: None = None, /) -> Any | None: ...

        @overload
        def get(self, key: str, default: Any, /) -> Any: ...

        def get(self, key: str, default: Any = None, /) -> Any:
            if key == "received_at":
                state["n"] += 1
                if state["n"] == 1:
                    raise RuntimeError("once")
            return super().get(key, default)

    with pytest.raises(RuntimeError, match="once"):
        _parse(Flaky([], _body()))
    assert state["n"] == 1


def test_unexpected_status_is_rejected_before_validation(trace: Trace) -> None:
    for status in (0, 202, 400, 500):
        assert _err(_body(), status) == (
            f"unexpected ingest status {status} for a result body"
        )
    assert trace == []


def test_unhashable_status_is_not_wrapped(trace: Trace) -> None:
    with pytest.raises(TypeError):
        _parse(_body(), cast(Any, [200]))


def test_validation_runs_before_the_object_check(trace: Trace) -> None:
    assert _err(["not", "a", "dict"]) == "ingest response was not a JSON object"
    assert trace == [("validate", 201)]


def test_validation_error_propagates_and_stops_parsing(
    monkeypatch: pytest.MonkeyPatch, trace: Trace
) -> None:
    def boom(body: Any, *, http_status: int) -> None:
        raise EconomicsResponseError("validator said no")

    monkeypatch.setattr(result_module, "validate_response_or_fail", boom)
    reads: Trace = []
    assert _err(TracedBody(reads, _body())) == "validator said no"
    assert reads == []


@pytest.mark.parametrize(
    ("over", "status", "message"),
    [
        ({"contract": "other"}, 201, "ingest response names a different contract"),
        (
            {"contract_version": "0"},
            201,
            "ingest response names a different contract version",
        ),
        (
            {"replayed": 1},
            201,
            "ingest response replayed flag is missing or non-boolean",
        ),
        (
            {"replayed": None},
            201,
            "ingest response replayed flag is missing or non-boolean",
        ),
        ({"replayed": True}, 201, "ingest response status and replayed flag disagree"),
        ({"replayed": False}, 200, "ingest response status and replayed flag disagree"),
        ({"replayed": True}, 422, "ingest response status and replayed flag disagree"),
        (
            {"batch_id": "nope"},
            201,
            "ingest response batch_id does not echo the request",
        ),
        (
            {"idempotency_key": "nope"},
            201,
            "ingest response idempotency_key does not echo the request",
        ),
        ({"received_at": ""}, 201, "ingest response received_at is missing"),
        ({"received_at": 5}, 201, "ingest response received_at is missing"),
        ({"received_at": None}, 201, "ingest response received_at is missing"),
        ({"counts": None}, 201, "ingest response counts are missing"),
        ({"counts": []}, 201, "ingest response counts are missing"),
        (
            {"counts": {"submitted": 3, "accepted": 2, "duplicate": -1, "rejected": 0}},
            201,
            "ingest response count 'duplicate' is invalid",
        ),
        (
            {
                "counts": {
                    "submitted": 3,
                    "accepted": 2,
                    "duplicate": 1,
                    "rejected": True,
                }
            },
            201,
            "ingest response count 'rejected' is invalid",
        ),
        (
            {"counts": {"submitted": 3, "accepted": 2, "duplicate": 1}},
            201,
            "ingest response count 'rejected' is invalid",
        ),
        (
            {"counts": {"submitted": 4, "accepted": 4, "duplicate": 0, "rejected": 0}},
            201,
            "ingest response submitted count does not match the batch",
        ),
        (
            {"counts": {"submitted": 3, "accepted": 2, "duplicate": 0, "rejected": 0}},
            201,
            "ingest response counts do not reconcile",
        ),
        (
            {"counts": {"submitted": 3, "accepted": 2, "duplicate": 2, "rejected": 0}},
            201,
            "ingest response counts do not reconcile",
        ),
        (
            {"counts": {"submitted": 3, "accepted": 3, "duplicate": 0, "rejected": 0}},
            422,
            "422 response is not an all-rejected batch",
        ),
        (
            {"counts": {"submitted": 3, "accepted": 0, "duplicate": 0, "rejected": 3}},
            201,
            "all-rejected batch must be reported as 422, not a fresh 201",
        ),
        (
            {"rejections": None},
            201,
            "ingest response rejections is missing or not a list",
        ),
        (
            {"rejections": [{"event_index": 0, "reason": "schema_violation"}]},
            201,
            "ingest response rejections length does not match count",
        ),
    ],
)
def test_exact_error_messages(
    trace: Trace, over: dict[str, Any], status: int, message: str
) -> None:
    body = _body(**over)
    if status == 200 and "replayed" not in over:
        body["replayed"] = True
    assert _err(body, status) == message


def test_check_order_pairs(trace: Trace) -> None:
    """Each earlier violation wins over every later one."""
    bad: dict[str, Any] = {
        "contract": "x",
        "contract_version": "x",
        "replayed": "x",
        "batch_id": "x",
        "idempotency_key": "x",
        "received_at": "",
        "counts": None,
        "rejections": None,
    }
    expected = [
        "ingest response names a different contract",
        "ingest response names a different contract version",
        "ingest response replayed flag is missing or non-boolean",
        "ingest response batch_id does not echo the request",
        "ingest response idempotency_key does not echo the request",
        "ingest response received_at is missing",
        "ingest response counts are missing",
        "ingest response rejections is missing or not a list",
    ]
    order = list(bad)
    for position, message in enumerate(expected):
        body = _body()
        for key in order[position:]:
            body[key] = bad[key]
        assert _err(body) == message


def test_status_replayed_mismatch_wins_over_batch_id(trace: Trace) -> None:
    body = _body(replayed=True, batch_id="nope")
    assert _err(body) == "ingest response status and replayed flag disagree"


def test_count_checks_order(trace: Trace) -> None:
    # submitted mismatch beats reconcile mismatch
    body = _body(counts={"submitted": 4, "accepted": 0, "duplicate": 0, "rejected": 0})
    assert _err(body) == "ingest response submitted count does not match the batch"
    # reconcile mismatch beats the 422 rule
    body = _body(counts={"submitted": 3, "accepted": 0, "duplicate": 0, "rejected": 1})
    assert _err(body, 422) == "ingest response counts do not reconcile"
    # 422 rule beats rejections parsing
    body = _body(rejections=None)
    assert _err(body, 422) == "422 response is not an all-rejected batch"


def test_422_boundaries(trace: Trace) -> None:
    empty = _body(counts={"submitted": 0, "accepted": 0, "duplicate": 0, "rejected": 0})
    assert _err(empty, 422, expected_submitted=0) == (
        "422 response is not an all-rejected batch"
    )
    one = _body(
        counts={"submitted": 1, "accepted": 0, "duplicate": 0, "rejected": 1},
        rejections=[{"event_index": 0, "reason": "schema_violation"}],
    )
    assert _parse(one, 422, expected_submitted=1).rejected == 1


def test_all_rejected_replay_200_is_allowed_and_201_is_not(trace: Trace) -> None:
    counts = {"submitted": 1, "accepted": 0, "duplicate": 0, "rejected": 1}
    rejections = [{"event_index": 0, "reason": "schema_violation"}]
    replay = _body(replayed=True, counts=counts, rejections=rejections)
    assert _parse(replay, 200, expected_submitted=1).any_rejected is True
    fresh = _body(counts=counts, rejections=rejections)
    assert _err(fresh, 201, expected_submitted=1) == (
        "all-rejected batch must be reported as 422, not a fresh 201"
    )


def test_empty_201_batch_is_not_all_rejected(trace: Trace) -> None:
    counts = {"submitted": 0, "accepted": 0, "duplicate": 0, "rejected": 0}
    result = _parse(_body(counts=counts), 201, expected_submitted=0)
    assert result.submitted == 0


def test_partially_rejected_201_is_allowed(trace: Trace) -> None:
    counts = {"submitted": 3, "accepted": 1, "duplicate": 0, "rejected": 2}
    rejections = [
        {"event_index": 1, "reason": "schema_violation"},
        {"event_index": 2, "reason": "schema_violation", "event_id": "e2"},
    ]
    result = _parse(_body(counts=counts, rejections=rejections))
    assert result.rejection_reasons == ("schema_violation", "schema_violation")


def test_stops_after_first_failure_without_later_helpers(trace: Trace) -> None:
    _err(_body(received_at=""))
    assert trace == [("validate", 201)]
    trace.clear()
    _err(_body(counts={"submitted": 9, "accepted": 9, "duplicate": 0, "rejected": 0}))
    assert trace == [("validate", 201), ("counts",)]


def test_rejection_errors_propagate_from_the_parser(trace: Trace) -> None:
    counts = {"submitted": 3, "accepted": 2, "duplicate": 0, "rejected": 1}
    body = _body(
        counts=counts,
        rejections=[{"event_index": 1, "reason": "schema_violation", "event_id": "zz"}],
    )
    assert _err(body) == (
        "ingest response rejection event_id does not match the request "
        "event at that index"
    )
    assert trace[-1][0] == "rejections"


def test_echo_comparison_uses_raw_values_not_their_string_form(trace: Trace) -> None:
    assert _err(_body(batch_id=5), expected_batch_id="5") == (
        "ingest response batch_id does not echo the request"
    )
    assert _err(_body(idempotency_key=5), expected_idempotency_key="5") == (
        "ingest response idempotency_key does not echo the request"
    )


def test_contract_constants_are_pinned_to_independent_literals() -> None:
    assert CONTRACT_ID == "economics_telemetry"
    assert CONTRACT_VERSION == "1.0.0"
