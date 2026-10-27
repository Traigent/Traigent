"""Characterization tests pinning process_record_linkage._send.

The transport and the bundle are recorders writing into one shared event
trace, so the order of the request, the opening handling and the result
recording is pinned together with the closed error codes that come back.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from traigent.cloud import process_record_linkage as plr
from traigent.cloud.process_record_linkage import (
    CommitmentLinkageClientError,
    ProcessRecordRequestError,
)

ROW_ID = "build_session_" + "0123456789abcdef" * 2
OPENINGS = {"dataset": b"\x01" * 32, "config_space": b"\x02" * 32}
ENCODED = {
    "dataset": base64.b64encode(b"\x01" * 32).decode("ascii"),
    "config_space": base64.b64encode(b"\x02" * 32).decode("ascii"),
}
RECORD: dict[str, Any] = {"certificate_id": "cert-1"}
LINKAGE: dict[str, Any] = {"digest": "linkage-1"}


class _Body(dict[str, Any]):
    """Response body recording reads into the shared trace."""

    def __init__(self, trace: list[Any], data: dict[str, Any]) -> None:
        super().__init__(data)
        self.trace = trace

    def get(self, key: str, default: Any = None) -> Any:
        self.trace.append(("body_get", key))
        return super().get(key, default)


class _Client:
    def __init__(
        self,
        trace: list[Any],
        finalize: list[Any],
        get: list[Any] | None = None,
    ) -> None:
        self.trace = trace
        self.finalize = list(finalize)
        self.get = list(get or [])

    def finalize_process_record_sync(
        self, build_session_row_id: str, payload: dict[str, Any]
    ) -> Any:
        self.trace.append(("finalize", build_session_row_id, dict(payload)))
        outcome = self.finalize.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def get_process_record_sync(self, build_session_row_id: str) -> Any:
        self.trace.append(("get", build_session_row_id))
        outcome = self.get.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Bundle:
    def __init__(self, trace: list[Any], openings: Any = None) -> None:
        self.trace = trace
        self._openings = dict(OPENINGS) if openings is None else openings

    @property
    def openings(self) -> Any:
        self.trace.append(("bundle_openings",))
        if isinstance(self._openings, BaseException):
            raise self._openings
        return self._openings

    def set_result(self, record: Any, linkage: Any) -> None:
        self.trace.append(("set_result", record, linkage))


def _send(
    client: Any,
    payload: dict[str, Any],
    bundle: Any,
    *,
    link: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = plr._send(client, ROW_ID, payload, bundle, link=link)
    return result


def _ok(trace: list[Any], data: Any = RECORD, linkage: Any = LINKAGE) -> _Body:
    return _Body(trace, {"data": data, "commitment_linkage": linkage})


# ------------------------------------------------------------------ success


def test_unlinked_send_without_bundle_returns_record_and_no_linkage() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    payload = {"refs": 1}
    result = _send(client, payload, None, link=False)
    assert result == {"record": RECORD, "commitment_linkage": None}
    assert list(result) == ["record", "commitment_linkage"]
    assert trace == [
        ("finalize", ROW_ID, {"refs": 1}),
        ("body_get", "data"),
        ("body_get", "commitment_linkage"),
    ]
    assert payload == {"refs": 1}


def test_unlinked_send_with_bundle_records_the_record_without_linkage() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    bundle = _Bundle(trace)
    result = _send(client, {"refs": 1}, bundle, link=False)
    assert result == {"record": RECORD, "commitment_linkage": None}
    assert trace == [
        ("finalize", ROW_ID, {"refs": 1}),
        ("body_get", "data"),
        ("body_get", "commitment_linkage"),
        ("set_result", RECORD, None),
    ]


def test_linked_send_sends_encoded_openings_and_records_both() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    bundle = _Bundle(trace)
    payload = {"refs": 1}
    result = _send(client, payload, bundle, link=True)
    assert result == {"record": RECORD, "commitment_linkage": LINKAGE}
    assert result["record"] is RECORD
    assert result["commitment_linkage"] is LINKAGE
    assert trace == [
        ("bundle_openings",),
        ("finalize", ROW_ID, {"refs": 1, "openings": ENCODED}),
        ("body_get", "data"),
        ("body_get", "commitment_linkage"),
        ("set_result", RECORD, LINKAGE),
    ]
    assert payload == {"refs": 1}
    assert "openings" not in payload


def test_linked_send_without_bundle_sends_no_openings() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    result = _send(client, {"refs": 1}, None, link=True)
    assert result == {"record": RECORD, "commitment_linkage": LINKAGE}
    assert trace[0] == ("finalize", ROW_ID, {"refs": 1})


def test_unlinked_send_never_reads_bundle_openings() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    _send(client, {}, _Bundle(trace), link=False)
    assert ("bundle_openings",) not in trace
    assert trace[0] == ("finalize", ROW_ID, {})


def test_unlinked_send_drops_openings_already_in_the_payload_afterwards() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    payload = {"openings": {"x": "y"}, "refs": 1}
    _send(client, payload, None, link=False)
    assert trace[0] == ("finalize", ROW_ID, {"openings": {"x": "y"}, "refs": 1})
    assert payload == {"refs": 1}


def test_linkage_value_is_read_even_when_unlinked() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace, linkage="junk")])
    result = _send(client, {}, None, link=False)
    assert result["commitment_linkage"] is None
    assert ("body_get", "commitment_linkage") in trace


# ------------------------------------------------------- response validation


@pytest.mark.parametrize("record", [None, [], "text", 0, ()])
def test_non_dict_record_is_refused_before_the_linkage_check(record: Any) -> None:
    trace: list[Any] = []
    bundle = _Bundle(trace)
    client = _Client(trace, [_ok(trace, data=record, linkage=None)])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, bundle, link=True)
    assert info.value.code == "BACKEND_REFUSED"
    assert not any(event[0] == "set_result" for event in trace)


@pytest.mark.parametrize("body", [None, [], "text", 5, ()])
def test_non_dict_body_is_refused(body: Any) -> None:
    trace: list[Any] = []
    bundle = _Bundle(trace)
    client = _Client(trace, [body])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, bundle, link=False)
    assert info.value.code == "BACKEND_REFUSED"
    assert not any(event[0] == "set_result" for event in trace)


def test_empty_dict_body_is_refused() -> None:
    client = _Client([], [{}])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, None, link=False)
    assert info.value.code == "BACKEND_REFUSED"


@pytest.mark.parametrize("linkage", [None, [], "text", 0, ()])
def test_linked_send_without_dict_linkage_records_the_record_then_fails(
    linkage: Any,
) -> None:
    trace: list[Any] = []
    bundle = _Bundle(trace)
    client = _Client(trace, [_ok(trace, linkage=linkage)])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, bundle, link=True)
    assert info.value.code == "LINKAGE_ABSENT"
    assert trace[-1] == ("set_result", RECORD, None)


def test_linked_send_without_linkage_and_without_bundle_just_fails() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace, linkage=None)])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, None, link=True)
    assert info.value.code == "LINKAGE_ABSENT"


def test_empty_dict_record_and_linkage_are_accepted() -> None:
    trace: list[Any] = []
    bundle = _Bundle(trace)
    client = _Client(trace, [_ok(trace, data={}, linkage={})])
    result = _send(client, {}, bundle, link=True)
    assert result == {"record": {}, "commitment_linkage": {}}
    assert trace[-1] == ("set_result", {}, {})


# ------------------------------------------------------- failure translation


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("COMMITMENT_LINKAGE_DISABLED", "COMMITMENT_LINKAGE_DISABLED"),
        ("commitment_linkage_opening_shape", "COMMITMENT_LINKAGE_OPENING_SHAPE"),
        ("Issuance_In_Progress", "ISSUANCE_IN_PROGRESS"),
        ("BACKEND_REFUSED", "BACKEND_REFUSED"),
        ("something_else", "BACKEND_REFUSED"),
        ("", "BACKEND_REFUSED"),
        (None, "BACKEND_REFUSED"),
        ("TRANSPORT_ERROR", "TRANSPORT_ERROR"),
    ],
)
def test_request_errors_map_to_closed_codes(raw: str | None, code: str) -> None:
    trace: list[Any] = []
    bundle = _Bundle(trace)
    client = _Client(trace, [ProcessRecordRequestError(400, raw)])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, bundle, link=True)
    assert info.value.code == code
    assert not any(event[0] in ("set_result", "get") for event in trace)


@pytest.mark.parametrize(
    "error", [RuntimeError("secret body"), ValueError("v"), OSError()]
)
def test_other_errors_become_transport_error_without_context(error: Exception) -> None:
    client = _Client([], [error])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, None, link=False)
    assert info.value.code == "TRANSPORT_ERROR"
    assert info.value.__context__ is None
    assert info.value.__cause__ is None
    assert "secret body" not in str(info.value)


def test_request_error_is_translated_without_context() -> None:
    client = _Client([], [ProcessRecordRequestError(409, "duplicate_issuance")])
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, None, link=False)
    assert info.value.code == "DUPLICATE_ISSUANCE"
    assert info.value.__context__ is None


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_propagate_and_still_drop_the_openings(
    exc_type: type[BaseException],
) -> None:
    trace: list[Any] = []
    client = _Client(trace, [exc_type()])
    payload = {"refs": 1}
    with pytest.raises(exc_type):
        _send(client, payload, _Bundle(trace), link=True)
    assert payload == {"refs": 1}
    assert not any(event[0] == "set_result" for event in trace)


def test_openings_are_dropped_after_a_translated_failure() -> None:
    trace: list[Any] = []
    client = _Client(trace, [RuntimeError("x")])
    payload = {"refs": 1}
    with pytest.raises(CommitmentLinkageClientError):
        _send(client, payload, _Bundle(trace), link=True)
    assert trace[1] == ("finalize", ROW_ID, {"refs": 1, "openings": ENCODED})
    assert payload == {"refs": 1}


# ----------------------------------------------------------------- duplicates


def test_duplicate_issuance_when_linked_fetches_the_existing_record() -> None:
    trace: list[Any] = []
    client = _Client(
        trace,
        [ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE")],
        [_ok(trace)],
    )
    bundle = _Bundle(trace)
    payload = {"refs": 1}
    result = _send(client, payload, bundle, link=True)
    assert result == {"record": RECORD, "commitment_linkage": LINKAGE}
    assert trace == [
        ("bundle_openings",),
        ("finalize", ROW_ID, {"refs": 1, "openings": ENCODED}),
        ("get", ROW_ID),
        ("body_get", "data"),
        ("body_get", "commitment_linkage"),
        ("set_result", RECORD, LINKAGE),
    ]
    assert payload == {"refs": 1}


def test_duplicate_issuance_when_linked_without_bundle_still_fetches() -> None:
    trace: list[Any] = []
    client = _Client(
        trace, [ProcessRecordRequestError(409, "duplicate_issuance")], [_ok(trace)]
    )
    result = _send(client, {}, None, link=True)
    assert result == {"record": RECORD, "commitment_linkage": LINKAGE}
    assert ("get", ROW_ID) in trace


def test_duplicate_issuance_when_unlinked_is_raised_without_fetching() -> None:
    trace: list[Any] = []
    client = _Client(
        trace, [ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE")], [_ok(trace)]
    )
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, _Bundle(trace), link=False)
    assert info.value.code == "DUPLICATE_ISSUANCE"
    assert not any(event[0] == "get" for event in trace)


@pytest.mark.parametrize(
    ("get_outcome", "code"),
    [
        (ProcessRecordRequestError(404, "backend_refused"), "BACKEND_REFUSED"),
        (
            ProcessRecordRequestError(503, "issuance_in_progress"),
            "ISSUANCE_IN_PROGRESS",
        ),
        (RuntimeError("boom"), "TRANSPORT_ERROR"),
    ],
)
def test_duplicate_fetch_failure_uses_the_closed_code_of_the_fetch(
    get_outcome: Exception, code: str
) -> None:
    trace: list[Any] = []
    client = _Client(
        trace, [ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE")], [get_outcome]
    )
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, _Bundle(trace), link=True)
    assert info.value.code == code
    assert [event[0] for event in trace].count("get") == 1
    assert not any(event[0] == "set_result" for event in trace)


def test_duplicate_fetch_base_exception_is_not_swallowed() -> None:
    client = _Client(
        [],
        [ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE")],
        [KeyboardInterrupt()],
    )
    with pytest.raises(KeyboardInterrupt):
        _send(client, {}, None, link=True)


@pytest.mark.parametrize("fetched", [None, [], {}])
def test_duplicate_fetch_with_unusable_body_is_refused(fetched: Any) -> None:
    trace: list[Any] = []
    client = _Client(
        trace, [ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE")], [fetched]
    )
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, _Bundle(trace), link=True)
    assert info.value.code == "BACKEND_REFUSED"


def test_duplicate_fetch_without_linkage_records_then_fails() -> None:
    trace: list[Any] = []
    client = _Client(
        trace,
        [ProcessRecordRequestError(409, "DUPLICATE_ISSUANCE")],
        [_ok(trace, linkage=None)],
    )
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, _Bundle(trace), link=True)
    assert info.value.code == "LINKAGE_ABSENT"
    assert trace[-1] == ("set_result", RECORD, None)


def test_other_failure_codes_do_not_trigger_a_fetch() -> None:
    trace: list[Any] = []
    client = _Client(
        trace, [ProcessRecordRequestError(409, "ISSUANCE_IN_PROGRESS")], [_ok(trace)]
    )
    with pytest.raises(CommitmentLinkageClientError) as info:
        _send(client, {}, _Bundle(trace), link=True)
    assert info.value.code == "ISSUANCE_IN_PROGRESS"
    assert not any(event[0] == "get" for event in trace)


# ----------------------------------------------------------- raising accessors


def test_bundle_openings_error_propagates_before_the_request() -> None:
    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    payload = {"refs": 1}
    with pytest.raises(RuntimeError, match="openings failed"):
        _send(
            client, payload, _Bundle(trace, RuntimeError("openings failed")), link=True
        )
    assert not any(event[0] == "finalize" for event in trace)
    assert payload == {"refs": 1}


def test_bundle_openings_items_error_propagates_before_the_request() -> None:
    class Openings(dict[str, bytes]):
        def items(self) -> Any:
            raise RuntimeError("items failed")

    trace: list[Any] = []
    client = _Client(trace, [_ok(trace)])
    payload = {"refs": 1}
    with pytest.raises(RuntimeError, match="items failed"):
        _send(client, payload, _Bundle(trace, Openings(OPENINGS)), link=True)
    assert not any(event[0] == "finalize" for event in trace)
    assert payload == {"refs": 1}


def test_set_result_error_propagates() -> None:
    trace: list[Any] = []

    class Bundle(_Bundle):
        def set_result(self, record: Any, linkage: Any) -> None:
            raise RuntimeError("persist failed")

    client = _Client(trace, [_ok(trace)])
    with pytest.raises(RuntimeError, match="persist failed"):
        _send(client, {}, Bundle(trace), link=True)


def test_body_get_error_propagates() -> None:
    class Body(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("get failed")

    client = _Client([], [Body(data=RECORD)])
    with pytest.raises(RuntimeError, match="get failed"):
        _send(client, {}, None, link=False)


def test_linkage_get_error_propagates_after_the_record_read() -> None:
    trace: list[Any] = []

    class Body(_Body):
        def get(self, key: str, default: Any = None) -> Any:
            if key == "commitment_linkage":
                raise RuntimeError("linkage get failed")
            return super().get(key, default)

    client = _Client(trace, [Body(trace, {"data": RECORD})])
    with pytest.raises(RuntimeError, match="linkage get failed"):
        _send(client, {}, _Bundle(trace), link=True)
    assert ("body_get", "data") in trace
    assert not any(event[0] == "set_result" for event in trace)
