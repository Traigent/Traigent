"""#3239: /execute and /evaluate enforce the Hybrid Mode request schema."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from traigent.wrapper.server import EVALUATE_PATH, EXECUTE_PATH, create_app
from traigent.wrapper.service import TraigentService

GOOD_ID = "550e8400-e29b-41d4-a716-446655440000"
LONG_ID = "a" * 257


class _Send:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int:
        return next(
            m["status"] for m in self.messages if m["type"] == "http.response.start"
        )

    @property
    def headers(self) -> list[tuple[bytes, bytes]]:
        return next(
            m["headers"] for m in self.messages if m["type"] == "http.response.start"
        )

    @property
    def raw_body(self) -> bytes:
        return next(
            m["body"] for m in self.messages if m["type"] == "http.response.body"
        )

    @property
    def body(self) -> dict:
        return json.loads(self.raw_body)


async def _post(app: Any, path: str, payload: dict) -> _Send:
    send = _Send()
    body = json.dumps(payload).encode()

    async def receive() -> dict:
        return {"body": body, "more_body": False}

    await app({"type": "http", "method": "POST", "path": path}, receive, send)
    return send


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def app(calls: list[str]) -> Any:
    svc = TraigentService(tunable_id="agent")

    @svc.execute
    def run(example_id, data, config):
        calls.append(example_id)
        return {"output": "ok"}

    @svc.evaluate
    def score(output, target):
        calls.append("evaluate")
        return {"accuracy": 1.0}

    return create_app(svc)


def _execute(**overrides: Any) -> dict:
    body: dict[str, Any] = {
        "request_id": GOOD_ID,
        "tunable_id": "agent",
        "config": {"temperature": 0.1},
        "examples": [{"example_id": "case_001", "data": {"q": "hi"}}],
    }
    for key, value in overrides.items():
        if value is _MISSING:
            body.pop(key, None)
        else:
            body[key] = value
    return body


def _evaluate(**overrides: Any) -> dict:
    body: dict[str, Any] = {
        "request_id": GOOD_ID,
        "tunable_id": "agent",
        "evaluations": [{"example_id": "case_001", "output": {"a": 1}, "target": {}}],
    }
    for key, value in overrides.items():
        if value is _MISSING:
            body.pop(key, None)
        else:
            body[key] = value
    return body


_MISSING = object()

MARKER = "MARKER-CLIENT-VALUE"

EXECUTE_BAD = {
    "missing_request_id": _execute(request_id=_MISSING),
    "non_uuid_request_id": _execute(request_id=f"nope-{MARKER}"),
    "non_string_request_id": _execute(request_id=7),
    "config_not_object": _execute(config=f"x-{MARKER}"),
    "config_missing": _execute(config=_MISSING),
    "config_too_many": _execute(config={f"k{i}": i for i in range(201)}),
    "tunable_missing": _execute(tunable_id=_MISSING),
    "timeout_too_big": _execute(timeout_ms=600001),
    "id_traversal": _execute(examples=[{"example_id": "../etc"}]),
    "id_too_long": _execute(examples=[{"example_id": LONG_ID}]),
    "id_integer": _execute(examples=[{"example_id": 1}]),
    "id_trailing_newline": _execute(examples=[{"example_id": "ok\n"}]),
    "id_slash": _execute(examples=[{"example_id": f"{MARKER}/x"}]),
    "id_backslash": _execute(examples=[{"example_id": "a\\b"}]),
    "id_dotdot": _execute(examples=[{"example_id": ".."}]),
    "id_empty": _execute(examples=[{"example_id": ""}]),
    "id_control_char": _execute(examples=[{"example_id": "a\x00b"}]),
    "id_inner_newline": _execute(examples=[{"example_id": "a\nb"}]),
}

EVALUATE_BAD = {
    "missing_request_id": _evaluate(request_id=_MISSING),
    "non_uuid_request_id": _evaluate(request_id=f"nope-{MARKER}"),
    "tunable_missing": _evaluate(tunable_id=_MISSING),
    "config_not_object": _evaluate(config=f"x-{MARKER}"),
    "config_too_many": _evaluate(config={f"k{i}": i for i in range(201)}),
    "timeout_too_big": _evaluate(timeout_ms=600001),
    "item_not_object": _evaluate(evaluations=[{"example_id": "ok"}, 42]),
    "id_traversal": _evaluate(evaluations=[{"example_id": "../etc"}]),
    "id_too_long": _evaluate(evaluations=[{"example_id": LONG_ID}]),
    "id_integer": _evaluate(evaluations=[{"example_id": 1}]),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(EXECUTE_BAD))
async def test_execute_rejects_invalid_request(app, calls, case) -> None:
    send = await _post(app, EXECUTE_PATH, EXECUTE_BAD[case])
    assert send.status == 400
    assert send.body["error"]["code"] == "INVALID_REQUEST"
    assert calls == []  # rejected before dispatch
    _assert_nothing_echoed(send, EXECUTE_BAD[case])


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(EVALUATE_BAD))
async def test_evaluate_rejects_invalid_request(app, calls, case) -> None:
    send = await _post(app, EVALUATE_PATH, EVALUATE_BAD[case])
    assert send.status == 400
    assert send.body["error"]["code"] == "INVALID_REQUEST"
    assert calls == []
    _assert_nothing_echoed(send, EVALUATE_BAD[case])


def _assert_nothing_echoed(send: _Send, payload: dict) -> None:
    haystack = send.raw_body + b"".join(k + b"=" + v for k, v in send.headers)
    for bad in (MARKER, LONG_ID, "../etc", "k200", "600001", "nope-"):
        assert bad.encode() not in haystack
    assert (
        b"x-request-id" not in haystack.lower() or payload.get("request_id") == GOOD_ID
    )


@pytest.mark.asyncio
async def test_execute_valid_request_still_succeeds(app, calls) -> None:
    send = await _post(app, EXECUTE_PATH, _execute(timeout_ms=600000))
    assert send.status == 200
    assert send.body["request_id"] == GOOD_ID
    assert calls == ["case_001"]


@pytest.mark.asyncio
async def test_execute_boundary_values_are_accepted(app) -> None:
    body = _execute(
        request_id=str(uuid.uuid4()),
        config={f"k{i}": i for i in range(200)},
        examples=[
            {"example_id": "a" + "b" * 255},
            {"example_id": "A.b_c-9"},
            {"example_id": "q 1"},
            {"example_id": "a:b"},
            {"example_id": "a..b"},
        ],
    )
    assert (await _post(app, EXECUTE_PATH, body)).status == 200


@pytest.mark.asyncio
async def test_evaluate_valid_request_still_succeeds(app, calls) -> None:
    send = await _post(app, EVALUATE_PATH, _evaluate())
    assert send.status == 200
    assert calls == ["evaluate"]
    assert (
        await _post(
            app, EVALUATE_PATH, _evaluate(config={}, request_id=str(uuid.uuid4()))
        )
    ).status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [EXECUTE_PATH, EVALUATE_PATH])
async def test_tunable_mismatch_message_does_not_echo_client_value(app, path) -> None:
    make = _execute if path == EXECUTE_PATH else _evaluate
    send = await _post(app, path, make(tunable_id="other-MARKER"))
    assert send.status == 400
    assert b"other-MARKER" not in send.raw_body
