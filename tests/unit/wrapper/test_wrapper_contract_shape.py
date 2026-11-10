"""The wrapper server follows the published hybrid request/response shape.

Contract points covered (Traigent#3140): ``inputs[]``/``input_id`` is the
primary request shape and wins over the deprecated ``examples[]``/``example_id``
alias; responses echo ``input_id`` (and keep ``example_id`` for the legacy SDK
client); a missing or wrong-shaped input array answers ``INVALID_INPUTS``; a
reused ``request_id`` with a different payload answers 409
``IDEMPOTENCY_CONFLICT``; an unknown ``tunable_id`` on config-space answers
404 ``UNKNOWN_TUNABLE_ID``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from traigent.wrapper.server import (
    CONFIG_SPACE_PATH,
    EVALUATE_PATH,
    EXECUTE_PATH,
    create_app,
)
from traigent.wrapper.service import TraigentService


async def _call(
    svc: TraigentService,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    query: str = "",
) -> tuple[int, dict[str, Any]]:
    app = create_app(svc)
    raw = json.dumps(body).encode() if body is not None else b""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"body": raw, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query.encode(),
        "headers": [],
    }
    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    payload = next(m["body"] for m in sent if m["type"] == "http.response.body")
    return status, json.loads(payload)


def _service() -> TraigentService:
    svc = TraigentService(tunable_id="my_agent")
    seen: list[str] = []

    @svc.execute
    def run(example_id, data, config):
        seen.append(example_id)
        return {"output": {"echo": example_id}}

    @svc.evaluate
    def score(output, target):
        return {"accuracy": 1.0}

    svc.seen = seen  # type: ignore[attr-defined]
    return svc


def _execute_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "tunable_id": "my_agent",
        "config": {},
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_execute_accepts_contract_inputs_and_echoes_input_id() -> None:
    svc = _service()
    status, body = await _call(
        svc,
        "POST",
        EXECUTE_PATH,
        _execute_body(inputs=[{"input_id": "input_001", "data": {"q": "x"}}]),
    )
    assert status == 200, body
    assert body["outputs"][0]["input_id"] == "input_001"
    # The deprecated key stays for the SDK's own legacy client.
    assert body["outputs"][0]["example_id"] == "input_001"
    assert svc.seen == ["input_001"]


@pytest.mark.asyncio
async def test_execute_inputs_take_precedence_over_examples() -> None:
    svc = _service()
    status, body = await _call(
        svc,
        "POST",
        EXECUTE_PATH,
        _execute_body(
            inputs=[{"input_id": "from_inputs"}],
            examples=[{"example_id": "from_examples"}],
        ),
    )
    assert status == 200, body
    assert [o["input_id"] for o in body["outputs"]] == ["from_inputs"]
    assert svc.seen == ["from_inputs"]


@pytest.mark.asyncio
async def test_execute_examples_alias_still_works_and_returns_input_id() -> None:
    svc = _service()
    status, body = await _call(
        svc,
        "POST",
        EXECUTE_PATH,
        _execute_body(examples=[{"example_id": "i1", "data": {}}]),
    )
    assert status == 200, body
    assert body["outputs"][0]["input_id"] == "i1"
    assert body["outputs"][0]["example_id"] == "i1"


@pytest.mark.asyncio
async def test_failed_item_still_carries_input_id() -> None:
    svc = TraigentService(tunable_id="my_agent")

    @svc.execute
    def run(example_id, data, config):
        raise RuntimeError("boom")

    status, body = await _call(
        svc, "POST", EXECUTE_PATH, _execute_body(inputs=[{"input_id": "bad_1"}])
    )
    assert status == 200, body
    assert body["outputs"][0]["input_id"] == "bad_1"
    assert "error" in body["outputs"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, [], "not-a-list", {"input_id": "a"}])
async def test_missing_or_wrong_shaped_inputs_use_invalid_inputs(bad: Any) -> None:
    svc = _service()
    body = _execute_body()
    if bad is not None:
        body["inputs"] = bad
    status, payload = await _call(svc, "POST", EXECUTE_PATH, body)
    assert status == 400
    assert payload["error"]["code"] == "INVALID_INPUTS"


@pytest.mark.asyncio
async def test_evaluate_echoes_request_input_id_not_a_random_one() -> None:
    svc = _service()
    status, body = await _call(
        svc,
        "POST",
        EVALUATE_PATH,
        {
            "request_id": "660e8400-e29b-41d4-a716-446655440001",
            "tunable_id": "my_agent",
            "evaluations": [
                {"input_id": "input_001", "output": {"a": 1}, "target": {"a": 1}}
            ],
        },
    )
    assert status == 200, body
    assert body["results"][0]["input_id"] == "input_001"
    assert body["results"][0]["example_id"] == "input_001"


@pytest.mark.asyncio
async def test_evaluate_legacy_example_id_still_echoed_as_input_id() -> None:
    svc = _service()
    status, body = await _call(
        svc,
        "POST",
        EVALUATE_PATH,
        {
            "request_id": "r-legacy",
            "tunable_id": "my_agent",
            "evaluations": [{"example_id": "e1", "output": {}, "target": {}}],
        },
    )
    assert status == 200, body
    assert body["results"][0]["input_id"] == "e1"


@pytest.mark.asyncio
async def test_request_id_reuse_with_different_payload_is_409() -> None:
    svc = _service()
    first = _execute_body(inputs=[{"input_id": "a"}])
    assert (await _call(svc, "POST", EXECUTE_PATH, first))[0] == 200
    second = _execute_body(inputs=[{"input_id": "b"}])
    status, payload = await _call(svc, "POST", EXECUTE_PATH, second)
    assert status == 409
    assert payload["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.asyncio
async def test_same_request_id_same_payload_replays() -> None:
    svc = _service()
    body = _execute_body(inputs=[{"input_id": "a"}])
    first = await _call(svc, "POST", EXECUTE_PATH, body)
    again = await _call(svc, "POST", EXECUTE_PATH, body)
    assert first[0] == again[0] == 200
    assert first[1] == again[1]


@pytest.mark.asyncio
async def test_config_space_unknown_tunable_id_is_404() -> None:
    svc = _service()
    status, payload = await _call(
        svc, "GET", CONFIG_SPACE_PATH, query="tunable_id=bogus_agent"
    )
    assert status == 404
    assert payload["error"]["code"] == "UNKNOWN_TUNABLE_ID"
    assert "bogus_agent" not in json.dumps(payload["error"].get("details") or {})


@pytest.mark.asyncio
async def test_config_space_known_or_absent_tunable_id_is_200() -> None:
    svc = _service()
    assert (await _call(svc, "GET", CONFIG_SPACE_PATH))[0] == 200
    status, _ = await _call(svc, "GET", CONFIG_SPACE_PATH, query="tunable_id=my_agent")
    assert status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("inputs", [None, "x", [], {"input_id": "a"}])
async def test_present_inputs_win_even_when_null_or_wrong_shaped(inputs: Any) -> None:
    svc = _service()
    status, payload = await _call(
        svc,
        "POST",
        EXECUTE_PATH,
        _execute_body(inputs=inputs, examples=[{"example_id": "legacy"}]),
    )
    assert status == 400
    assert payload["error"]["code"] == "INVALID_INPUTS"
    assert svc.seen == []


@pytest.mark.asyncio
async def test_absent_inputs_key_falls_back_to_examples() -> None:
    svc = _service()
    status, body = await _call(
        svc, "POST", EXECUTE_PATH, _execute_body(examples=[{"example_id": "legacy"}])
    )
    assert status == 200, body
    assert svc.seen == ["legacy"]


def _gated_service() -> tuple[TraigentService, Any, Any, list[str]]:
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()
    runs: list[str] = []
    svc = TraigentService(tunable_id="my_agent")

    @svc.execute
    async def run(example_id, data, config):
        runs.append(example_id)
        started.set()
        await release.wait()
        return {"output": {"ok": True}}

    return svc, started, release, runs


@pytest.mark.asyncio
async def test_concurrent_same_id_different_payload_is_409_while_first_runs() -> None:
    import asyncio

    svc, started, release, runs = _gated_service()
    first = asyncio.create_task(
        svc.handle_execute(_execute_body(inputs=[{"input_id": "a"}]))
    )
    await started.wait()
    from traigent.wrapper.errors import IdempotencyConflictError

    with pytest.raises(IdempotencyConflictError):
        await svc.handle_execute(_execute_body(inputs=[{"input_id": "b"}]))
    release.set()
    result = await first
    assert result["outputs"][0]["input_id"] == "a"
    assert runs == ["a"]


@pytest.mark.asyncio
async def test_concurrent_same_id_same_payload_runs_handler_once() -> None:
    import asyncio

    svc, started, release, runs = _gated_service()
    body = _execute_body(inputs=[{"input_id": "a"}])
    first = asyncio.create_task(svc.handle_execute(body))
    await started.wait()
    second = asyncio.create_task(svc.handle_execute(dict(body)))
    await asyncio.sleep(0)
    release.set()
    r1, r2 = await asyncio.gather(first, second)
    assert r1 == r2
    assert runs == ["a"]


@pytest.mark.asyncio
async def test_reservation_released_when_handler_raises_so_retry_runs() -> None:
    from traigent.wrapper.errors import ServiceUnavailableError

    calls: list[int] = []
    svc = TraigentService(tunable_id="my_agent")

    @svc.execute
    def run(example_id, data, config):
        calls.append(1)
        if len(calls) == 1:
            raise ServiceUnavailableError("down")
        return {"output": {}}

    body = _execute_body(inputs=[{"input_id": "a"}])
    with pytest.raises(ServiceUnavailableError):
        await svc.handle_execute(body)
    result = await svc.handle_execute(body)
    assert result["status"] == "completed"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_waiter_gets_the_first_failure_and_handler_runs_once() -> None:
    import asyncio

    from traigent.wrapper.errors import ServiceUnavailableError

    started = asyncio.Event()
    release = asyncio.Event()
    runs: list[str] = []
    svc = TraigentService(tunable_id="my_agent")

    @svc.execute
    async def run(example_id, data, config):
        runs.append(example_id)
        started.set()
        await release.wait()
        raise ServiceUnavailableError("upstream down")

    body = _execute_body(inputs=[{"input_id": "a"}])
    first = asyncio.create_task(svc.handle_execute(body))
    await started.wait()
    waiter = asyncio.create_task(svc.handle_execute(dict(body)))
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(ServiceUnavailableError):
        await first
    with pytest.raises(ServiceUnavailableError):
        await waiter
    assert runs == ["a"]
    assert svc._inflight_requests == {}


@pytest.mark.asyncio
async def test_waiter_gets_retryable_error_when_first_is_cancelled() -> None:
    import asyncio

    from traigent.wrapper.errors import ServiceUnavailableError

    svc, started, release, runs = _gated_service()
    body = _execute_body(inputs=[{"input_id": "a"}])
    first = asyncio.create_task(svc.handle_execute(body))
    await started.wait()
    waiter = asyncio.create_task(svc.handle_execute(dict(body)))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(ServiceUnavailableError) as info:
        await waiter
    assert info.value.error_code == "REQUEST_CANCELLED"
    assert runs == ["a"]
    assert svc._inflight_requests == {}


@pytest.mark.asyncio
async def test_inflight_reservations_are_capped_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import traigent.wrapper.service as service_module
    from traigent.wrapper.errors import ServiceUnavailableError

    monkeypatch.setattr(service_module, "MAX_INFLIGHT_REQUESTS", 2)
    svc, started, release, runs = _gated_service()
    tasks = [
        asyncio.create_task(
            svc.handle_execute(
                _execute_body(request_id=f"id-{i}", inputs=[{"input_id": f"x{i}"}])
            )
        )
        for i in range(2)
    ]
    await started.wait()
    await asyncio.sleep(0)
    assert len(svc._inflight_requests) == 2
    with pytest.raises(ServiceUnavailableError) as info:
        await svc.handle_execute(
            _execute_body(request_id="id-3", inputs=[{"input_id": "x3"}])
        )
    assert info.value.error_code == "TOO_MANY_IN_FLIGHT_REQUESTS"
    assert len(svc._inflight_requests) == 2
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert svc._inflight_requests == {}
