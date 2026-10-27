"""Characterization tests pinning the ASGI app built by ``create_app``."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from traigent.wrapper.errors import BadRequestError, HybridAPIError
from traigent.wrapper.server import create_app

TRACE = [(b"traceparent", b"00-abc"), (b"x-other", b"1"), (b"TraceState", b"k=v")]


class _Service:
    def __init__(self, execute=None, evaluate=None, alive=True):
        self.calls: list[Any] = []
        self._execute = execute
        self._evaluate = evaluate
        self._alive = alive

    def get_capabilities(self):
        self.calls.append("caps")
        return {"caps": True}

    def get_config_space(self):
        self.calls.append("space")
        return {"space": True}

    def get_health(self):
        self.calls.append("health")
        return {"status": "ok"}

    def handle_keep_alive(self, session_id):
        self.calls.append(("alive", session_id))
        return self._alive

    async def handle_execute(self, request):
        self.calls.append(("execute", request))
        return await _outcome(self._execute, request)

    async def handle_evaluate(self, request):
        self.calls.append(("evaluate", request))
        return await _outcome(self._evaluate, request)


async def _outcome(spec, request):
    if spec is None:
        return {"request_id": request.get("request_id"), "ok": True}
    if isinstance(spec, BaseException):
        raise spec
    return spec


def _call(method, path, body=b"", service=None, scope_type="http", headers=TRACE):
    service = service or _Service()
    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"body": body, "more_body": False}

    scope = {"type": scope_type, "method": method, "path": path, "headers": headers}
    asyncio.run(create_app(service)(scope, receive, send))
    out = []
    for message in sent:
        if message["type"] == "http.response.start":
            out.append(
                (
                    message["status"],
                    sorted((k.decode(), v.decode()) for k, v in message["headers"]),
                )
            )
        else:
            out.append(json.loads(message["body"]))
    return out, service.calls


J = lambda d: json.dumps(d).encode()  # noqa: E731

CASES = {
    "lifespan": ("GET", "/traigent/v1/capabilities", b"", None, "lifespan"),
    "caps": ("GET", "/traigent/v1/capabilities", b"", None, "http"),
    "caps_post": ("POST", "/traigent/v1/capabilities", b"", None, "http"),
    "space": ("GET", "/traigent/v1/config-space", b"", None, "http"),
    "health": ("GET", "/traigent/v1/health", b"", None, "http"),
    "health_post": ("POST", "/traigent/v1/health", b"", None, "http"),
    "unknown": ("GET", "/nope", b"", None, "http"),
    "execute": ("POST", "/traigent/v1/execute", J({"request_id": "r1"}), None, "http"),
    "execute_get": ("GET", "/traigent/v1/execute", b"", None, "http"),
    "execute_empty_body": ("POST", "/traigent/v1/execute", b"", None, "http"),
    "execute_timeout_ok": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": 7, "timeout_ms": 1000}),
        None,
        "http",
    ),
    "execute_timeout_bad": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": "r", "timeout_ms": 5}),
        None,
        "http",
    ),
    "execute_timeout_type": (
        "POST",
        "/traigent/v1/execute",
        J({"timeout_ms": "x"}),
        None,
        "http",
    ),
    "execute_times_out": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": "r", "timeout_ms": 1000}),
        _Service(execute=TimeoutError()),
        "http",
    ),
    "execute_non_dict": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": "r"}),
        _Service(execute=[1]),
        "http",
    ),
    "execute_hybrid_error": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": "r"}),
        _Service(
            execute=HybridAPIError(
                "teapot",
                status_code=418,
                error_code="TEA",
                details={"a": 1},
                headers={"X-Extra": "e"},
            )
        ),
        "http",
    ),
    "execute_bad_request_no_id": (
        "POST",
        "/traigent/v1/execute",
        J({"zzz": 1}),
        _Service(execute=BadRequestError("nope")),
        "http",
    ),
    "execute_value_error": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": "r"}),
        _Service(execute=ValueError("bad value")),
        "http",
    ),
    "execute_internal_error": (
        "POST",
        "/traigent/v1/execute",
        J({"request_id": "r"}),
        _Service(execute=RuntimeError("kaboom")),
        "http",
    ),
    "execute_internal_error_empty_request": (
        "POST",
        "/traigent/v1/execute",
        b"",
        _Service(execute=RuntimeError("kaboom")),
        "http",
    ),
    "execute_invalid_json": ("POST", "/traigent/v1/execute", b"{bad", None, "http"),
    "execute_json_array": ("POST", "/traigent/v1/execute", b"[1]", None, "http"),
    "evaluate": (
        "POST",
        "/traigent/v1/evaluate",
        J({"request_id": "e1"}),
        None,
        "http",
    ),
    "evaluate_timeout": (
        "POST",
        "/traigent/v1/evaluate",
        J({"request_id": "e", "timeout_ms": 2000}),
        None,
        "http",
    ),
    "evaluate_error": (
        "POST",
        "/traigent/v1/evaluate",
        J({"request_id": "e"}),
        _Service(evaluate=KeyError("k")),
        "http",
    ),
    "keep_alive_yes": (
        "POST",
        "/traigent/v1/keep-alive",
        J({"session_id": "s1"}),
        None,
        "http",
    ),
    "keep_alive_no": (
        "POST",
        "/traigent/v1/keep-alive",
        J({"session_id": "s1"}),
        _Service(alive=False),
        "http",
    ),
    "keep_alive_get": ("GET", "/traigent/v1/keep-alive", b"", None, "http"),
    "keep_alive_bad_json": ("POST", "/traigent/v1/keep-alive", b"nope", None, "http"),
}


def _snapshot(name):
    method, path, body, service, scope_type = CASES[name]
    if service is not None:
        # Fresh copy so call logs never leak between runs.
        service = _Service(service._execute, service._evaluate, service._alive)
    out, calls = _call(method, path, body, service, scope_type)
    return {"out": repr(out), "calls": repr(calls)}


GOLDEN: dict = {
    "lifespan": {"out": "[]", "calls": "[]"},
    "caps": {
        "out": "[(200, [('content-length', '14'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'caps': True}]",
        "calls": "['caps']",
    },
    "caps_post": {
        "out": "[(404, [('content-length', '88'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'error': {'code': "
        "'NOT_FOUND', 'message': 'Not found: POST /traigent/v1/capabilities'}}]",
        "calls": "[]",
    },
    "space": {
        "out": "[(200, [('content-length', '15'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'space': True}]",
        "calls": "['space']",
    },
    "health": {
        "out": "[(200, [('content-length', '16'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'status': 'ok'}]",
        "calls": "['health']",
    },
    "health_post": {
        "out": "[(404, [('content-length', '82'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'error': {'code': "
        "'NOT_FOUND', 'message': 'Not found: POST /traigent/v1/health'}}]",
        "calls": "[]",
    },
    "unknown": {
        "out": "[(404, [('content-length', '67'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'error': {'code': "
        "'NOT_FOUND', 'message': 'Not found: GET /nope'}}]",
        "calls": "[]",
    },
    "execute": {
        "out": "[(200, [('content-length', '32'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v'), ('x-traigent-request-id', "
        "'r1')]), {'request_id': 'r1', 'ok': True}]",
        "calls": "[('execute', {'request_id': 'r1'})]",
    },
    "execute_get": {
        "out": "[(404, [('content-length', '82'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'error': {'code': "
        "'NOT_FOUND', 'message': 'Not found: GET /traigent/v1/execute'}}]",
        "calls": "[]",
    },
    "execute_empty_body": {
        "out": "[(200, [('content-length', '32'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'request_id': None, 'ok': True}]",
        "calls": "[('execute', {})]",
    },
    "execute_timeout_ok": {
        "out": "[(200, [('content-length', '29'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', '7')]), {'request_id': 7, 'ok': "
        "True}]",
        "calls": "[('execute', {'request_id': 7, 'timeout_ms': 1000})]",
    },
    "execute_timeout_bad": {
        "out": "[(400, [('content-length', '79'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', 'r')]), {'error': {'code': "
        "'INVALID_REQUEST', 'message': 'timeout_ms must be >= 1000'}}]",
        "calls": "[]",
    },
    "execute_timeout_type": {
        "out": "[(400, [('content-length', '82'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'error': {'code': 'INVALID_REQUEST', 'message': "
        "'timeout_ms must be an integer'}}]",
        "calls": "[]",
    },
    "execute_times_out": {
        "out": "[(408, [('content-length', '152'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', 'r')]), {'error': {'code': "
        "'REQUEST_TIMEOUT', 'message': 'Request timed out after 1000ms', "
        "'details': {'endpoint': '/traigent/v1/execute', 'timeout_ms': "
        "1000}}}]",
        "calls": "[('execute', {'request_id': 'r', 'timeout_ms': 1000})]",
    },
    "execute_non_dict": {
        "out": "[(400, [('content-length', '102'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', 'r')]), {'error': {'code': "
        "'INVALID_REQUEST', 'message': 'Wrapper service handler must return a "
        "JSON object'}}]",
        "calls": "[('execute', {'request_id': 'r'})]",
    },
    "execute_hybrid_error": {
        "out": "[(418, [('content-length', '68'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-extra', 'e'), ('x-traigent-request-id', 'r')]), "
        "{'error': {'code': 'TEA', 'message': 'teapot', 'details': {'a': "
        "1}}}]",
        "calls": "[('execute', {'request_id': 'r'})]",
    },
    "execute_bad_request_no_id": {
        "out": "[(400, [('content-length', '57'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), "
        "('tracestate', 'k=v')]), {'error': {'code': "
        "'INVALID_REQUEST', 'message': 'nope'}}]",
        "calls": "[('execute', {'zzz': 1})]",
    },
    "execute_value_error": {
        "out": "[(400, [('content-length', '62'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', 'r')]), {'error': {'code': "
        "'INVALID_REQUEST', 'message': 'bad value'}}]",
        "calls": "[('execute', {'request_id': 'r'})]",
    },
    "execute_internal_error": {
        "out": "[(500, [('content-length', '58'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', 'r')]), {'error': {'code': "
        "'INTERNAL_ERROR', 'message': 'kaboom'}}]",
        "calls": "[('execute', {'request_id': 'r'})]",
    },
    "execute_internal_error_empty_request": {
        "out": "[(500, [('content-length', '58'), "
        "('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'error': {'code': 'INTERNAL_ERROR', "
        "'message': 'kaboom'}}]",
        "calls": "[('execute', {})]",
    },
    "execute_invalid_json": {
        "out": "[(400, [('content-length', '139'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'error': {'code': 'INVALID_JSON', 'message': 'Invalid "
        "JSON: Expecting property name enclosed in double quotes: line 1 "
        "column 2 (char 1)'}}]",
        "calls": "[]",
    },
    "execute_json_array": {
        "out": "[(400, [('content-length', '87'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'error': {'code': 'INVALID_REQUEST', 'message': "
        "'Request body must be a JSON object'}}]",
        "calls": "[]",
    },
    "evaluate": {
        "out": "[(200, [('content-length', '32'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v'), ('x-traigent-request-id', "
        "'e1')]), {'request_id': 'e1', 'ok': True}]",
        "calls": "[('evaluate', {'request_id': 'e1'})]",
    },
    "evaluate_timeout": {
        "out": "[(200, [('content-length', '31'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v'), ('x-traigent-request-id', 'e')]), {'request_id': 'e', 'ok': "
        "True}]",
        "calls": "[('evaluate', {'request_id': 'e', 'timeout_ms': 2000})]",
    },
    "evaluate_error": {
        "out": "[(500, [('content-length', '55'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', 'k=v'), "
        "('x-traigent-request-id', 'e')]), {'error': {'code': 'INTERNAL_ERROR', "
        "'message': \"'k'\"}}]",
        "calls": "[('evaluate', {'request_id': 'e'})]",
    },
    "keep_alive_yes": {
        "out": "[(200, [('content-length', '39'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'status': 'alive', 'session_id': 's1'}]",
        "calls": "[('alive', 's1')]",
    },
    "keep_alive_no": {
        "out": "[(404, [('content-length', '76'), ('content-type', 'application/json'), "
        "('traceparent', '00-abc'), ('tracestate', 'k=v')]), {'error': {'code': "
        "'SESSION_NOT_FOUND', 'message': 'Session not found: s1'}}]",
        "calls": "[('alive', 's1')]",
    },
    "keep_alive_get": {
        "out": "[(404, [('content-length', '85'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'error': {'code': 'NOT_FOUND', 'message': 'Not found: GET "
        "/traigent/v1/keep-alive'}}]",
        "calls": "[]",
    },
    "keep_alive_bad_json": {
        "out": "[(400, [('content-length', '105'), ('content-type', "
        "'application/json'), ('traceparent', '00-abc'), ('tracestate', "
        "'k=v')]), {'error': {'code': 'INVALID_JSON', 'message': 'Invalid "
        "JSON: Expecting value: line 1 column 1 (char 0)'}}]",
        "calls": "[]",
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_create_app_responses(name):
    assert _snapshot(name) == GOLDEN[name]
