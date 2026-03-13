"""Unit tests for wrapper HTTP server module.

Tests for the ASGI application, request routing, body reading,
JSON response sending, and server runner functions.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.wrapper.server import (
    CAPABILITIES_PATH,
    CONFIG_SPACE_PATH,
    EVALUATE_PATH,
    EXECUTE_PATH,
    HEALTH_PATH,
    KEEP_ALIVE_PATH,
    _extract_trace_headers,
    _get_timeout_ms,
    _response_headers_with_request_id,
    create_app,
    read_body,
    run_server,
    send_json_response,
)
from traigent.wrapper.service import TraigentService


# ---------------------------------------------------------------------------
# Helper to simulate ASGI scope / receive / send
# ---------------------------------------------------------------------------
def _make_scope(method: str, path: str) -> dict:
    """Create an ASGI HTTP scope dict."""
    return {"type": "http", "method": method, "path": path}


def _make_receive(body: bytes = b"") -> AsyncMock:
    """Create an ASGI receive callable returning the given body."""
    receive = AsyncMock()
    receive.return_value = {"body": body, "more_body": False}
    return receive


def _make_receive_chunked(chunks: list[bytes]) -> AsyncMock:
    """Create an ASGI receive callable returning body in multiple chunks."""
    messages = []
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        messages.append({"body": chunk, "more_body": not is_last})
    receive = AsyncMock(side_effect=messages)
    return receive


class _SendCollector:
    """Collects messages sent via ASGI send."""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int:
        for msg in self.messages:
            if msg["type"] == "http.response.start":
                return msg["status"]
        raise ValueError("No response.start message found")

    @property
    def body_json(self) -> dict:
        for msg in self.messages:
            if msg["type"] == "http.response.body":
                return json.loads(msg["body"])
        raise ValueError("No response.body message found")


# ---------------------------------------------------------------------------
# Path constant tests
# ---------------------------------------------------------------------------
class TestPathConstants:
    """Tests for the endpoint path constants."""

    def test_capabilities_path(self) -> None:
        assert CAPABILITIES_PATH == "/traigent/v1/capabilities"

    def test_config_space_path(self) -> None:
        assert CONFIG_SPACE_PATH == "/traigent/v1/config-space"

    def test_execute_path(self) -> None:
        assert EXECUTE_PATH == "/traigent/v1/execute"

    def test_evaluate_path(self) -> None:
        assert EVALUATE_PATH == "/traigent/v1/evaluate"

    def test_health_path(self) -> None:
        assert HEALTH_PATH == "/traigent/v1/health"

    def test_keep_alive_path(self) -> None:
        assert KEEP_ALIVE_PATH == "/traigent/v1/keep-alive"


# ---------------------------------------------------------------------------
# read_body tests
# ---------------------------------------------------------------------------
class TestReadBody:
    """Tests for the read_body helper."""

    @pytest.mark.asyncio
    async def test_single_chunk(self) -> None:
        """Test reading body from a single message."""
        receive = _make_receive(b'{"key": "value"}')
        body = await read_body(receive)
        assert body == b'{"key": "value"}'

    @pytest.mark.asyncio
    async def test_multi_chunk(self) -> None:
        """Test reading body from multiple chunks."""
        receive = _make_receive_chunked([b'{"ke', b'y": ', b'"val"}'])
        body = await read_body(receive)
        assert body == b'{"key": "val"}'

    @pytest.mark.asyncio
    async def test_empty_body(self) -> None:
        """Test reading empty body."""
        receive = _make_receive(b"")
        body = await read_body(receive)
        assert body == b""

    @pytest.mark.asyncio
    async def test_missing_body_key(self) -> None:
        """Test that missing 'body' key defaults to empty bytes."""
        receive = AsyncMock(return_value={"more_body": False})
        body = await read_body(receive)
        assert body == b""


# ---------------------------------------------------------------------------
# send_json_response tests
# ---------------------------------------------------------------------------
class TestSendJsonResponse:
    """Tests for the send_json_response helper."""

    @pytest.mark.asyncio
    async def test_sends_status_and_headers(self) -> None:
        """Test that response includes status and JSON content-type."""
        send = _SendCollector()
        await send_json_response(send, 200, {"ok": True})
        assert send.status == 200
        start_msg = send.messages[0]
        headers = {h[0]: h[1] for h in start_msg["headers"]}
        assert headers[b"content-type"] == b"application/json"

    @pytest.mark.asyncio
    async def test_sends_body(self) -> None:
        """Test that the JSON body is sent correctly."""
        send = _SendCollector()
        await send_json_response(send, 201, {"created": True})
        assert send.body_json == {"created": True}

    @pytest.mark.asyncio
    async def test_content_length_header(self) -> None:
        """Test that content-length header matches body size."""
        send = _SendCollector()
        data = {"field": "value"}
        await send_json_response(send, 200, data)
        body_bytes = json.dumps(data).encode("utf-8")
        start_msg = send.messages[0]
        headers = {h[0]: h[1] for h in start_msg["headers"]}
        assert headers[b"content-length"] == str(len(body_bytes)).encode()

    @pytest.mark.asyncio
    async def test_error_status_code(self) -> None:
        """Test sending an error response."""
        send = _SendCollector()
        await send_json_response(send, 500, {"error": "internal"})
        assert send.status == 500
        assert send.body_json["error"] == "internal"


# ---------------------------------------------------------------------------
# create_app route tests
# ---------------------------------------------------------------------------
class TestCreateAppRoutes:
    """Tests for the ASGI app created by create_app."""

    @pytest.fixture
    def service(self) -> TraigentService:
        """Create a TraigentService with handlers registered."""
        svc = TraigentService(
            tunable_id="test_svc",
            version="1.0",
            supports_keep_alive=True,
        )

        @svc.tvars
        def cfg():
            return {"model": {"type": "enum", "values": ["gpt-4"]}}

        @svc.execute
        def run(example_id, data, config):
            return {"output": "ok", "cost_usd": 0.01}

        @svc.evaluate
        def score(output, target, config):
            return {"accuracy": 1.0}

        return svc

    @pytest.fixture
    def app(self, service: TraigentService):
        """Create ASGI app from service."""
        return create_app(service)

    # --- GET /traigent/v1/capabilities ---
    @pytest.mark.asyncio
    async def test_capabilities_route(self, app, service) -> None:
        """Test GET capabilities endpoint."""
        send = _SendCollector()
        await app(
            _make_scope("GET", CAPABILITIES_PATH),
            _make_receive(),
            send,
        )
        assert send.status == 200
        body = send.body_json
        assert body["version"] == "1.0"
        assert body["supports_evaluate"] is True

    # --- GET /traigent/v1/config-space ---
    @pytest.mark.asyncio
    async def test_config_space_route(self, app, service) -> None:
        """Test GET config-space endpoint."""
        send = _SendCollector()
        await app(
            _make_scope("GET", CONFIG_SPACE_PATH),
            _make_receive(),
            send,
        )
        assert send.status == 200
        body = send.body_json
        assert body["tunable_id"] == "test_svc"
        assert len(body["tvars"]) == 1

    @pytest.mark.asyncio
    async def test_config_space_route_includes_estimated_tokens_when_configured(
        self,
    ) -> None:
        """Configured wrapper token estimates should be exposed via config-space."""
        service = TraigentService(
            tunable_id="test_svc",
            estimated_tokens_per_example={"input_tokens": 100, "output_tokens": 50},
        )

        @service.tvars
        def config_space():
            return {"model": {"type": "enum", "values": ["gpt-4"]}}

        app = create_app(service)
        send = _SendCollector()
        await app(
            _make_scope("GET", CONFIG_SPACE_PATH),
            _make_receive(),
            send,
        )
        assert send.status == 200
        body = send.body_json
        assert body["estimated_tokens_per_example"] == {
            "input_tokens": 100,
            "output_tokens": 50,
        }

    # --- POST /traigent/v1/execute ---
    @pytest.mark.asyncio
    async def test_execute_route(self, app, service) -> None:
        """Test POST execute endpoint."""
        request_body = json.dumps(
            {
                "benchmark_id": "bench_001",
                "config": {"model": "gpt-4"},
                "examples": [{"example_id": "i1", "data": {"q": "hi"}}],
            }
        ).encode()
        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(request_body),
            send,
        )
        assert send.status == 200
        body = send.body_json
        assert body["status"] == "completed"
        assert len(body["outputs"]) == 1

    # --- POST /traigent/v1/evaluate ---
    @pytest.mark.asyncio
    async def test_evaluate_route(self, app, service) -> None:
        """Test POST evaluate endpoint."""
        request_body = json.dumps(
            {
                "benchmark_id": "bench_001",
                "evaluations": [{"example_id": "e1", "output": "a", "target": "a"}],
            }
        ).encode()
        send = _SendCollector()
        await app(
            _make_scope("POST", EVALUATE_PATH),
            _make_receive(request_body),
            send,
        )
        assert send.status == 200
        body = send.body_json
        assert body["status"] == "completed"

    # --- GET /traigent/v1/health ---
    @pytest.mark.asyncio
    async def test_health_route(self, app, service) -> None:
        """Test GET health endpoint."""
        send = _SendCollector()
        await app(
            _make_scope("GET", HEALTH_PATH),
            _make_receive(),
            send,
        )
        assert send.status == 200
        body = send.body_json
        assert body["status"] == "healthy"

    # --- POST /traigent/v1/keep-alive ---
    @pytest.mark.asyncio
    async def test_keep_alive_existing_session(self, app, service) -> None:
        """Test keep-alive with a valid session."""
        sid = service.create_session()
        request_body = json.dumps({"session_id": sid}).encode()
        send = _SendCollector()
        await app(
            _make_scope("POST", KEEP_ALIVE_PATH),
            _make_receive(request_body),
            send,
        )
        assert send.status == 200
        assert send.body_json["status"] == "alive"
        assert send.body_json["session_id"] == sid

    @pytest.mark.asyncio
    async def test_keep_alive_missing_session(self, app, service) -> None:
        """Test keep-alive with a non-existent session returns 404."""
        request_body = json.dumps({"session_id": "nonexistent"}).encode()
        send = _SendCollector()
        await app(
            _make_scope("POST", KEEP_ALIVE_PATH),
            _make_receive(request_body),
            send,
        )
        # Wrapper now auto-creates keep-alive sessions for stateful integrations.
        assert send.status == 200
        assert send.body_json["status"] == "alive"
        assert send.body_json["session_id"] == "nonexistent"

    # --- 404 for unknown routes ---
    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self, app) -> None:
        """Test that unknown routes return 404."""
        send = _SendCollector()
        await app(
            _make_scope("GET", "/unknown/path"),
            _make_receive(),
            send,
        )
        assert send.status == 404
        assert send.body_json["error"]["code"] == "NOT_FOUND"
        assert "Not found" in send.body_json["error"]["message"]

    @pytest.mark.asyncio
    async def test_wrong_method_returns_404(self, app) -> None:
        """Test that wrong HTTP method returns 404."""
        send = _SendCollector()
        await app(
            _make_scope("POST", CAPABILITIES_PATH),  # Should be GET
            _make_receive(),
            send,
        )
        assert send.status == 404

    # --- Non-HTTP scope is ignored ---
    @pytest.mark.asyncio
    async def test_non_http_scope_ignored(self, app) -> None:
        """Test that non-HTTP scopes are ignored."""
        send = _SendCollector()
        await app(
            {"type": "websocket", "path": "/ws"},
            _make_receive(),
            send,
        )
        assert len(send.messages) == 0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------
class TestCreateAppErrorHandling:
    """Tests for error handling in the ASGI app."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self) -> None:
        """Test that invalid JSON in request body returns 400."""
        svc = TraigentService()

        @svc.execute
        def run(example_id, data, config):
            return {"output": "ok"}

        app = create_app(svc)
        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(b"not valid json{{{"),
            send,
        )
        assert send.status == 400
        assert send.body_json["error"]["code"] == "INVALID_JSON"
        assert "Invalid JSON" in send.body_json["error"]["message"]

    @pytest.mark.asyncio
    async def test_value_error_returns_400(self) -> None:
        """Test that ValueError from handler returns 400."""
        svc = TraigentService()
        # No execute handler -> handle_execute raises ValueError
        app = create_app(svc)
        send = _SendCollector()
        request_body = json.dumps({"examples": []}).encode()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(request_body),
            send,
        )
        assert send.status == 400
        assert send.body_json["error"]["code"] == "INVALID_REQUEST"
        assert "No execute handler" in send.body_json["error"]["message"]

    @pytest.mark.asyncio
    async def test_generic_exception_returns_500(self) -> None:
        """Test that unexpected exceptions return 500."""
        svc = TraigentService()
        # Mock get_capabilities to raise an unexpected error
        svc.get_capabilities = MagicMock(side_effect=RuntimeError("unexpected"))
        app = create_app(svc)
        send = _SendCollector()
        await app(
            _make_scope("GET", CAPABILITIES_PATH),
            _make_receive(),
            send,
        )
        assert send.status == 500
        assert send.body_json["error"]["code"] == "INTERNAL_ERROR"
        assert "unexpected" in send.body_json["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_body_treated_as_empty_dict(self) -> None:
        """Test that empty request body is treated as empty dict.

        An empty body becomes ``{}``, which lacks ``benchmark_id``.
        The service returns a failed response with INVALID_BENCHMARK_ID
        (not a raised exception), so the server sends it back as 200.
        """
        svc = TraigentService()

        @svc.execute
        def run(example_id, data, config):
            return {"output": "ok"}

        app = create_app(svc)
        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(b""),
            send,
        )
        assert send.status == 200
        assert send.body_json["status"] == "failed"
        assert send.body_json["error"]["code"] == "INVALID_BENCHMARK_ID"


# ---------------------------------------------------------------------------
# run_server tests
# ---------------------------------------------------------------------------
class TestRunServer:
    """Tests for the run_server function."""

    @patch("traigent.wrapper.server.logger")
    def test_uvicorn_server(self, mock_logger: MagicMock) -> None:
        """Test that run_server with uvicorn calls uvicorn.run."""
        mock_uvicorn = MagicMock()
        app = MagicMock()

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            run_server(app, host="127.0.0.1", port=9090, server="uvicorn")

        mock_uvicorn.run.assert_called_once_with(
            app, host="127.0.0.1", port=9090, log_level="info"
        )

    @patch("traigent.wrapper.server.logger")
    def test_uvicorn_not_installed(self, mock_logger: MagicMock) -> None:
        """Test that missing uvicorn raises ImportError."""
        app = MagicMock()

        with patch.dict("sys.modules", {"uvicorn": None}):
            with pytest.raises(ImportError, match="uvicorn is required"):
                run_server(app, server="uvicorn")

    @patch("traigent.wrapper.server.logger")
    def test_hypercorn_server(self, mock_logger: MagicMock) -> None:
        """Test that run_server with hypercorn calls hypercorn serve."""
        mock_hypercorn_asyncio = MagicMock()
        mock_hypercorn_config = MagicMock()
        mock_config_instance = MagicMock()
        mock_hypercorn_config.Config.return_value = mock_config_instance

        app = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "hypercorn": MagicMock(),
                "hypercorn.asyncio": mock_hypercorn_asyncio,
                "hypercorn.config": mock_hypercorn_config,
            },
        ):
            with patch("asyncio.run") as mock_asyncio_run:
                run_server(app, host="0.0.0.0", port=8080, server="hypercorn")

                mock_asyncio_run.assert_called_once()
                mock_hypercorn_config.Config.assert_called_once()
                assert mock_config_instance.bind == ["0.0.0.0:8080"]

    @patch("traigent.wrapper.server.logger")
    def test_hypercorn_not_installed(self, mock_logger: MagicMock) -> None:
        """Test that missing hypercorn raises ImportError."""
        app = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "hypercorn": None,
                "hypercorn.asyncio": None,
                "hypercorn.config": None,
            },
        ):
            with pytest.raises(ImportError, match="hypercorn is required"):
                run_server(app, server="hypercorn")

    @patch("traigent.wrapper.server.logger")
    def test_unknown_server_raises_value_error(self, mock_logger: MagicMock) -> None:
        """Test that unknown server name raises ValueError."""
        app = MagicMock()
        with pytest.raises(ValueError, match="Unknown server"):
            run_server(app, server="gunicorn")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _make_error_response tests
# ---------------------------------------------------------------------------
class TestMakeErrorResponse:
    """Tests for _make_error_response helper."""

    def test_without_details(self) -> None:
        """Test error response without details."""
        from traigent.wrapper.server import _make_error_response

        result = _make_error_response("ERR_CODE", "Something went wrong")
        assert result == {
            "error": {"code": "ERR_CODE", "message": "Something went wrong"}
        }

    def test_with_details(self) -> None:
        """Test error response with details included."""
        from traigent.wrapper.server import _make_error_response

        result = _make_error_response(
            "VALIDATION_ERROR",
            "Bad input",
            details={"field": "temperature", "reason": "out of range"},
        )
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert result["error"]["details"]["field"] == "temperature"


# ---------------------------------------------------------------------------
# Keep-alive 404 path test
# ---------------------------------------------------------------------------
class TestKeepAlive404:
    """Test keep-alive returns 404 when session is not found."""

    @pytest.mark.asyncio
    async def test_keep_alive_returns_404_when_session_not_found(self) -> None:
        """Keep-alive for unknown session with keep_alive disabled returns 404."""
        svc = TraigentService(
            tunable_id="test_svc",
            supports_keep_alive=False,
        )
        app = create_app(svc)
        request_body = json.dumps({"session_id": "nonexistent"}).encode()
        send = _SendCollector()
        await app(
            _make_scope("POST", KEEP_ALIVE_PATH),
            _make_receive(request_body),
            send,
        )
        assert send.status == 404
        assert send.body_json["error"]["code"] == "SESSION_NOT_FOUND"


# ---------------------------------------------------------------------------
# Tests for server helper functions
# ---------------------------------------------------------------------------


class TestExtractTraceHeaders:
    def test_extracts_traceparent_and_tracestate(self) -> None:
        scope = {
            "headers": [
                (b"traceparent", b"00-abc-def-01"),
                (b"tracestate", b"vendor=val"),
                (b"content-type", b"application/json"),
            ]
        }
        result = _extract_trace_headers(scope)
        assert result == {
            "traceparent": "00-abc-def-01",
            "tracestate": "vendor=val",
        }

    def test_empty_headers(self) -> None:
        assert _extract_trace_headers({"headers": []}) == {}

    def test_no_headers_key(self) -> None:
        assert _extract_trace_headers({}) == {}


class TestResponseHeadersWithRequestId:
    def test_adds_request_id(self) -> None:
        result = _response_headers_with_request_id(
            {"content-type": "application/json"}, "req-123"
        )
        assert result["x-traigent-request-id"] == "req-123"
        assert result["content-type"] == "application/json"

    def test_none_request_id_omits_header(self) -> None:
        result = _response_headers_with_request_id({}, None)
        assert "x-traigent-request-id" not in result


class TestGetTimeoutMs:
    def test_missing_key_returns_default(self) -> None:
        assert _get_timeout_ms({}, default_ms=5000) == 5000

    def test_null_value_returns_default(self) -> None:
        assert _get_timeout_ms({"timeout_ms": None}, default_ms=5000) == 5000

    def test_valid_value(self) -> None:
        assert _get_timeout_ms({"timeout_ms": 3000}, default_ms=None) == 3000

    def test_bool_raises(self) -> None:
        with pytest.raises(ValueError, match="must be an integer"):
            _get_timeout_ms({"timeout_ms": True}, default_ms=None)

    def test_below_minimum_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1000"):
            _get_timeout_ms({"timeout_ms": 500}, default_ms=None)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for HybridAPIError and timeout handling in ASGI app."""

    @pytest.mark.asyncio
    async def test_hybrid_api_error_caught_and_returned(self) -> None:
        """HybridAPIError from handler should be caught and returned as JSON."""
        from traigent.wrapper.errors import BadRequestError

        svc = TraigentService()

        @svc.execute
        def run(example_id, data, config):
            raise BadRequestError(
                "Invalid input format",
                error_code="VALIDATION_ERROR",
                details={"field": "temperature"},
            )

        app = create_app(svc)
        request_body = json.dumps(
            {
                "tunable_id": "default",
                "benchmark_id": "bench_001",
                "examples": [{"example_id": "i1", "data": {}}],
            }
        ).encode()

        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(request_body),
            send,
        )

        assert send.status == 400
        assert send.body_json["error"]["code"] == "VALIDATION_ERROR"
        assert "Invalid input format" in send.body_json["error"]["message"]
        assert send.body_json["error"]["details"]["field"] == "temperature"

    @pytest.mark.asyncio
    async def test_execute_timeout_raises_request_timeout_error(self) -> None:
        """Execute handler exceeding timeout should raise RequestTimeoutError."""
        import asyncio

        from traigent.wrapper.errors import RequestTimeoutError

        svc = TraigentService()

        @svc.execute
        async def run(example_id, data, config):
            await asyncio.sleep(2)  # Longer than timeout
            return {"output": "done"}

        app = create_app(svc)
        request_body = json.dumps(
            {
                "tunable_id": "default",
                "benchmark_id": "bench_001",
                "timeout_ms": 1000,  # 1 second timeout
                "examples": [{"example_id": "i1", "data": {}}],
            }
        ).encode()

        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(request_body),
            send,
        )

        assert send.status == 408
        assert "timed out" in send.body_json["error"]["message"].lower()
        assert send.body_json["error"]["code"] == "REQUEST_TIMEOUT"

    @pytest.mark.asyncio
    async def test_hybrid_api_error_with_custom_headers(self) -> None:
        """HybridAPIError with custom headers should include them in response."""
        from traigent.wrapper.errors import RateLimitError

        svc = TraigentService()

        @svc.execute
        def run(example_id, data, config):
            raise RateLimitError(retry_after=60)

        app = create_app(svc)
        request_body = json.dumps(
            {
                "tunable_id": "default",
                "benchmark_id": "bench_001",
                "examples": [{"example_id": "i1", "data": {}}],
            }
        ).encode()

        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(request_body),
            send,
        )

        assert send.status == 429
        assert send.body_json["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_evaluate_timeout_raises_request_timeout_error(self) -> None:
        """Evaluate handler exceeding timeout should raise RequestTimeoutError."""
        import asyncio

        svc = TraigentService()

        @svc.evaluate
        async def score(output, target, config):
            await asyncio.sleep(2)  # Longer than timeout
            return {"accuracy": 1.0}

        app = create_app(svc)
        request_body = json.dumps(
            {
                "tunable_id": "default",
                "benchmark_id": "bench_001",
                "timeout_ms": 1000,  # 1 second timeout
                "evaluations": [{"example_id": "e1", "output": "a", "target": "a"}],
            }
        ).encode()

        send = _SendCollector()
        await app(
            _make_scope("POST", EVALUATE_PATH),
            _make_receive(request_body),
            send,
        )

        assert send.status == 408
        assert "timed out" in send.body_json["error"]["message"].lower()
        assert send.body_json["error"]["code"] == "REQUEST_TIMEOUT"

    @pytest.mark.asyncio
    async def test_hybrid_api_error_with_custom_headers(self) -> None:
        """HybridAPIError with custom headers should include them in response."""
        from traigent.wrapper.errors import RateLimitError

        svc = TraigentService()

        @svc.execute
        def run(example_id, data, config):
            raise RateLimitError(retry_after=60)

        app = create_app(svc)
        request_body = json.dumps(
            {
                "tunable_id": "default",
                "benchmark_id": "bench_001",
                "examples": [{"example_id": "i1", "data": {}}],
            }
        ).encode()

        send = _SendCollector()
        await app(
            _make_scope("POST", EXECUTE_PATH),
            _make_receive(request_body),
            send,
        )

        assert send.status == 429
        assert send.body_json["error"]["code"] == "RATE_LIMITED"
