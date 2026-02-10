"""HTTP server for TraigentService.

Provides an ASGI application that exposes Traigent hybrid API endpoints.
Can run with uvicorn or hypercorn.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION CLIENT-WRAPPER-SDK

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from traigent.utils.logging import get_logger
from traigent.wrapper.errors import HybridAPIError, RequestTimeoutError

if TYPE_CHECKING:
    from traigent.wrapper.service import TraigentService

logger = get_logger(__name__)


# API endpoint paths
CAPABILITIES_PATH = "/traigent/v1/capabilities"
CONFIG_SPACE_PATH = "/traigent/v1/config-space"
EXECUTE_PATH = "/traigent/v1/execute"
EVALUATE_PATH = "/traigent/v1/evaluate"
HEALTH_PATH = "/traigent/v1/health"
KEEP_ALIVE_PATH = "/traigent/v1/keep-alive"


def _make_error_response(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build contract-compliant error payload."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def _extract_trace_headers(scope: dict[str, Any]) -> dict[str, str]:
    """Extract W3C trace headers for optional trace propagation."""
    headers: dict[str, str] = {}
    for header_name, header_value in scope.get("headers", []):
        name = header_name.decode("latin-1").lower()
        if name in {"traceparent", "tracestate"}:
            headers[name] = header_value.decode("latin-1")
    return headers


def _response_headers_with_request_id(
    headers: dict[str, str],
    request_id: Any | None,
) -> dict[str, str]:
    """Attach correlation header when request_id is available."""
    merged = dict(headers)
    if request_id is not None:
        merged["x-traigent-request-id"] = str(request_id)
    return merged


def _get_timeout_ms(
    request: dict[str, Any],
    *,
    default_ms: int | None,
) -> int | None:
    """Parse timeout_ms field with contract-compliant validation."""
    if "timeout_ms" not in request:
        return default_ms

    timeout_ms = request.get("timeout_ms")
    if timeout_ms is None:
        return default_ms

    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
        raise ValueError("timeout_ms must be an integer")

    if timeout_ms < 1000:
        raise ValueError("timeout_ms must be >= 1000")

    return timeout_ms


def create_app(service: TraigentService) -> Callable[..., Any]:
    """Create ASGI application for TraigentService.

    Args:
        service: TraigentService instance to wrap.

    Returns:
        ASGI application callable.
    """

    async def app(scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        """ASGI application entry point."""
        if scope["type"] != "http":
            return

        path = scope["path"]
        method = scope["method"]
        trace_headers = _extract_trace_headers(scope)
        request: dict[str, Any] | None = None

        # Route to handler
        try:
            if path == CAPABILITIES_PATH and method == "GET":
                response = service.get_capabilities()
                await send_json_response(
                    send,
                    200,
                    response,
                    headers=trace_headers,
                )

            elif path == CONFIG_SPACE_PATH and method == "GET":
                response = service.get_config_space()
                await send_json_response(
                    send,
                    200,
                    response,
                    headers=trace_headers,
                )

            elif path == EXECUTE_PATH and method == "POST":
                body = await read_body(receive)
                request = json.loads(body) if body else {}
                timeout_ms = _get_timeout_ms(request, default_ms=30000)

                try:
                    response = await asyncio.wait_for(
                        service.handle_execute(request),
                        timeout=(timeout_ms / 1000.0) if timeout_ms else None,
                    )
                except TimeoutError as exc:
                    raise RequestTimeoutError(
                        f"Request timed out after {timeout_ms}ms",
                        details={
                            "endpoint": EXECUTE_PATH,
                            "timeout_ms": timeout_ms,
                        },
                    ) from exc

                response_headers = _response_headers_with_request_id(
                    trace_headers,
                    response.get("request_id"),
                )
                await send_json_response(send, 200, response, headers=response_headers)

            elif path == EVALUATE_PATH and method == "POST":
                body = await read_body(receive)
                request = json.loads(body) if body else {}
                timeout_ms = _get_timeout_ms(request, default_ms=None)

                if timeout_ms is not None:
                    try:
                        response = await asyncio.wait_for(
                            service.handle_evaluate(request),
                            timeout=timeout_ms / 1000.0,
                        )
                    except TimeoutError as exc:
                        raise RequestTimeoutError(
                            f"Request timed out after {timeout_ms}ms",
                            details={
                                "endpoint": EVALUATE_PATH,
                                "timeout_ms": timeout_ms,
                            },
                        ) from exc
                else:
                    response = await service.handle_evaluate(request)

                response_headers = _response_headers_with_request_id(
                    trace_headers,
                    response.get("request_id"),
                )
                await send_json_response(send, 200, response, headers=response_headers)

            elif path == HEALTH_PATH and method == "GET":
                response = service.get_health()
                await send_json_response(
                    send,
                    200,
                    response,
                    headers=trace_headers,
                )

            elif path == KEEP_ALIVE_PATH and method == "POST":
                body = await read_body(receive)
                request = json.loads(body) if body else {}
                session_id = request.get("session_id", "")
                alive = service.handle_keep_alive(session_id)
                if alive:
                    await send_json_response(
                        send,
                        200,
                        {
                            "status": "alive",
                            "session_id": session_id,
                        },
                        headers=trace_headers,
                    )
                else:
                    await send_json_response(
                        send,
                        404,
                        _make_error_response(
                            "SESSION_NOT_FOUND",
                            f"Session not found: {session_id}",
                        ),
                        headers=trace_headers,
                    )

            else:
                await send_json_response(
                    send,
                    404,
                    _make_error_response(
                        "NOT_FOUND",
                        f"Not found: {method} {path}",
                    ),
                    headers=trace_headers,
                )

        except HybridAPIError as e:
            logger.warning(
                "Hybrid API handler returned explicit error: status=%s code=%s message=%s",
                e.status_code,
                e.error_code,
                str(e),
            )
            error_headers = _response_headers_with_request_id(
                trace_headers,
                request.get("request_id") if request else None,
            )
            if e.headers:
                error_headers.update(e.headers)
            await send_json_response(
                send,
                e.status_code,
                _make_error_response(e.error_code, str(e), details=e.details),
                headers=error_headers,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            await send_json_response(
                send,
                400,
                _make_error_response("INVALID_JSON", f"Invalid JSON: {e}"),
                headers=trace_headers,
            )

        except ValueError as e:
            logger.error(f"Request error: {e}")
            error_headers = _response_headers_with_request_id(
                trace_headers,
                request.get("request_id") if request else None,
            )
            await send_json_response(
                send,
                400,
                _make_error_response("INVALID_REQUEST", str(e)),
                headers=error_headers,
            )

        except Exception as e:
            logger.error(f"Internal error: {e}")
            error_headers = _response_headers_with_request_id(
                trace_headers,
                request.get("request_id") if request else None,
            )
            await send_json_response(
                send,
                500,
                _make_error_response("INTERNAL_ERROR", str(e)),
                headers=error_headers,
            )

    return app


async def read_body(receive: Callable) -> bytes:
    """Read full request body from ASGI receive."""
    body = b""
    more_body = True

    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)

    return body


async def send_json_response(
    send: Callable,
    status: int,
    data: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> None:
    """Send JSON response via ASGI send."""
    body = json.dumps(data).encode("utf-8")
    response_headers: list[list[bytes]] = [
        [b"content-type", b"application/json"],
        [b"content-length", str(len(body)).encode()],
    ]
    if headers:
        for name, value in headers.items():
            response_headers.append(
                [name.encode("latin-1").lower(), str(value).encode("latin-1")]
            )

    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": response_headers,
        }
    )

    await send(
        {
            "type": "http.response.body",
            "body": body,
        }
    )


def run_server(
    app: Callable,
    host: str = "0.0.0.0",  # nosec B104 - intentional for server binding
    port: int = 8080,
    server: Literal["uvicorn", "hypercorn"] = "uvicorn",
) -> None:
    """Run ASGI server.

    Args:
        app: ASGI application to run.
        host: Host to bind to.
        port: Port to listen on.
        server: Server implementation to use.

    Raises:
        ImportError: If requested server is not installed.
    """
    logger.info(f"Starting Traigent API server on {host}:{port}")

    if server == "uvicorn":
        try:
            import uvicorn

            uvicorn.run(app, host=host, port=port, log_level="info")
        except ImportError as e:
            raise ImportError(
                "uvicorn is required to run the server. "
                "Install with: pip install uvicorn"
            ) from e

    elif server == "hypercorn":
        try:
            import asyncio

            from hypercorn.asyncio import serve
            from hypercorn.config import Config

            config = Config()
            config.bind = [f"{host}:{port}"]
            asyncio.run(serve(app, config))
        except ImportError as e:
            raise ImportError(
                "hypercorn is required to run the server. "
                "Install with: pip install hypercorn"
            ) from e

    else:
        raise ValueError(f"Unknown server: {server}")
