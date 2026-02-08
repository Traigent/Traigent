"""HTTP server for TraigentService.

Provides an ASGI application that exposes Traigent hybrid API endpoints.
Can run with uvicorn or hypercorn.
"""

# Traceability: HYBRID-MODE-OPTIMIZATION CLIENT-WRAPPER-SDK

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from traigent.utils.logging import get_logger

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

        # Route to handler
        try:
            if path == CAPABILITIES_PATH and method == "GET":
                response = service.get_capabilities()
                await send_json_response(send, 200, response)

            elif path == CONFIG_SPACE_PATH and method == "GET":
                response = service.get_config_space()
                await send_json_response(send, 200, response)

            elif path == EXECUTE_PATH and method == "POST":
                body = await read_body(receive)
                request = json.loads(body) if body else {}
                response = await service.handle_execute(request)
                await send_json_response(send, 200, response)

            elif path == EVALUATE_PATH and method == "POST":
                body = await read_body(receive)
                request = json.loads(body) if body else {}
                response = await service.handle_evaluate(request)
                await send_json_response(send, 200, response)

            elif path == HEALTH_PATH and method == "GET":
                response = service.get_health()
                await send_json_response(send, 200, response)

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
                    )
                else:
                    await send_json_response(
                        send,
                        404,
                        _make_error_response(
                            "SESSION_NOT_FOUND",
                            f"Session not found: {session_id}",
                        ),
                    )

            else:
                await send_json_response(
                    send,
                    404,
                    _make_error_response(
                        "NOT_FOUND",
                        f"Not found: {method} {path}",
                    ),
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            await send_json_response(
                send,
                400,
                _make_error_response("INVALID_JSON", f"Invalid JSON: {e}"),
            )

        except ValueError as e:
            logger.error(f"Request error: {e}")
            await send_json_response(
                send,
                400,
                _make_error_response("INVALID_REQUEST", str(e)),
            )

        except Exception as e:
            logger.error(f"Internal error: {e}")
            await send_json_response(
                send,
                500,
                _make_error_response("INTERNAL_ERROR", str(e)),
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
) -> None:
    """Send JSON response via ASGI send."""
    body = json.dumps(data).encode("utf-8")

    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
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
