"""Tool implementations for the agent-facing Traigent analytics MCP server.

These tools are a **thin authenticated client** over the backend analytics
endpoints (via :class:`traigent.cloud.analytics_client.BackendAnalyticsClient`)
plus a local chart-render helper
(:func:`traigent.analytics.render.render_chart`).

Design rules enforced here:

* Every cloud tool requires an explicit ``project_id`` (and ``run_id`` where
  relevant) — there is no implicit "latest run".
* The MCP never trusts a caller-supplied ``tenant_id``; tenancy is owned by the
  backend and derived from the authenticated principal. No tool accepts a
  tenant argument.
* Tools really call the client (no fake completion). HTTP is only ever mocked in
  tests, never here.
* Credentials and backend transport errors are reported as structured failures,
  never raised into the MCP payload (an error message could embed a response
  body or URL).
"""

# Traceability: CONC-Layer-Infra CONC-Security FUNC-ANALYTICS

from __future__ import annotations

from typing import Any, cast

from traigent.cloud.analytics_client import normalize_decision_intent
from traigent.utils.logging import get_logger

try:
    import httpx
except ImportError:  # pragma: no cover - BackendAnalyticsClient reports this first
    _HTTP_STATUS_ERRORS: tuple[type[BaseException], ...] = ()
    _HTTP_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = ()
else:
    _HTTP_STATUS_ERRORS = (httpx.HTTPStatusError,)
    _HTTP_TRANSPORT_ERRORS = (httpx.TransportError,)

logger = get_logger(__name__)

ANALYTICS_TOOL_NAMES: tuple[str, ...] = (
    "health_check",
    "auth_status",
    "analytics_get_run_report",
    "analytics_get_project_overview",
    "analytics_compare_runs",
    "analytics_get_run_decision_brief",
    "analytics_get_single_run_pareto",
    "analytics_get_correlation_matrix",
    "analytics_get_run_leaderboard",
    "analytics_get_parameter_insights",
    "analytics_get_example_insights",
    "analytics_render_chart",
)

_SUPPORTED_CHART_KINDS: tuple[str, ...] = ("run_pareto", "run_correlations")


def _failure(
    message: str,
    *,
    code: str = "invalid_input",
    http_status: int | None = None,
    backend_url: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "code": code,
        "message": message,
        "http_status": http_status,
        "backend_url": (
            backend_url if backend_url is not None else _resolved_backend_url()
        ),
    }


def _auth_failure(*, backend_url: str | None = None) -> dict[str, Any]:
    return _failure(
        "Analytics authentication failed; configure valid Traigent credentials.",
        code="authentication_failed",
        backend_url=backend_url,
    )


def _require_identifier(value: str | None, *, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise _ToolInputError(f"{field} is required and must be a non-empty string.")
    return text


class _ToolInputError(ValueError):
    """Raised for user-correctable analytics MCP tool input errors."""


async def _new_analytics_client() -> Any:
    """Construct a backend analytics read client using SDK credentials.

    Reuses the same AuthManager-backed analytics credential resolver as
    ``client.analytics`` before constructing the low-level read client.
    """
    from traigent.cloud.analytics_auth import resolve_analytics_read_client_credentials
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    credential_kwargs = await resolve_analytics_read_client_credentials()
    return BackendAnalyticsClient(
        api_key=credential_kwargs.get("api_key"),
        jwt_token=credential_kwargs.get("jwt_token"),
    )


def _resolved_backend_url(client: Any | None = None) -> str | None:
    candidate = getattr(client, "backend_url", None)
    if not candidate:
        try:
            from traigent.config.backend_config import BackendConfig

            candidate = BackendConfig.get_backend_url()
        except Exception:  # pragma: no cover - defensive structured fallback
            return None
    return _sanitize_url(str(candidate))


def _response_payload(response: Any | None) -> Any:
    if response is None:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _iter_payload_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_payload_text(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_payload_text(value)
    elif isinstance(payload, str):
        yield payload


def _signals_not_ready(response: Any | None) -> bool:
    payload = _response_payload(response)
    for text in _iter_payload_text(payload):
        normalized = text.lower().replace("_", " ").replace("-", " ")
        if (
            "not ready" in normalized
            or "not yet computed" in normalized
            or "not computed" in normalized
            or "not yet available" in normalized
            or "compute to trigger" in normalized
            or "analytics pending" in normalized
        ):
            return True
    return False


def _http_status_failure(
    exc: BaseException,
    *,
    what: str,
    backend_url: str | None,
) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    status = int(status) if isinstance(status, int) else None

    if status == 401:
        return _failure(
            "Analytics authentication failed; configure valid Traigent credentials.",
            code="authentication_failed",
            http_status=status,
            backend_url=backend_url,
        )
    if status == 403:
        return _failure(
            "Analytics request was forbidden by the backend; check project permissions "
            "or whether a WAF/Cloudflare rule is blocking the request.",
            code="forbidden",
            http_status=status,
            backend_url=backend_url,
        )
    if status == 404:
        if _signals_not_ready(response):
            return _failure(
                f"The backend found the run, but {what} analytics are not ready yet.",
                code="not_ready",
                http_status=status,
                backend_url=backend_url,
            )
        return _failure(
            f"The requested {what} was not found on the configured backend.",
            code="not_found",
            http_status=status,
            backend_url=backend_url,
        )

    return _failure(
        f"Could not retrieve {what} from the backend.",
        code="backend_unavailable",
        http_status=status,
        backend_url=backend_url,
    )


async def _call_backend(coro_factory: Any, *, what: str) -> dict[str, Any]:
    """Run a backend read coroutine and normalize failures to structured output.

    ``coro_factory`` is a callable that takes the opened client and returns the
    awaitable. Transport / auth / contract errors become ``ok=False`` payloads
    with a generic message — raw exception text (which may embed URLs or
    response bodies) never reaches the caller.
    """
    from traigent.cloud.analytics_client import AnalyticsClientError
    from traigent.cloud.auth import InvalidCredentialsError

    client = None
    try:
        client = await _new_analytics_client()
    except ImportError as exc:
        return _failure(str(exc), code="dependency_missing")
    except InvalidCredentialsError as exc:
        logger.debug("Analytics %s authentication failed: %s", what, exc)
        return _auth_failure()

    backend_url = _resolved_backend_url(client)
    try:
        async with client as reader:
            data = await coro_factory(reader)
    except AnalyticsClientError:
        return _failure(
            f"The backend returned a malformed {what} response.",
            code="malformed_response",
            backend_url=backend_url,
        )
    except InvalidCredentialsError as exc:
        logger.debug("Analytics %s authentication failed: %s", what, exc)
        return _auth_failure(backend_url=backend_url)
    except _HTTP_STATUS_ERRORS as exc:
        logger.debug("Analytics %s request returned HTTP status", what, exc_info=True)
        return _http_status_failure(exc, what=what, backend_url=backend_url)
    except _HTTP_TRANSPORT_ERRORS as exc:
        logger.debug("Analytics %s transport failed: %s", what, exc)
        return _failure(
            f"Could not retrieve {what} from the backend.",
            code="backend_unavailable",
            backend_url=backend_url,
        )
    except Exception as exc:  # noqa: BLE001 - normalize all transport failures
        # Do not surface raw exception text: it can contain the backend URL,
        # auth header echoes, or response bodies. Log at debug for operators.
        logger.debug("Analytics %s request failed: %s", what, exc)
        return _failure(
            f"Could not retrieve {what} from the backend.",
            code="backend_unavailable",
            backend_url=backend_url,
        )

    return {"ok": True, what.replace(" ", "_"): data}


async def health_check_tool() -> dict[str, Any]:
    """Report whether the analytics MCP can reach its dependencies (no network).

    Reports SDK/import readiness and resolved (sanitized) backend URL. Does NOT
    make a network call or return any credential material.
    """
    from traigent.config.backend_config import BackendConfig

    httpx_ready = False
    try:
        from traigent.cloud.analytics_client import HTTPX_AVAILABLE

        httpx_ready = bool(HTTPX_AVAILABLE)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("analytics_client import failed during health_check: %s", exc)

    matplotlib_ready = False
    try:
        import importlib.util

        matplotlib_ready = importlib.util.find_spec("matplotlib") is not None
    except Exception:  # pragma: no cover - defensive
        matplotlib_ready = False

    backend_url = _sanitize_url(BackendConfig.get_backend_url())
    return {
        "ok": True,
        "service": "traigent-analytics-mcp",
        "backend_url": backend_url,
        "httpx_available": httpx_ready,
        "chart_rendering_available": matplotlib_ready,
        "supported_chart_kinds": list(_SUPPORTED_CHART_KINDS),
    }


def _sanitize_url(url: str | None) -> str | None:
    """Strip userinfo and query material from a URL before returning it."""
    if not url:
        return url
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        parts = parts._replace(netloc=host)
    return urlunsplit(parts._replace(query="", fragment=""))


def _mask_api_key(api_key: str | None) -> dict[str, Any]:
    if not api_key:
        return {"present": False, "prefix": None, "last4": None}
    prefix = api_key[:4]
    last4 = api_key[-4:] if len(api_key) > 8 else None
    return {"present": True, "prefix": prefix, "last4": last4}


async def auth_status_tool() -> dict[str, Any]:
    """Report local auth posture for the analytics MCP (masked; no network).

    API key material is masked to prefix and last4 only.
    """
    import os

    from traigent.cloud.credential_manager import CredentialManager

    credentials = CredentialManager.get_credentials()
    api_key = credentials.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        api_key = None
    jwt_token = credentials.get("jwt_token")
    if jwt_token is not None and not isinstance(jwt_token, str):
        jwt_token = None
    if isinstance(jwt_token, str) and not jwt_token.strip():
        jwt_token = None

    credential_source = credentials.get("source") or "none"
    if not api_key and not jwt_token:
        env_jwt_token = os.getenv("TRAIGENT_JWT_TOKEN")
        if env_jwt_token:
            jwt_token = env_jwt_token
            credential_source = "environment"

    credential_present = bool(api_key or jwt_token)
    authenticated = bool(api_key)
    if jwt_token and not api_key:
        from traigent.cloud.analytics_auth import (
            resolve_analytics_read_client_credentials,
        )
        from traigent.cloud.auth import InvalidCredentialsError

        try:
            credential_kwargs = await resolve_analytics_read_client_credentials()
        except InvalidCredentialsError as exc:
            logger.debug("Analytics JWT credential validation failed: %s", exc)
            authenticated = False
        else:
            authenticated = bool(credential_kwargs.get("jwt_token"))

    return {
        "ok": True,
        "authenticated": authenticated,
        "credential_present": credential_present,
        "credential_source": credential_source if credential_present else "none",
        "auth_type": "api_key" if api_key else ("jwt" if jwt_token else "none"),
        "api_key": _mask_api_key(api_key),
        "backend_url": _sanitize_url(credentials.get("backend_url")),
    }


async def analytics_get_run_report_tool(project_id: str, run_id: str) -> dict[str, Any]:
    """Fetch the backend's full analytics report for one run."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_run_report(pid, rid), what="run report"
    )


async def analytics_get_project_overview_tool(project_id: str) -> dict[str, Any]:
    """Fetch the backend's cross-run overview for a project."""
    try:
        pid = _require_identifier(project_id, field="project_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_project_overview(pid), what="project overview"
    )


async def analytics_compare_runs_tool(
    project_id: str, run_ids: list[str]
) -> dict[str, Any]:
    """Compare two or more runs within a project."""
    try:
        pid = _require_identifier(project_id, field="project_id")
    except _ToolInputError as exc:
        return _failure(str(exc))

    if not isinstance(run_ids, list):
        return _failure("run_ids must be a list of run identifiers.")
    cleaned = [str(rid).strip() for rid in run_ids if str(rid).strip()]
    if len(cleaned) < 2:
        return _failure("compare_runs requires at least two run_ids.")

    return await _call_backend(
        lambda reader: reader.compare_runs(pid, cleaned), what="run comparison"
    )


async def analytics_get_run_decision_brief_tool(
    project_id: str,
    run_id: str,
    intent: str = "iterate",
) -> dict[str, Any]:
    """Fetch the backend's decision brief (decision_payload v0) for a run."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    try:
        normalized_intent = normalize_decision_intent(intent)
    except ValueError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_run_decision_brief(pid, rid, normalized_intent),
        what="decision brief",
    )


async def analytics_get_single_run_pareto_tool(
    project_id: str,
    run_id: str,
    x_measure: str = "cost",
    y_measure: str = "quality",
    request_count: int = 1,
) -> dict[str, Any]:
    """Fetch the backend's Pareto frontier (run_pareto v0) for one run."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_single_run_pareto(
            pid,
            rid,
            x_measure=x_measure,
            y_measure=y_measure,
            request_count=request_count,
        ),
        what="single run pareto",
    )


async def analytics_get_correlation_matrix_tool(
    project_id: str,
    run_id: str,
    method: str = "pearson",
    min_sample: int = 3,
) -> dict[str, Any]:
    """Fetch the backend's correlation matrix (run_correlations v0)."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_correlation_matrix(
            pid,
            rid,
            method=method,
            min_sample=min_sample,
        ),
        what="correlation matrix",
    )


async def analytics_get_run_leaderboard_tool(
    project_id: str,
    run_id: str,
    objective: str = "weighted",
    weights: dict[str, object] | str | None = None,
    constraints: dict[str, object] | str | None = None,
    request_count: int = 1,
    limit: int = 50,
) -> dict[str, Any]:
    """Fetch the backend's ranked configuration leaderboard for one run."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_run_leaderboard(
            pid,
            rid,
            objective=objective,
            weights=weights,
            constraints=constraints,
            request_count=request_count,
            limit=limit,
        ),
        what="run leaderboard",
    )


async def analytics_get_parameter_insights_tool(
    project_id: str,
    run_id: str,
    target_measure: str = "quality",
    min_trials: int = 10,
    top_k: int = 10,
) -> dict[str, Any]:
    """Fetch the backend's parameter-importance insights for one run."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_parameter_insights(
            pid,
            rid,
            target_measure=target_measure,
            min_trials=min_trials,
            top_k=top_k,
        ),
        what="parameter insights",
    )


async def analytics_get_example_insights_tool(
    project_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Fetch the backend's privacy-bounded example insights for one run."""
    try:
        pid = _require_identifier(project_id, field="project_id")
        rid = _require_identifier(run_id, field="run_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_example_insights(pid, rid),
        what="example insights",
    )


def analytics_render_chart_tool(
    payload: dict[str, Any],
    kind: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Render a canonical analytics payload to an image file; return its path.

    The payload must be a backend-produced ``run_pareto`` or
    ``run_correlations`` document. Pixels are rendered from the payload's
    numbers; no analytics are recomputed.
    """
    if kind not in _SUPPORTED_CHART_KINDS:
        return _failure(
            f"kind must be one of {list(_SUPPORTED_CHART_KINDS)}.",
        )
    if not isinstance(payload, dict):
        return _failure(
            "payload must be a JSON object (the canonical analytics document)."
        )

    from traigent.analytics.render import ChartRenderError, render_chart

    try:
        path = render_chart(payload, cast(Any, kind), output_path)
    except ChartRenderError as exc:
        return _failure(str(exc), code="render_failed")
    except Exception as exc:  # noqa: BLE001 - defensive; surface as structured failure
        logger.debug("Chart render failed: %s", exc)
        return _failure("Chart rendering failed.", code="render_failed")

    return {"ok": True, "kind": kind, "chart_path": path}
