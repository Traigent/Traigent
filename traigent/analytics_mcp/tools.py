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

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

ANALYTICS_TOOL_NAMES: tuple[str, ...] = (
    "health_check",
    "auth_status",
    "analytics_get_run_report",
    "analytics_get_project_overview",
    "analytics_compare_runs",
    "analytics_get_run_decision_brief",
    "analytics_render_chart",
)

_SUPPORTED_CHART_KINDS: tuple[str, ...] = ("run_pareto", "run_correlations")


def _failure(message: str, *, code: str = "invalid_input") -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message}


def _require_identifier(value: str | None, *, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise _ToolInputError(f"{field} is required and must be a non-empty string.")
    return text


class _ToolInputError(ValueError):
    """Raised for user-correctable analytics MCP tool input errors."""


def _new_analytics_client() -> Any:
    """Construct a backend analytics read client using SDK credentials.

    Reuses :class:`BackendAnalyticsClient`, which resolves the backend URL and
    API key through the SDK's existing credential plumbing.
    """
    from traigent.cloud.analytics_client import BackendAnalyticsClient

    return BackendAnalyticsClient()


async def _call_backend(coro_factory: Any, *, what: str) -> dict[str, Any]:
    """Run a backend read coroutine and normalize failures to structured output.

    ``coro_factory`` is a callable that takes the opened client and returns the
    awaitable. Transport / auth / contract errors become ``ok=False`` payloads
    with a generic message — raw exception text (which may embed URLs or
    response bodies) never reaches the caller.
    """
    from traigent.cloud.analytics_client import AnalyticsClientError

    try:
        client = _new_analytics_client()
    except ImportError as exc:
        return _failure(str(exc), code="dependency_missing")

    try:
        async with client as reader:
            data = await coro_factory(reader)
    except AnalyticsClientError:
        return _failure(
            f"The backend returned a malformed {what} response.",
            code="malformed_response",
        )
    except Exception as exc:  # noqa: BLE001 - normalize all transport failures
        # Do not surface raw exception text: it can contain the backend URL,
        # auth header echoes, or response bodies. Log at debug for operators.
        logger.debug("Analytics %s request failed: %s", what, exc)
        return _failure(
            f"Could not retrieve {what} from the backend.",
            code="backend_unavailable",
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
    """Strip any userinfo (``user:pass@``) from a URL before returning it."""
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
    return urlunsplit(parts)


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
    from traigent.cloud.credential_manager import CredentialManager

    credentials = CredentialManager.get_credentials()
    api_key = credentials.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        api_key = None
    authenticated = bool(api_key or credentials.get("jwt_token"))
    return {
        "ok": True,
        "authenticated": authenticated,
        "credential_source": credentials.get("source") or "none",
        "auth_type": (
            "api_key"
            if api_key
            else ("jwt" if credentials.get("jwt_token") else "none")
        ),
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
    normalized_intent = (intent or "").strip() or "iterate"
    return await _call_backend(
        lambda reader: reader.get_run_decision_brief(pid, rid, normalized_intent),
        what="decision brief",
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
