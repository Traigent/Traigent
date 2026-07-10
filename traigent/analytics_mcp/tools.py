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

import math
from datetime import datetime, timedelta
from typing import Any, cast

from traigent.cloud.analytics_client import (
    _project_observability_analysis_insights,
    normalize_decision_intent,
)
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
    "observability_search_traces",
    "observability_list_issues",
    "observability_get_issue",
    "observability_get_trace_slice",
    "observability_get_tool_analysis",
    "observability_get_analysis_insights",
    "observability_compare_cohorts",
    "observability_get_related_changes",
    "observability_build_change_brief",
    "analytics_list_experiment_groups",
    "analytics_get_experiment_group",
    "analytics_list_experiment_group_configuration_runs",
)

_SUPPORTED_CHART_KINDS: tuple[str, ...] = ("run_pareto", "run_correlations")
_OBSERVABILITY_MAX_WINDOW = timedelta(days=31)
_OBSERVABILITY_METRICS = frozenset(
    {
        "quality_score",
        "cost_usd",
        "latency_ms",
        "error_rate",
        "retry_rate",
        "fallback_rate",
        "input_tokens",
        "output_tokens",
    }
)
_AGGREGATE_RECOMMENDATION_RATIONALES = {
    "tool_reliability": "Measured tool failures indicate a reliability change should be tested.",
    "retry_policy": "Measured retry behavior indicates a bounded retry-policy change should be tested.",
    "tool_routing": "Measured tool outcomes indicate a routing change should be tested.",
    "recurring_issue": "Repeated issue evidence indicates a targeted remediation should be tested.",
    "behavioral_variation": "Observed structural variation indicates a control change should be tested.",
}


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


def _bounded_identifier(value: str | None, *, field: str) -> str:
    text = _require_identifier(value, field=field)
    if len(text) > 128:
        raise _ToolInputError(f"{field} must be at most 128 characters.")
    return text


def _bounded_page(page: int, per_page: int) -> tuple[int, int]:
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise _ToolInputError("page must be an integer of at least 1.")
    if (
        not isinstance(per_page, int)
        or isinstance(per_page, bool)
        or not 1 <= per_page <= 100
    ):
        raise _ToolInputError("per_page must be an integer between 1 and 100.")
    return page, per_page


def _bounded_limit(limit: int, *, maximum: int) -> int:
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= maximum
    ):
        raise _ToolInputError(f"limit must be an integer between 1 and {maximum}.")
    return limit


def _parse_window_time(value: str | None, *, field: str) -> datetime:
    text = _require_identifier(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _ToolInputError(f"{field} must be an ISO 8601 date-time.") from exc
    if parsed.utcoffset() is None:
        raise _ToolInputError(f"{field} must include a UTC offset.")
    return parsed


def _bounded_time_window(start_time: str, end_time: str) -> tuple[str, str]:
    start = _parse_window_time(start_time, field="start_time")
    end = _parse_window_time(end_time, field="end_time")
    if end <= start:
        raise _ToolInputError("end_time must be later than start_time.")
    if end - start > _OBSERVABILITY_MAX_WINDOW:
        raise _ToolInputError("observability time windows cannot exceed 31 days.")
    return start_time.strip(), end_time.strip()


def _bounded_optional_text(
    value: str | None, *, field: str, maximum: int
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > maximum:
        raise _ToolInputError(f"{field} must be at most {maximum} characters.")
    return text


class _ToolInputError(ValueError):
    """Raised for user-correctable analytics MCP tool input errors."""


def _bounded_identifier_list(value: object, *, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list):
        raise _ToolInputError(f"{field} must be a list.")
    if len(value) > maximum:
        raise _ToolInputError(f"{field} must contain at most {maximum} values.")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise _ToolInputError(f"{field} values must be strings.")
        cleaned.append(_bounded_identifier(item, field=field))
    if len(set(cleaned)) != len(cleaned):
        raise _ToolInputError(f"{field} values must be unique.")
    return cleaned


def _bounded_cohort(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _ToolInputError(f"{field} must be an object.")
    allowed = {
        "start_time",
        "end_time",
        "execution_context",
        "trace_statuses",
        "variant_ids",
        "issue_ids",
        "environment",
        "sample_limit",
    }
    extra = sorted(set(value).difference(allowed))
    if extra:
        raise _ToolInputError(
            f"{field} contains unsupported field(s): {', '.join(extra)}."
        )
    start_time, end_time = _bounded_time_window(
        str(value.get("start_time") or ""), str(value.get("end_time") or "")
    )
    statuses = _bounded_identifier_list(
        value.get("trace_statuses", []), field=f"{field}.trace_statuses", maximum=4
    )
    if set(statuses).difference({"running", "completed", "failed", "rejected"}):
        raise _ToolInputError(f"{field}.trace_statuses contains an unsupported status.")
    sample_limit = value.get("sample_limit", 5000)
    if (
        not isinstance(sample_limit, int)
        or isinstance(sample_limit, bool)
        or not 1 <= sample_limit <= 5000
    ):
        raise _ToolInputError(
            f"{field}.sample_limit must be an integer between 1 and 5000."
        )
    clean: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "trace_statuses": statuses,
        "variant_ids": _bounded_identifier_list(
            value.get("variant_ids", []),
            field=f"{field}.variant_ids",
            maximum=100,
        ),
        "issue_ids": _bounded_identifier_list(
            value.get("issue_ids", []),
            field=f"{field}.issue_ids",
            maximum=100,
        ),
        "environment": _bounded_optional_text(
            cast(str | None, value.get("environment")),
            field=f"{field}.environment",
            maximum=64,
        ),
        "sample_limit": sample_limit,
    }
    context = value.get("execution_context")
    if context is not None:
        if not isinstance(context, dict):
            raise _ToolInputError(f"{field}.execution_context must be an object.")
        allowed_context = {
            "schema_version",
            "agent_id",
            "agent_version",
            "release_id",
            "deployment_id",
            "code_revision",
            "configuration_id",
            "configuration_version",
            "prompt_id",
            "prompt_version",
            "toolset_id",
            "toolset_version",
            "evaluator_id",
            "evaluator_version",
            "dataset_id",
            "dataset_version",
            "experiment_run_id",
            "configuration_run_id",
            "optimization_run_id",
            "intervention_id",
        }
        context_extra = sorted(set(context).difference(allowed_context))
        if context_extra:
            raise _ToolInputError(
                f"{field}.execution_context contains unsupported field(s): "
                + ", ".join(context_extra)
                + "."
            )
        clean_context: dict[str, object] = {"schema_version": "1.0"}
        for key, context_value in context.items():
            if key == "schema_version":
                if context_value != "1.0":
                    raise _ToolInputError(
                        f"{field}.execution_context.schema_version must be '1.0'."
                    )
                continue
            if context_value is None:
                clean_context[key] = None
                continue
            if not isinstance(context_value, str):
                raise _ToolInputError(
                    f"{field}.execution_context.{key} must be an identifier or null."
                )
            clean_context[key] = _bounded_identifier(
                context_value, field=f"{field}.execution_context.{key}"
            )
        clean["execution_context"] = clean_context
    return clean


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

    return {"ok": True, what.replace(" ", "_"): _to_plain_payload(data)}


async def _call_observability_backend(
    coro_factory: Any, *, what: str
) -> dict[str, Any]:
    """Call a reader and reapply the content-free allowlist at the MCP edge."""
    result = await _call_backend(coro_factory, what=what)
    if result.get("ok") is not True:
        return result
    from traigent.cloud.analytics_client import _project_content_free_observability

    key = what.replace(" ", "_")
    return {
        "ok": True,
        key: _project_content_free_observability(result.get(key)),
    }


def _to_plain_payload(data: Any) -> Any:
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return data


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
    """Fetch the backend's privacy-bounded example insights for one run.

    This is a pass-through surface for the backend payload. Responses may
    include coarse "examples to review" rows with opaque refs and enum-only
    review metadata, but never raw signal values or scores.
    """
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


async def analytics_list_experiment_groups_tool(
    project_id: str,
    agent_id: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """List experiment groups/cohorts for a project.

    A cohort is a **source-preserving grouped view** keyed by
    ``(agent_id + dataset_id)``. Rows remain individual source runs (join on
    ``experiment_run_id`` / ``configuration_run_id`` / ``experiment_id``, never
    on config hash). Grouping is NOT a merged analytics run.
    """
    try:
        pid = _require_identifier(project_id, field="project_id")
    except _ToolInputError as exc:
        return _failure(str(exc))

    aid = str(agent_id).strip() if agent_id is not None else None
    did = str(dataset_id).strip() if dataset_id is not None else None
    if aid == "":
        aid = None
    if did == "":
        did = None

    return await _call_backend(
        lambda reader: reader.list_experiment_groups(
            pid,
            agent_id=aid,
            dataset_id=did,
        ),
        what="experiment groups",
    )


async def analytics_get_experiment_group_tool(
    project_id: str,
    group_id: str,
) -> dict[str, Any]:
    """Fetch one experiment group/cohort detail.

    A cohort is a **source-preserving grouped view** keyed by
    ``(agent_id + dataset_id)``. Rows remain individual source runs (join on
    ``experiment_run_id`` / ``configuration_run_id`` / ``experiment_id``, never
    on config hash). Grouping is NOT a merged analytics run.
    """
    try:
        pid = _require_identifier(project_id, field="project_id")
        gid = _require_identifier(group_id, field="group_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.get_experiment_group(gid, pid),
        what="experiment group",
    )


async def analytics_list_experiment_group_configuration_runs_tool(
    project_id: str,
    group_id: str,
) -> dict[str, Any]:
    """List configuration runs for one experiment group/cohort.

    A cohort is a **source-preserving grouped view** keyed by
    ``(agent_id + dataset_id)``. Rows remain individual source runs (join on
    ``experiment_run_id`` / ``configuration_run_id`` / ``experiment_id``, never
    on config hash). Grouping is NOT a merged analytics run.

    This returns the "one aggregated multi-run table": one row per
    configuration-run with ``configuration`` + ``measures`` + source ids.
    """
    try:
        pid = _require_identifier(project_id, field="project_id")
        gid = _require_identifier(group_id, field="group_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_backend(
        lambda reader: reader.list_experiment_group_configuration_runs(gid, pid),
        what="experiment group configuration runs",
    )


async def observability_search_traces_tool(
    project_id: str,
    start_time: str,
    end_time: str,
    page: int = 1,
    per_page: int = 50,
    status: str | None = None,
    environment: str | None = None,
    release: str | None = None,
) -> dict[str, Any]:
    """Search trace summaries through a bounded, content-free projection."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        start, end = _bounded_time_window(start_time, end_time)
        page, per_page = _bounded_page(page, per_page)
        clean_status = _bounded_optional_text(status, field="status", maximum=32)
        if clean_status not in {None, "running", "completed", "failed", "rejected"}:
            raise _ToolInputError("status contains an unsupported trace status.")
        clean_environment = _bounded_optional_text(
            environment, field="environment", maximum=64
        )
        clean_release = _bounded_optional_text(release, field="release", maximum=128)
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.search_observability_traces(
            pid,
            start_time=start,
            end_time=end,
            page=page,
            per_page=per_page,
            status=clean_status,
            environment=clean_environment,
            release=clean_release,
        ),
        what="observability trace search",
    )


async def observability_list_issues_tool(
    project_id: str,
    page: int = 1,
    per_page: int = 50,
    state: str | None = None,
    detector_family: str | None = None,
    severity: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """List durable recurring issues without raw trace content."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        page, per_page = _bounded_page(page, per_page)
        clean_state = _bounded_optional_text(state, field="state", maximum=32)
        clean_detector = _bounded_optional_text(
            detector_family, field="detector_family", maximum=32
        )
        clean_severity = _bounded_optional_text(severity, field="severity", maximum=16)
        clean_search = _bounded_optional_text(search, field="search", maximum=128)
        if clean_state not in {None, "open", "acknowledged", "resolved", "ignored"}:
            raise _ToolInputError("state contains an unsupported issue state.")
        if clean_detector not in {
            None,
            "explicit_error",
            "loop",
            "retry",
            "fallback",
            "dead_end",
        }:
            raise _ToolInputError(
                "detector_family contains an unsupported detector family."
            )
        if clean_severity not in {None, "info", "warning", "error", "critical"}:
            raise _ToolInputError("severity contains an unsupported severity.")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.list_observability_issues(
            pid,
            page=page,
            per_page=per_page,
            state=clean_state,
            detector_family=clean_detector,
            severity=clean_severity,
            search=clean_search,
        ),
        what="observability issues",
    )


async def observability_get_issue_tool(
    project_id: str,
    issue_id: str,
    occurrence_page: int = 1,
    occurrences_per_page: int = 50,
) -> dict[str, Any]:
    """Get one issue and bounded immutable occurrence evidence."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        iid = _bounded_identifier(issue_id, field="issue_id")
        occurrence_page, occurrences_per_page = _bounded_page(
            occurrence_page, occurrences_per_page
        )
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.get_observability_issue(
            pid,
            iid,
            occurrence_page=occurrence_page,
            occurrences_per_page=occurrences_per_page,
        ),
        what="observability issue",
    )


async def observability_get_trace_slice_tool(
    project_id: str,
    trace_id: str,
    cursor: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Get a cursor-bounded content-free trace projection."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        tid = _bounded_identifier(trace_id, field="trace_id")
        clean_cursor = _bounded_optional_text(cursor, field="cursor", maximum=512)
        limit = _bounded_limit(limit, maximum=500)
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.get_observability_trace_slice(
            pid, tid, cursor=clean_cursor, limit=limit
        ),
        what="observability trace slice",
    )


async def observability_get_tool_analysis_tool(
    project_id: str,
    start_time: str,
    end_time: str,
    limit: int = 50,
) -> dict[str, Any]:
    """Get bounded tool execution aggregates without correctness claims."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        start, end = _bounded_time_window(start_time, end_time)
        limit = _bounded_limit(limit, maximum=100)
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.get_observability_tool_analysis(
            pid, start_time=start, end_time=end, limit=limit
        ),
        what="observability tool analysis",
    )


async def observability_get_analysis_insights_tool(
    project_id: str,
    start_time: str,
    end_time: str,
    limit: int = 20,
) -> dict[str, Any]:
    """Get bounded content-free conformance facts and non-causal guidance."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        start, end = _bounded_time_window(start_time, end_time)
        limit = _bounded_limit(limit, maximum=100)
    except _ToolInputError as exc:
        return _failure(str(exc))

    async def _read_insights(reader: Any) -> dict[str, Any]:
        payload = await reader.get_observability_analysis_insights(
            pid, start_time=start, end_time=end, limit=limit
        )
        return _project_observability_analysis_insights(payload, for_mcp=True)

    return await _call_backend(
        _read_insights,
        what="observability analysis insights",
    )


async def observability_compare_cohorts_tool(
    project_id: str,
    reference: dict[str, object],
    comparison: dict[str, object],
    metrics: list[str],
) -> dict[str, Any]:
    """Compare bounded reference and comparison cohorts using aggregate metrics."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        clean_reference = _bounded_cohort(reference, field="reference")
        clean_comparison = _bounded_cohort(comparison, field="comparison")
        clean_metrics = _bounded_identifier_list(metrics, field="metrics", maximum=8)
        if not clean_metrics:
            raise _ToolInputError("metrics must contain at least one value.")
        unsupported = sorted(set(clean_metrics).difference(_OBSERVABILITY_METRICS))
        if unsupported:
            raise _ToolInputError(
                "metrics contains unsupported value(s): " + ", ".join(unsupported)
            )
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.compare_observability_cohorts(
            pid,
            reference=clean_reference,
            comparison=clean_comparison,
            metrics=clean_metrics,
        ),
        what="observability cohort comparison",
    )


async def observability_get_related_changes_tool(
    project_id: str,
    trace_id: str,
) -> dict[str, Any]:
    """Get content-free lineage links related to a trace, without causal claims."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        tid = _bounded_identifier(trace_id, field="trace_id")
    except _ToolInputError as exc:
        return _failure(str(exc))
    return await _call_observability_backend(
        lambda reader: reader.get_observability_related_changes(pid, tid),
        what="observability related changes",
    )


def _change_brief_guidance(
    evidence: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build bounded, deterministic guidance from already-projected facts."""
    hypotheses: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []

    def evidence_count(value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return 0
        return max(0, int(value))

    analysis = evidence.get("trace_analysis")
    if not isinstance(analysis, dict):
        analysis = {}

    analysis_status = analysis.get("analysis_status")
    if analysis_status != "completed":
        hypotheses.append(
            {
                "category": "analysis_completeness",
                "assessment": "unverified",
                "statement": "Trace derivation is incomplete; structural conclusions may be missing.",
                "evidence": {
                    "analysis_status": analysis_status,
                    "failure_code": analysis.get("failure_code"),
                },
            }
        )
        recommendations.append(
            {
                "action": "restore_trace_derivation",
                "rationale": "Complete structural evidence is required before attributing a failure.",
                "verification": "Re-run this brief after analysis_status becomes completed.",
            }
        )

    tool_summaries = analysis.get("tool_summaries")
    if isinstance(tool_summaries, list):
        for summary in tool_summaries[:20]:
            if not isinstance(summary, dict):
                continue
            failures = evidence_count(summary.get("failure_count"))
            retries = evidence_count(summary.get("retry_count"))
            fallbacks = evidence_count(summary.get("fallback_count"))
            if failures + retries + fallbacks == 0:
                continue
            tool_id = summary.get("normalized_tool_id")
            hypotheses.append(
                {
                    "category": "tool_execution_reliability",
                    "assessment": "unverified",
                    "statement": "A tool execution pattern may contribute to this trace's failure or recovery path.",
                    "evidence": {
                        "normalized_tool_id": tool_id,
                        "attempt_count": evidence_count(summary.get("attempt_count")),
                        "failure_count": failures,
                        "retry_count": retries,
                        "fallback_count": fallbacks,
                    },
                }
            )
            recommendations.append(
                {
                    "action": "test_tool_contract_change",
                    "target_id": tool_id,
                    "rationale": "The trace contains measured tool failures, retries, or fallbacks.",
                    "verification": "Compare error_rate, retry_rate, fallback_rate, latency_ms, and cost_usd before and after the change.",
                }
            )

    repeat_groups = analysis.get("repeat_groups")
    if isinstance(repeat_groups, list) and repeat_groups:
        hypotheses.append(
            {
                "category": "loop_control",
                "assessment": "unverified",
                "statement": "Repeated structural subsequences may indicate ineffective retry or planning control.",
                "evidence": {
                    "repeat_group_ids": [
                        group.get("id")
                        for group in repeat_groups[:20]
                        if isinstance(group, dict)
                    ]
                },
            }
        )
        recommendations.append(
            {
                "action": "test_loop_guard",
                "rationale": "The trace contains server-derived repeated subsequences.",
                "verification": "Compare retry_rate, latency_ms, cost_usd, and quality_score on matched cohorts.",
            }
        )

    issue_ids = analysis.get("issue_ids")
    if isinstance(issue_ids, list) and issue_ids:
        recommendations.append(
            {
                "action": "inspect_recurring_issues",
                "target_ids": issue_ids[:20],
                "rationale": "Durable issue occurrences connect this trace to repeated behavior.",
                "verification": "Use observability_get_issue for immutable occurrence evidence before changing code.",
            }
        )

    lineage = evidence.get("related_changes")
    links = lineage.get("links") if isinstance(lineage, dict) else None
    if isinstance(links, list) and links:
        hypotheses.append(
            {
                "category": "change_attribution",
                "assessment": "unverified",
                "statement": "Linked releases, prompts, configurations, or toolsets are candidate variables, not established causes.",
                "evidence": {
                    "candidate_changes": [
                        {
                            key: link.get(key)
                            for key in (
                                "resource_type",
                                "resource_id",
                                "resource_version",
                            )
                        }
                        for link in links[:20]
                        if isinstance(link, dict)
                    ]
                },
            }
        )

    aggregate_insights = evidence.get("analysis_insights")
    if isinstance(aggregate_insights, dict):
        conformance = aggregate_insights.get("conformance")
        if isinstance(conformance, dict) and evidence_count(
            conformance.get("alternate_trace_count")
        ):
            hypotheses.append(
                {
                    "category": "structural_conformance",
                    "assessment": "descriptive_non_causal",
                    "statement": "Alternate structures differ from the dominant observed variant; the baseline is descriptive, not an intended-workflow assertion.",
                    "evidence": {
                        key: conformance.get(key)
                        for key in (
                            "baseline_type",
                            "baseline_variant_id",
                            "sampled_trace_count",
                            "conformance_rate",
                            "alternate_trace_count",
                            "alternate_rate",
                            "alternate_variant_count",
                            "sample_truncated",
                        )
                    },
                }
            )
        aggregate_recommendations = aggregate_insights.get("recommendations")
        if isinstance(aggregate_recommendations, list):
            for recommendation in aggregate_recommendations[:20]:
                if not isinstance(recommendation, dict):
                    continue
                category = recommendation.get("category")
                if not isinstance(category, str):
                    continue
                rationale = _AGGREGATE_RECOMMENDATION_RATIONALES.get(category)
                if rationale is None:
                    continue
                recommendations.append(
                    {
                        "action": "evaluate_aggregate_recommendation",
                        "target_id": recommendation.get("id"),
                        "category": category,
                        "priority": recommendation.get("priority"),
                        "confidence": recommendation.get("confidence"),
                        "subject": recommendation.get("subject"),
                        "rationale": rationale,
                        "evidence": recommendation.get("evidence"),
                        "verification": recommendation.get("measurement"),
                        "caveat": "Observed association only; validate with a bounded before/after cohort before accepting a change.",
                    }
                )

    cohort = evidence.get("cohort_comparison")
    if isinstance(cohort, dict):
        recommendations.append(
            {
                "action": "review_measured_cohort_deltas",
                "rationale": "A bounded before/after comparison is included in this brief.",
                "verification": "Accept a change only when selected quality/reliability metrics improve without violating cost or latency constraints.",
            }
        )

    projection = evidence.get("trace_slice")
    projection_items = projection.get("items") if isinstance(projection, dict) else None
    if not projection_items:
        recommendations.append(
            {
                "action": "improve_trace_instrumentation",
                "rationale": "No content-free observation projection was available for this trace.",
                "verification": "Emit typed observations with terminal status and stable tool_name, then rebuild the brief.",
            }
        )
    if not hypotheses:
        hypotheses.append(
            {
                "category": "insufficient_evidence",
                "assessment": "unverified",
                "statement": "The available content-free evidence does not identify a concrete failure mechanism.",
                "evidence": {},
            }
        )
    return hypotheses[:25], recommendations[:25]


async def observability_build_change_brief_tool(
    project_id: str,
    trace_id: str,
    start_time: str,
    end_time: str,
    reference: dict[str, object] | None = None,
    comparison: dict[str, object] | None = None,
    metrics: list[str] | None = None,
    trace_limit: int = 200,
    tool_limit: int = 50,
    insights_limit: int = 20,
) -> dict[str, Any]:
    """Compose a privacy-bounded, explicitly non-causal change brief."""
    try:
        pid = _bounded_identifier(project_id, field="project_id")
        tid = _bounded_identifier(trace_id, field="trace_id")
        start, end = _bounded_time_window(start_time, end_time)
        trace_limit = _bounded_limit(trace_limit, maximum=500)
        tool_limit = _bounded_limit(tool_limit, maximum=100)
        insights_limit = _bounded_limit(insights_limit, maximum=100)
        if (reference is None) != (comparison is None):
            raise _ToolInputError(
                "reference and comparison must either both be provided or both be omitted."
            )
        clean_reference = (
            _bounded_cohort(reference, field="reference")
            if reference is not None
            else None
        )
        clean_comparison = (
            _bounded_cohort(comparison, field="comparison")
            if comparison is not None
            else None
        )
        clean_metrics = _bounded_identifier_list(
            metrics or [], field="metrics", maximum=8
        )
        if clean_reference is not None and not clean_metrics:
            clean_metrics = [
                "error_rate",
                "retry_rate",
                "fallback_rate",
                "latency_ms",
                "cost_usd",
            ]
        unsupported = sorted(set(clean_metrics).difference(_OBSERVABILITY_METRICS))
        if unsupported:
            raise _ToolInputError(
                "metrics contains unsupported value(s): " + ", ".join(unsupported)
            )
        if clean_reference is None and clean_metrics:
            raise _ToolInputError(
                "metrics require both reference and comparison cohorts."
            )
    except _ToolInputError as exc:
        return _failure(str(exc))

    async def _read_evidence(reader: Any) -> dict[str, Any]:
        evidence = {
            "trace_analysis": await reader.get_observability_trace_analysis(pid, tid),
            "trace_slice": await reader.get_observability_trace_slice(
                pid, tid, cursor=None, limit=trace_limit
            ),
            "related_changes": await reader.get_observability_related_changes(pid, tid),
            "tool_analysis": await reader.get_observability_tool_analysis(
                pid, start_time=start, end_time=end, limit=tool_limit
            ),
            "analysis_insights": _project_observability_analysis_insights(
                await reader.get_observability_analysis_insights(
                    pid, start_time=start, end_time=end, limit=insights_limit
                ),
                for_mcp=True,
            ),
        }
        if clean_reference is not None and clean_comparison is not None:
            evidence["cohort_comparison"] = await reader.compare_observability_cohorts(
                pid,
                reference=clean_reference,
                comparison=clean_comparison,
                metrics=clean_metrics,
            )
        return evidence

    result = await _call_backend(_read_evidence, what="observability change evidence")
    if result.get("ok") is not True:
        return result
    from traigent.cloud.analytics_client import _project_content_free_observability

    raw_evidence = result.get("observability_change_evidence")
    if not isinstance(raw_evidence, dict):
        return _failure(
            "The backend returned malformed observability change evidence.",
            code="malformed_response",
        )
    evidence: dict[str, Any] = {
        key: _project_content_free_observability(raw_evidence.get(key))
        for key in (
            "trace_analysis",
            "trace_slice",
            "related_changes",
            "tool_analysis",
            "cohort_comparison",
        )
        if key in raw_evidence
    }
    if "analysis_insights" in raw_evidence:
        evidence["analysis_insights"] = raw_evidence["analysis_insights"]
    hypotheses, recommendations = _change_brief_guidance(evidence)
    return {
        "ok": True,
        "observability_change_brief": {
            "project_id": pid,
            "trace_id": tid,
            "assessment": "evidence_only_non_causal",
            "evidence": evidence,
            "hypotheses": hypotheses,
            "recommendations": recommendations,
        },
    }
