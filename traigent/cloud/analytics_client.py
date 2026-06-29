"""Read-only client for backend optimization-results analytics.

This module powers the ``client.analytics`` read namespace and the
``traigent-analytics-mcp`` server. It is a **thin authenticated read client**:
the backend owns all analytics intelligence, so the SDK only

* sends the request with the user's existing credentials,
* validates that the response is the platform success envelope, unwraps ``data``
  (and, for the frozen v0 contracts, validates that the required payload keys
  are present), and
* returns the backend payload unchanged for aggregate readers, or a freshly
  projected allowlist for IP-sensitive per-example surfaces.

It deliberately reuses the SDK's existing credential plumbing
(:func:`traigent.cloud.auth._build_api_key_auth_headers`,
:meth:`traigent.config.backend_config.BackendConfig.get_backend_url`) rather
than adding a second API-key auth path. Tenancy is owned by the backend and
derived from the authenticated principal; this client never sends a
caller-supplied ``tenant_id``.

Wired analytics endpoints:

* ``GET /api/v1/analytics/runs/{run_id}/report``
* ``GET /api/v1/analytics/dashboards/optimization-overview``
* ``POST /api/v1/optimization-comparisons``
* ``GET /api/v1/analytics/runs/{run_id}/decision-payload``
* ``GET /api/v1/analytics/runs/{run_id}/pareto``
* ``GET /api/v1/analytics/runs/{run_id}/correlations``
* ``GET /api/v1/analytics/runs/{run_id}/leaderboard``
* ``GET /api/v1/analytics/runs/{run_id}/parameter-insights``
* ``GET /api/v1/analytics/runs/{run_id}/example-insights``
"""

# Traceability: CONC-Layer-Infra CONC-Security FUNC-CLOUD-HYBRID FUNC-ANALYTICS REQ-CLOUD-009

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import quote

from traigent.cloud.url_security import validate_cloud_base_url
from traigent.cloud.user_agent import get_sdk_user_agent
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when httpx is absent
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]


SUPPORTED_DECISION_INTENTS: tuple[str, ...] = ("iterate", "deploy", "debug", "report")
_DEFAULT_DECISION_INTENT = "iterate"

# Frozen v0 contract keys the SDK asserts before returning a payload. The
# backend may add fields; the SDK only fails closed when a required key is
# missing (a malformed/partial response must never look like success).
_DECISION_PAYLOAD_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "project_id",
        "intent",
        "headline",
        "confidence",
        "recommended_action",
        "evidence",
        "drilldowns",
        "warnings",
    }
)
_RUN_PARETO_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "project_id",
        "measures",
        "frontier",
        "dominated",
        "shape",
        "warnings",
    }
)
_RUN_CORRELATIONS_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "method",
        "sample_size",
        "measure_correlations",
        "parameter_correlations",
        "warnings",
    }
)
_RUN_LEADERBOARD_REQUIRED_KEYS = frozenset({"run_id", "ranking_basis", "configs"})
_RUN_PARAMETER_INSIGHTS_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "target_measure",
        "min_trials",
        "drivers",
        "interactions",
        "warnings",
    }
)
_RUN_EXAMPLE_INSIGHTS_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "privacy_mode",
        "summary",
        "example_rows",
        "cohorts",
        "recommendations",
        "redactions",
    }
)
_EXAMPLE_INSIGHT_REF_PATTERN = re.compile(r"^exref_[0-9a-f]{16}$")
_EXAMPLE_INSIGHT_REVIEW_PRIORITIES = frozenset({"critical", "high", "medium", "low"})
_EXAMPLE_INSIGHT_DIFFICULTY_BUCKETS = frozenset({"low", "medium", "high", "unknown"})
_EXAMPLE_INSIGHT_SUSPICIOUS_FLAGS = frozenset(
    {
        "low_agent_strength_correlation",
        "anomalous_low_success",
        "high_response_variance",
        "possible_mislabel",
        "redundant_pattern",
        "low_sample_support",
    }
)
_EXAMPLE_INSIGHT_RECOMMENDED_ACTIONS = frozenset(
    {
        "review_label",
        "clarify_expected_output",
        "increase_repetitions",
        "replace_or_rewrite",
        "keep_as_hard_case",
        "remove_redundant",
        "inspect_evaluator",
    }
)

# Allowlists for nested aggregate fields in get_example_insights() — derived from
# BackendAnalyticsClient's RunExampleInsights Pydantic schema (ExampleInsightsSummary,
# ExampleInsightsCohort, ExampleInsightsRecommendation, ExampleInsightsRedactions in
# shared_infrastructure/schemas/run_analytics.py) and from the extended fixture in
# tests/unit/cloud/test_analytics_client.py (which documents count keys that the backend
# schema will expose in subsequent releases). All fields are coarse-aggregated, backend-
# redacted, and contain no proprietary per-example signals.
_EXAMPLE_INSIGHTS_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "example_count",
        "weak_example_count",
        "unstable_example_count",
        "suspicious_example_count",
        "notable_example_count",
        "stable_example_count",
        "dataset_quality",
    }
)
_EXAMPLE_INSIGHTS_COHORT_KEYS: frozenset[str] = frozenset(
    {"kind", "count", "impact", "safe_example_refs", "recommendation"}
)
_EXAMPLE_INSIGHTS_RECOMMENDATION_KEYS: frozenset[str] = frozenset({"action", "reason"})
_EXAMPLE_INSIGHTS_REDACTIONS_KEYS: frozenset[str] = frozenset(
    {"raw_proprietary_signals_hidden", "raw_prompt_text_hidden_by_default"}
)


class AnalyticsClientError(RuntimeError):
    """Raised when the analytics backend returns a malformed response.

    Transport errors (connection failures, non-2xx status) are surfaced as the
    underlying :class:`httpx.HTTPError` so callers can distinguish "the backend
    said no" from "the backend lied about the shape".
    """


def _require_object(payload: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AnalyticsClientError(
            f"Malformed {what} response: expected a JSON object."
        )
    return cast(dict[str, Any], payload)


def _unwrap_success_data(payload: Any, *, what: str) -> dict[str, Any]:
    envelope = _require_object(payload, what=what)
    if envelope.get("success") is not True or "data" not in envelope:
        raise AnalyticsClientError(
            f"Malformed {what} response: expected Traigent success envelope with data."
        )
    return _require_object(envelope["data"], what=what)


def _require_keys(
    payload: dict[str, Any], required: frozenset[str], *, what: str
) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise AnalyticsClientError(
            f"Malformed {what} response: missing required key(s): {', '.join(missing)}."
        )


def _project_example_insight_row(row: Any, *, index: int) -> dict[str, Any]:
    prefix = f"Malformed example insights response: example_rows[{index}]"
    if not isinstance(row, dict):
        raise AnalyticsClientError(f"{prefix} must be an object.")

    safe_example_ref = row.get("safe_example_ref")
    if not isinstance(safe_example_ref, str) or not _EXAMPLE_INSIGHT_REF_PATTERN.match(
        safe_example_ref
    ):
        raise AnalyticsClientError(
            f"{prefix}.safe_example_ref must match ^exref_[0-9a-f]{{16}}$."
        )

    review_priority = row.get("review_priority")
    if review_priority not in _EXAMPLE_INSIGHT_REVIEW_PRIORITIES:
        raise AnalyticsClientError(
            f"{prefix}.review_priority must be one of: "
            f"{', '.join(sorted(_EXAMPLE_INSIGHT_REVIEW_PRIORITIES))}."
        )

    difficulty_bucket = row.get("difficulty_bucket")
    if difficulty_bucket not in _EXAMPLE_INSIGHT_DIFFICULTY_BUCKETS:
        raise AnalyticsClientError(
            f"{prefix}.difficulty_bucket must be one of: "
            f"{', '.join(sorted(_EXAMPLE_INSIGHT_DIFFICULTY_BUCKETS))}."
        )

    suspicious_flags = row.get("suspicious_flags")
    if not isinstance(suspicious_flags, list):
        raise AnalyticsClientError(f"{prefix}.suspicious_flags must be a list.")
    invalid_flags = [
        flag
        for flag in suspicious_flags
        if flag not in _EXAMPLE_INSIGHT_SUSPICIOUS_FLAGS
    ]
    if invalid_flags:
        raise AnalyticsClientError(
            f"{prefix}.suspicious_flags contains unsupported value(s): "
            f"{', '.join(str(flag) for flag in invalid_flags)}."
        )

    recommended_action = row.get("recommended_action")
    if recommended_action not in _EXAMPLE_INSIGHT_RECOMMENDED_ACTIONS:
        raise AnalyticsClientError(
            f"{prefix}.recommended_action must be one of: "
            f"{', '.join(sorted(_EXAMPLE_INSIGHT_RECOMMENDED_ACTIONS))}."
        )

    return {
        "safe_example_ref": safe_example_ref,
        "review_priority": review_priority,
        "difficulty_bucket": difficulty_bucket,
        "suspicious_flags": list(suspicious_flags),
        "recommended_action": recommended_action,
    }


def _project_aggregate(data: Any, keys: frozenset[str], *, what: str) -> dict[str, Any]:
    """Project a mapping to its known-safe keys, dropping unknown/raw keys.

    Unknown nested keys can carry proprietary signals or raw backend state.
    This function is the nested-aggregate counterpart of the per-row construction
    in :func:`_project_example_insight_row`: both rebuild output by allowlist so
    nothing unknown can pass through regardless of what the backend sends.
    """
    if not isinstance(data, dict):
        raise AnalyticsClientError(
            f"Malformed example insights response: {what} must be an object."
        )
    return {k: v for k, v in data.items() if k in keys}


def _quote_segment(value: str, *, field: str) -> str:
    """URL-encode a path segment, rejecting empty values up front.

    The MCP tool layer requires an explicit ``project_id`` / ``run_id`` (no
    implicit "latest run"), so an empty identifier here is a programming error
    rather than user input — surface it loudly instead of building a request
    against ``.../projects//overview``.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return quote(text, safe="")


def _json_object_query_value(
    value: Mapping[str, object] | str | None, *, field: str
) -> str | None:
    """Serialize a leaderboard JSON-object query value."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-object serializable.") from exc


def _without_none(params: dict[str, str | None]) -> dict[str, str]:
    return {key: value for key, value in params.items() if value is not None}


def normalize_decision_intent(intent: str | None = None) -> str:
    """Return a supported decision-payload intent or raise ``ValueError``."""
    normalized = (intent or "").strip() or _DEFAULT_DECISION_INTENT
    if normalized not in SUPPORTED_DECISION_INTENTS:
        allowed = ", ".join(SUPPORTED_DECISION_INTENTS)
        raise ValueError(f"intent must be one of: {allowed}.")
    return normalized


class BackendAnalyticsClient:
    """Async-first read client for backend optimization-results analytics.

    Thread Safety: safe for concurrent use (``httpx.AsyncClient`` is
    thread-safe).
    """

    def __init__(
        self,
        backend_url: str | None = None,
        api_key: str | None = None,
        jwt_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the analytics read client.

        Args:
            backend_url: Backend origin URL. Defaults to the SDK's resolved
                backend URL (env / stored CLI credentials / cloud default).
            api_key: Explicit API key. When ``None`` the SDK's existing
                credential resolution is used (``TRAIGENT_API_KEY`` /
                stored CLI credentials / dev-mode key).
            jwt_token: Explicit JWT bearer token. If ``api_key`` is also
                provided, ``api_key`` wins to match the JS SDK header builder.
                API-key values are never treated as JWTs based on string shape.
            timeout: Per-request timeout in seconds.

        Raises:
            ImportError: If ``httpx`` is not installed.
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for BackendAnalyticsClient. "
                "Install with: pip install 'traigent[hybrid]'"
            )

        # Import lazily to avoid import cycles through the cloud package.
        from traigent.config.backend_config import (
            BackendConfig,
            get_no_credentials_hint,
        )

        resolved_url = backend_url or BackendConfig.get_backend_url()
        resolved_url = resolved_url.rstrip("/")
        self.backend_url = validate_cloud_base_url(
            resolved_url, purpose="analytics request"
        )
        self.timeout = timeout

        self.api_key = api_key if api_key is not None else None
        self.jwt_token = jwt_token
        if self.api_key is None and self.jwt_token is None:
            self.api_key = self._resolve_api_key()
        if not self.api_key and not self.jwt_token:
            logger.warning(
                "No API key or JWT found for BackendAnalyticsClient. %s",
                get_no_credentials_hint(),
            )

        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _resolve_api_key() -> str | None:
        """Resolve the API key through the SDK's existing credential path."""
        from traigent.cloud.credential_manager import CredentialManager

        # CredentialManager.get_api_key() honors env vars, stored CLI
        # credentials, and (only in explicit dev mode) TRAIGENT_DEV_API_KEY.
        return cast("str | None", CredentialManager.get_api_key())

    def _auth_headers(self) -> dict[str, str]:
        """Build auth headers using the canonical credential helper.

        Reuses :func:`traigent.cloud.auth._build_api_key_auth_headers`, the same
        API-key header builder used by ``AuthManager`` and other analytics
        clients. API keys always route to ``X-API-Key`` and win when both
        credential types are present; JWTs are explicit and route to
        ``Authorization: Bearer``.
        """
        from traigent.cloud.auth import _build_api_key_auth_headers

        api_key_headers: dict[str, str] = _build_api_key_auth_headers(self.api_key)
        if api_key_headers:
            return api_key_headers
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        return {}

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client, failing closed when offline."""
        from traigent.utils.env_config import raise_if_backend_offline

        raise_if_backend_offline("BackendAnalyticsClient request")
        if self._client is None:
            headers = self._auth_headers()
            headers.setdefault("User-Agent", get_sdk_user_agent())
            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> BackendAnalyticsClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @staticmethod
    def _project_headers(project_id: str) -> dict[str, str]:
        return {"X-Project-Id": _require_non_empty(project_id, field="project_id")}

    async def _get_json(
        self,
        path: str,
        *,
        what: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.get(path, headers=headers, params=params)
        response.raise_for_status()
        return _unwrap_success_data(response.json(), what=what)

    # === Read methods (the client.analytics surface) ===

    async def get_run_report(self, project_id: str, run_id: str) -> dict[str, Any]:
        """Return the backend's full analytics report for one run.

        Args:
            project_id: Owning project identifier (explicit; no implicit
                "latest").
            run_id: Experiment-run identifier.

        Returns:
            The backend report payload (returned unchanged).
        """
        path = f"/api/v1/analytics/runs/{_quote_segment(run_id, field='run_id')}/report"
        return await self._get_json(
            path,
            what="run report",
            headers=self._project_headers(project_id),
        )

    async def get_project_overview(self, project_id: str) -> dict[str, Any]:
        """Return the backend's cross-run overview for a project.

        Args:
            project_id: Project identifier (explicit).

        Returns:
            The backend overview payload (returned unchanged).
        """
        path = "/api/v1/analytics/dashboards/optimization-overview"
        return await self._get_json(
            path,
            what="project overview",
            headers=self._project_headers(project_id),
        )

    async def compare_runs(
        self, project_id: str, run_ids: Sequence[str]
    ) -> dict[str, Any]:
        """Compare two or more runs within a project.

        Args:
            project_id: Project identifier (explicit).
            run_ids: Run identifiers to compare (at least two).

        Returns:
            The backend comparison payload (returned unchanged).
        """
        cleaned = [str(run_id).strip() for run_id in run_ids]
        cleaned = [run_id for run_id in cleaned if run_id]
        if len(cleaned) < 2:
            raise ValueError("compare_runs requires at least two run_ids.")

        path = "/api/v1/optimization-comparisons"
        client = self._get_client()
        response = await client.post(
            path,
            json={"run_ids": cleaned},
            headers=self._project_headers(project_id),
        )
        response.raise_for_status()
        return _unwrap_success_data(response.json(), what="run comparison")

    async def get_run_decision_brief(
        self,
        project_id: str,
        run_id: str,
        intent: str = _DEFAULT_DECISION_INTENT,
    ) -> dict[str, Any]:
        """Return the backend's decision brief (decision_payload v0) for a run.

        Endpoint: ``GET /api/v1/analytics/runs/{run_id}/decision-payload``.

        The frozen v0 ``decision_payload`` contract is asserted before the
        payload is returned: a partial/malformed brief must never be presented
        as a confident recommendation.

        Args:
            project_id: Project identifier (explicit). Sent via
                ``X-Project-Id`` so the backend can scope/validate against the
                run's owning project.
            run_id: Experiment-run identifier.
            intent: One of ``iterate`` / ``deploy`` / ``debug`` / ``report``.

        Returns:
            The decision_payload v0 dict (returned unchanged on success).
        """
        normalized_intent = normalize_decision_intent(intent)
        path = (
            "/api/v1/analytics/runs/"
            f"{_quote_segment(run_id, field='run_id')}/decision-payload"
        )
        payload = await self._get_json(
            path,
            what="decision brief",
            headers=self._project_headers(project_id),
            params={"intent": normalized_intent},
        )
        _require_keys(payload, _DECISION_PAYLOAD_REQUIRED_KEYS, what="decision brief")
        return payload

    async def get_single_run_pareto(
        self,
        project_id: str,
        run_id: str,
        *,
        x_measure: str = "cost",
        y_measure: str = "quality",
        request_count: int = 1,
    ) -> dict[str, Any]:
        """Return the backend's run_pareto v0 payload for one run."""
        path = f"/api/v1/analytics/runs/{_quote_segment(run_id, field='run_id')}/pareto"
        payload = await self._get_json(
            path,
            what="single-run Pareto",
            headers=self._project_headers(project_id),
            params={
                "x_measure": str(x_measure),
                "y_measure": str(y_measure),
                "request_count": str(request_count),
            },
        )
        _require_keys(payload, _RUN_PARETO_REQUIRED_KEYS, what="single-run Pareto")
        return payload

    async def get_correlation_matrix(
        self,
        project_id: str,
        run_id: str,
        *,
        method: str = "pearson",
        min_sample: int = 3,
    ) -> dict[str, Any]:
        """Return the backend's run_correlations v0 payload for one run."""
        path = (
            "/api/v1/analytics/runs/"
            f"{_quote_segment(run_id, field='run_id')}/correlations"
        )
        payload = await self._get_json(
            path,
            what="correlation matrix",
            headers=self._project_headers(project_id),
            params={"method": str(method), "min_sample": str(min_sample)},
        )
        _require_keys(
            payload, _RUN_CORRELATIONS_REQUIRED_KEYS, what="correlation matrix"
        )
        return payload

    async def get_run_leaderboard(
        self,
        project_id: str,
        run_id: str,
        *,
        objective: str = "weighted",
        weights: Mapping[str, object] | str | None = None,
        constraints: Mapping[str, object] | str | None = None,
        request_count: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return the backend's run_leaderboard v0 payload for one run."""
        path = (
            "/api/v1/analytics/runs/"
            f"{_quote_segment(run_id, field='run_id')}/leaderboard"
        )
        payload = await self._get_json(
            path,
            what="run leaderboard",
            headers=self._project_headers(project_id),
            params=_without_none(
                {
                    "objective": str(objective),
                    "weights": _json_object_query_value(weights, field="weights"),
                    "constraints": _json_object_query_value(
                        constraints, field="constraints"
                    ),
                    "request_count": str(request_count),
                    "limit": str(limit),
                }
            ),
        )
        _require_keys(payload, _RUN_LEADERBOARD_REQUIRED_KEYS, what="run leaderboard")
        return payload

    async def get_parameter_insights(
        self,
        project_id: str,
        run_id: str,
        *,
        target_measure: str = "quality",
        min_trials: int = 10,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return the backend's run_parameter_insights v0 payload for one run."""
        path = (
            "/api/v1/analytics/runs/"
            f"{_quote_segment(run_id, field='run_id')}/parameter-insights"
        )
        payload = await self._get_json(
            path,
            what="parameter insights",
            headers=self._project_headers(project_id),
            params={
                "target_measure": str(target_measure),
                "min_trials": str(min_trials),
                "top_k": str(top_k),
            },
        )
        _require_keys(
            payload,
            _RUN_PARAMETER_INSIGHTS_REQUIRED_KEYS,
            what="parameter insights",
        )
        return payload

    async def get_example_insights(
        self,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Return allowlisted privacy-bounded example insights for one run.

        The payload is freshly constructed after validating the stable top-level
        contract. ``example_rows`` is the only per-example surface and is rebuilt
        from projected safe fields so raw signals cannot pass through even on a
        backend regression. ``summary``, ``cohorts``, ``recommendations``, and
        ``redactions`` are backend-redacted, templated, canary-enforced safe
        aggregate guidance and are forwarded as-is. This reader deliberately
        allowlists by construction, unlike the other thin pass-through readers in
        this module, because it is the IP-sensitive per-example endpoint.
        """
        path = (
            "/api/v1/analytics/runs/"
            f"{_quote_segment(run_id, field='run_id')}/example-insights"
        )
        payload = await self._get_json(
            path,
            what="example insights",
            headers=self._project_headers(project_id),
        )
        _require_keys(
            payload,
            _RUN_EXAMPLE_INSIGHTS_REQUIRED_KEYS,
            what="example insights",
        )
        rows = payload["example_rows"]
        if not isinstance(rows, list):
            raise AnalyticsClientError(
                "Malformed example insights response: example_rows must be a list."
            )
        if len(rows) > 100:
            raise AnalyticsClientError(
                "Malformed example insights response: example_rows must contain "
                "at most 100 rows."
            )
        cohorts_raw = payload["cohorts"]
        if not isinstance(cohorts_raw, list):
            raise AnalyticsClientError(
                "Malformed example insights response: cohorts must be a list."
            )
        recommendations_raw = payload["recommendations"]
        if not isinstance(recommendations_raw, list):
            raise AnalyticsClientError(
                "Malformed example insights response: recommendations must be a list."
            )

        return {
            "run_id": payload["run_id"],
            "privacy_mode": payload["privacy_mode"],
            "summary": _project_aggregate(
                payload["summary"],
                _EXAMPLE_INSIGHTS_SUMMARY_KEYS,
                what="summary",
            ),
            "example_rows": [
                _project_example_insight_row(row, index=index)
                for index, row in enumerate(rows)
            ],
            "cohorts": [
                _project_aggregate(
                    cohort,
                    _EXAMPLE_INSIGHTS_COHORT_KEYS,
                    what=f"cohorts[{i}]",
                )
                for i, cohort in enumerate(cohorts_raw)
            ],
            "recommendations": [
                _project_aggregate(
                    rec,
                    _EXAMPLE_INSIGHTS_RECOMMENDATION_KEYS,
                    what=f"recommendations[{i}]",
                )
                for i, rec in enumerate(recommendations_raw)
            ],
            "redactions": _project_aggregate(
                payload["redactions"],
                _EXAMPLE_INSIGHTS_REDACTIONS_KEYS,
                what="redactions",
            ),
        }


def _require_non_empty(value: str, *, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text
