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
* ``GET /api/v1/experiment-groups``
* ``GET /api/v1/experiment-groups/{group_id}``
* ``GET /api/v1/experiment-groups/{group_id}/configuration-runs``
* ``GET /api/v1beta/projects/{project_id}/observability/traces`` (safe projection)
* ``GET /api/v1beta/projects/{project_id}/observability/issues[/{issue_id}]``
* ``GET /api/v1beta/projects/{project_id}/observability/variants[/{variant_id}]``
* ``GET /api/v1beta/projects/{project_id}/observability/traces/{trace_id}/analysis``
* ``GET /api/v1beta/projects/{project_id}/observability/traces/{trace_id}/projection``
* ``GET /api/v1beta/projects/{project_id}/observability/traces/{trace_id}/lineage``
* ``GET /api/v1beta/projects/{project_id}/observability/analysis/tools``
* ``POST /api/v1beta/projects/{project_id}/observability/analysis/cohorts/compare``
"""

# Traceability: CONC-Layer-Infra CONC-Security FUNC-CLOUD-HYBRID FUNC-ANALYTICS REQ-CLOUD-009

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

from traigent.cloud.dtos import (
    ExperimentGroupDetailDTO,
    ExperimentGroupsPageDTO,
    GroupedConfigurationRunsPageDTO,
)
from traigent.cloud.url_security import validate_cloud_base_url
from traigent.cloud.user_agent import get_sdk_user_agent
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.observability.analytics_dtos import (
        ObservabilityCohortComparisonDTO,
        ObservabilityIssueDetailDTO,
        ObservabilityIssueListDTO,
        ObservabilityLineageDTO,
        ObservabilityToolAnalysisDTO,
        ObservabilityTraceAnalysisDTO,
        ObservabilityTraceProjectionDTO,
        ObservabilityTraceSearchDTO,
        ObservabilityVariantDetailDTO,
        ObservabilityVariantListDTO,
    )

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
_EXAMPLE_INSIGHT_MAX_SAFE_EXAMPLE_REFS = 50
_EXAMPLE_INSIGHTS_DATASET_QUALITIES = frozenset({"low", "medium", "high"})
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

# Observability MCP responses are reconstructed from this closed key vocabulary.
# The backend schemas are content-free already; this second projection prevents a
# future backend field such as input_data, output_data, metadata, error text, or a
# comment from silently becoming agent-visible through the SDK.
_OBSERVABILITY_CONTENT_FREE_KEYS: frozenset[str] = frozenset(
    {
        "absolute_delta",
        "affected_trace_count",
        "algorithm",
        "analysis_status",
        "agent_id",
        "agent_version",
        "assessment",
        "attempt_count",
        "boundary_trace_ids",
        "comparison",
        "canonical_event_count",
        "collapsed_observation_count",
        "content_included",
        "configuration_id",
        "configuration_run_id",
        "configuration_version",
        "cost_usd",
        "created_at",
        "critical_path",
        "code_revision",
        "completed",
        "depth",
        "derivation",
        "derivation_run_id",
        "derived_at",
        "deriver",
        "deriver_version",
        "detected_at",
        "detector_family",
        "detector_rule_version",
        "deltas",
        "deployment_id",
        "display_label",
        "duration_ms",
        "end_observation_id",
        "end_sequence_index",
        "end_time",
        "ended_at",
        "environment",
        "evaluator_id",
        "evaluator_version",
        "error_category",
        "evidence",
        "evidence_type",
        "execution_context",
        "experiment_run_id",
        "failure_code",
        "failed",
        "failure_count",
        "failure_rate",
        "fallback_count",
        "fallback_rate",
        "fingerprint",
        "fingerprint_spec_version",
        "first_seen_at",
        "generated_at",
        "has_more",
        "has_next",
        "id",
        "input_digest",
        "input_revision",
        "input_tokens",
        "intervention_id",
        "is_boundary",
        "is_critical_path",
        "is_representative",
        "issue",
        "issue_id",
        "issue_ids",
        "items",
        "iteration_count",
        "last_seen_at",
        "links",
        "matched_pair_count",
        "mean",
        "median",
        "metric",
        "metrics",
        "next_cursor",
        "normalized_model_id",
        "normalized_tool_id",
        "observation_count",
        "observation_count_per_iteration",
        "observation_id",
        "observation_ids",
        "occurrence_count",
        "occurrence_page",
        "occurrences",
        "occurrences_per_page",
        "optimization_run_id",
        "output_tokens",
        "p50_latency_ms",
        "p95",
        "p95_latency_ms",
        "page",
        "parent_observation_id",
        "per_page",
        "problem_signature",
        "project_id",
        "prompt_id",
        "prompt_version",
        "projection_mode",
        "rejected",
        "relative_delta",
        "release",
        "relationship",
        "release_id",
        "repeat_count",
        "repeat_group_id",
        "repeat_groups",
        "reopen_count",
        "representative_trace_id",
        "reference",
        "resource_id",
        "resource_type",
        "resource_version",
        "retry_count",
        "retry_rate",
        "root_count",
        "running",
        "sample_count",
        "semantic_kind",
        "schema_version",
        "sequence_fingerprint",
        "sequence_index",
        "severity",
        "signature_spec_version",
        "start_observation_id",
        "start_sequence_index",
        "start_time",
        "started_at",
        "state",
        "state_changed_at",
        "status",
        "status_counts",
        "success_count",
        "superseded_by_issue_id",
        "tool_summaries",
        "toolset_id",
        "toolset_version",
        "total",
        "total_cost_usd",
        "total_input_tokens",
        "total_latency_ms",
        "total_occurrences",
        "total_output_tokens",
        "total_tokens",
        "total_traces",
        "trace_count",
        "trace_id",
        "trace_page",
        "traces",
        "traces_per_page",
        "updated_at",
        "value",
        "variant",
        "variant_id",
        "variant_ids",
        "version",
        "dataset_id",
        "dataset_version",
    }
)

_OBSERVABILITY_MAX_PAGE_SIZE = 100
_OBSERVABILITY_MAX_TRACE_SLICE = 500
_OBSERVABILITY_MAX_WINDOW = timedelta(days=31)
_OBSERVABILITY_ISSUE_LIST_REQUIRED_KEYS = frozenset(
    {"items", "page", "per_page", "total", "generated_at"}
)
_OBSERVABILITY_ISSUE_DETAIL_REQUIRED_KEYS = frozenset(
    {
        "issue",
        "occurrences",
        "occurrence_page",
        "occurrences_per_page",
        "total_occurrences",
        "variant_ids",
        "generated_at",
    }
)
_OBSERVABILITY_VARIANT_LIST_REQUIRED_KEYS = _OBSERVABILITY_ISSUE_LIST_REQUIRED_KEYS
_OBSERVABILITY_VARIANT_DETAIL_REQUIRED_KEYS = frozenset(
    {
        "variant",
        "traces",
        "trace_page",
        "traces_per_page",
        "total_traces",
        "generated_at",
    }
)
_OBSERVABILITY_TRACE_ANALYSIS_REQUIRED_KEYS = frozenset(
    {
        "project_id",
        "trace_id",
        "analysis_status",
        "failure_code",
        "fingerprint",
        "variant_id",
        "critical_path",
        "repeat_groups",
        "tool_summaries",
        "issue_ids",
        "derivation",
    }
)
_OBSERVABILITY_TRACE_PROJECTION_REQUIRED_KEYS = frozenset(
    {
        "project_id",
        "trace_id",
        "projection_mode",
        "content_included",
        "items",
        "next_cursor",
        "has_more",
        "generated_at",
    }
)
_OBSERVABILITY_TOOL_ANALYSIS_REQUIRED_KEYS = frozenset(
    {"project_id", "start_time", "end_time", "items", "generated_at"}
)
_OBSERVABILITY_COHORT_COMPARISON_REQUIRED_KEYS = frozenset(
    {
        "project_id",
        "reference",
        "comparison",
        "matched_pair_count",
        "deltas",
        "generated_at",
    }
)
_OBSERVABILITY_LINEAGE_REQUIRED_KEYS = frozenset(
    {"project_id", "trace_id", "execution_context", "links", "generated_at"}
)
_OBSERVABILITY_COHORT_METRICS = frozenset(
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
    nothing unknown can pass through regardless of what the backend sends. Note
    that this allowlists by *key* only — callers whose values carry a specific
    contract (e.g. ``summary.dataset_quality``, ``cohorts[].safe_example_refs``)
    must additionally validate those values themselves (see
    :func:`_project_example_insights_summary` and
    :func:`_project_example_insights_cohort`); an unvalidated value allowlisted
    by key alone could still smuggle raw/proprietary content through.
    """
    if not isinstance(data, dict):
        raise AnalyticsClientError(
            f"Malformed example insights response: {what} must be an object."
        )
    return {k: v for k, v in data.items() if k in keys}


def _assert_allowed_string(value: Any, allowed: frozenset[str], *, what: str) -> str:
    """Validate that ``value`` is one of a fixed set of enum strings.

    Mirrors the JS SDK's ``assertAllowedString`` (traigent-js analytics-client.ts)
    used to close the same value-validation gap for cross-SDK parity.
    """
    if not isinstance(value, str) or value not in allowed:
        raise AnalyticsClientError(
            f"Malformed example insights response: {what} must be one of: "
            f"{', '.join(sorted(allowed))}."
        )
    return value


def _assert_safe_example_refs(value: Any, *, what: str) -> list[str]:
    """Validate a cohort's ``safe_example_refs`` list.

    Each element must be an opaque ``exref_`` reference matching the backend
    contract (``run_analytics_service._safe_example_refs``), and the list must
    not exceed the backend's cap. Mirrors the JS SDK's
    ``assertSafeExampleRefs`` for cross-SDK parity (Traigent#1662).
    """
    if not isinstance(value, list):
        raise AnalyticsClientError(
            f"Malformed example insights response: {what} must be a list."
        )
    if len(value) > _EXAMPLE_INSIGHT_MAX_SAFE_EXAMPLE_REFS:
        raise AnalyticsClientError(
            f"Malformed example insights response: {what} must contain at most "
            f"{_EXAMPLE_INSIGHT_MAX_SAFE_EXAMPLE_REFS} refs."
        )
    for ref in value:
        if not isinstance(ref, str) or not _EXAMPLE_INSIGHT_REF_PATTERN.match(ref):
            raise AnalyticsClientError(
                f"Malformed example insights response: {what} must only contain "
                "refs matching ^exref_[0-9a-f]{16}$."
            )
    return list(value)


def _project_example_insights_summary(data: Any) -> dict[str, Any]:
    """Project + value-validate the ``summary`` aggregate.

    ``dataset_quality`` is required and must be one of the backend's frozen
    ``Literal["low", "medium", "high"]`` enum values; a backend regression
    emitting a raw/unexpected value must fail closed rather than pass through.
    """
    summary = _project_aggregate(data, _EXAMPLE_INSIGHTS_SUMMARY_KEYS, what="summary")
    summary["dataset_quality"] = _assert_allowed_string(
        summary.get("dataset_quality"),
        _EXAMPLE_INSIGHTS_DATASET_QUALITIES,
        what="summary.dataset_quality",
    )
    return summary


def _project_example_insights_cohort(cohort: Any, *, index: int) -> dict[str, Any]:
    """Project + value-validate one ``cohorts[]`` aggregate.

    ``safe_example_refs`` is optional per cohort, but when present each element
    must be a well-formed opaque ref within the backend's cap — a backend
    regression placing raw prompt text/emails in this list must not pass
    through the SDK.
    """
    projected = _project_aggregate(
        cohort, _EXAMPLE_INSIGHTS_COHORT_KEYS, what=f"cohorts[{index}]"
    )
    if "safe_example_refs" in projected:
        projected["safe_example_refs"] = _assert_safe_example_refs(
            projected["safe_example_refs"],
            what=f"cohorts[{index}].safe_example_refs",
        )
    return projected


def _project_content_free_observability(data: Any) -> Any:
    """Recursively rebuild an observability payload from aggregate-safe keys."""
    if isinstance(data, dict):
        return {
            key: _project_content_free_observability(value)
            for key, value in data.items()
            if key in _OBSERVABILITY_CONTENT_FREE_KEYS
        }
    if isinstance(data, list):
        return [_project_content_free_observability(value) for value in data]
    return data


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

    async def _post_json(
        self,
        path: str,
        *,
        what: str,
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.post(path, json=json_body)
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
        aggregate guidance, projected by key allowlist. ``summary.dataset_quality``
        and ``cohorts[].safe_example_refs`` additionally get per-value shape
        validation (enum / opaque-ref-pattern + length cap): a backend regression
        placing raw content in those two value families must fail closed rather
        than pass through (Traigent#1662). This reader deliberately allowlists by
        construction, unlike the other thin pass-through readers in this module,
        because it is the IP-sensitive per-example endpoint.
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
            "summary": _project_example_insights_summary(payload["summary"]),
            "example_rows": [
                _project_example_insight_row(row, index=index)
                for index, row in enumerate(rows)
            ],
            "cohorts": [
                _project_example_insights_cohort(cohort, index=i)
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

    # === Content-free observability analysis readers ===

    async def search_observability_traces(
        self,
        project_id: str,
        *,
        start_time: str,
        end_time: str,
        page: int = 1,
        per_page: int = 50,
        status: str | None = None,
        environment: str | None = None,
        release: str | None = None,
    ) -> ObservabilityTraceSearchDTO:
        """Return a bounded, content-free projection of matching trace summaries."""
        _validate_observability_page(page, per_page)
        start_time, end_time = _validate_observability_time_window(
            start_time, end_time
        )
        payload = await self._get_json(
            f"{_observability_path(project_id)}/traces",
            what="observability trace search",
            params=_without_none(
                {
                    "start_time_from": start_time,
                    "start_time_to": end_time,
                    "page": str(page),
                    "per_page": str(per_page),
                    "status": status,
                    "environment": environment,
                    "release": release,
                }
            ),
        )
        items = _require_bounded_list(
            payload.get("items"),
            what="observability trace search items",
            maximum=per_page,
        )
        pagination = payload.get("pagination")
        if pagination is not None and not isinstance(pagination, dict):
            raise AnalyticsClientError(
                "Malformed observability trace search response: pagination must be an object."
            )
        page_info = cast(dict[str, Any], pagination or {})
        safe_items = []
        for item in items:
            if not isinstance(item, dict):
                raise AnalyticsClientError(
                    "Malformed observability trace search response: each item must be an object."
                )
            safe_items.append(
                {
                    key: item[key]
                    for key in (
                        "id",
                        "status",
                        "environment",
                        "release",
                        "started_at",
                        "ended_at",
                        "observation_count",
                        "total_input_tokens",
                        "total_output_tokens",
                        "total_tokens",
                        "total_cost_usd",
                        "total_latency_ms",
                    )
                    if key in item
                }
            )
        return {
            "items": safe_items,
            "page": int(page_info.get("page", payload.get("page", page))),
            "per_page": int(
                page_info.get("per_page", payload.get("per_page", per_page))
            ),
            "total": int(page_info.get("total", payload.get("total", 0))),
            "has_more": bool(
                page_info.get("has_next", payload.get("has_more", False))
            ),
        }

    async def list_observability_issues(
        self,
        project_id: str,
        *,
        page: int = 1,
        per_page: int = 50,
        state: str | None = None,
        detector_family: str | None = None,
        severity: str | None = None,
        search: str | None = None,
    ) -> ObservabilityIssueListDTO:
        """Return a bounded page of durable, content-free issue summaries."""
        _validate_observability_page(page, per_page)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/issues",
            what="observability issues",
            params=_without_none(
                {
                    "page": str(page),
                    "per_page": str(per_page),
                    "state": state,
                    "detector_family": detector_family,
                    "severity": severity,
                    "search": search,
                }
            ),
        )
        _require_keys(
            payload,
            _OBSERVABILITY_ISSUE_LIST_REQUIRED_KEYS,
            what="observability issues",
        )
        _require_bounded_list(
            payload.get("items"), what="observability issue items", maximum=per_page
        )
        return cast(
            "ObservabilityIssueListDTO",
            _project_content_free_observability(payload),
        )

    async def get_observability_issue(
        self,
        project_id: str,
        issue_id: str,
        *,
        occurrence_page: int = 1,
        occurrences_per_page: int = 50,
    ) -> ObservabilityIssueDetailDTO:
        """Return one issue and bounded immutable occurrence evidence."""
        _validate_observability_page(occurrence_page, occurrences_per_page)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/issues/"
            f"{_quote_segment(issue_id, field='issue_id')}",
            what="observability issue",
            params={
                "occurrence_page": str(occurrence_page),
                "occurrences_per_page": str(occurrences_per_page),
            },
        )
        _require_keys(
            payload,
            _OBSERVABILITY_ISSUE_DETAIL_REQUIRED_KEYS,
            what="observability issue",
        )
        _require_bounded_list(
            payload.get("occurrences"),
            what="observability issue occurrences",
            maximum=occurrences_per_page,
        )
        return cast(
            "ObservabilityIssueDetailDTO",
            _project_content_free_observability(payload),
        )

    async def list_observability_variants(
        self,
        project_id: str,
        *,
        page: int = 1,
        per_page: int = 50,
        search: str | None = None,
    ) -> ObservabilityVariantListDTO:
        """Return exact structural trace variants for a project."""
        _validate_observability_page(page, per_page)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/variants",
            what="observability variants",
            params=_without_none(
                {"page": str(page), "per_page": str(per_page), "search": search}
            ),
        )
        _require_keys(
            payload,
            _OBSERVABILITY_VARIANT_LIST_REQUIRED_KEYS,
            what="observability variants",
        )
        _require_bounded_list(
            payload.get("items"), what="observability variant items", maximum=per_page
        )
        return cast(
            "ObservabilityVariantListDTO",
            _project_content_free_observability(payload),
        )

    async def get_observability_variant(
        self,
        project_id: str,
        variant_id: str,
        *,
        trace_page: int = 1,
        traces_per_page: int = 50,
    ) -> ObservabilityVariantDetailDTO:
        """Return one exact structural variant and bounded trace references."""
        _validate_observability_page(trace_page, traces_per_page)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/variants/"
            f"{_quote_segment(variant_id, field='variant_id')}",
            what="observability variant",
            params={
                "trace_page": str(trace_page),
                "traces_per_page": str(traces_per_page),
            },
        )
        _require_keys(
            payload,
            _OBSERVABILITY_VARIANT_DETAIL_REQUIRED_KEYS,
            what="observability variant",
        )
        _require_bounded_list(
            payload.get("traces"),
            what="observability variant traces",
            maximum=traces_per_page,
        )
        return cast(
            "ObservabilityVariantDetailDTO",
            _project_content_free_observability(payload),
        )

    async def get_observability_trace_analysis(
        self, project_id: str, trace_id: str
    ) -> ObservabilityTraceAnalysisDTO:
        """Return server-derived content-free structural analysis for one trace."""
        payload = await self._get_json(
            f"{_observability_path(project_id)}/traces/"
            f"{_quote_segment(trace_id, field='trace_id')}/analysis",
            what="observability trace analysis",
        )
        _require_keys(
            payload,
            _OBSERVABILITY_TRACE_ANALYSIS_REQUIRED_KEYS,
            what="observability trace analysis",
        )
        return cast(
            "ObservabilityTraceAnalysisDTO",
            _project_content_free_observability(payload),
        )

    async def get_observability_trace_slice(
        self,
        project_id: str,
        trace_id: str,
        *,
        cursor: str | None = None,
        limit: int = 200,
    ) -> ObservabilityTraceProjectionDTO:
        """Return a bounded content-free observation projection for one trace."""
        if not 1 <= limit <= _OBSERVABILITY_MAX_TRACE_SLICE:
            raise ValueError(
                f"limit must be between 1 and {_OBSERVABILITY_MAX_TRACE_SLICE}."
            )
        payload = await self._get_json(
            f"{_observability_path(project_id)}/traces/"
            f"{_quote_segment(trace_id, field='trace_id')}/projection",
            what="observability trace slice",
            params=_without_none({"cursor": cursor, "limit": str(limit)}),
        )
        _require_keys(
            payload,
            _OBSERVABILITY_TRACE_PROJECTION_REQUIRED_KEYS,
            what="observability trace slice",
        )
        items = _require_bounded_list(
            payload.get("items"), what="observability trace slice items", maximum=limit
        )
        if payload.get("projection_mode") != "content_free" or payload.get(
            "content_included"
        ) is not False:
            raise AnalyticsClientError(
                "Malformed observability trace slice response: content-free markers are required."
            )
        if any(not isinstance(item, dict) for item in items):
            raise AnalyticsClientError(
                "Malformed observability trace slice response: each item must be an object."
            )
        return cast(
            "ObservabilityTraceProjectionDTO",
            _project_content_free_observability(payload),
        )

    async def get_observability_tool_analysis(
        self,
        project_id: str,
        *,
        start_time: str,
        end_time: str,
        limit: int = 50,
    ) -> ObservabilityToolAnalysisDTO:
        """Return bounded tool execution aggregates without semantic claims."""
        if not 1 <= limit <= _OBSERVABILITY_MAX_PAGE_SIZE:
            raise ValueError(
                f"limit must be between 1 and {_OBSERVABILITY_MAX_PAGE_SIZE}."
            )
        start_time, end_time = _validate_observability_time_window(
            start_time, end_time
        )
        payload = await self._get_json(
            f"{_observability_path(project_id)}/analysis/tools",
            what="observability tool analysis",
            params={
                "start_time": start_time,
                "end_time": end_time,
                "limit": str(limit),
            },
        )
        _require_keys(
            payload,
            _OBSERVABILITY_TOOL_ANALYSIS_REQUIRED_KEYS,
            what="observability tool analysis",
        )
        _require_bounded_list(
            payload.get("items"), what="observability tool analysis items", maximum=limit
        )
        return cast(
            "ObservabilityToolAnalysisDTO",
            _project_content_free_observability(payload),
        )

    async def compare_observability_cohorts(
        self,
        project_id: str,
        *,
        reference: Mapping[str, object],
        comparison: Mapping[str, object],
        metrics: Sequence[str],
    ) -> ObservabilityCohortComparisonDTO:
        """Compare two validated, bounded trace cohorts using aggregate metrics."""
        clean_reference = _validate_observability_cohort(reference, field="reference")
        clean_comparison = _validate_observability_cohort(
            comparison, field="comparison"
        )
        clean_metrics = [str(metric).strip() for metric in metrics]
        if not 1 <= len(clean_metrics) <= 8 or len(set(clean_metrics)) != len(
            clean_metrics
        ):
            raise ValueError("metrics must contain 1 to 8 unique values.")
        unsupported = sorted(
            set(clean_metrics).difference(_OBSERVABILITY_COHORT_METRICS)
        )
        if unsupported:
            raise ValueError(
                "metrics contains unsupported value(s): " + ", ".join(unsupported)
            )
        payload = await self._post_json(
            f"{_observability_path(project_id)}/analysis/cohorts/compare",
            what="observability cohort comparison",
            json_body={
                "reference": clean_reference,
                "comparison": clean_comparison,
                "metrics": clean_metrics,
            },
        )
        _require_keys(
            payload,
            _OBSERVABILITY_COHORT_COMPARISON_REQUIRED_KEYS,
            what="observability cohort comparison",
        )
        return cast(
            "ObservabilityCohortComparisonDTO",
            _project_content_free_observability(payload),
        )

    async def get_observability_related_changes(
        self, project_id: str, trace_id: str
    ) -> ObservabilityLineageDTO:
        """Return content-free lineage links; links are not causal attribution."""
        payload = await self._get_json(
            f"{_observability_path(project_id)}/traces/"
            f"{_quote_segment(trace_id, field='trace_id')}/lineage",
            what="observability related changes",
        )
        _require_keys(
            payload,
            _OBSERVABILITY_LINEAGE_REQUIRED_KEYS,
            what="observability related changes",
        )
        return cast(
            "ObservabilityLineageDTO",
            _project_content_free_observability(payload),
        )

    async def list_experiment_groups(
        self,
        project_id: str,
        *,
        agent_id: str | None = None,
        dataset_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> ExperimentGroupsPageDTO:
        """Return experiment groups/cohorts visible to the authenticated user."""
        payload = await self._get_json(
            "/api/v1/experiment-groups",
            what="experiment groups",
            headers=self._project_headers(project_id),
            params=_without_none(
                {
                    "agent_id": str(agent_id) if agent_id is not None else None,
                    "dataset_id": str(dataset_id) if dataset_id is not None else None,
                    "page": str(page),
                    "page_size": str(page_size),
                }
            ),
        )
        return ExperimentGroupsPageDTO.from_dict(payload)

    async def get_experiment_group(
        self,
        group_id: str,
        project_id: str,
    ) -> ExperimentGroupDetailDTO:
        """Return one experiment group/cohort detail payload."""
        path = f"/api/v1/experiment-groups/{_quote_segment(group_id, field='group_id')}"
        payload = await self._get_json(
            path,
            what="experiment group",
            headers=self._project_headers(project_id),
        )
        return ExperimentGroupDetailDTO.from_dict(payload)

    async def list_experiment_group_configuration_runs(
        self,
        group_id: str,
        project_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> GroupedConfigurationRunsPageDTO:
        """Return source-preserving configuration rows for one group/cohort."""
        path = (
            "/api/v1/experiment-groups/"
            f"{_quote_segment(group_id, field='group_id')}/configuration-runs"
        )
        payload = await self._get_json(
            path,
            what="experiment group configuration runs",
            headers=self._project_headers(project_id),
            params={"page": str(page), "page_size": str(page_size)},
        )
        return GroupedConfigurationRunsPageDTO.from_dict(payload)


def _require_non_empty(value: str, *, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def _observability_path(project_id: str) -> str:
    return (
        "/api/v1beta/projects/"
        f"{_quote_segment(project_id, field='project_id')}/observability"
    )


def _validate_observability_page(page: int, per_page: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1.")
    if not 1 <= per_page <= _OBSERVABILITY_MAX_PAGE_SIZE:
        raise ValueError(
            f"per_page must be between 1 and {_OBSERVABILITY_MAX_PAGE_SIZE}."
        )


def _require_bounded_list(value: Any, *, what: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise AnalyticsClientError(f"Malformed {what} response: expected a list.")
    if len(value) > maximum:
        raise AnalyticsClientError(
            f"Malformed {what} response: expected at most {maximum} items."
        )
    return value


def _parse_observability_time(value: str, *, field: str) -> datetime:
    text = _require_non_empty(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 date-time.") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset.")
    return parsed


def _validate_observability_time_window(
    start_time: str, end_time: str
) -> tuple[str, str]:
    start = _parse_observability_time(start_time, field="start_time")
    end = _parse_observability_time(end_time, field="end_time")
    if end <= start:
        raise ValueError("end_time must be later than start_time.")
    if end - start > _OBSERVABILITY_MAX_WINDOW:
        raise ValueError("observability time windows cannot exceed 31 days.")
    return start_time.strip(), end_time.strip()


def _validate_identifier_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list.")
    result = [str(item).strip() for item in value]
    if len(result) > 100 or len(set(result)) != len(result):
        raise ValueError(f"{field} must contain at most 100 unique identifiers.")
    if any(not item or len(item) > 128 for item in result):
        raise ValueError(f"{field} contains an invalid identifier.")
    return result


def _validate_observability_cohort(
    cohort: Mapping[str, object], *, field: str
) -> dict[str, object]:
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
    extra = sorted(set(cohort).difference(allowed))
    if extra:
        raise ValueError(f"{field} contains unsupported field(s): {', '.join(extra)}.")
    start_time, end_time = _validate_observability_time_window(
        str(cohort.get("start_time") or ""), str(cohort.get("end_time") or "")
    )
    statuses = _validate_identifier_list(
        cohort.get("trace_statuses", []), field=f"{field}.trace_statuses"
    )
    allowed_statuses = {"running", "completed", "failed", "rejected"}
    if len(statuses) > 4 or set(statuses).difference(allowed_statuses):
        raise ValueError(
            f"{field}.trace_statuses must use running/completed/failed/rejected."
        )
    sample_limit = cohort.get("sample_limit", 5000)
    if not isinstance(sample_limit, int) or isinstance(sample_limit, bool):
        raise ValueError(f"{field}.sample_limit must be an integer.")
    if not 1 <= sample_limit <= 5000:
        raise ValueError(f"{field}.sample_limit must be between 1 and 5000.")
    environment = cohort.get("environment")
    if environment is not None and (
        not isinstance(environment, str)
        or not environment.strip()
        or len(environment.strip()) > 64
    ):
        raise ValueError(f"{field}.environment must be null or 1 to 64 characters.")

    result: dict[str, object] = {
        "start_time": start_time,
        "end_time": end_time,
        "trace_statuses": statuses,
        "variant_ids": _validate_identifier_list(
            cohort.get("variant_ids", []), field=f"{field}.variant_ids"
        ),
        "issue_ids": _validate_identifier_list(
            cohort.get("issue_ids", []), field=f"{field}.issue_ids"
        ),
        "environment": environment.strip() if isinstance(environment, str) else None,
        "sample_limit": sample_limit,
    }
    execution_context = cohort.get("execution_context")
    if execution_context is not None:
        if not isinstance(execution_context, Mapping):
            raise ValueError(f"{field}.execution_context must be an object.")
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
        context_extra = sorted(set(execution_context).difference(allowed_context))
        if context_extra:
            raise ValueError(
                f"{field}.execution_context contains unsupported field(s): "
                + ", ".join(context_extra)
                + "."
            )
        clean_context: dict[str, object] = {"schema_version": "1.0"}
        for key, value in execution_context.items():
            if key == "schema_version":
                if value != "1.0":
                    raise ValueError(
                        f"{field}.execution_context.schema_version must be '1.0'."
                    )
                continue
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value.strip()) > 128
            ):
                raise ValueError(
                    f"{field}.execution_context.{key} must be null or a bounded identifier."
                )
            clean_context[key] = value.strip() if isinstance(value, str) else None
        result["execution_context"] = clean_context
    return result
