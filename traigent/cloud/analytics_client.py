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
* ``GET /api/v1beta/projects/{project_id}/observability/analysis/insights``
* ``POST /api/v1beta/projects/{project_id}/observability/analysis/cohorts/compare``
"""

# Traceability: CONC-Layer-Infra CONC-Security FUNC-CLOUD-HYBRID FUNC-ANALYTICS REQ-CLOUD-009

from __future__ import annotations

import json
import math
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
        ObservabilityAnalysisInsightsDTO,
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

# MCP-visible observability responses use endpoint-specific, nesting-specific
# reconstruction below. A shared recursive key vocabulary is intentionally not
# used: an otherwise safe key at the wrong nesting must fail closed.
_OBSERVABILITY_TRACE_STATUSES = frozenset(
    {"running", "completed", "failed", "rejected"}
)
_OBSERVABILITY_TRACE_STATUS_ALIASES = {
    "success": "completed",
    "error": "failed",
    "timeout": "failed",
    "cancelled": "rejected",
}
_OBSERVABILITY_OBSERVATION_KINDS = frozenset(
    {
        "span",
        "generation",
        "event",
        "tool_call",
        "agent",
        "chain",
        "tool",
        "retriever",
        "evaluator",
        "embedding",
        "guardrail",
    }
)
_OBSERVABILITY_EXECUTION_CONTEXT_KEYS = frozenset(
    {
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
)
_OBSERVABILITY_HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")

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
_OBSERVABILITY_ANALYSIS_INSIGHTS_REQUIRED_KEYS = frozenset(
    {
        "project_id",
        "start_time",
        "end_time",
        "content_included",
        "conformance",
        "recommendations",
        "limitations",
        "generated_at",
    }
)
_OBSERVABILITY_CONFORMANCE_REQUIRED_KEYS = frozenset(
    {
        "baseline_type",
        "baseline_variant_id",
        "analyzed_trace_count",
        "sampled_trace_count",
        "total_trace_count",
        "analysis_coverage",
        "sample_coverage",
        "conforming_trace_count",
        "conformance_rate",
        "alternate_trace_count",
        "alternate_rate",
        "alternate_variant_count",
        "deviations",
        "sample_truncated",
        "interpretation",
    }
)
_OBSERVABILITY_DEVIATION_REQUIRED_KEYS = frozenset(
    {
        "variant_id",
        "trace_count",
        "failed_trace_count",
        "representative_trace_id",
        "evidence_trace_ids",
        "share",
    }
)
_OBSERVABILITY_RECOMMENDATION_REQUIRED_KEYS = frozenset(
    {
        "id",
        "category",
        "priority",
        "confidence",
        "subject",
        "evidence",
        "suggested_action",
        "measurement",
    }
)
_OBSERVABILITY_MEASUREMENT_REQUIRED_KEYS = frozenset(
    {"comparison", "metrics", "intervention_context_key"}
)
_OBSERVABILITY_RECOMMENDATION_EVIDENCE_KEYS = frozenset(
    {
        "normalized_tool_id",
        "trace_count",
        "attempt_count",
        "issue_ids",
        "failure_count",
        "failure_rate",
        "retry_count",
        "retry_rate",
        "fallback_count",
        "fallback_rate",
        "issue_id",
        "detector_family",
        "occurrence_count",
        "affected_trace_count",
        "baseline_variant_id",
        "analyzed_trace_count",
        "sampled_trace_count",
        "alternate_trace_count",
        "alternate_rate",
        "alternate_variant_count",
    }
)
_OBSERVABILITY_RECOMMENDATION_CATEGORIES = frozenset(
    {
        "tool_reliability",
        "retry_policy",
        "tool_routing",
        "recurring_issue",
        "behavioral_variation",
    }
)
_OBSERVABILITY_RECOMMENDATION_PRIORITIES = frozenset({"high", "medium", "low"})
_OBSERVABILITY_DETECTOR_FAMILIES = frozenset(
    {"explicit_error", "loop", "retry", "fallback", "dead_end"}
)
_OBSERVABILITY_SAFE_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,127}$"
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


def _exact_observability_object(
    payload: Any,
    *,
    keys: frozenset[str],
    what: str,
    require_all: bool = True,
) -> dict[str, Any]:
    source = _require_object(payload, what=what)
    _require_exact_keys(source, keys, what=what, require_all=require_all)
    return source


def _project_observability_trace_search(payload: Any) -> dict[str, Any]:
    """Rebuild the safe subset of the ordinary trace-summary response."""
    source = _require_object(payload, what="observability trace search")
    allowed_top = frozenset(
        {"items", "pagination", "page", "per_page", "total", "has_more"}
    )
    _require_exact_keys(
        source, allowed_top, what="observability trace search", require_all=False
    )
    items = _require_bounded_list(
        source.get("items"), what="observability trace search items", maximum=100
    )
    safe_item_keys = frozenset(
        {
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
        }
    )
    known_content_keys = frozenset(
        {
            "tenant_id",
            "project_id",
            "session_id",
            "name",
            "user_id",
            "custom_trace_id",
            "tags",
            "metadata",
            "execution_context",
            "input_data",
            "output_data",
            "is_bookmarked",
            "bookmarked_at",
            "bookmarked_by",
            "is_published",
            "published_at",
            "published_by",
            "otel_trace_id",
            "otel_parent_span_id",
            "root_observation_count",
            "created_at",
            "updated_at",
            "sum_observation_latency_ms",
            "wall_clock_latency_ms",
            "user_display_name",
            "bookmarked_by_display_name",
            "published_by_display_name",
        }
    )
    projected_items: list[dict[str, Any]] = []
    for index, value in enumerate(items):
        item = _require_object(value, what=f"observability trace search items[{index}]")
        unsupported = sorted(set(item).difference(safe_item_keys | known_content_keys))
        if unsupported:
            raise AnalyticsClientError(
                "Malformed observability trace search response: an unsupported key "
                f"is present in items[{index}]."
            )
        if "id" not in item:
            raise AnalyticsClientError(
                f"Malformed observability trace search items[{index}] response: "
                "missing required key(s): id."
            )
        _require_safe_observability_identifier(item["id"], what=f"items[{index}].id")
        if "status" in item:
            raw_status = item["status"]
            item_status = (
                _OBSERVABILITY_TRACE_STATUS_ALIASES.get(raw_status, raw_status)
                if isinstance(raw_status, str)
                else raw_status
            )
            _require_observability_enum(
                item_status,
                _OBSERVABILITY_TRACE_STATUSES,
                what=f"items[{index}].status",
            )
            item = {**item, "status": item_status}
        if "environment" in item:
            _require_safe_observability_identifier(
                item["environment"],
                what=f"items[{index}].environment",
                nullable=True,
                maximum=64,
            )
        if "release" in item:
            _require_safe_observability_identifier(
                item["release"], what=f"items[{index}].release", nullable=True
            )
        for field in ("started_at", "ended_at"):
            if field in item and item[field] is not None:
                _require_observability_datetime(
                    item[field], what=f"items[{index}].{field}"
                )
        for field in (
            "observation_count",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_latency_ms",
        ):
            if field in item:
                _require_observability_integer(
                    item[field], what=f"items[{index}].{field}"
                )
        if "total_cost_usd" in item:
            _require_non_negative_observability_number(
                item["total_cost_usd"], what=f"items[{index}].total_cost_usd"
            )
        projected_items.append(
            {key: item[key] for key in safe_item_keys if key in item}
        )

    pagination = source.get("pagination")
    if pagination is not None:
        page_info = _exact_observability_object(
            pagination,
            keys=frozenset(
                {"page", "per_page", "total", "total_pages", "has_next", "has_prev"}
            ),
            what="observability trace search pagination",
            require_all=False,
        )
    else:
        page_info = source
    page = page_info.get("page", 1)
    per_page = page_info.get("per_page", len(items) or 1)
    total = page_info.get("total", len(items))
    has_more = page_info.get("has_next", source.get("has_more", False))
    _require_observability_integer(page, what="trace search page", minimum=1)
    _require_observability_integer(per_page, what="trace search per_page", minimum=1)
    _require_observability_integer(total, what="trace search total")
    if "total_pages" in page_info:
        _require_observability_integer(
            page_info["total_pages"], what="trace search total_pages", minimum=1
        )
    for field in ("has_next", "has_prev"):
        if field in page_info and not isinstance(page_info[field], bool):
            raise AnalyticsClientError(
                f"Malformed observability trace search response: {field} must be a boolean."
            )
    if not isinstance(has_more, bool):
        raise AnalyticsClientError(
            "Malformed observability trace search response: has_more must be a boolean."
        )
    return {
        "items": projected_items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "has_more": has_more,
    }


def _project_observability_issue(payload: Any, *, what: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "id",
            "project_id",
            "detector_family",
            "problem_signature",
            "signature_spec_version",
            "state",
            "severity",
            "occurrence_count",
            "affected_trace_count",
            "reopen_count",
            "first_seen_at",
            "last_seen_at",
            "created_at",
            "updated_at",
            "state_changed_at",
            "superseded_by_issue_id",
            "version",
        }
    )
    source = _exact_observability_object(payload, keys=keys, what=what)
    for field in ("id", "project_id"):
        _require_safe_observability_identifier(source[field], what=f"{what}.{field}")
    _require_safe_observability_identifier(
        source["signature_spec_version"],
        what=f"{what}.signature_spec_version",
        maximum=64,
    )
    _require_observability_enum(
        source["detector_family"],
        _OBSERVABILITY_DETECTOR_FAMILIES,
        what=f"{what}.detector_family",
    )
    _require_observability_hash(
        source["problem_signature"], what=f"{what}.problem_signature"
    )
    _require_observability_enum(
        source["state"],
        frozenset({"open", "acknowledged", "resolved", "ignored"}),
        what=f"{what}.state",
    )
    _require_observability_enum(
        source["severity"],
        frozenset({"info", "warning", "error", "critical"}),
        what=f"{what}.severity",
    )
    for field, minimum in (
        ("occurrence_count", 1),
        ("affected_trace_count", 1),
        ("reopen_count", 0),
        ("version", 1),
    ):
        _require_observability_integer(
            source[field], what=f"{what}.{field}", minimum=minimum
        )
    for field in (
        "first_seen_at",
        "last_seen_at",
        "created_at",
        "updated_at",
        "state_changed_at",
    ):
        _require_observability_datetime(source[field], what=f"{what}.{field}")
    _require_safe_observability_identifier(
        source["superseded_by_issue_id"],
        what=f"{what}.superseded_by_issue_id",
        nullable=True,
    )
    return {key: source[key] for key in keys}


def _project_observability_fingerprint(payload: Any, *, what: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "value",
            "algorithm",
            "fingerprint_spec_version",
            "canonical_event_count",
            "root_count",
        }
    )
    source = _exact_observability_object(payload, keys=keys, what=what)
    _require_observability_hash(source["value"], what=f"{what}.value")
    if source["algorithm"] != "sha256":
        raise AnalyticsClientError(
            f"Malformed {what} response: algorithm must be sha256."
        )
    _require_safe_observability_identifier(
        source["fingerprint_spec_version"],
        what=f"{what}.fingerprint_spec_version",
        maximum=64,
    )
    for field in ("canonical_event_count", "root_count"):
        _require_observability_integer(source[field], what=f"{what}.{field}")
    return {key: source[key] for key in keys}


def _project_observability_derivation(payload: Any, *, what: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "derivation_run_id",
            "deriver",
            "deriver_version",
            "input_revision",
            "input_digest",
            "derived_at",
        }
    )
    source = _exact_observability_object(payload, keys=keys, what=what)
    _require_safe_observability_identifier(
        source["derivation_run_id"], what=f"{what}.derivation_run_id"
    )
    _require_safe_observability_identifier(
        source["deriver_version"], what=f"{what}.deriver_version", maximum=64
    )
    _require_observability_enum(
        source["deriver"],
        frozenset(
            {
                "structural_analysis",
                "variant_assignment",
                "issue_detection",
                "lineage_projection",
                "cohort_analysis",
            }
        ),
        what=f"{what}.deriver",
    )
    _require_observability_integer(
        source["input_revision"], what=f"{what}.input_revision", minimum=1
    )
    _require_observability_hash(source["input_digest"], what=f"{what}.input_digest")
    _require_observability_datetime(source["derived_at"], what=f"{what}.derived_at")
    return {key: source[key] for key in keys}


def _project_observability_issue_evidence(payload: Any, *, what: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "evidence_type",
            "trace_id",
            "observation_id",
            "start_observation_id",
            "end_observation_id",
            "start_sequence_index",
            "end_sequence_index",
            "repeat_count",
            "error_category",
        }
    )
    source = _exact_observability_object(payload, keys=keys, what=what)
    evidence_type = _require_observability_enum(
        source["evidence_type"],
        frozenset(
            {
                "explicit_error",
                "repeated_subsequence",
                "retry_sequence",
                "fallback_transition",
                "terminal_dead_end",
            }
        ),
        what=f"{what}.evidence_type",
    )
    _require_safe_observability_identifier(source["trace_id"], what=f"{what}.trace_id")
    for field in ("observation_id", "start_observation_id", "end_observation_id"):
        _require_safe_observability_identifier(
            source[field], what=f"{what}.{field}", nullable=True
        )
    for field in ("start_sequence_index", "end_sequence_index"):
        if source[field] is not None:
            _require_observability_integer(
                source[field], what=f"{what}.{field}", minimum=0, maximum=1_000_000
            )
    if source["repeat_count"] is not None:
        _require_observability_integer(
            source["repeat_count"],
            what=f"{what}.repeat_count",
            minimum=2,
            maximum=1_000_000,
        )
    if source["error_category"] is not None:
        _require_observability_enum(
            source["error_category"],
            frozenset(
                {
                    "authentication",
                    "authorization",
                    "timeout",
                    "rate_limit",
                    "validation",
                    "model",
                    "tool",
                    "dependency",
                    "cancelled",
                    "unknown",
                }
            ),
            what=f"{what}.error_category",
        )
    if (
        evidence_type in {"explicit_error", "terminal_dead_end"}
        and source["observation_id"] is None
    ):
        raise AnalyticsClientError(
            f"Malformed {what} response: observation_id is required for {evidence_type}."
        )
    if evidence_type in {
        "repeated_subsequence",
        "retry_sequence",
        "fallback_transition",
    }:
        required_coordinates = (
            "start_observation_id",
            "end_observation_id",
            "start_sequence_index",
            "end_sequence_index",
        )
        if any(source[field] is None for field in required_coordinates):
            raise AnalyticsClientError(
                f"Malformed {what} response: structural evidence coordinates are required."
            )
    if evidence_type == "repeated_subsequence" and source["repeat_count"] is None:
        raise AnalyticsClientError(
            f"Malformed {what} response: repeat_count is required for repeated_subsequence."
        )
    return {key: source[key] for key in keys}


def _project_observability_issue_occurrence(
    payload: Any, *, index: int
) -> dict[str, Any]:
    what = f"observability issue occurrences[{index}]"
    keys = frozenset(
        {
            "id",
            "issue_id",
            "project_id",
            "trace_id",
            "variant_id",
            "detector_rule_version",
            "detected_at",
            "fingerprint",
            "derivation",
            "evidence",
        }
    )
    source = _exact_observability_object(payload, keys=keys, what=what)
    for field in ("id", "issue_id", "project_id", "trace_id"):
        _require_safe_observability_identifier(source[field], what=f"{what}.{field}")
    _require_safe_observability_identifier(
        source["detector_rule_version"],
        what=f"{what}.detector_rule_version",
        maximum=64,
    )
    _require_safe_observability_identifier(
        source["variant_id"], what=f"{what}.variant_id", nullable=True
    )
    _require_observability_datetime(source["detected_at"], what=f"{what}.detected_at")
    evidence = _require_bounded_list(
        source["evidence"], what=f"{what}.evidence", maximum=16
    )
    if not evidence:
        raise AnalyticsClientError(
            f"Malformed {what} response: evidence must not be empty."
        )
    return {
        "id": source["id"],
        "issue_id": source["issue_id"],
        "project_id": source["project_id"],
        "trace_id": source["trace_id"],
        "variant_id": source["variant_id"],
        "detector_rule_version": source["detector_rule_version"],
        "detected_at": source["detected_at"],
        "fingerprint": _project_observability_fingerprint(
            source["fingerprint"], what=f"{what}.fingerprint"
        ),
        "derivation": _project_observability_derivation(
            source["derivation"], what=f"{what}.derivation"
        ),
        "evidence": [
            _project_observability_issue_evidence(
                value, what=f"{what}.evidence[{evidence_index}]"
            )
            for evidence_index, value in enumerate(evidence)
        ],
    }


def _project_observability_issue_list(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_ISSUE_LIST_REQUIRED_KEYS,
        what="observability issues",
    )
    items = _require_bounded_list(
        source["items"], what="observability issue items", maximum=100
    )
    page = _require_observability_integer(source["page"], what="issues.page", minimum=1)
    per_page = _require_observability_integer(
        source["per_page"], what="issues.per_page", minimum=1, maximum=100
    )
    total = _require_observability_integer(source["total"], what="issues.total")
    _require_observability_datetime(source["generated_at"], what="issues.generated_at")
    return {
        "items": [
            _project_observability_issue(
                value, what=f"observability issue items[{index}]"
            )
            for index, value in enumerate(items)
        ],
        "page": page,
        "per_page": per_page,
        "total": total,
        "generated_at": source["generated_at"],
    }


def _project_observability_issue_detail(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_ISSUE_DETAIL_REQUIRED_KEYS,
        what="observability issue",
    )
    occurrences = _require_bounded_list(
        source["occurrences"], what="observability issue occurrences", maximum=100
    )
    occurrence_page = _require_observability_integer(
        source["occurrence_page"], what="issue.occurrence_page", minimum=1
    )
    occurrences_per_page = _require_observability_integer(
        source["occurrences_per_page"],
        what="issue.occurrences_per_page",
        minimum=1,
        maximum=100,
    )
    total_occurrences = _require_observability_integer(
        source["total_occurrences"], what="issue.total_occurrences"
    )
    variant_ids = _require_bounded_list(
        source["variant_ids"], what="issue.variant_ids", maximum=100
    )
    _require_unique_observability_identifiers(variant_ids, what="issue.variant_ids")
    _require_observability_datetime(source["generated_at"], what="issue.generated_at")
    return {
        "issue": _project_observability_issue(
            source["issue"], what="observability issue.issue"
        ),
        "occurrences": [
            _project_observability_issue_occurrence(value, index=index)
            for index, value in enumerate(occurrences)
        ],
        "occurrence_page": occurrence_page,
        "occurrences_per_page": occurrences_per_page,
        "total_occurrences": total_occurrences,
        "variant_ids": list(variant_ids),
        "generated_at": source["generated_at"],
    }


def _project_observability_variant(payload: Any, *, what: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "id",
            "project_id",
            "display_label",
            "fingerprint",
            "trace_count",
            "first_seen_at",
            "last_seen_at",
            "representative_trace_id",
            "boundary_trace_ids",
            "status_counts",
            "derivation",
        }
    )
    source = _exact_observability_object(payload, keys=keys, what=what)
    for field in ("id", "project_id", "representative_trace_id"):
        _require_safe_observability_identifier(source[field], what=f"{what}.{field}")
    if not isinstance(source["display_label"], str) or not re.fullmatch(
        r"Variant [A-F0-9]{8}", source["display_label"]
    ):
        raise AnalyticsClientError(
            f"Malformed {what} response: display_label is invalid."
        )
    _require_observability_integer(
        source["trace_count"], what=f"{what}.trace_count", minimum=1
    )
    for field in ("first_seen_at", "last_seen_at"):
        _require_observability_datetime(source[field], what=f"{what}.{field}")
    boundary_ids = _require_bounded_list(
        source["boundary_trace_ids"], what=f"{what}.boundary_trace_ids", maximum=4
    )
    _require_unique_observability_identifiers(
        boundary_ids, what=f"{what}.boundary_trace_ids"
    )
    status_counts = _exact_observability_object(
        source["status_counts"],
        keys=_OBSERVABILITY_TRACE_STATUSES,
        what=f"{what}.status_counts",
    )
    for field in _OBSERVABILITY_TRACE_STATUSES:
        _require_observability_integer(
            status_counts[field], what=f"{what}.status_counts.{field}"
        )
    return {
        "id": source["id"],
        "project_id": source["project_id"],
        "display_label": source["display_label"],
        "fingerprint": _project_observability_fingerprint(
            source["fingerprint"], what=f"{what}.fingerprint"
        ),
        "trace_count": source["trace_count"],
        "first_seen_at": source["first_seen_at"],
        "last_seen_at": source["last_seen_at"],
        "representative_trace_id": source["representative_trace_id"],
        "boundary_trace_ids": list(boundary_ids),
        "status_counts": {
            key: status_counts[key] for key in _OBSERVABILITY_TRACE_STATUSES
        },
        "derivation": _project_observability_derivation(
            source["derivation"], what=f"{what}.derivation"
        ),
    }


def _project_observability_variant_list(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_VARIANT_LIST_REQUIRED_KEYS,
        what="observability variants",
    )
    items = _require_bounded_list(
        source["items"], what="observability variant items", maximum=100
    )
    page = _require_observability_integer(
        source["page"], what="variants.page", minimum=1
    )
    per_page = _require_observability_integer(
        source["per_page"], what="variants.per_page", minimum=1, maximum=100
    )
    total = _require_observability_integer(source["total"], what="variants.total")
    _require_observability_datetime(
        source["generated_at"], what="variants.generated_at"
    )
    return {
        "items": [
            _project_observability_variant(value, what=f"variants.items[{index}]")
            for index, value in enumerate(items)
        ],
        "page": page,
        "per_page": per_page,
        "total": total,
        "generated_at": source["generated_at"],
    }


def _project_observability_variant_detail(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_VARIANT_DETAIL_REQUIRED_KEYS,
        what="observability variant",
    )
    traces = _require_bounded_list(
        source["traces"], what="observability variant traces", maximum=100
    )
    trace_keys = frozenset(
        {"trace_id", "status", "started_at", "is_representative", "is_boundary"}
    )
    projected_traces: list[dict[str, Any]] = []
    for index, value in enumerate(traces):
        what = f"variant.traces[{index}]"
        trace = _exact_observability_object(value, keys=trace_keys, what=what)
        _require_safe_observability_identifier(
            trace["trace_id"], what=f"{what}.trace_id"
        )
        _require_observability_enum(
            trace["status"], _OBSERVABILITY_TRACE_STATUSES, what=f"{what}.status"
        )
        if trace["started_at"] is not None:
            _require_observability_datetime(
                trace["started_at"], what=f"{what}.started_at"
            )
        for field in ("is_representative", "is_boundary"):
            if not isinstance(trace[field], bool):
                raise AnalyticsClientError(
                    f"Malformed {what} response: {field} must be a boolean."
                )
        projected_traces.append({key: trace[key] for key in trace_keys})
    trace_page = _require_observability_integer(
        source["trace_page"], what="variant.trace_page", minimum=1
    )
    traces_per_page = _require_observability_integer(
        source["traces_per_page"],
        what="variant.traces_per_page",
        minimum=1,
        maximum=100,
    )
    total_traces = _require_observability_integer(
        source["total_traces"], what="variant.total_traces"
    )
    _require_observability_datetime(source["generated_at"], what="variant.generated_at")
    return {
        "variant": _project_observability_variant(
            source["variant"], what="observability variant.variant"
        ),
        "traces": projected_traces,
        "trace_page": trace_page,
        "traces_per_page": traces_per_page,
        "total_traces": total_traces,
        "generated_at": source["generated_at"],
    }


def _project_observability_trace_analysis(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_TRACE_ANALYSIS_REQUIRED_KEYS,
        what="observability trace analysis",
    )
    for field in ("project_id", "trace_id"):
        _require_safe_observability_identifier(
            source[field], what=f"trace_analysis.{field}"
        )
    _require_observability_enum(
        source["analysis_status"],
        frozenset({"pending", "running", "completed", "failed", "stale"}),
        what="trace_analysis.analysis_status",
    )
    if source["failure_code"] is not None:
        _require_observability_enum(
            source["failure_code"],
            frozenset(
                {
                    "trace_too_large",
                    "invalid_structure",
                    "derivation_timeout",
                    "internal_error",
                }
            ),
            what="trace_analysis.failure_code",
        )
    _require_safe_observability_identifier(
        source["variant_id"], what="trace_analysis.variant_id", nullable=True
    )
    critical_path = _exact_observability_object(
        source["critical_path"],
        keys=frozenset({"observation_ids", "duration_ms", "observation_count"}),
        what="observability trace analysis critical_path",
    )
    observation_ids = _require_bounded_list(
        critical_path["observation_ids"],
        what="trace_analysis.critical_path.observation_ids",
        maximum=2000,
    )
    _require_unique_observability_identifiers(
        observation_ids, what="trace_analysis.critical_path.observation_ids"
    )
    _require_observability_integer(
        critical_path["duration_ms"], what="trace_analysis.critical_path.duration_ms"
    )
    _require_observability_integer(
        critical_path["observation_count"],
        what="trace_analysis.critical_path.observation_count",
        maximum=2000,
    )
    repeat_groups = _require_bounded_list(
        source["repeat_groups"], what="trace_analysis.repeat_groups", maximum=500
    )
    repeat_keys = frozenset(
        {
            "id",
            "parent_observation_id",
            "start_sequence_index",
            "end_sequence_index",
            "sequence_fingerprint",
            "iteration_count",
            "observation_count_per_iteration",
            "collapsed_observation_count",
        }
    )
    projected_repeat_groups: list[dict[str, Any]] = []
    for index, value in enumerate(repeat_groups):
        what = f"trace_analysis.repeat_groups[{index}]"
        group = _exact_observability_object(value, keys=repeat_keys, what=what)
        _require_safe_observability_identifier(group["id"], what=f"{what}.id")
        _require_safe_observability_identifier(
            group["parent_observation_id"],
            what=f"{what}.parent_observation_id",
            nullable=True,
        )
        for field in ("start_sequence_index", "end_sequence_index"):
            _require_observability_integer(
                group[field], what=f"{what}.{field}", maximum=1_000_000
            )
        _require_observability_hash(
            group["sequence_fingerprint"], what=f"{what}.sequence_fingerprint"
        )
        _require_observability_integer(
            group["iteration_count"],
            what=f"{what}.iteration_count",
            minimum=2,
            maximum=1_000_000,
        )
        _require_observability_integer(
            group["observation_count_per_iteration"],
            what=f"{what}.observation_count_per_iteration",
            minimum=1,
            maximum=2000,
        )
        _require_observability_integer(
            group["collapsed_observation_count"],
            what=f"{what}.collapsed_observation_count",
            minimum=2,
            maximum=1_000_000,
        )
        projected_repeat_groups.append({key: group[key] for key in repeat_keys})

    tool_summaries = _require_bounded_list(
        source["tool_summaries"], what="trace_analysis.tool_summaries", maximum=256
    )
    tool_keys = frozenset(
        {
            "normalized_tool_id",
            "attempt_count",
            "success_count",
            "failure_count",
            "retry_count",
            "fallback_count",
            "total_latency_ms",
            "total_cost_usd",
        }
    )
    projected_tools: list[dict[str, Any]] = []
    for index, value in enumerate(tool_summaries):
        what = f"trace_analysis.tool_summaries[{index}]"
        tool = _exact_observability_object(value, keys=tool_keys, what=what)
        _require_safe_observability_identifier(
            tool["normalized_tool_id"], what=f"{what}.normalized_tool_id"
        )
        _require_observability_integer(
            tool["attempt_count"], what=f"{what}.attempt_count", minimum=1
        )
        for field in (
            "success_count",
            "failure_count",
            "retry_count",
            "fallback_count",
            "total_latency_ms",
        ):
            _require_observability_integer(tool[field], what=f"{what}.{field}")
        _require_non_negative_observability_number(
            tool["total_cost_usd"], what=f"{what}.total_cost_usd"
        )
        projected_tools.append({key: tool[key] for key in tool_keys})

    issue_ids = _require_bounded_list(
        source["issue_ids"], what="trace_analysis.issue_ids", maximum=100
    )
    _require_unique_observability_identifiers(
        issue_ids, what="trace_analysis.issue_ids"
    )
    fingerprint = (
        None
        if source["fingerprint"] is None
        else _project_observability_fingerprint(
            source["fingerprint"], what="trace_analysis.fingerprint"
        )
    )
    derivation = (
        None
        if source["derivation"] is None
        else _project_observability_derivation(
            source["derivation"], what="trace_analysis.derivation"
        )
    )
    return {
        "project_id": source["project_id"],
        "trace_id": source["trace_id"],
        "analysis_status": source["analysis_status"],
        "failure_code": source["failure_code"],
        "fingerprint": fingerprint,
        "variant_id": source["variant_id"],
        "critical_path": {
            "observation_ids": list(observation_ids),
            "duration_ms": critical_path["duration_ms"],
            "observation_count": critical_path["observation_count"],
        },
        "repeat_groups": projected_repeat_groups,
        "tool_summaries": projected_tools,
        "issue_ids": list(issue_ids),
        "derivation": derivation,
    }


def _project_observability_trace_slice(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_TRACE_PROJECTION_REQUIRED_KEYS,
        what="observability trace slice",
    )
    for field in ("project_id", "trace_id"):
        _require_safe_observability_identifier(
            source[field], what=f"trace_slice.{field}"
        )
    if (
        source["projection_mode"] != "content_free"
        or source["content_included"] is not False
    ):
        raise AnalyticsClientError(
            "Malformed observability trace slice response: content-free markers are required."
        )
    items = _require_bounded_list(
        source["items"], what="observability trace slice items", maximum=500
    )
    item_keys = frozenset(
        {
            "observation_id",
            "parent_observation_id",
            "semantic_kind",
            "status",
            "sequence_index",
            "depth",
            "duration_ms",
            "normalized_tool_id",
            "normalized_model_id",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "is_critical_path",
            "repeat_group_id",
        }
    )
    projected_items: list[dict[str, Any]] = []
    for index, value in enumerate(items):
        what = f"trace_slice.items[{index}]"
        item = _exact_observability_object(value, keys=item_keys, what=what)
        _require_safe_observability_identifier(
            item["observation_id"], what=f"{what}.observation_id"
        )
        for field in (
            "parent_observation_id",
            "normalized_tool_id",
            "normalized_model_id",
            "repeat_group_id",
        ):
            _require_safe_observability_identifier(
                item[field], what=f"{what}.{field}", nullable=True
            )
        _require_observability_enum(
            item["semantic_kind"],
            _OBSERVABILITY_OBSERVATION_KINDS,
            what=f"{what}.semantic_kind",
        )
        _require_observability_enum(
            item["status"], _OBSERVABILITY_TRACE_STATUSES, what=f"{what}.status"
        )
        _require_observability_integer(
            item["sequence_index"], what=f"{what}.sequence_index", maximum=1_000_000
        )
        _require_observability_integer(
            item["depth"], what=f"{what}.depth", maximum=2000
        )
        if item["duration_ms"] is not None:
            _require_observability_integer(
                item["duration_ms"], what=f"{what}.duration_ms"
            )
        for field in ("input_tokens", "output_tokens"):
            _require_observability_integer(item[field], what=f"{what}.{field}")
        _require_non_negative_observability_number(
            item["cost_usd"], what=f"{what}.cost_usd"
        )
        if not isinstance(item["is_critical_path"], bool):
            raise AnalyticsClientError(
                f"Malformed {what} response: is_critical_path must be a boolean."
            )
        projected_items.append({key: item[key] for key in item_keys})
    next_cursor = source["next_cursor"]
    if next_cursor is not None:
        if (
            not isinstance(next_cursor, str)
            or not next_cursor
            or len(next_cursor) > 512
        ):
            raise AnalyticsClientError(
                "Malformed observability trace slice response: next_cursor must be null or a non-empty string of at most 512 characters."
            )
    if not isinstance(source["has_more"], bool):
        raise AnalyticsClientError(
            "Malformed observability trace slice response: has_more must be a boolean."
        )
    _require_observability_datetime(
        source["generated_at"], what="trace_slice.generated_at"
    )
    return {
        "project_id": source["project_id"],
        "trace_id": source["trace_id"],
        "projection_mode": "content_free",
        "content_included": False,
        "items": projected_items,
        "next_cursor": next_cursor,
        "has_more": source["has_more"],
        "generated_at": source["generated_at"],
    }


def _project_observability_tool_analysis(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_TOOL_ANALYSIS_REQUIRED_KEYS,
        what="observability tool analysis",
    )
    _require_safe_observability_identifier(
        source["project_id"], what="tool_analysis.project_id"
    )
    for field in ("start_time", "end_time", "generated_at"):
        _require_observability_datetime(source[field], what=f"tool_analysis.{field}")
    items = _require_bounded_list(
        source["items"], what="observability tool analysis items", maximum=100
    )
    item_keys = frozenset(
        {
            "normalized_tool_id",
            "trace_count",
            "attempt_count",
            "success_count",
            "failure_count",
            "retry_count",
            "fallback_count",
            "failure_rate",
            "retry_rate",
            "fallback_rate",
            "p50_latency_ms",
            "p95_latency_ms",
            "total_cost_usd",
            "issue_ids",
        }
    )
    projected_items: list[dict[str, Any]] = []
    for index, value in enumerate(items):
        what = f"tool_analysis.items[{index}]"
        item = _exact_observability_object(value, keys=item_keys, what=what)
        _require_safe_observability_identifier(
            item["normalized_tool_id"], what=f"{what}.normalized_tool_id"
        )
        for field in ("trace_count", "attempt_count"):
            _require_observability_integer(
                item[field], what=f"{what}.{field}", minimum=1
            )
        for field in (
            "success_count",
            "failure_count",
            "retry_count",
            "fallback_count",
        ):
            _require_observability_integer(item[field], what=f"{what}.{field}")
        for field in ("failure_rate", "retry_rate", "fallback_rate"):
            _require_observability_number(
                item[field], what=f"{what}.{field}", minimum=0, maximum=1
            )
        for field in ("p50_latency_ms", "p95_latency_ms", "total_cost_usd"):
            _require_non_negative_observability_number(
                item[field], what=f"{what}.{field}"
            )
        issue_ids = _require_bounded_list(
            item["issue_ids"], what=f"{what}.issue_ids", maximum=100
        )
        _require_unique_observability_identifiers(issue_ids, what=f"{what}.issue_ids")
        projected_items.append(
            {
                **{key: item[key] for key in item_keys if key != "issue_ids"},
                "issue_ids": list(issue_ids),
            }
        )
    return {
        "project_id": source["project_id"],
        "start_time": source["start_time"],
        "end_time": source["end_time"],
        "items": projected_items,
        "generated_at": source["generated_at"],
    }


def _project_observability_cohort_summary(payload: Any, *, what: str) -> dict[str, Any]:
    source = _exact_observability_object(
        payload, keys=frozenset({"trace_count", "metrics"}), what=what
    )
    trace_count = _require_observability_integer(
        source["trace_count"], what=f"{what}.trace_count"
    )
    metrics = _require_bounded_list(
        source["metrics"], what=f"{what}.metrics", maximum=8
    )
    metric_keys = frozenset({"metric", "sample_count", "mean", "median", "p95"})
    projected_metrics: list[dict[str, Any]] = []
    seen_metrics: set[str] = set()
    for index, value in enumerate(metrics):
        metric_what = f"{what}.metrics[{index}]"
        metric = _exact_observability_object(value, keys=metric_keys, what=metric_what)
        name = _require_observability_enum(
            metric["metric"],
            _OBSERVABILITY_COHORT_METRICS,
            what=f"{metric_what}.metric",
        )
        if name in seen_metrics:
            raise AnalyticsClientError(
                f"Malformed {what} response: metrics must be unique."
            )
        seen_metrics.add(name)
        _require_observability_integer(
            metric["sample_count"], what=f"{metric_what}.sample_count"
        )
        for field in ("mean", "median", "p95"):
            if metric[field] is not None:
                _require_finite_observability_number(
                    metric[field], what=f"{metric_what}.{field}"
                )
        projected_metrics.append({key: metric[key] for key in metric_keys})
    return {"trace_count": trace_count, "metrics": projected_metrics}


def _project_observability_cohort_comparison(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_COHORT_COMPARISON_REQUIRED_KEYS,
        what="observability cohort comparison",
    )
    _require_safe_observability_identifier(
        source["project_id"], what="cohort_comparison.project_id"
    )
    matched_pair_count = _require_observability_integer(
        source["matched_pair_count"], what="cohort_comparison.matched_pair_count"
    )
    deltas = _require_bounded_list(
        source["deltas"], what="cohort_comparison.deltas", maximum=8
    )
    delta_keys = frozenset({"metric", "absolute_delta", "relative_delta", "assessment"})
    projected_deltas: list[dict[str, Any]] = []
    seen_metrics: set[str] = set()
    for index, value in enumerate(deltas):
        what = f"cohort_comparison.deltas[{index}]"
        delta = _exact_observability_object(value, keys=delta_keys, what=what)
        metric = _require_observability_enum(
            delta["metric"], _OBSERVABILITY_COHORT_METRICS, what=f"{what}.metric"
        )
        if metric in seen_metrics:
            raise AnalyticsClientError(
                "Malformed observability cohort comparison response: delta metrics must be unique."
            )
        seen_metrics.add(metric)
        for field in ("absolute_delta", "relative_delta"):
            if delta[field] is not None:
                _require_finite_observability_number(
                    delta[field], what=f"{what}.{field}"
                )
        _require_observability_enum(
            delta["assessment"],
            frozenset({"improved", "worse", "unchanged", "inconclusive"}),
            what=f"{what}.assessment",
        )
        projected_deltas.append({key: delta[key] for key in delta_keys})
    _require_observability_datetime(
        source["generated_at"], what="cohort_comparison.generated_at"
    )
    return {
        "project_id": source["project_id"],
        "reference": _project_observability_cohort_summary(
            source["reference"], what="cohort_comparison.reference"
        ),
        "comparison": _project_observability_cohort_summary(
            source["comparison"], what="cohort_comparison.comparison"
        ),
        "matched_pair_count": matched_pair_count,
        "deltas": projected_deltas,
        "generated_at": source["generated_at"],
    }


def _project_observability_execution_context(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_EXECUTION_CONTEXT_KEYS,
        what="observability lineage execution_context",
        require_all=False,
    )
    if source.get("schema_version") != "1.0":
        raise AnalyticsClientError(
            "Malformed observability lineage response: execution_context.schema_version must be 1.0."
        )
    for field in _OBSERVABILITY_EXECUTION_CONTEXT_KEYS - {"schema_version"}:
        if field in source:
            _require_safe_observability_identifier(
                source[field], what=f"lineage.execution_context.{field}", nullable=True
            )
    return {
        key: source[key]
        for key in _OBSERVABILITY_EXECUTION_CONTEXT_KEYS
        if key in source
    }


def _project_observability_lineage(payload: Any) -> dict[str, Any]:
    source = _exact_observability_object(
        payload,
        keys=_OBSERVABILITY_LINEAGE_REQUIRED_KEYS,
        what="observability related changes",
    )
    for field in ("project_id", "trace_id"):
        _require_safe_observability_identifier(source[field], what=f"lineage.{field}")
    links = _require_bounded_list(
        source["links"], what="observability lineage links", maximum=32
    )
    link_keys = frozenset(
        {"resource_type", "resource_id", "resource_version", "relationship"}
    )
    resource_types = frozenset(
        {
            "agent",
            "release",
            "deployment",
            "code_revision",
            "configuration",
            "prompt",
            "toolset",
            "evaluator",
            "dataset",
            "experiment_run",
            "configuration_run",
            "optimization_run",
            "intervention",
        }
    )
    relationships = frozenset(
        {
            "executed_by",
            "released_as",
            "deployed_as",
            "built_from",
            "configured_by",
            "used_prompt",
            "used_toolset",
            "evaluated_by",
            "sourced_from",
            "measured_in",
            "optimized_in",
            "changed_by",
        }
    )
    projected_links: list[dict[str, Any]] = []
    for index, value in enumerate(links):
        what = f"lineage.links[{index}]"
        link = _exact_observability_object(value, keys=link_keys, what=what)
        _require_observability_enum(
            link["resource_type"], resource_types, what=f"{what}.resource_type"
        )
        _require_safe_observability_identifier(
            link["resource_id"], what=f"{what}.resource_id"
        )
        _require_safe_observability_identifier(
            link["resource_version"], what=f"{what}.resource_version", nullable=True
        )
        _require_observability_enum(
            link["relationship"], relationships, what=f"{what}.relationship"
        )
        projected_links.append({key: link[key] for key in link_keys})
    _require_observability_datetime(source["generated_at"], what="lineage.generated_at")
    return {
        "project_id": source["project_id"],
        "trace_id": source["trace_id"],
        "execution_context": _project_observability_execution_context(
            source["execution_context"]
        ),
        "links": projected_links,
        "generated_at": source["generated_at"],
    }


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
            from traigent.cloud.auth import _strip_trace_context_headers

            headers = self._auth_headers()
            headers.setdefault("User-Agent", get_sdk_user_agent())
            # The cached client is long-lived: never freeze trace-context
            # headers (traceparent/tracestate) into its defaults -- they would
            # outlive the span and go stale. Trace context rides per-request
            # via _request_headers(). Strip defensively to enforce the
            # invariant even if a future _auth_headers() ever leaked one.
            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                headers=_strip_trace_context_headers(headers),
                timeout=self.timeout,
            )
        return self._client

    def _request_headers(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        """Build a fresh per-request header dict carrying the active trace context.

        Trace context (``traceparent``/``tracestate``) must ride on per-request
        headers, never on the cached client's long-lived default headers (that
        would freeze one span's context onto every later request). When no
        OpenTelemetry span is active -- or the ``tracing`` extra is not
        installed -- injection is a no-op and the returned dict is byte-for-byte
        the caller's headers. See
        :func:`traigent.cloud.auth._inject_trace_context`.
        """
        from traigent.cloud.auth import _inject_trace_context

        request_headers = dict(headers or {})
        _inject_trace_context(request_headers)
        return request_headers

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
        response = await client.get(
            path, headers=self._request_headers(headers), params=params
        )
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
        response = await client.post(
            path, headers=self._request_headers(), json=json_body
        )
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
            headers=self._request_headers(self._project_headers(project_id)),
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
        start_time, end_time = _validate_observability_time_window(start_time, end_time)
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
        return cast(
            "ObservabilityTraceSearchDTO",
            _project_observability_trace_search(payload),
        )

    async def list_observability_issues(
        self,
        project_id: str,
        *,
        page: int = 1,
        per_page: int = 50,
        state: str | None = None,
        detector_family: str | None = None,
        severity: str | None = None,
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
                }
            ),
        )
        return cast(
            "ObservabilityIssueListDTO",
            _project_observability_issue_list(payload),
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
        return cast(
            "ObservabilityIssueDetailDTO",
            _project_observability_issue_detail(payload),
        )

    async def list_observability_variants(
        self,
        project_id: str,
        *,
        page: int = 1,
        per_page: int = 50,
    ) -> ObservabilityVariantListDTO:
        """Return exact structural trace variants for a project."""
        _validate_observability_page(page, per_page)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/variants",
            what="observability variants",
            params={"page": str(page), "per_page": str(per_page)},
        )
        return cast(
            "ObservabilityVariantListDTO",
            _project_observability_variant_list(payload),
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
        return cast(
            "ObservabilityVariantDetailDTO",
            _project_observability_variant_detail(payload),
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
        return cast(
            "ObservabilityTraceAnalysisDTO",
            _project_observability_trace_analysis(payload),
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
        return cast(
            "ObservabilityTraceProjectionDTO",
            _project_observability_trace_slice(payload),
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
        start_time, end_time = _validate_observability_time_window(start_time, end_time)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/analysis/tools",
            what="observability tool analysis",
            params={
                "start_time": start_time,
                "end_time": end_time,
                "limit": str(limit),
            },
        )
        return cast(
            "ObservabilityToolAnalysisDTO",
            _project_observability_tool_analysis(payload),
        )

    async def get_observability_analysis_insights(
        self,
        project_id: str,
        *,
        start_time: str,
        end_time: str,
        limit: int = 20,
    ) -> ObservabilityAnalysisInsightsDTO:
        """Return bounded structural conformance facts and non-causal guidance."""
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit must be between 1 and 100.")
        start_time, end_time = _validate_observability_time_window(start_time, end_time)
        payload = await self._get_json(
            f"{_observability_path(project_id)}/analysis/insights",
            what="observability analysis insights",
            params={
                "start_time": start_time,
                "end_time": end_time,
                "limit": str(limit),
            },
        )
        return cast(
            "ObservabilityAnalysisInsightsDTO",
            _project_observability_analysis_insights(payload),
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
        return cast(
            "ObservabilityCohortComparisonDTO",
            _project_observability_cohort_comparison(payload),
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
        return cast(
            "ObservabilityLineageDTO",
            _project_observability_lineage(payload),
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


def _validate_observability_analysis_insights_payload(
    payload: dict[str, Any],
    *,
    for_mcp: bool = False,
) -> None:
    what = "observability analysis insights"
    required_payload_keys = _OBSERVABILITY_ANALYSIS_INSIGHTS_REQUIRED_KEYS - (
        frozenset({"limitations"}) if for_mcp else frozenset()
    )
    _require_keys(payload, required_payload_keys, what=what)
    _require_exact_keys(
        payload,
        _OBSERVABILITY_ANALYSIS_INSIGHTS_REQUIRED_KEYS,
        what=what,
        require_all=False,
    )
    _require_safe_observability_identifier(payload.get("project_id"), what="project_id")
    _require_observability_datetime(payload.get("start_time"), what="start_time")
    _require_observability_datetime(payload.get("end_time"), what="end_time")
    _require_observability_datetime(payload.get("generated_at"), what="generated_at")
    if payload.get("content_included") is not False:
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            "content_included must be false."
        )

    conformance = _require_object(
        payload.get("conformance"), what=f"{what} conformance"
    )
    required_conformance_keys = _OBSERVABILITY_CONFORMANCE_REQUIRED_KEYS - (
        frozenset({"interpretation"}) if for_mcp else frozenset()
    )
    _require_keys(conformance, required_conformance_keys, what=f"{what} conformance")
    _require_exact_keys(
        conformance,
        _OBSERVABILITY_CONFORMANCE_REQUIRED_KEYS,
        what=f"{what} conformance",
        require_all=False,
    )
    if conformance.get("baseline_type") != "observed_dominant_variant":
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            "baseline_type must be observed_dominant_variant."
        )
    _require_safe_observability_identifier(
        conformance.get("baseline_variant_id"),
        what="conformance.baseline_variant_id",
        nullable=True,
    )
    for field in (
        "analyzed_trace_count",
        "sampled_trace_count",
        "total_trace_count",
        "conforming_trace_count",
        "alternate_trace_count",
        "alternate_variant_count",
    ):
        _require_observability_integer(
            conformance.get(field), what=f"conformance.{field}"
        )
    for field in ("analysis_coverage", "sample_coverage"):
        _require_observability_number(
            conformance.get(field), what=f"conformance.{field}", minimum=0, maximum=1
        )
    for field in ("conformance_rate", "alternate_rate"):
        value = conformance.get(field)
        if value is not None:
            _require_observability_number(
                value, what=f"conformance.{field}", minimum=0, maximum=1
            )
    if not isinstance(conformance.get("sample_truncated"), bool):
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            "conformance.sample_truncated must be a boolean."
        )
    if "interpretation" in conformance:
        _require_observability_text(
            conformance["interpretation"],
            what="conformance.interpretation",
            maximum=512,
        )
    deviations = _require_bounded_list(
        conformance.get("deviations"), what=f"{what} deviations", maximum=100
    )
    for index, value in enumerate(deviations):
        deviation = _require_object(value, what=f"{what} deviations[{index}]")
        _require_exact_keys(
            deviation,
            _OBSERVABILITY_DEVIATION_REQUIRED_KEYS,
            what=f"{what} deviations[{index}]",
        )
        _require_safe_observability_identifier(
            deviation.get("variant_id"), what=f"deviations[{index}].variant_id"
        )
        _require_observability_integer(
            deviation.get("trace_count"),
            what=f"deviations[{index}].trace_count",
            minimum=1,
        )
        _require_observability_integer(
            deviation.get("failed_trace_count"),
            what=f"deviations[{index}].failed_trace_count",
        )
        _require_safe_observability_identifier(
            deviation.get("representative_trace_id"),
            what=f"deviations[{index}].representative_trace_id",
        )
        evidence_trace_ids = _require_bounded_list(
            deviation.get("evidence_trace_ids"),
            what=f"{what} deviations[{index}] evidence_trace_ids",
            maximum=3,
        )
        _require_unique_observability_identifiers(
            evidence_trace_ids, what=f"deviations[{index}].evidence_trace_ids"
        )
        _require_observability_number(
            deviation.get("share"),
            what=f"deviations[{index}].share",
            minimum=0,
            maximum=1,
        )

    recommendations = _require_bounded_list(
        payload.get("recommendations"), what=f"{what} recommendations", maximum=100
    )
    for index, value in enumerate(recommendations):
        recommendation = _require_object(value, what=f"{what} recommendations[{index}]")
        required_recommendation_keys = _OBSERVABILITY_RECOMMENDATION_REQUIRED_KEYS - (
            frozenset({"suggested_action"}) if for_mcp else frozenset()
        )
        _require_keys(
            recommendation,
            required_recommendation_keys,
            what=f"{what} recommendations[{index}]",
        )
        _require_exact_keys(
            recommendation,
            _OBSERVABILITY_RECOMMENDATION_REQUIRED_KEYS,
            what=f"{what} recommendations[{index}]",
            require_all=False,
        )
        _require_safe_observability_identifier(
            recommendation.get("id"), what=f"recommendations[{index}].id"
        )
        _require_observability_enum(
            recommendation.get("category"),
            _OBSERVABILITY_RECOMMENDATION_CATEGORIES,
            what=f"recommendations[{index}].category",
        )
        _require_observability_enum(
            recommendation.get("priority"),
            _OBSERVABILITY_RECOMMENDATION_PRIORITIES,
            what=f"recommendations[{index}].priority",
        )
        _require_observability_number(
            recommendation.get("confidence"),
            what=f"recommendations[{index}].confidence",
            minimum=0,
            maximum=1,
        )
        _require_safe_observability_identifier(
            recommendation.get("subject"), what=f"recommendations[{index}].subject"
        )
        if "suggested_action" in recommendation:
            _require_observability_text(
                recommendation["suggested_action"],
                what=f"recommendations[{index}].suggested_action",
                maximum=512,
            )
        evidence = _require_object(
            recommendation.get("evidence"),
            what=f"{what} recommendations[{index}] evidence",
        )
        _require_exact_keys(
            evidence,
            _OBSERVABILITY_RECOMMENDATION_EVIDENCE_KEYS,
            what=f"{what} recommendations[{index}] evidence",
            require_all=False,
        )
        _validate_observability_recommendation_evidence(evidence, index=index)
        measurement = _require_object(
            recommendation.get("measurement"),
            what=f"{what} recommendations[{index}] measurement",
        )
        _require_exact_keys(
            measurement,
            _OBSERVABILITY_MEASUREMENT_REQUIRED_KEYS,
            what=f"{what} recommendations[{index}] measurement",
        )
        if (
            measurement.get("comparison") != "before_after_cohorts"
            or measurement.get("intervention_context_key") != "intervention_id"
        ):
            raise AnalyticsClientError(
                "Malformed observability analysis insights response: "
                "recommendation measurement constants are invalid."
            )
        metrics = _require_bounded_list(
            measurement.get("metrics"),
            what=f"{what} recommendations[{index}] metrics",
            maximum=8,
        )
        if (
            not metrics
            or any(not isinstance(metric, str) for metric in metrics)
            or set(metrics).difference(_OBSERVABILITY_COHORT_METRICS)
        ):
            raise AnalyticsClientError(
                "Malformed observability analysis insights response: "
                "recommendation metrics must contain supported values."
            )
        if len(set(metrics)) != len(metrics):
            raise AnalyticsClientError(
                "Malformed observability analysis insights response: "
                "recommendation metrics must contain unique values."
            )

    if "limitations" in payload:
        limitations = _require_bounded_list(
            payload["limitations"], what=f"{what} limitations", maximum=8
        )
        if not limitations:
            raise AnalyticsClientError(
                "Malformed observability analysis insights response: "
                "limitations must contain non-empty strings."
            )
        for index, limitation in enumerate(limitations):
            _require_observability_text(
                limitation, what=f"limitations[{index}]", maximum=512
            )


def _require_exact_keys(
    payload: dict[str, Any],
    allowed: frozenset[str],
    *,
    what: str,
    require_all: bool = True,
) -> None:
    if require_all:
        _require_keys(payload, allowed, what=what)
    extra = sorted(payload.keys() - allowed)
    if extra:
        raise AnalyticsClientError(
            f"Malformed {what} response: unsupported key(s) are present."
        )


def _require_safe_observability_identifier(
    value: Any,
    *,
    what: str,
    nullable: bool = False,
    maximum: int = 128,
) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or not _OBSERVABILITY_SAFE_IDENTIFIER_PATTERN.fullmatch(value)
        or len(value) > maximum
    ):
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must be a content-free identifier of at most {maximum} characters."
        )
    return value


def _require_observability_datetime(value: Any, *, what: str) -> str:
    if not isinstance(value, str):
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must be an ISO 8601 date-time."
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must be an ISO 8601 date-time."
        ) from exc
    if parsed.utcoffset() is None:
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must include a UTC offset."
        )
    return value


def _require_observability_integer(
    value: Any, *, what: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise AnalyticsClientError(
            "Malformed observability response: "
            f"{what} must be an integer greater than or equal to {minimum}{upper}."
        )
    return value


def _require_observability_number(
    value: Any, *, what: str, minimum: float, maximum: float
) -> int | float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must be a finite number between {minimum} and {maximum}."
        )
    return value


def _require_finite_observability_number(value: Any, *, what: str) -> int | float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise AnalyticsClientError(
            f"Malformed observability response: {what} must be a finite number."
        )
    return value


def _require_non_negative_observability_number(value: Any, *, what: str) -> int | float:
    number = _require_finite_observability_number(value, what=what)
    if number < 0:
        raise AnalyticsClientError(
            f"Malformed observability response: {what} must be non-negative."
        )
    return number


def _require_observability_hash(value: Any, *, what: str) -> str:
    if not isinstance(value, str) or not _OBSERVABILITY_HASH_PATTERN.fullmatch(value):
        raise AnalyticsClientError(
            f"Malformed observability response: {what} must be a lowercase sha256 digest."
        )
    return value


def _require_observability_text(value: Any, *, what: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must be a non-empty string of at most {maximum} characters."
        )
    return value


def _require_observability_enum(
    value: Any, allowed: frozenset[str], *, what: str
) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must be one of: {', '.join(sorted(allowed))}."
        )
    return value


def _require_unique_observability_identifiers(values: list[Any], *, what: str) -> None:
    identifiers = [
        _require_safe_observability_identifier(value, what=f"{what}[{index}]")
        for index, value in enumerate(values)
    ]
    if len(set(identifiers)) != len(identifiers):
        raise AnalyticsClientError(
            "Malformed observability analysis insights response: "
            f"{what} must contain unique values."
        )


def _validate_observability_recommendation_evidence(
    evidence: dict[str, Any], *, index: int
) -> None:
    prefix = f"recommendations[{index}].evidence"
    for field in ("normalized_tool_id", "issue_id"):
        if field in evidence:
            _require_safe_observability_identifier(
                evidence[field], what=f"{prefix}.{field}"
            )
    if "baseline_variant_id" in evidence:
        _require_safe_observability_identifier(
            evidence["baseline_variant_id"],
            what=f"{prefix}.baseline_variant_id",
            nullable=True,
        )
    if "issue_ids" in evidence:
        issue_ids = _require_bounded_list(
            evidence["issue_ids"], what=f"{prefix}.issue_ids", maximum=100
        )
        _require_unique_observability_identifiers(issue_ids, what=f"{prefix}.issue_ids")
    for field in (
        "trace_count",
        "attempt_count",
        "failure_count",
        "retry_count",
        "fallback_count",
        "occurrence_count",
        "affected_trace_count",
        "analyzed_trace_count",
        "sampled_trace_count",
        "alternate_trace_count",
        "alternate_variant_count",
    ):
        if field in evidence:
            _require_observability_integer(evidence[field], what=f"{prefix}.{field}")
    for field in ("failure_rate", "retry_rate", "fallback_rate", "alternate_rate"):
        if field in evidence:
            _require_observability_number(
                evidence[field], what=f"{prefix}.{field}", minimum=0, maximum=1
            )
    if "detector_family" in evidence:
        _require_observability_enum(
            evidence["detector_family"],
            _OBSERVABILITY_DETECTOR_FAMILIES,
            what=f"{prefix}.detector_family",
        )


def _project_observability_analysis_insights(
    payload: Any, *, for_mcp: bool = False
) -> dict[str, Any]:
    """Validate and rebuild analysis insights according to their exact nesting."""
    source = _require_object(payload, what="observability analysis insights")
    _validate_observability_analysis_insights_payload(source, for_mcp=for_mcp)
    conformance = cast(dict[str, Any], source["conformance"])
    deviations = cast(list[dict[str, Any]], conformance["deviations"])
    projected_conformance = {
        key: conformance[key]
        for key in _OBSERVABILITY_CONFORMANCE_REQUIRED_KEYS
        if key not in {"deviations", "interpretation"}
    }
    projected_conformance["deviations"] = [
        {
            **{
                key: deviation[key]
                for key in _OBSERVABILITY_DEVIATION_REQUIRED_KEYS
                if key != "evidence_trace_ids"
            },
            "evidence_trace_ids": list(deviation["evidence_trace_ids"]),
        }
        for deviation in deviations
    ]
    if not for_mcp:
        projected_conformance["interpretation"] = conformance["interpretation"]

    projected_recommendations: list[dict[str, Any]] = []
    for recommendation in cast(list[dict[str, Any]], source["recommendations"]):
        evidence = cast(dict[str, Any], recommendation["evidence"])
        measurement = cast(dict[str, Any], recommendation["measurement"])
        projected = {
            key: recommendation[key]
            for key in (
                "id",
                "category",
                "priority",
                "confidence",
                "subject",
            )
        }
        projected["evidence"] = {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in evidence.items()
        }
        projected["measurement"] = {
            "comparison": measurement["comparison"],
            "metrics": list(measurement["metrics"]),
            "intervention_context_key": measurement["intervention_context_key"],
        }
        if not for_mcp:
            projected["suggested_action"] = recommendation["suggested_action"]
        projected_recommendations.append(projected)

    projected_payload = {
        "project_id": source["project_id"],
        "start_time": source["start_time"],
        "end_time": source["end_time"],
        "content_included": False,
        "conformance": projected_conformance,
        "recommendations": projected_recommendations,
        "generated_at": source["generated_at"],
    }
    if not for_mcp:
        projected_payload["limitations"] = list(source["limitations"])
    return projected_payload


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
