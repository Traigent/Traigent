"""Typed JSON contracts for content-free observability analysis readers."""

from __future__ import annotations

from typing import Any, TypedDict


class ObservabilityIssueDTO(TypedDict):
    id: str
    project_id: str
    detector_family: str
    problem_signature: str
    signature_spec_version: str
    state: str
    severity: str
    occurrence_count: int
    affected_trace_count: int
    reopen_count: int
    first_seen_at: str
    last_seen_at: str
    created_at: str
    updated_at: str
    state_changed_at: str
    superseded_by_issue_id: str | None
    version: int


class ObservabilityIssueListDTO(TypedDict):
    items: list[ObservabilityIssueDTO]
    page: int
    per_page: int
    total: int
    generated_at: str


class ObservabilityIssueDetailDTO(TypedDict):
    issue: ObservabilityIssueDTO
    occurrences: list[dict[str, Any]]
    occurrence_page: int
    occurrences_per_page: int
    total_occurrences: int
    variant_ids: list[str]
    generated_at: str


class ObservabilityTraceVariantDTO(TypedDict):
    id: str
    project_id: str
    display_label: str
    fingerprint: dict[str, Any]
    trace_count: int
    first_seen_at: str
    last_seen_at: str
    representative_trace_id: str
    boundary_trace_ids: list[str]
    status_counts: dict[str, int]
    derivation: dict[str, Any]


class ObservabilityVariantListDTO(TypedDict):
    items: list[ObservabilityTraceVariantDTO]
    page: int
    per_page: int
    total: int
    generated_at: str


class ObservabilityVariantDetailDTO(TypedDict):
    variant: ObservabilityTraceVariantDTO
    traces: list[dict[str, Any]]
    trace_page: int
    traces_per_page: int
    total_traces: int
    generated_at: str


class ObservabilityTraceSearchDTO(TypedDict):
    items: list[dict[str, Any]]
    page: int
    per_page: int
    total: int
    has_more: bool


class ObservabilityTraceAnalysisDTO(TypedDict):
    project_id: str
    trace_id: str
    analysis_status: str
    failure_code: str | None
    fingerprint: dict[str, Any] | None
    variant_id: str | None
    critical_path: dict[str, Any]
    repeat_groups: list[dict[str, Any]]
    tool_summaries: list[dict[str, Any]]
    issue_ids: list[str]
    derivation: dict[str, Any] | None


class ObservabilityTraceProjectionDTO(TypedDict):
    project_id: str
    trace_id: str
    projection_mode: str
    content_included: bool
    items: list[dict[str, Any]]
    next_cursor: str | None
    has_more: bool
    generated_at: str


class ObservabilityToolAnalysisDTO(TypedDict):
    project_id: str
    start_time: str
    end_time: str
    items: list[dict[str, Any]]
    generated_at: str


class ObservabilityCohortComparisonDTO(TypedDict):
    project_id: str
    reference: dict[str, Any]
    comparison: dict[str, Any]
    matched_pair_count: int
    deltas: list[dict[str, Any]]
    generated_at: str


class ObservabilityLineageDTO(TypedDict):
    project_id: str
    trace_id: str
    execution_context: dict[str, Any]
    links: list[dict[str, Any]]
    generated_at: str


__all__ = [
    "ObservabilityCohortComparisonDTO",
    "ObservabilityIssueDTO",
    "ObservabilityIssueDetailDTO",
    "ObservabilityIssueListDTO",
    "ObservabilityLineageDTO",
    "ObservabilityToolAnalysisDTO",
    "ObservabilityTraceAnalysisDTO",
    "ObservabilityTraceProjectionDTO",
    "ObservabilityTraceSearchDTO",
    "ObservabilityTraceVariantDTO",
    "ObservabilityVariantDetailDTO",
    "ObservabilityVariantListDTO",
]
