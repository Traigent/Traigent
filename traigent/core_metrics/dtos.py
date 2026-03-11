"""DTOs for project-scoped analytics and export responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalyticsContextDTO:
    tenant_id: str
    project_id: str
    generated_at: str
    privacy_classification: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalyticsContextDTO:
        return cls(
            tenant_id=str(payload["tenant_id"]),
            project_id=str(payload["project_id"]),
            generated_at=str(payload["generated_at"]),
            privacy_classification=str(payload["privacy_classification"]),
        )


@dataclass(frozen=True)
class CoreEntityCountsDTO:
    agents: int
    benchmarks: int
    measures: int
    experiments: int
    experiment_runs: int
    configuration_runs: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoreEntityCountsDTO:
        return cls(
            agents=int(payload["agents"]),
            benchmarks=int(payload["benchmarks"]),
            measures=int(payload["measures"]),
            experiments=int(payload["experiments"]),
            experiment_runs=int(payload["experiment_runs"]),
            configuration_runs=int(payload["configuration_runs"]),
        )


@dataclass(frozen=True)
class StatusBreakdownsDTO:
    experiments: dict[str, int]
    experiment_runs: dict[str, int]
    configuration_runs: dict[str, int]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StatusBreakdownsDTO:
        return cls(
            experiments={
                str(key): int(value) for key, value in payload["experiments"].items()
            },
            experiment_runs={
                str(key): int(value)
                for key, value in payload["experiment_runs"].items()
            },
            configuration_runs={
                str(key): int(value)
                for key, value in payload["configuration_runs"].items()
            },
        )


@dataclass(frozen=True)
class UsageSummaryDTO:
    experiment_runs: int
    configuration_runs: int
    total_cost_usd: float
    avg_cost_usd: float
    total_tokens: int
    avg_latency_ms: float
    p95_latency_ms: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UsageSummaryDTO:
        return cls(
            experiment_runs=int(payload["experiment_runs"]),
            configuration_runs=int(payload["configuration_runs"]),
            total_cost_usd=float(payload["total_cost_usd"]),
            avg_cost_usd=float(payload["avg_cost_usd"]),
            total_tokens=int(payload["total_tokens"]),
            avg_latency_ms=float(payload["avg_latency_ms"]),
            p95_latency_ms=float(payload["p95_latency_ms"]),
        )


@dataclass(frozen=True)
class MeasureSummaryDTO:
    measure_key: str
    measure_id: str | None
    label: str
    value_type: str
    sample_count: int
    mean: float
    min: float
    max: float
    privacy_classification: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasureSummaryDTO:
        return cls(
            measure_key=str(payload["measure_key"]),
            measure_id=(
                str(payload["measure_id"])
                if payload.get("measure_id") is not None
                else None
            ),
            label=str(payload["label"]),
            value_type=str(payload["value_type"]),
            sample_count=int(payload["sample_count"]),
            mean=float(payload["mean"]),
            min=float(payload["min"]),
            max=float(payload["max"]),
            privacy_classification=str(payload["privacy_classification"]),
        )


@dataclass(frozen=True)
class TrendPointDTO:
    bucket_start: str
    bucket_label: str
    value: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrendPointDTO:
        return cls(
            bucket_start=str(payload["bucket_start"]),
            bucket_label=str(payload["bucket_label"]),
            value=int(payload["value"]),
        )


@dataclass(frozen=True)
class TrendSeriesDTO:
    series_key: str
    label: str
    unit: str
    points: list[TrendPointDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrendSeriesDTO:
        return cls(
            series_key=str(payload["series_key"]),
            label=str(payload["label"]),
            unit=str(payload["unit"]),
            points=[TrendPointDTO.from_dict(item) for item in payload["points"]],
        )


@dataclass(frozen=True)
class HistogramBucketDTO:
    lower_bound: float
    upper_bound: float
    count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HistogramBucketDTO:
        return cls(
            lower_bound=float(payload["lower_bound"]),
            upper_bound=float(payload["upper_bound"]),
            count=int(payload["count"]),
        )


@dataclass(frozen=True)
class FineTuningManifestRecordDTO:
    record_id: str
    experiment_id: str
    experiment_run_id: str
    configuration_run_id: str
    input_hash: str | None
    output_hash: str | None
    input_ref: str | None
    output_ref: str | None
    input_content: str | None
    output_content: str | None
    materialization: str
    measure_summary: dict[str, float | None]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FineTuningManifestRecordDTO:
        return cls(
            record_id=str(payload["record_id"]),
            experiment_id=str(payload["experiment_id"]),
            experiment_run_id=str(payload["experiment_run_id"]),
            configuration_run_id=str(payload["configuration_run_id"]),
            input_hash=(
                str(payload["input_hash"])
                if payload.get("input_hash") is not None
                else None
            ),
            output_hash=(
                str(payload["output_hash"])
                if payload.get("output_hash") is not None
                else None
            ),
            input_ref=(
                str(payload["input_ref"])
                if payload.get("input_ref") is not None
                else None
            ),
            output_ref=(
                str(payload["output_ref"])
                if payload.get("output_ref") is not None
                else None
            ),
            input_content=(
                str(payload["input_content"])
                if payload.get("input_content") is not None
                else None
            ),
            output_content=(
                str(payload["output_content"])
                if payload.get("output_content") is not None
                else None
            ),
            materialization=str(payload["materialization"]),
            measure_summary={
                str(key): (float(value) if value is not None else None)
                for key, value in dict(payload["measure_summary"]).items()
            },
            metadata=dict(payload["metadata"]),
        )


@dataclass(frozen=True)
class ProjectAnalyticsSummaryDTO:
    context: AnalyticsContextDTO
    range_days: int
    entity_counts: CoreEntityCountsDTO
    status_breakdowns: StatusBreakdownsDTO
    usage_summary: UsageSummaryDTO
    measure_summaries: list[MeasureSummaryDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectAnalyticsSummaryDTO:
        return cls(
            context=AnalyticsContextDTO.from_dict(payload["context"]),
            range_days=int(payload["range_days"]),
            entity_counts=CoreEntityCountsDTO.from_dict(payload["entity_counts"]),
            status_breakdowns=StatusBreakdownsDTO.from_dict(
                payload["status_breakdowns"]
            ),
            usage_summary=UsageSummaryDTO.from_dict(payload["usage_summary"]),
            measure_summaries=[
                MeasureSummaryDTO.from_dict(item)
                for item in payload["measure_summaries"]
            ],
        )


@dataclass(frozen=True)
class ProjectAnalyticsTrendDTO:
    context: AnalyticsContextDTO
    metric_id: str
    experiment_id: str | None
    range_days: int
    requested_bucket: str | None
    resolved_bucket: str
    series: list[TrendSeriesDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectAnalyticsTrendDTO:
        return cls(
            context=AnalyticsContextDTO.from_dict(payload["context"]),
            metric_id=str(payload["metric_id"]),
            experiment_id=(
                str(payload["experiment_id"])
                if payload.get("experiment_id") is not None
                else None
            ),
            range_days=int(payload["range_days"]),
            requested_bucket=(
                str(payload["requested_bucket"])
                if payload.get("requested_bucket") is not None
                else None
            ),
            resolved_bucket=str(payload["resolved_bucket"]),
            series=[TrendSeriesDTO.from_dict(item) for item in payload["series"]],
        )


@dataclass(frozen=True)
class ProjectMeasureDistributionDTO:
    context: AnalyticsContextDTO
    measure_key: str
    measure_id: str | None
    label: str
    experiment_id: str | None
    value_type: str
    sample_count: int
    mean: float | None
    min: float | None
    max: float | None
    bucket_count: int
    histogram: list[HistogramBucketDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectMeasureDistributionDTO:
        return cls(
            context=AnalyticsContextDTO.from_dict(payload["context"]),
            measure_key=str(payload["measure_key"]),
            measure_id=(
                str(payload["measure_id"])
                if payload.get("measure_id") is not None
                else None
            ),
            label=str(payload["label"]),
            experiment_id=(
                str(payload["experiment_id"])
                if payload.get("experiment_id") is not None
                else None
            ),
            value_type=str(payload["value_type"]),
            sample_count=int(payload["sample_count"]),
            mean=float(payload["mean"]) if payload.get("mean") is not None else None,
            min=float(payload["min"]) if payload.get("min") is not None else None,
            max=float(payload["max"]) if payload.get("max") is not None else None,
            bucket_count=int(payload["bucket_count"]),
            histogram=[
                HistogramBucketDTO.from_dict(item) for item in payload["histogram"]
            ],
        )


@dataclass(frozen=True)
class FineTuningManifestDTO:
    context: AnalyticsContextDTO
    export_mode: str
    privacy_mode: bool
    include_content: bool
    record_count: int
    records: list[FineTuningManifestRecordDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FineTuningManifestDTO:
        return cls(
            context=AnalyticsContextDTO.from_dict(payload["context"]),
            export_mode=str(payload["export_mode"]),
            privacy_mode=bool(payload["privacy_mode"]),
            include_content=bool(payload["include_content"]),
            record_count=int(payload["record_count"]),
            records=[
                FineTuningManifestRecordDTO.from_dict(item)
                for item in payload["records"]
            ],
        )


@dataclass(frozen=True)
class DailyCountPointDTO:
    date: str
    count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DailyCountPointDTO:
        return cls(
            date=str(payload.get("date", "")),
            count=int(payload.get("count", 0)),
        )


@dataclass(frozen=True)
class MeasureAggregateSummaryDTO:
    count: int
    mean: float
    min: float
    max: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MeasureAggregateSummaryDTO:
        return cls(
            count=int(payload.get("count", 0)),
            mean=float(payload.get("mean", 0.0)),
            min=float(payload.get("min", 0.0)),
            max=float(payload.get("max", 0.0)),
        )


@dataclass(frozen=True)
class CoreMetricsOverviewDTO:
    tenant_id: str
    project_id: str
    entities: CoreEntityCountsDTO
    experiment_statuses: dict[str, int]
    experiment_run_statuses: dict[str, int]
    configuration_run_statuses: dict[str, int]
    recent_run_volume: list[DailyCountPointDTO]
    measure_summary: dict[str, MeasureAggregateSummaryDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoreMetricsOverviewDTO:
        statuses = payload.get("statuses") or {}
        return cls(
            tenant_id=str(payload.get("tenant_id", "")),
            project_id=str(payload.get("project_id", "")),
            entities=CoreEntityCountsDTO.from_dict(payload.get("entities") or {}),
            experiment_statuses={
                str(key): int(value)
                for key, value in (statuses.get("experiments") or {}).items()
            },
            experiment_run_statuses={
                str(key): int(value)
                for key, value in (statuses.get("experiment_runs") or {}).items()
            },
            configuration_run_statuses={
                str(key): int(value)
                for key, value in (statuses.get("configuration_runs") or {}).items()
            },
            recent_run_volume=[
                DailyCountPointDTO.from_dict(item)
                for item in payload.get("recent_run_volume") or []
            ],
            measure_summary={
                str(key): MeasureAggregateSummaryDTO.from_dict(value or {})
                for key, value in (payload.get("measure_summary") or {}).items()
            },
        )


@dataclass(frozen=True)
class CoreExperimentTrendDTO:
    experiment_id: str
    experiment_name: str
    project_id: str
    runs_total: int
    configuration_runs_total: int
    daily_runs: list[DailyCountPointDTO]
    daily_configuration_runs: list[DailyCountPointDTO]
    measure_summary: dict[str, MeasureAggregateSummaryDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoreExperimentTrendDTO:
        return cls(
            experiment_id=str(payload.get("experiment_id", "")),
            experiment_name=str(payload.get("experiment_name", "")),
            project_id=str(payload.get("project_id", "")),
            runs_total=int(payload.get("runs_total", 0)),
            configuration_runs_total=int(payload.get("configuration_runs_total", 0)),
            daily_runs=[
                DailyCountPointDTO.from_dict(item)
                for item in payload.get("daily_runs") or []
            ],
            daily_configuration_runs=[
                DailyCountPointDTO.from_dict(item)
                for item in payload.get("daily_configuration_runs") or []
            ],
            measure_summary={
                str(key): MeasureAggregateSummaryDTO.from_dict(value or {})
                for key, value in (payload.get("measure_summary") or {}).items()
            },
        )


@dataclass(frozen=True)
class FineTuningExportDTO:
    content: str
    filename: str
    content_type: str
