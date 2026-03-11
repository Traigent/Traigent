"""DTOs for core metrics and export responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
            agents=int(payload.get("agents", 0)),
            benchmarks=int(payload.get("benchmarks", 0)),
            measures=int(payload.get("measures", 0)),
            experiments=int(payload.get("experiments", 0)),
            experiment_runs=int(payload.get("experiment_runs", 0)),
            configuration_runs=int(payload.get("configuration_runs", 0)),
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
