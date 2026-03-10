"""DTOs for the Traigent evaluation operations client."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from traigent.prompts.dtos import PaginationInfo


def _coalesce_not_none(candidate: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return value
    return None


class MeasureValueType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class EvaluationTargetType(str, Enum):
    CONFIGURATION_RUN = "configuration_run"
    EXPERIMENT_RUN = "experiment_run"
    OBSERVABILITY_OBSERVATION = "observability_observation"
    OBSERVABILITY_TRACE = "observability_trace"


class ScoreSource(str, Enum):
    MANUAL = "manual"
    EVALUATOR = "evaluator"


class EvaluatorRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnnotationQueueStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class AnnotationQueueItemStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class JudgeConfigDTO:
    instructions: str
    model_id: str
    context_type: str = "none"
    context_source: str | None = None
    parameters: dict[str, Any] | None = None
    scoring_rubric: dict[str, Any] | None = None
    response_format_instructions: str | None = None
    max_budget: float | None = None

    def __post_init__(self) -> None:
        normalized_parameters = dict(self.parameters or {})
        normalized_parameters["model_id"] = self.model_id
        object.__setattr__(self, "instructions", self.instructions.strip())
        object.__setattr__(self, "context_type", self.context_type.strip() or "none")
        object.__setattr__(self, "parameters", normalized_parameters or None)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JudgeConfigDTO:
        return cls(
            instructions=str(payload.get("instructions", "")),
            model_id=str(payload.get("model_id", "")),
            context_type=str(payload.get("context_type", "none")),
            context_source=payload.get("context_source"),
            parameters=dict(payload.get("parameters") or {}),
            scoring_rubric=(
                dict(payload.get("scoring_rubric") or {})
                if payload.get("scoring_rubric") is not None
                else None
            ),
            response_format_instructions=payload.get("response_format_instructions"),
            max_budget=(
                float(payload["max_budget"])
                if payload.get("max_budget") is not None
                else None
            ),
        )

    @classmethod
    def from_benchmark_payload(
        cls, payload: dict[str, Any] | None
    ) -> JudgeConfigDTO | None:
        if not payload:
            return None

        parameters = payload.get("parameters")
        if parameters is None and payload.get("modelParameters"):
            model_parameters = payload.get("modelParameters") or []
            if isinstance(model_parameters, list) and model_parameters:
                candidate = model_parameters[0]
                if isinstance(candidate, dict):
                    parameters = {
                        "temperature": _coalesce_not_none(candidate, "temperature"),
                        "max_tokens": _coalesce_not_none(
                            candidate, "max_tokens", "maxTokens"
                        ),
                        "top_p": _coalesce_not_none(candidate, "top_p", "topP"),
                        "top_k": _coalesce_not_none(candidate, "top_k", "topK"),
                        "frequency_penalty": _coalesce_not_none(
                            candidate, "frequency_penalty", "frequencyPenalty"
                        ),
                        "presence_penalty": _coalesce_not_none(
                            candidate, "presence_penalty", "presencePenalty"
                        ),
                        "repetition_penalty": _coalesce_not_none(
                            candidate, "repetition_penalty", "repetitionPenalty"
                        ),
                        "stop_sequences": _coalesce_not_none(
                            candidate, "stop_sequences", "stopSequences"
                        ),
                    }

        return cls(
            instructions=str(payload.get("instructions", "")),
            model_id=str(payload.get("model_id") or payload.get("model") or ""),
            context_type=str(
                payload.get("context_type")
                or (payload.get("context") or {}).get("type")
                or "none"
            ),
            context_source=payload.get("context_source")
            or (payload.get("context") or {}).get("source"),
            parameters=dict(parameters or {}),
            scoring_rubric=(
                dict(payload.get("scoring_rubric") or {})
                if payload.get("scoring_rubric") is not None
                else None
            ),
            response_format_instructions=payload.get("response_format_instructions"),
            max_budget=(
                float(payload["max_budget"])
                if payload.get("max_budget") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "instructions": self.instructions,
            "model_id": self.model_id,
            "context_type": self.context_type,
            "context_source": self.context_source,
        }
        if self.parameters is not None:
            payload["parameters"] = dict(self.parameters)
        if self.scoring_rubric is not None:
            payload["scoring_rubric"] = dict(self.scoring_rubric)
        if self.response_format_instructions is not None:
            payload["response_format_instructions"] = self.response_format_instructions
        if self.max_budget is not None:
            payload["max_budget"] = self.max_budget
        return payload

    def to_benchmark_payload(self) -> dict[str, Any]:
        """Return a benchmark-compatible payload with both normalized and legacy aliases."""
        benchmark_parameters: dict[str, Any] | None = None
        if self.parameters is not None:
            benchmark_parameters = {
                "model_id": self.model_id,
                "temperature": self.parameters.get("temperature"),
                "maxTokens": self.parameters.get("max_tokens"),
                "topP": self.parameters.get("top_p"),
                "topK": self.parameters.get("top_k"),
                "frequencyPenalty": self.parameters.get("frequency_penalty"),
                "presencePenalty": self.parameters.get("presence_penalty"),
                "repetitionPenalty": self.parameters.get("repetition_penalty"),
                "stopSequences": self.parameters.get("stop_sequences"),
            }

        payload: dict[str, Any] = {
            "instructions": self.instructions,
            "model": self.model_id,
            "model_id": self.model_id,
            "context": {
                "type": self.context_type,
                "source": self.context_source,
            },
            "context_type": self.context_type,
            "context_source": self.context_source,
        }
        if self.parameters is not None:
            payload["parameters"] = dict(self.parameters)
            payload["modelParameters"] = [benchmark_parameters]
        if self.scoring_rubric is not None:
            payload["scoring_rubric"] = dict(self.scoring_rubric)
        if self.response_format_instructions is not None:
            payload["response_format_instructions"] = self.response_format_instructions
        if self.max_budget is not None:
            payload["max_budget"] = self.max_budget
        return payload


@dataclass(frozen=True)
class EvaluationTargetRefDTO:
    target_type: EvaluationTargetType
    target_id: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationTargetRefDTO:
        return cls(
            target_type=EvaluationTargetType(payload.get("target_type")),
            target_id=str(payload.get("target_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type.value,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class EvaluatorDefinitionDTO:
    id: str
    name: str
    description: str | None
    measure_id: str
    primary_measure_id: str
    target_type: EvaluationTargetType
    judge_config: JudgeConfigDTO
    sampling_rate: float
    target_filters: dict[str, Any]
    is_active: bool
    measure: dict[str, Any] | None
    created_by: str | None
    updated_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluatorDefinitionDTO:
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            description=payload.get("description"),
            measure_id=str(payload.get("measure_id", "")),
            primary_measure_id=str(
                payload.get("primary_measure_id") or payload.get("measure_id", "")
            ),
            target_type=EvaluationTargetType(payload.get("target_type")),
            judge_config=JudgeConfigDTO.from_dict(payload.get("judge_config") or {}),
            sampling_rate=float(payload.get("sampling_rate", 1.0)),
            target_filters=dict(payload.get("target_filters") or {}),
            is_active=bool(payload.get("is_active", True)),
            measure=(
                dict(payload.get("measure") or {}) if payload.get("measure") else None
            ),
            created_by=payload.get("created_by"),
            updated_by=payload.get("updated_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class EvaluatorRunDTO:
    id: str
    evaluator_id: str
    measure_id: str
    target_type: EvaluationTargetType
    target_id: str
    status: EvaluatorRunStatus
    source: str
    input_snapshot: dict[str, Any]
    output_payload: dict[str, Any] | None
    score_record_ids: list[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    error_message: str | None
    observability_trace_id: str | None
    job_id: str | None
    created_by: str | None
    started_at: str | None
    completed_at: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluatorRunDTO:
        return cls(
            id=str(payload.get("id", "")),
            evaluator_id=str(payload.get("evaluator_id", "")),
            measure_id=str(payload.get("measure_id", "")),
            target_type=EvaluationTargetType(payload.get("target_type")),
            target_id=str(payload.get("target_id", "")),
            status=EvaluatorRunStatus(
                payload.get("status", EvaluatorRunStatus.PENDING.value)
            ),
            source=str(payload.get("source", "")),
            input_snapshot=dict(payload.get("input_snapshot") or {}),
            output_payload=(
                dict(payload.get("output_payload") or {})
                if payload.get("output_payload") is not None
                else None
            ),
            score_record_ids=[
                str(item) for item in payload.get("score_record_ids") or []
            ],
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            latency_ms=int(payload.get("latency_ms", 0)),
            error_message=payload.get("error_message"),
            observability_trace_id=payload.get("observability_trace_id"),
            job_id=payload.get("job_id"),
            created_by=payload.get("created_by"),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class BackfillResultDTO:
    evaluator_id: str
    target_type: EvaluationTargetType
    submitted_count: int
    skipped_count: int
    skipped_target_ids: list[str]
    items: list[EvaluatorRunDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackfillResultDTO:
        return cls(
            evaluator_id=str(payload.get("evaluator_id", "")),
            target_type=EvaluationTargetType(payload.get("target_type")),
            submitted_count=int(payload.get("submitted_count", 0)),
            skipped_count=int(payload.get("skipped_count", 0)),
            skipped_target_ids=[
                str(item) for item in payload.get("skipped_target_ids") or []
            ],
            items=[
                EvaluatorRunDTO.from_dict(item) for item in payload.get("items") or []
            ],
        )


@dataclass(frozen=True)
class ScoreRecordDTO:
    id: str
    measure_id: str
    evaluator_run_id: str | None
    target_type: EvaluationTargetType
    target_id: str
    source: ScoreSource
    value: float | str | bool | None
    numeric_value: float | None
    categorical_value: str | None
    boolean_value: bool | None
    value_type: MeasureValueType | None
    measure_label: str | None
    categories: list[str]
    comment: str | None
    correction_output: Any
    metadata: dict[str, Any]
    actor_user_id: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreRecordDTO:
        value_type = payload.get("value_type")
        return cls(
            id=str(payload.get("id", "")),
            measure_id=str(payload.get("measure_id", "")),
            evaluator_run_id=payload.get("evaluator_run_id"),
            target_type=EvaluationTargetType(payload.get("target_type")),
            target_id=str(payload.get("target_id", "")),
            source=ScoreSource(payload.get("source", ScoreSource.MANUAL.value)),
            value=payload.get("value"),
            numeric_value=(
                float(payload["numeric_value"])
                if payload.get("numeric_value") is not None
                else None
            ),
            categorical_value=payload.get("categorical_value"),
            boolean_value=payload.get("boolean_value"),
            value_type=MeasureValueType(value_type) if value_type else None,
            measure_label=payload.get("measure_label"),
            categories=[str(item) for item in payload.get("categories") or []],
            comment=payload.get("comment"),
            correction_output=payload.get("correction_output"),
            metadata=dict(payload.get("metadata") or {}),
            actor_user_id=payload.get("actor_user_id"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class AnnotationQueueDTO:
    id: str
    name: str
    description: str | None
    target_type: EvaluationTargetType
    measure_ids: list[str]
    status: AnnotationQueueStatus
    measures: list[dict[str, Any]]
    counts: dict[str, int]
    created_by: str | None
    updated_by: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationQueueDTO:
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            description=payload.get("description"),
            target_type=EvaluationTargetType(payload.get("target_type")),
            measure_ids=[str(item) for item in payload.get("measure_ids") or []],
            status=AnnotationQueueStatus(
                payload.get("status", AnnotationQueueStatus.ACTIVE.value)
            ),
            measures=[dict(item) for item in payload.get("measures") or []],
            counts={
                str(key): int(value)
                for key, value in dict(payload.get("counts") or {}).items()
            },
            created_by=payload.get("created_by"),
            updated_by=payload.get("updated_by"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class AnnotationQueueItemDTO:
    id: str
    queue_id: str
    target_type: EvaluationTargetType
    target_id: str
    target_snapshot: dict[str, Any]
    status: AnnotationQueueItemStatus
    assigned_user_id: str | None
    note: str | None
    score_record_ids: list[str]
    created_by: str | None
    completed_by: str | None
    completed_at: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationQueueItemDTO:
        return cls(
            id=str(payload.get("id", "")),
            queue_id=str(payload.get("queue_id", "")),
            target_type=EvaluationTargetType(payload.get("target_type")),
            target_id=str(payload.get("target_id", "")),
            target_snapshot=dict(payload.get("target_snapshot") or {}),
            status=AnnotationQueueItemStatus(
                payload.get("status", AnnotationQueueItemStatus.PENDING.value)
            ),
            assigned_user_id=payload.get("assigned_user_id"),
            note=payload.get("note"),
            score_record_ids=[
                str(item) for item in payload.get("score_record_ids") or []
            ],
            created_by=payload.get("created_by"),
            completed_by=payload.get("completed_by"),
            completed_at=payload.get("completed_at"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )


@dataclass(frozen=True)
class EvaluatorListResponse:
    items: list[EvaluatorDefinitionDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluatorListResponse:
        return cls(
            items=[
                EvaluatorDefinitionDTO.from_dict(item)
                for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class EvaluatorRunListResponse:
    items: list[EvaluatorRunDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluatorRunListResponse:
        return cls(
            items=[
                EvaluatorRunDTO.from_dict(item) for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class ScoreRecordListResponse:
    items: list[ScoreRecordDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScoreRecordListResponse:
        return cls(
            items=[
                ScoreRecordDTO.from_dict(item) for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class AnnotationQueueListResponse:
    items: list[AnnotationQueueDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationQueueListResponse:
        return cls(
            items=[
                AnnotationQueueDTO.from_dict(item)
                for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )


@dataclass(frozen=True)
class AnnotationQueueItemListResponse:
    items: list[AnnotationQueueItemDTO]
    pagination: PaginationInfo

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationQueueItemListResponse:
        return cls(
            items=[
                AnnotationQueueItemDTO.from_dict(item)
                for item in payload.get("items") or []
            ],
            pagination=PaginationInfo.from_dict(payload.get("pagination") or {}),
        )
