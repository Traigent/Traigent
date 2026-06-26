"""DTOs for the Traigent evaluation operations client."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from traigent.prompts.dtos import PaginationInfo


def _coalesce_not_none(candidate: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return value
    return None


def _require_target_type(payload: dict[str, Any]) -> EvaluationTargetType:
    raw = payload.get("target_type")
    if raw is None or raw == "":
        raise ValueError("payload is missing required field 'target_type'")
    return EvaluationTargetType(str(raw))


class MeasureValueType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class EvaluationTargetType(StrEnum):
    CONFIGURATION_RUN = "configuration_run"
    EXPERIMENT_RUN = "experiment_run"
    OBSERVABILITY_OBSERVATION = "observability_observation"
    OBSERVABILITY_TRACE = "observability_trace"


class ScoreSource(StrEnum):
    MANUAL = "manual"
    EVALUATOR = "evaluator"


class EvaluatorRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnnotationQueueStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class AnnotationQueueItemStatus(StrEnum):
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
            target_type=_require_target_type(payload),
            target_id=str(payload.get("target_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type.value,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class RecommendedEvaluatorSpecDTO:
    evaluator_key: str
    display_name: str
    measure_key: str
    target_type: str
    target_field: str
    priority: str
    rationale: str
    config: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecommendedEvaluatorSpecDTO:
        return cls(
            evaluator_key=str(payload.get("evaluator_key", "")),
            display_name=str(payload.get("display_name", "")),
            measure_key=str(payload.get("measure_key", "")),
            target_type=str(payload.get("target_type", "")),
            target_field=str(payload.get("target_field", "")),
            priority=str(payload.get("priority", "")),
            rationale=str(payload.get("rationale", "")),
            config=dict(payload.get("config") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evaluator_key": self.evaluator_key,
            "display_name": self.display_name,
            "measure_key": self.measure_key,
            "target_type": self.target_type,
            "target_field": self.target_field,
            "priority": self.priority,
            "rationale": self.rationale,
            "config": dict(self.config),
        }
        return payload


@dataclass(frozen=True)
class RecommendedEvaluatorPlanDTO:
    plan_id: str
    spec_version: str
    status: str
    evaluation_dataset_id: str
    source_trace_id: str
    evaluators: list[RecommendedEvaluatorSpecDTO]
    execution: dict[str, Any]
    warnings: list[str]
    provenance: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecommendedEvaluatorPlanDTO:
        return cls(
            plan_id=str(payload.get("plan_id", "")),
            spec_version=str(payload.get("spec_version", "")),
            status=str(payload.get("status", "")),
            evaluation_dataset_id=str(payload.get("evaluation_dataset_id", "")),
            source_trace_id=str(payload.get("source_trace_id", "")),
            evaluators=[
                RecommendedEvaluatorSpecDTO.from_dict(item)
                for item in payload.get("evaluators") or []
            ],
            execution=dict(payload.get("execution") or {}),
            warnings=[str(item) for item in payload.get("warnings") or []],
            provenance=dict(payload.get("provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "spec_version": self.spec_version,
            "status": self.status,
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "source_trace_id": self.source_trace_id,
            "evaluators": [item.to_dict() for item in self.evaluators],
            "execution": dict(self.execution),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class EvaluationDatasetExampleCandidateDTO:
    evaluation_dataset_id: str
    source_trace_id: str
    input_text: str
    expected_output: str | None
    metadata: dict[str, Any]
    lineage: dict[str, Any]
    recommended_evaluator_plan: RecommendedEvaluatorPlanDTO

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationDatasetExampleCandidateDTO:
        return cls(
            evaluation_dataset_id=str(payload.get("evaluation_dataset_id", "")),
            source_trace_id=str(payload.get("source_trace_id", "")),
            input_text=str(payload.get("input_text", "")),
            expected_output=(
                str(payload["expected_output"])
                if payload.get("expected_output") is not None
                else None
            ),
            metadata=dict(payload.get("metadata") or {}),
            lineage=dict(payload.get("lineage") or {}),
            recommended_evaluator_plan=RecommendedEvaluatorPlanDTO.from_dict(
                payload.get("recommended_evaluator_plan") or {}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "source_trace_id": self.source_trace_id,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "metadata": dict(self.metadata),
            "lineage": dict(self.lineage),
            "recommended_evaluator_plan": self.recommended_evaluator_plan.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationDatasetExampleFromTraceDTO(EvaluationDatasetExampleCandidateDTO):
    example_id: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationDatasetExampleFromTraceDTO:
        candidate = EvaluationDatasetExampleCandidateDTO.from_dict(payload)
        return cls(
            evaluation_dataset_id=candidate.evaluation_dataset_id,
            source_trace_id=candidate.source_trace_id,
            input_text=candidate.input_text,
            expected_output=candidate.expected_output,
            metadata=candidate.metadata,
            lineage=candidate.lineage,
            recommended_evaluator_plan=candidate.recommended_evaluator_plan,
            example_id=str(payload.get("example_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["example_id"] = self.example_id
        return payload


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
            target_type=_require_target_type(payload),
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
            target_type=_require_target_type(payload),
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
            target_type=_require_target_type(payload),
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
    # tenant_id / project_id are present in the backend ScoreRecord.to_dict()
    # runtime shape (src/models/score_record.py) — the dict returned by
    # create_manual_score and carried in complete_annotation_queue_item's
    # `scores`. Kept so the DTO is a strict superset of the prior raw dict (#1444).
    tenant_id: str | None
    project_id: str | None
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
            tenant_id=payload.get("tenant_id"),
            project_id=payload.get("project_id"),
            measure_id=str(payload.get("measure_id", "")),
            evaluator_run_id=payload.get("evaluator_run_id"),
            target_type=_require_target_type(payload),
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
            target_type=_require_target_type(payload),
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
    # tenant_id / project_id are present in the backend AnnotationQueueItem.to_dict()
    # runtime shape (src/models/annotation_queue_item.py). They are carried so the
    # DTO is a strict superset of the raw dict previously returned by
    # add_annotation_queue_items / complete_annotation_queue_item (#1444).
    tenant_id: str | None
    project_id: str | None
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
            tenant_id=payload.get("tenant_id"),
            project_id=payload.get("project_id"),
            queue_id=str(payload.get("queue_id", "")),
            target_type=_require_target_type(payload),
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


@dataclass(frozen=True)
class AnnotationQueueItemCreateResultDTO:
    """Return type for :meth:`EvaluationClient.add_annotation_queue_items`.

    Matches the backend response shape:
    ``{"queue_id": str, "created_count": int, "items": [<AnnotationQueueItem>]}``.
    Parity with JS ``AnnotationQueueItemCreateResult`` (AnnotationQueueItemCreateResultSchema).
    """

    queue_id: str
    created_count: int
    items: list[AnnotationQueueItemDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationQueueItemCreateResultDTO:
        return cls(
            queue_id=str(payload.get("queue_id", "")),
            created_count=int(payload.get("created_count", 0)),
            items=[
                AnnotationQueueItemDTO.from_dict(item)
                for item in payload.get("items") or []
            ],
        )


@dataclass(frozen=True)
class AnnotationQueueItemCompleteResultDTO:
    """Return type for :meth:`EvaluationClient.complete_annotation_queue_item`.

    Matches the backend response shape:
    ``{"queue_id": str, "item": <AnnotationQueueItem>, "scores": [<ScoreRecord>]}``.
    Parity with JS ``AnnotationQueueItemCompleteResult`` (AnnotationQueueItemCompleteResultSchema).
    """

    queue_id: str
    item: AnnotationQueueItemDTO
    scores: list[ScoreRecordDTO]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnotationQueueItemCompleteResultDTO:
        return cls(
            queue_id=str(payload.get("queue_id", "")),
            item=AnnotationQueueItemDTO.from_dict(payload.get("item") or {}),
            scores=[ScoreRecordDTO.from_dict(s) for s in payload.get("scores") or []],
        )


@dataclass(frozen=True)
class TypedMeasureDTO:
    """Return type for :meth:`EvaluationClient.create_typed_measure` and
    :meth:`EvaluationClient.update_typed_measure`.

    Mirrors the backend ``Measure.to_dict()`` shape.  The ``measure_id`` field is
    a backward-compatibility alias for ``id`` preserved by the backend.
    """

    id: str
    measure_id: str
    label: str
    measure_type: str
    version: str
    value_type: str
    is_custom: bool
    tenant_id: str | None
    owner_user_id: str | None
    project_id: str | None
    description: str | None
    category: str | None
    evaluation_method: str | None
    target_aspect: str | None
    inverse: bool | None
    unit: str | None
    criteria: str | None
    linked_evaluator_count: int
    domain: list[float]
    agent_types: list[dict[str, Any]]
    python_packages: list[Any]
    measure_parameters: dict[str, Any]
    categories: list[str]
    target_types: list[str]
    allowed_score_sources: list[str]
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TypedMeasureDTO:
        return cls(
            id=str(payload.get("id", "")),
            measure_id=str(payload.get("measure_id") or payload.get("id", "")),
            label=str(payload.get("label", "")),
            measure_type=str(payload.get("measure_type", "")),
            version=str(payload.get("version", "1.0.0")),
            value_type=str(payload.get("value_type") or "numeric"),
            is_custom=bool(payload.get("is_custom", False)),
            tenant_id=payload.get("tenant_id"),
            owner_user_id=payload.get("owner_user_id"),
            project_id=payload.get("project_id"),
            description=payload.get("description"),
            category=payload.get("category"),
            evaluation_method=payload.get("evaluation_method"),
            target_aspect=payload.get("target_aspect"),
            inverse=payload.get("inverse"),
            unit=payload.get("unit"),
            criteria=payload.get("criteria"),
            linked_evaluator_count=int(payload.get("linked_evaluator_count", 0)),
            domain=list(payload.get("domain") or [0.0, 1.0]),
            agent_types=list(payload.get("agent_types") or []),
            python_packages=list(payload.get("python_packages") or []),
            measure_parameters=dict(payload.get("measure_parameters") or {}),
            categories=[str(c) for c in payload.get("categories") or []],
            target_types=[str(t) for t in payload.get("target_types") or []],
            allowed_score_sources=[
                str(s) for s in payload.get("allowed_score_sources") or []
            ],
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
        )
