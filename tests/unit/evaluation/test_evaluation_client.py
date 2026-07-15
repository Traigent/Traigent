from __future__ import annotations

from unittest.mock import patch
from urllib import error

import pytest

from traigent.evaluation import (
    AnnotationQueueItemCompleteResultDTO,
    AnnotationQueueItemCreateResultDTO,
    AnnotationQueueItemDTO,
    AnnotationQueueItemStatus,
    AnnotationQueueStatus,
    EvaluationClient,
    EvaluationTargetRefDTO,
    EvaluationTargetType,
    EvaluatorRunStatus,
    JudgeConfigDTO,
    MeasureValueType,
    ScoreRecordDTO,
    TypedMeasureDTO,
)
from traigent.utils.exceptions import AuthenticationError, TraigentConnectionError


@pytest.fixture(autouse=True)
def _online_backend(jwt_development_mode, monkeypatch):
    """These tests exercise the online client request path, so they opt out of
    the suite-wide TRAIGENT_OFFLINE_MODE=true default (#1068). The transport is
    mocked, so no real egress occurs; depends on jwt_development_mode so this
    override runs after it."""
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


def test_evaluation_client_lists_and_mutates_evaluators():
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        if method == "GET" and path.startswith("/api/v1beta/evaluators?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "evaluator_1",
                            "name": "Trace Quality Judge",
                            "description": "Shared judge",
                            "measure_id": "quality_score",
                            "primary_measure_id": "quality_score",
                            "target_type": "observability_trace",
                            "judge_config": {
                                "instructions": "Judge quality",
                                "model_id": "gpt-4.1-mini",
                                "context_type": "none",
                                "parameters": {
                                    "model_id": "gpt-4.1-mini",
                                    "temperature": 0.2,
                                },
                            },
                            "sampling_rate": 1.0,
                            "target_filters": {},
                            "is_active": True,
                            "measure": {
                                "id": "quality_score",
                                "label": "Quality Score",
                                "value_type": "numeric",
                                "categories": [],
                                "target_types": ["observability_trace"],
                            },
                            "created_at": "2026-03-10T12:00:00+00:00",
                            "updated_at": "2026-03-10T12:05:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }

        if method == "PATCH":
            return {
                "data": {
                    "id": "evaluator_1",
                    "name": "Trace Quality Judge",
                    "description": "Updated judge",
                    "measure_id": "quality_score",
                    "primary_measure_id": "quality_score",
                    "target_type": "observability_trace",
                    "judge_config": {
                        "instructions": "Judge quality",
                        "model_id": "gpt-4.1-mini",
                        "context_type": "none",
                    },
                    "sampling_rate": 0.5,
                    "target_filters": {},
                    "is_active": False,
                    "measure": None,
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "updated_at": "2026-03-10T12:10:00+00:00",
                }
            }

        return {
            "data": {
                "id": "evaluator_1",
                "name": "Trace Quality Judge",
                "description": "Shared judge",
                "measure_id": "quality_score",
                "primary_measure_id": "quality_score",
                "target_type": "observability_trace",
                "judge_config": {
                    "instructions": "Judge quality",
                    "model_id": "gpt-4.1-mini",
                    "context_type": "none",
                    "parameters": {
                        "model_id": "gpt-4.1-mini",
                        "temperature": 0.2,
                        "max_tokens": 256,
                    },
                },
                "sampling_rate": 1.0,
                "target_filters": {"environment": "production"},
                "is_active": True,
                "measure": {
                    "id": "quality_score",
                    "label": "Quality Score",
                    "value_type": "numeric",
                    "categories": [],
                    "target_types": ["observability_trace"],
                },
                "created_at": "2026-03-10T12:00:00+00:00",
                "updated_at": "2026-03-10T12:05:00+00:00",
            }
        }

    client = EvaluationClient(request_sender=request_sender)
    judge_config = JudgeConfigDTO(
        instructions="Judge quality",
        model_id="gpt-4.1-mini",
        parameters={"temperature": 0.2, "max_tokens": 256},
    )

    listed = client.list_evaluators(
        search="Trace",
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE,
    )
    created = client.create_evaluator(
        name="Trace Quality Judge",
        measure_id="quality_score",
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE,
        judge_config=judge_config,
        description="Shared judge",
        target_filters={"environment": "production"},
    )
    detail = client.get_evaluator("evaluator_1")
    updated = client.update_evaluator(
        "evaluator_1",
        description="Updated judge",
        sampling_rate=0.5,
        is_active=False,
    )

    assert calls[0] == (
        "GET",
        "/api/v1beta/evaluators?page=1&per_page=20&search=Trace&target_type=observability_trace",
        None,
    )
    assert calls[1][0] == "POST"
    assert calls[1][1] == "/api/v1beta/evaluators"
    assert calls[1][2]["judge_config"]["parameters"]["model_id"] == "gpt-4.1-mini"
    assert calls[2] == ("GET", "/api/v1beta/evaluators/evaluator_1", None)
    assert calls[3] == (
        "PATCH",
        "/api/v1beta/evaluators/evaluator_1",
        {"description": "Updated judge", "sampling_rate": 0.5, "is_active": False},
    )
    assert listed.items[0].target_type == EvaluationTargetType.OBSERVABILITY_TRACE
    assert created.measure["value_type"] == MeasureValueType.NUMERIC.value
    assert detail.judge_config.parameters["model_id"] == "gpt-4.1-mini"
    assert updated.is_active is False


def test_evaluation_client_scores_polling_and_benchmark_helpers(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []
    run_statuses = iter(["pending", "running", "completed"])

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        if path == "/api/v1beta/evaluators/evaluator_1/execute":
            return {
                "data": {
                    "id": "evalrun_1",
                    "evaluator_id": "evaluator_1",
                    "measure_id": "quality_score",
                    "target_type": "observability_trace",
                    "target_id": "trace_1",
                    "status": "pending",
                    "source": "evaluator",
                    "input_snapshot": {},
                    "output_payload": None,
                    "score_record_ids": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": 0,
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "updated_at": "2026-03-10T12:00:00+00:00",
                }
            }
        if path == "/api/v1beta/evaluator-runs/evalrun_1":
            status = next(run_statuses)
            return {
                "data": {
                    "id": "evalrun_1",
                    "evaluator_id": "evaluator_1",
                    "measure_id": "quality_score",
                    "target_type": "observability_trace",
                    "target_id": "trace_1",
                    "status": status,
                    "source": "evaluator",
                    "input_snapshot": {},
                    "output_payload": (
                        {"numeric_value": 0.9} if status == "completed" else None
                    ),
                    "score_record_ids": ["score_1"] if status == "completed" else [],
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.002,
                    "latency_ms": 120,
                    "observability_trace_id": (
                        "trace_evalrun_1" if status == "completed" else None
                    ),
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "updated_at": "2026-03-10T12:01:00+00:00",
                }
            }
        if path.startswith("/api/v1beta/scores?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "score_1",
                            "measure_id": "quality_score",
                            "target_type": "observability_trace",
                            "target_id": "trace_1",
                            "source": "manual",
                            "value": "pass",
                            "numeric_value": None,
                            "categorical_value": "pass",
                            "boolean_value": None,
                            "value_type": "categorical",
                            "measure_label": "Quality Score",
                            "categories": ["pass", "needs_review", "fail"],
                            "comment": "Reviewed",
                            "metadata": {"reviewer": "alice"},
                            "created_at": "2026-03-10T12:03:00+00:00",
                            "updated_at": "2026-03-10T12:03:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        return {
            "data": {
                "id": "score_1",
                "measure_id": "quality_score",
                "target_type": "observability_trace",
                "target_id": "trace_1",
                "source": "manual",
                "value": "pass",
                "numeric_value": None,
                "categorical_value": "pass",
                "boolean_value": None,
                "value_type": "categorical",
                "measure_label": "Quality Score",
                "categories": ["pass", "needs_review", "fail"],
                "comment": "Reviewed",
                "metadata": {"reviewer": "alice"},
                "created_at": "2026-03-10T12:03:00+00:00",
                "updated_at": "2026-03-10T12:03:00+00:00",
            }
        }

    monkeypatch.setattr("traigent.evaluation.client.time.sleep", lambda _: None)

    client = EvaluationClient(request_sender=request_sender)
    target = EvaluationTargetRefDTO(
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE,
        target_id="trace_1",
    )

    benchmark_payload = {
        "instructions": "Judge the trace response",
        "model": "gpt-4.1-mini",
        "context": {"type": "document", "source": "rubric.md"},
        "modelParameters": [
            {
                "temperature": 0.3,
                "maxTokens": 200,
                "topP": 0.0,
                "frequencyPenalty": 0.0,
                "presencePenalty": 0.0,
                "repetitionPenalty": 0.0,
            }
        ],
    }
    judge_config = client.judge_config_from_benchmark_payload(benchmark_payload)
    execute_result = client.execute_evaluator("evaluator_1", target=target)
    completed_run = client.wait_for_evaluator_run(
        "evalrun_1", max_attempts=3, interval_seconds=0.01
    )
    created_score = client.create_score(
        measure_id="quality_score",
        target=target,
        categorical_value="pass",
        comment="Reviewed",
        metadata={"reviewer": "alice"},
    )
    listed_scores = client.list_scores(target=target)

    assert judge_config is not None
    assert judge_config.parameters["model_id"] == "gpt-4.1-mini"
    assert judge_config.parameters["top_p"] == 0.0
    assert judge_config.parameters["frequency_penalty"] == 0.0
    benchmark_round_trip = client.judge_config_to_benchmark_payload(judge_config)
    assert benchmark_round_trip["context_type"] == "document"
    assert benchmark_round_trip["context"]["type"] == "document"
    assert benchmark_round_trip["model"] == "gpt-4.1-mini"
    assert benchmark_round_trip["modelParameters"][0]["maxTokens"] == 200
    assert benchmark_round_trip["modelParameters"][0]["topP"] == 0.0
    assert benchmark_round_trip["modelParameters"][0]["frequencyPenalty"] == 0.0
    assert execute_result.status == EvaluatorRunStatus.PENDING
    assert completed_run.status == EvaluatorRunStatus.COMPLETED
    assert completed_run.score_record_ids == ["score_1"]
    assert created_score.categorical_value == "pass"
    assert listed_scores.items[0].categories == ["pass", "needs_review", "fail"]
    assert calls[0] == (
        "POST",
        "/api/v1beta/evaluators/evaluator_1/execute",
        {"target_type": "observability_trace", "target_id": "trace_1"},
    )


def test_evaluation_client_updates_typed_measures_and_maps_errors(monkeypatch):
    measure_calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        measure_calls.append((method, path, payload))
        return {
            "id": payload.get("id", "measure_quality"),
            "label": payload.get("label", "Quality Score"),
            "value_type": payload.get("value_type", "numeric"),
            "categories": payload.get("categories", []),
            "target_types": payload.get("target_types", ["observability_trace"]),
        }

    client = EvaluationClient(request_sender=request_sender)
    created_measure = client.create_typed_measure(
        {
            "label": "Quality Score",
            "measure_type": "quality",
            "category": "Quality",
            "value_type": "categorical",
            "categories": ["pass", "fail"],
            "target_types": ["observability_trace"],
        }
    )
    updated_measure = client.update_typed_measure(
        "measure_quality",
        {
            "label": "Quality Score",
            "measure_type": "quality",
            "category": "Quality",
            "value_type": "boolean",
            "target_types": ["observability_trace"],
        },
    )

    assert measure_calls[0][1] == "/api/v1beta/measures"
    assert measure_calls[1][1] == "/api/v1beta/measures/measure_quality"
    assert created_measure.value_type == "categorical"
    assert updated_measure.value_type == "boolean"

    auth_client = EvaluationClient()
    monkeypatch.setattr(
        "traigent.evaluation.client.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            error.HTTPError(
                url="https://example.com",
                code=401,
                msg="unauthorized",
                hdrs=None,
                fp=None,
            )
        ),
    )
    with pytest.raises(AuthenticationError):
        auth_client.get_evaluator("blocked")

    network_client = EvaluationClient()
    monkeypatch.setattr(
        "traigent.evaluation.client.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error.URLError("offline")),
    )
    with pytest.raises(TraigentConnectionError):
        network_client.get_evaluator("offline")


def test_evaluation_client_handles_backfill_retry_and_annotation_queues():
    calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        if path == "/api/v1beta/evaluators/evaluator_1/backfill":
            return {
                "data": {
                    "evaluator_id": "evaluator_1",
                    "target_type": "observability_trace",
                    "submitted_count": 1,
                    "skipped_count": 1,
                    "skipped_target_ids": ["trace_existing"],
                    "items": [
                        {
                            "id": "evalrun_backfill_1",
                            "evaluator_id": "evaluator_1",
                            "measure_id": "quality_score",
                            "target_type": "observability_trace",
                            "target_id": "trace_new",
                            "status": "pending",
                            "source": "backfill",
                            "input_snapshot": {},
                            "output_payload": None,
                            "score_record_ids": [],
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "cost_usd": 0.0,
                            "latency_ms": 0,
                            "created_at": "2026-03-10T12:00:00+00:00",
                            "updated_at": "2026-03-10T12:00:00+00:00",
                        }
                    ],
                }
            }
        if path == "/api/v1beta/evaluator-runs/evalrun_1/retry":
            return {
                "data": {
                    "id": "evalrun_retry_1",
                    "evaluator_id": "evaluator_1",
                    "measure_id": "quality_score",
                    "target_type": "observability_trace",
                    "target_id": "trace_1",
                    "status": "pending",
                    "source": "retry",
                    "input_snapshot": {},
                    "output_payload": None,
                    "score_record_ids": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": 0,
                    "created_at": "2026-03-10T12:05:00+00:00",
                    "updated_at": "2026-03-10T12:05:00+00:00",
                }
            }
        if method == "GET" and path.startswith("/api/v1beta/annotation-queues?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "queue_1",
                            "name": "Human trace review",
                            "description": "Manual queue",
                            "target_type": "observability_trace",
                            "measure_ids": ["quality_score"],
                            "status": "active",
                            "measures": [
                                {
                                    "id": "quality_score",
                                    "label": "Quality Score",
                                    "value_type": "categorical",
                                    "categories": ["pass", "fail"],
                                    "target_types": ["observability_trace"],
                                }
                            ],
                            "created_at": "2026-03-10T12:00:00+00:00",
                            "updated_at": "2026-03-10T12:00:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        if path == "/api/v1beta/annotation-queues":
            return {
                "data": {
                    "id": "queue_1",
                    "name": "Human trace review",
                    "description": "Manual queue",
                    "target_type": "observability_trace",
                    "measure_ids": ["quality_score"],
                    "status": "active",
                    "measures": [
                        {
                            "id": "quality_score",
                            "label": "Quality Score",
                            "value_type": "categorical",
                            "categories": ["pass", "fail"],
                            "target_types": ["observability_trace"],
                        }
                    ],
                    "created_at": "2026-03-10T12:00:00+00:00",
                    "updated_at": "2026-03-10T12:00:00+00:00",
                }
            }
        if method == "GET" and path.startswith(
            "/api/v1beta/annotation-queues/queue_1/items?"
        ):
            return {
                "data": {
                    "items": [
                        {
                            "id": "queueitem_1",
                            "queue_id": "queue_1",
                            "target_type": "observability_trace",
                            "target_id": "trace_1",
                            "target_snapshot": {"output": {"answer": "Paris"}},
                            "status": "pending",
                            "assigned_user_id": "annotator_1",
                            "score_record_ids": [],
                            "created_at": "2026-03-10T12:02:00+00:00",
                            "updated_at": "2026-03-10T12:02:00+00:00",
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "per_page": 20,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": False,
                    },
                }
            }
        if path == "/api/v1beta/annotation-queues/queue_1/items":
            return {
                "data": {
                    "queue_id": "queue_1",
                    "created_count": 1,
                    "items": [
                        {
                            "id": "queueitem_1",
                            "queue_id": "queue_1",
                            "target_type": "observability_trace",
                            "target_id": "trace_1",
                            "target_snapshot": {"output": {"answer": "Paris"}},
                            "status": "pending",
                            "assigned_user_id": "annotator_1",
                            "score_record_ids": [],
                            "created_at": "2026-03-10T12:02:00+00:00",
                            "updated_at": "2026-03-10T12:02:00+00:00",
                        }
                    ],
                }
            }
        if path == "/api/v1beta/annotation-queues/queue_1/next":
            return {
                "data": {
                    "id": "queueitem_1",
                    "queue_id": "queue_1",
                    "target_type": "observability_trace",
                    "target_id": "trace_1",
                    "target_snapshot": {"output": {"answer": "Paris"}},
                    "status": "pending",
                    "assigned_user_id": "annotator_1",
                    "score_record_ids": [],
                    "created_at": "2026-03-10T12:02:00+00:00",
                    "updated_at": "2026-03-10T12:02:00+00:00",
                }
            }
        if path == "/api/v1beta/annotation-queues/items/queueitem_1":
            return {
                "data": {
                    "id": "queueitem_1",
                    "queue_id": "queue_1",
                    "target_type": "observability_trace",
                    "target_id": "trace_1",
                    "target_snapshot": {"output": {"answer": "Paris"}},
                    "status": "in_progress",
                    "assigned_user_id": "annotator_1",
                    "score_record_ids": [],
                    "created_at": "2026-03-10T12:02:00+00:00",
                    "updated_at": "2026-03-10T12:03:00+00:00",
                }
            }
        if path == "/api/v1beta/annotation-queues/items/queueitem_1/complete":
            return {
                "data": {
                    "queue_id": "queue_1",
                    "item": {
                        "id": "queueitem_1",
                        "queue_id": "queue_1",
                        "target_type": "observability_trace",
                        "target_id": "trace_1",
                        "status": "completed",
                        "score_record_ids": ["score_1"],
                    },
                    "scores": [
                        {
                            "id": "score_1",
                            "measure_id": "quality_score",
                            "target_type": "observability_trace",
                            "target_id": "trace_1",
                            "source": "manual",
                            "categorical_value": "pass",
                            "value": "pass",
                        }
                    ],
                }
            }
        return {"data": {"items": []}}

    client = EvaluationClient(request_sender=request_sender)
    backfill = client.backfill_evaluator(
        "evaluator_1",
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE,
        filters={"search": "trace"},
        skip_existing_scores=True,
    )
    retried = client.retry_evaluator_run("evalrun_1")
    queues = client.list_annotation_queues(
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE,
        status=AnnotationQueueStatus.ACTIVE,
    )
    queue = client.create_annotation_queue(
        name="Human trace review",
        target_type=EvaluationTargetType.OBSERVABILITY_TRACE,
        measure_ids=["quality_score"],
    )
    added_items = client.add_annotation_queue_items(
        "queue_1",
        targets=[
            {
                "target_type": "observability_trace",
                "target_id": "trace_1",
            }
        ],
        assigned_user_id="annotator_1",
    )
    queue_items = client.list_annotation_queue_items(
        "queue_1",
        status=AnnotationQueueItemStatus.PENDING,
        assigned_user_id="annotator_1",
    )
    next_item = client.get_next_annotation_queue_item("queue_1")
    updated_item = client.update_annotation_queue_item(
        "queueitem_1",
        status=AnnotationQueueItemStatus.IN_PROGRESS,
        note="Investigating",
    )
    completed = client.complete_annotation_queue_item(
        "queueitem_1",
        scores=[{"measure_id": "quality_score", "categorical_value": "pass"}],
        note="Complete + next",
    )

    assert backfill.submitted_count == 1
    assert backfill.skipped_target_ids == ["trace_existing"]
    assert retried.source == "retry"
    assert queues.items[0].status == AnnotationQueueStatus.ACTIVE
    assert queue.measure_ids == ["quality_score"]
    assert added_items.queue_id == "queue_1"
    assert added_items.created_count == 1
    assert added_items.items[0].target_id == "trace_1"
    assert queue_items.items[0].assigned_user_id == "annotator_1"
    assert next_item is not None
    assert next_item.id == "queueitem_1"
    assert updated_item.status == AnnotationQueueItemStatus.IN_PROGRESS
    assert completed.queue_id == "queue_1"
    assert completed.item.status == AnnotationQueueItemStatus.COMPLETED

    assert calls[0] == (
        "POST",
        "/api/v1beta/evaluators/evaluator_1/backfill",
        {
            "filters": {"search": "trace"},
            "limit": 25,
            "skip_existing_scores": True,
            "target_type": "observability_trace",
        },
    )
    assert calls[1] == ("POST", "/api/v1beta/evaluator-runs/evalrun_1/retry", {})


def test_evaluation_client_trace_to_dataset_example_helpers():
    calls: list[tuple[str, str, dict | None]] = []
    recommended_plan = {
        "plan_id": "evalplan_012345abcdef",
        "spec_version": "2026-05-21.v1",
        "status": "operator_review_required",
        "evaluation_dataset_id": "eval_dataset_001",
        "source_trace_id": "trace_001",
        "evaluators": [
            {
                "evaluator_key": "expected_output_alignment",
                "display_name": "Expected Output Alignment",
                "measure_key": "expected_output_alignment",
                "target_type": "evaluation_dataset_example",
                "target_field": "expected_output",
                "priority": "required",
                "rationale": "Judge alignment.",
            }
        ],
        "execution": {
            "mode": "manual_review_before_run",
            "can_autorun": False,
            "suggested_sample_size": 1,
        },
        "warnings": ["Review before running evaluators."],
        "provenance": {
            "source": "trace_to_evaluation_dataset_example",
            "signals": ["trace_io", "expected_output_present"],
        },
    }

    def request_sender(method: str, path: str, payload: dict | None):
        calls.append((method, path, payload))
        data = {
            "evaluation_dataset_id": "eval_dataset_001",
            "source_trace_id": "trace_001",
            "input_text": "What is the release gate?",
            "expected_output": "operator review",
            "metadata": {"source_trace_id": "trace_001"},
            "lineage": {"source_trace_id": "trace_001"},
            "recommended_evaluator_plan": recommended_plan,
        }
        if path.endswith("/examples/from-trace"):
            data["example_id"] = "example_001"
        return {"data": data}

    client = EvaluationClient(request_sender=request_sender)

    candidate = client.preview_evaluation_dataset_example_from_trace(
        "trace_001",
        evaluation_dataset_id="eval_dataset_001",
        input_text="What is the release gate?",
        corrected_expected_output="operator review",
    )
    created = client.create_evaluation_dataset_example_from_trace(
        "eval_dataset_001",
        source_trace_id="trace_001",
        input_text=candidate.input_text,
        corrected_expected_output=candidate.expected_output,
    )

    assert calls[0] == (
        "POST",
        "/api/v1beta/observability/traces/trace_001/evaluation-dataset-example-candidate",
        {
            "evaluation_dataset_id": "eval_dataset_001",
            "input_text": "What is the release gate?",
            "corrected_expected_output": "operator review",
        },
    )
    assert calls[1] == (
        "POST",
        "/api/v1beta/evaluation-datasets/eval_dataset_001/examples/from-trace",
        {
            "source_trace_id": "trace_001",
            "input_text": "What is the release gate?",
            "corrected_expected_output": "operator review",
        },
    )
    assert candidate.recommended_evaluator_plan.plan_id == "evalplan_012345abcdef"
    assert candidate.recommended_evaluator_plan.spec_version == "2026-05-21.v1"
    assert candidate.recommended_evaluator_plan.evaluators[0].priority == "required"
    assert candidate.recommended_evaluator_plan.evaluators[0].to_dict()["config"] == {}
    assert created.example_id == "example_001"

    with pytest.raises(ValueError, match="mutually exclusive"):
        client.preview_evaluation_dataset_example_from_trace(
            "trace_001",
            evaluation_dataset_id="eval_dataset_001",
            expected_output="raw",
            corrected_expected_output="corrected",
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        client.create_evaluation_dataset_example_from_trace(
            "eval_dataset_001",
            source_trace_id="trace_001",
            expected_output="raw",
            corrected_expected_output="corrected",
        )


def test_get_next_annotation_queue_item_returns_none_for_empty_queue():
    client = EvaluationClient(
        request_sender=lambda method, path, payload: {"data": None}
    )

    assert client.get_next_annotation_queue_item("queue_1") is None


# ---------------------------------------------------------------------------
# Issue #1444 — typed DTO surface for previously-raw-dict methods
# ---------------------------------------------------------------------------


def test_add_annotation_queue_items_returns_typed_dto():
    """add_annotation_queue_items must return AnnotationQueueItemCreateResultDTO."""

    def request_sender(method: str, path: str, payload: dict | None):
        return {
            "data": {
                "queue_id": "q_abc",
                "created_count": 2,
                "items": [
                    {
                        "id": "item_1",
                        # tenant_id/project_id are part of the backend
                        # AnnotationQueueItem.to_dict() runtime shape — must
                        # survive into the DTO (no data loss vs raw dict, #1444).
                        "tenant_id": "tenant_42",
                        "project_id": "project_7",
                        "queue_id": "q_abc",
                        "target_type": "observability_trace",
                        "target_id": "trace_x",
                        "target_snapshot": {},
                        "status": "pending",
                        "assigned_user_id": None,
                        "score_record_ids": [],
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "id": "item_2",
                        "tenant_id": "tenant_42",
                        "project_id": "project_7",
                        "queue_id": "q_abc",
                        "target_type": "observability_trace",
                        "target_id": "trace_y",
                        "target_snapshot": {},
                        "status": "pending",
                        "assigned_user_id": None,
                        "score_record_ids": [],
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    },
                ],
            }
        }

    client = EvaluationClient(request_sender=request_sender)
    result = client.add_annotation_queue_items(
        "q_abc",
        targets=[
            {"target_type": "observability_trace", "target_id": "trace_x"},
            {"target_type": "observability_trace", "target_id": "trace_y"},
        ],
    )

    # Must be the typed DTO, not a raw dict
    assert isinstance(result, AnnotationQueueItemCreateResultDTO)
    assert result.queue_id == "q_abc"
    assert result.created_count == 2
    assert len(result.items) == 2
    assert result.items[0].target_id == "trace_x"
    assert result.items[1].target_id == "trace_y"
    assert result.items[0].status == AnnotationQueueItemStatus.PENDING
    # Regression guard: tenant_id/project_id must NOT be dropped by the DTO.
    assert result.items[0].tenant_id == "tenant_42"
    assert result.items[0].project_id == "project_7"
    assert result.items[1].tenant_id == "tenant_42"
    assert result.items[1].project_id == "project_7"


def test_complete_annotation_queue_item_returns_typed_dto():
    """complete_annotation_queue_item must return AnnotationQueueItemCompleteResultDTO."""

    def request_sender(method: str, path: str, payload: dict | None):
        return {
            "data": {
                "queue_id": "q_abc",
                "item": {
                    "id": "item_1",
                    "tenant_id": "tenant_42",
                    "project_id": "project_7",
                    "queue_id": "q_abc",
                    "target_type": "observability_trace",
                    "target_id": "trace_x",
                    "target_snapshot": {},
                    "status": "completed",
                    "assigned_user_id": "user_1",
                    "score_record_ids": ["score_1"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:01:00+00:00",
                },
                "scores": [
                    {
                        "id": "score_1",
                        # tenant_id/project_id are part of the backend
                        # ScoreRecord.to_dict() runtime shape (create_manual_score)
                        # — must survive into the DTO (no data loss, #1444).
                        "tenant_id": "tenant_42",
                        "project_id": "project_7",
                        "measure_id": "quality_score",
                        "target_type": "observability_trace",
                        "target_id": "trace_x",
                        "source": "manual",
                        "numeric_value": 0.9,
                        "value": 0.9,
                    }
                ],
            }
        }

    client = EvaluationClient(request_sender=request_sender)
    result = client.complete_annotation_queue_item(
        "item_1",
        scores=[{"measure_id": "quality_score", "numeric_value": 0.9}],
    )

    # Must be the typed DTO, not a raw dict
    assert isinstance(result, AnnotationQueueItemCompleteResultDTO)
    assert result.queue_id == "q_abc"
    assert result.item.id == "item_1"
    assert result.item.status == AnnotationQueueItemStatus.COMPLETED
    assert result.item.score_record_ids == ["score_1"]
    assert len(result.scores) == 1
    assert result.scores[0].id == "score_1"
    assert result.scores[0].numeric_value == 0.9
    # Regression guard: tenant_id/project_id must NOT be dropped by the DTOs
    # (nested item AND nested scores).
    assert result.item.tenant_id == "tenant_42"
    assert result.item.project_id == "project_7"
    assert result.scores[0].tenant_id == "tenant_42"
    assert result.scores[0].project_id == "project_7"


def test_create_typed_measure_returns_typed_dto():
    """create_typed_measure must return TypedMeasureDTO with correct field access."""

    def request_sender(method: str, path: str, payload: dict | None):
        # Simulates the direct JSON response from the measures API (no data wrapper)
        return {
            "id": "measure_abc",
            "measure_id": "measure_abc",
            "label": "Accuracy",
            "measure_type": "accuracy",
            "version": "1.0.0",
            "value_type": "numeric",
            "is_custom": True,
            "categories": [],
            "target_types": ["observability_trace"],
            "allowed_score_sources": ["manual"],
            "domain": [0.0, 1.0],
            "agent_types": [],
            "python_packages": [],
            "measure_parameters": {},
            "linked_evaluator_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }

    client = EvaluationClient(request_sender=request_sender)
    result = client.create_typed_measure(
        {"label": "Accuracy", "measure_type": "accuracy"}
    )

    # Must be the typed DTO, not a raw dict
    assert isinstance(result, TypedMeasureDTO)
    assert result.id == "measure_abc"
    assert result.label == "Accuracy"
    assert result.measure_type == "accuracy"
    assert result.value_type == "numeric"
    assert result.is_custom is True
    assert result.domain == [0.0, 1.0]


def test_update_typed_measure_returns_typed_dto():
    """update_typed_measure must return TypedMeasureDTO with correct field access."""

    def request_sender(method: str, path: str, payload: dict | None):
        return {
            "id": "measure_abc",
            "measure_id": "measure_abc",
            "label": "Accuracy v2",
            "measure_type": "accuracy",
            "version": "1.1.0",
            "value_type": "numeric",
            "is_custom": True,
            "categories": [],
            "target_types": ["observability_trace"],
            "allowed_score_sources": ["manual", "evaluator"],
            "domain": [0.0, 1.0],
            "agent_types": [],
            "python_packages": [],
            "measure_parameters": {},
            "linked_evaluator_count": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        }

    client = EvaluationClient(request_sender=request_sender)
    result = client.update_typed_measure(
        "measure_abc",
        {"label": "Accuracy v2", "measure_type": "accuracy"},
    )

    assert isinstance(result, TypedMeasureDTO)
    assert result.id == "measure_abc"
    assert result.label == "Accuracy v2"
    assert result.version == "1.1.0"
    assert result.linked_evaluator_count == 1
    assert result.allowed_score_sources == ["manual", "evaluator"]


# ---------------------------------------------------------------------------
# Issue #1444 — no-data-loss: error field + forward-compat `extra` retention
# ---------------------------------------------------------------------------


def test_typed_measure_dto_carries_error_fallback_field():
    """TypedMeasureDTO must surface the backend Measure.to_dict() EXCEPTION-path
    `error` key (src/models/measure.py), not drop it."""
    payload = {
        "id": "measure_x",
        "measure_id": "measure_x",
        "tenant_id": "tenant_1",
        "owner_user_id": "owner_1",
        "project_id": "project_1",
        "label": "Broken Measure",
        "description": "partial",
        "error": "Failed to convert all fields",
    }

    dto = TypedMeasureDTO.from_dict(payload)

    assert dto.error == "Failed to convert all fields"
    # Happy-path payloads (no error key) leave it None.
    happy = TypedMeasureDTO.from_dict(
        {"id": "m", "label": "ok", "measure_type": "accuracy"}
    )
    assert happy.error is None


def test_annotation_queue_item_dto_retains_unknown_keys_in_extra():
    """A backend key the DTO does not model must survive into `.extra`, never
    be silently dropped (forward-compat guard, #1444)."""
    payload = {
        "id": "item_1",
        "tenant_id": "tenant_1",
        "project_id": "project_1",
        "queue_id": "q_1",
        "target_type": "observability_trace",
        "target_id": "trace_1",
        "status": "pending",
        # A future/unmodeled backend field:
        "future_backend_field": {"nested": 42},
    }

    dto = AnnotationQueueItemDTO.from_dict(payload)

    assert dto.extra == {"future_backend_field": {"nested": 42}}
    # Known keys must NOT leak into extra.
    assert "id" not in dto.extra
    assert "tenant_id" not in dto.extra


def test_score_record_dto_retains_unknown_keys_in_extra():
    """ScoreRecordDTO retains unmodeled backend keys in `.extra` (#1444)."""
    payload = {
        "id": "score_1",
        "tenant_id": "tenant_1",
        "project_id": "project_1",
        "measure_id": "quality_score",
        "target_type": "observability_trace",
        "target_id": "trace_1",
        "source": "manual",
        "numeric_value": 0.9,
        # Unmodeled future field:
        "data_type": "NUMERIC",
    }

    dto = ScoreRecordDTO.from_dict(payload)

    assert dto.extra == {"data_type": "NUMERIC"}
    assert "measure_id" not in dto.extra


def test_typed_measure_dto_retains_unknown_keys_in_extra():
    """TypedMeasureDTO retains unmodeled backend keys in `.extra` (#1444)."""
    payload = {
        "id": "measure_x",
        "label": "Accuracy",
        "measure_type": "accuracy",
        # Unmodeled future field:
        "experimental_weight": 0.5,
    }

    dto = TypedMeasureDTO.from_dict(payload)

    assert dto.extra == {"experimental_weight": 0.5}
    # The `error` field is a modeled key, so it never lands in extra.
    assert "error" not in dto.extra


def test_result_dtos_retain_unknown_top_level_keys_in_extra():
    """Both result DTOs retain unmodeled top-level result keys in `.extra` (#1444)."""
    create = AnnotationQueueItemCreateResultDTO.from_dict(
        {
            "queue_id": "q_1",
            "created_count": 0,
            "items": [],
            "request_id": "req_123",
        }
    )
    assert create.extra == {"request_id": "req_123"}

    complete = AnnotationQueueItemCompleteResultDTO.from_dict(
        {
            "queue_id": "q_1",
            "item": {
                "id": "item_1",
                "queue_id": "q_1",
                "target_type": "observability_trace",
                "target_id": "trace_1",
                "status": "completed",
            },
            "scores": [],
            "audit_token": "tok_456",
        }
    )
    assert complete.extra == {"audit_token": "tok_456"}


# --- SDK #1893: W3C traceparent injection on evaluation backend calls --------
# _request_json_sync builds headers per-request via config.build_headers()
# (a fresh dict) -- a safe per-request injection site (no cached headers).

import re  # noqa: E402
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from urllib import request as _urllib_request  # noqa: E402

from opentelemetry.sdk.trace import TracerProvider as _SdkTracerProvider  # noqa: E402

from traigent.evaluation.config import EvaluationConfig  # noqa: E402

_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-0[0-9a-f]$")


@contextmanager
def _recording_span():
    provider = _SdkTracerProvider()
    tracer = provider.get_tracer("test-sdk-1893-evaluation")
    with tracer.start_as_current_span("test-span") as span:
        yield span


def _header_ci(req, name):
    """Case-insensitive header lookup on a urllib Request (which capitalizes
    header names internally)."""
    for key, value in req.header_items():
        if key.lower() == name.lower():
            return value
    return None


class _FakeResponse:
    status = 200

    def read(self):
        return b'{"data": {"ok": true}}'

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(monkeypatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req, *args, **kwargs):
        captured["request"] = req
        return _FakeResponse()

    monkeypatch.setattr("traigent.evaluation.client.request.urlopen", fake_urlopen)
    return captured


def test_evaluation_request_headers_carry_traceparent(monkeypatch):
    captured = _capture_urlopen(monkeypatch)
    client = EvaluationClient()

    with _recording_span() as span:
        client._request_json_sync("GET", "/api/v1beta/ping", None)
        expected_trace_id = format(span.get_span_context().trace_id, "032x")

    req = captured["request"]
    tp = _header_ci(req, "traceparent")
    assert tp is not None
    assert _TRACEPARENT_RE.match(tp), tp
    assert tp.split("-")[1] == expected_trace_id


def test_evaluation_request_no_span_headers_byte_identical(monkeypatch):
    captured = _capture_urlopen(monkeypatch)
    client = EvaluationClient()

    client._request_json_sync("POST", "/api/v1beta/ping", {"a": 1})

    req = captured["request"]
    assert _header_ci(req, "traceparent") is None
    assert _header_ci(req, "tracestate") is None
    # Byte-identical to the pre-#1893 header set (built directly from config).
    control = _urllib_request.Request(
        f"{client.config.backend_origin}/api/v1beta/ping",
        headers=client.config.build_headers(),
        method="POST",
    )
    assert sorted(req.header_items()) == sorted(control.header_items())


def test_evaluation_request_caller_supplied_traceparent_not_overridden(monkeypatch):
    caller_tp = "00-" + "b" * 32 + "-" + "c" * 16 + "-01"
    captured = _capture_urlopen(monkeypatch)
    client = EvaluationClient(
        config=EvaluationConfig(extra_headers={"traceparent": caller_tp})
    )

    with _recording_span():
        client._request_json_sync("GET", "/api/v1beta/ping", None)

    req = captured["request"]
    assert _header_ci(req, "traceparent") == caller_tp


def test_evaluation_request_noop_without_opentelemetry(monkeypatch):
    captured = _capture_urlopen(monkeypatch)
    client = EvaluationClient()

    with patch.dict(
        sys.modules,
        {
            "opentelemetry": None,
            "opentelemetry.trace.propagation.tracecontext": None,
        },
    ):
        with _recording_span():
            client._request_json_sync("GET", "/api/v1beta/ping", None)

    req = captured["request"]
    assert _header_ci(req, "traceparent") is None
