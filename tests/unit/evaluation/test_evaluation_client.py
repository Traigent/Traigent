from __future__ import annotations

from urllib import error

import pytest

from traigent.evaluation import (
    AnnotationQueueItemStatus,
    AnnotationQueueStatus,
    EvaluationClient,
    EvaluationTargetRefDTO,
    EvaluationTargetType,
    EvaluatorRunStatus,
    JudgeConfigDTO,
    MeasureValueType,
)
from traigent.utils.exceptions import AuthenticationError, TraigentConnectionError


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
                    "output_payload": {"numeric_value": 0.9} if status == "completed" else None,
                    "score_record_ids": ["score_1"] if status == "completed" else [],
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.002,
                    "latency_ms": 120,
                    "observability_trace_id": "trace_evalrun_1" if status == "completed" else None,
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
    completed_run = client.wait_for_evaluator_run("evalrun_1", max_attempts=3, interval_seconds=0.01)
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
    assert created_measure["value_type"] == "categorical"
    assert updated_measure["value_type"] == "boolean"

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
        if method == "GET" and path.startswith("/api/v1beta/annotation-queues/queue_1/items?"):
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
                    ]
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
    assert added_items["items"][0]["target_id"] == "trace_1"
    assert queue_items.items[0].assigned_user_id == "annotator_1"
    assert next_item is not None
    assert next_item.id == "queueitem_1"
    assert updated_item.status == AnnotationQueueItemStatus.IN_PROGRESS
    assert completed["item"]["status"] == "completed"

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


def test_get_next_annotation_queue_item_returns_none_for_empty_queue():
    client = EvaluationClient(request_sender=lambda method, path, payload: {"data": None})

    assert client.get_next_annotation_queue_item("queue_1") is None
