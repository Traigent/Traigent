from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from traigent.observability import (
    ObservabilityClient,
    ObservabilityConfig,
    ObservationType,
    PromptReferenceDTO,
    ThumbRating,
    observe,
)
from traigent.observability.decorators import set_default_observability_client
from traigent.observability.dtos import ObservationDTO
from traigent.utils.exceptions import AuthenticationError, ClientError


def test_observability_client_flushes_trace_payloads():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=2,
            max_buffer_age=5.0,
            max_queue_size=10,
        ),
        sender=sender,
    )

    trace_id = client.start_trace(
        "customer-support",
        trace_id="trace_test_001",
        session_id="session_test_001",
        user_id="user_001",
        tags=["demo"],
        metadata={"source": "unit-test"},
        prompt_reference=PromptReferenceDTO(
            name="support/welcome",
            version=2,
            label="latest",
            variables={"customer_name": "Ada"},
        ),
    )
    root_observation_id = client.record_observation(
        trace_id,
        observation_id="obs_root",
        name="root-span",
        observation_type=ObservationType.SPAN,
        input_tokens=10,
        output_tokens=5,
    )
    client.record_observation(
        trace_id,
        observation_id="obs_child",
        parent_observation_id=root_observation_id,
        name="llm-call",
        observation_type=ObservationType.GENERATION,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.0025,
        model_name="gpt-4.1-mini",
        metadata={"provider": "openai"},
        prompt_reference={
            "name": "support/welcome",
            "version": 1,
            "label": "production",
            "variables": {"customer_name": "Ada"},
        },
    )
    client.end_trace(trace_id, output_data={"answer": "hello"})

    result = client.flush()
    client.close()

    assert result.success is True
    assert result.items_sent >= 1
    assert sent_batches

    sent_trace = sent_batches[-1][-1]
    assert sent_trace["id"] == "trace_test_001"
    assert sent_trace["session_id"] == "session_test_001"
    assert sent_trace["prompt_reference"]["name"] == "support/welcome"
    assert sent_trace["observations"][0]["id"] == "obs_root"
    assert sent_trace["observations"][0]["children"][0]["id"] == "obs_child"
    assert sent_trace["observations"][0]["children"][0]["type"] == "generation"
    assert (
        sent_trace["observations"][0]["children"][0]["prompt_reference"]["version"] == 1
    )


def test_record_observation_update_preserves_tokens_and_cost_when_omitted():
    sent_batches: list[list[dict]] = []

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: sent_batches.append(traces),
    )

    trace_id = client.start_trace("preserve-metrics", trace_id="trace_preserve")
    observation_id = client.record_observation(
        trace_id,
        observation_id="obs_preserve",
        name="llm-call",
        observation_type=ObservationType.GENERATION,
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        cost_usd=0.004,
    )
    client.record_observation(
        trace_id,
        observation_id=observation_id,
        name="llm-call",
        observation_type=ObservationType.GENERATION,
        status="completed",
    )
    client.end_trace(trace_id)

    result = client.flush()
    client.close()

    assert result.success is True
    observation = sent_batches[-1][-1]["observations"][0]
    assert observation["input_tokens"] == 11
    assert observation["output_tokens"] == 7
    assert observation["total_tokens"] == 18
    assert observation["cost_usd"] == 0.004


def test_observability_client_tracks_dropped_payloads_when_buffer_is_full():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=100,
            max_buffer_age=999.0,
            max_queue_size=1,
        ),
        sender=sender,
    )

    trace_a = client.start_trace("trace-a", trace_id="trace_a")
    trace_b = client.start_trace("trace-b", trace_id="trace_b")
    client.end_trace(trace_a)
    client.end_trace(trace_b)

    result = client.close()
    stats = client.get_stats()

    assert stats["dropped_items"] >= 1
    assert result.items_dropped >= 1


def test_observe_decorator_creates_nested_observations():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )
    set_default_observability_client(client)

    @observe(
        "inner-operation", client=client, observation_type=ObservationType.TOOL_CALL
    )
    def inner(value: int) -> int:
        return value + 1

    @observe("outer-operation", client=client)
    def outer(value: int) -> int:
        return inner(value) * 2

    assert outer(2) == 6

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["name"] == "outer-operation"
    root_observation = trace_payload["observations"][0]
    assert root_observation["name"] == "outer-operation"
    assert root_observation["children"][0]["name"] == "inner-operation"
    assert root_observation["children"][0]["type"] == "tool_call"


def test_observe_decorator_can_redact_inputs():
    sent_batches: list[list[dict]] = []

    def sender(traces):
        sent_batches.append(traces)

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=sender,
    )

    @observe("sensitive-operation", client=client, redact_input=True)
    def sensitive(password: str) -> str:
        return f"len={len(password)}"

    assert sensitive("super-secret-password") == "len=21"

    result = client.flush()
    client.close()

    assert result.success is True
    trace_payload = sent_batches[-1][-1]
    assert trace_payload["input_data"] == {"redacted": True}
    assert trace_payload["observations"][0]["input_data"] == {"redacted": True}


def test_observability_client_flush_surfaces_backend_warnings():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: {
            "warnings": [
                "Prompt reference could not be resolved for trace 'trace_warn': support/missing (label=latest)"
            ]
        },
    )

    trace_id = client.start_trace("warn-trace", trace_id="trace_warn")
    client.end_trace(trace_id)

    result = client.flush()
    client.close()

    assert result.success is True
    assert result.warnings == [
        "Prompt reference could not be resolved for trace 'trace_warn': support/missing (label=latest)"
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 10_001, "batch_size"),
        ("max_queue_size", 1_000_001, "max_queue_size"),
        ("max_buffer_age", 3601.0, "max_buffer_age"),
        ("flush_timeout", 601.0, "flush_timeout"),
        ("request_timeout", 601.0, "request_timeout"),
    ],
)
def test_observability_config_rejects_unbounded_values(field, value, message):
    kwargs = {
        "backend_origin": "http://localhost:5000",
        "api_key": "test-key",  # pragma: allowlist secret
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        ObservabilityConfig(**kwargs)


def test_observation_dto_rejects_negative_values():
    with pytest.raises(ValueError, match="input_tokens"):
        ObservationDTO(
            id="obs_bad",
            type=ObservationType.SPAN,
            name="bad-observation",
            input_tokens=-1,
        )


def test_observability_client_collaboration_helpers_follow_backend_contract():
    request_calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        request_calls.append((method, path, payload))
        if method == "GET" and path == "/traces/trace_sdk/comments":
            return {
                "data": {
                    "trace_id": "trace_sdk",
                    "count": 1,
                    "items": [
                        {
                            "id": "comment_1",
                            "trace_id": "trace_sdk",
                            "author_user_id": "sdk-user",
                            "content": "Investigate the answer format.",
                            "created_at": "2026-03-10T14:10:00+00:00",
                            "updated_at": "2026-03-10T14:10:00+00:00",
                        }
                    ],
                }
            }
        if method == "POST" and path == "/traces/trace_sdk/comments":
            return {
                "data": {
                    "id": "comment_2",
                    "trace_id": "trace_sdk",
                    "author_user_id": "sdk-user",
                    "content": payload["content"],
                    "created_at": "2026-03-10T14:11:00+00:00",
                    "updated_at": "2026-03-10T14:11:00+00:00",
                }
            }
        if method == "PUT" and path == "/traces/trace_sdk/feedback":
            return {
                "data": {
                    "feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_sdk",
                        "author_user_id": "sdk-user",
                        "rating": payload["rating"],
                        "comment": payload["comment"],
                        "correction_output": payload["correction_output"],
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                    "summary": {
                        "up_count": 1 if payload["rating"] == "up" else 0,
                        "down_count": 1 if payload["rating"] == "down" else 0,
                    },
                }
            }
        if method == "PATCH" and path == "/traces/trace_sdk/collaboration":
            return {
                "data": {
                    "is_bookmarked": bool(payload.get("is_bookmarked")),
                    "bookmarked_at": (
                        "2026-03-10T14:13:00+00:00"
                        if payload.get("is_bookmarked")
                        else None
                    ),
                    "bookmarked_by": (
                        "sdk-user" if payload.get("is_bookmarked") else None
                    ),
                    "is_published": bool(payload.get("is_published")),
                    "published_at": (
                        "2026-03-10T14:14:00+00:00"
                        if payload.get("is_published")
                        else None
                    ),
                    "published_by": "sdk-user" if payload.get("is_published") else None,
                    "comment_count": 2,
                    "feedback_summary": {"up_count": 1, "down_count": 0},
                    "current_user_feedback": {
                        "id": "feedback_1",
                        "trace_id": "trace_sdk",
                        "author_user_id": "sdk-user",
                        "rating": "up",
                        "comment": "Approved",
                        "correction_output": {"answer": "Approved answer"},
                        "created_at": "2026-03-10T14:12:00+00:00",
                        "updated_at": "2026-03-10T14:12:00+00:00",
                    },
                }
            }
        raise AssertionError(f"Unexpected SDK request: {method} {path}")

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=request_sender,
    )

    comments = client.list_comments("trace_sdk")
    created_comment = client.add_comment("trace_sdk", "Ship this after QA review.")
    feedback = client.submit_feedback(
        "trace_sdk",
        ThumbRating.UP,
        comment="Approved",
        correction_output={"answer": "Approved answer"},
    )
    bookmarked = client.set_bookmarked("trace_sdk", True)
    published = client.set_published("trace_sdk", True)
    client.close()

    assert comments.count == 1
    assert comments.items[0].content == "Investigate the answer format."
    assert created_comment.content == "Ship this after QA review."
    assert feedback.feedback.rating is ThumbRating.UP
    assert feedback.summary.up_count == 1
    assert bookmarked.is_bookmarked is True
    assert bookmarked.current_user_feedback is not None
    assert published.is_published is True
    assert request_calls == [
        ("GET", "/traces/trace_sdk/comments", None),
        (
            "POST",
            "/traces/trace_sdk/comments",
            {"content": "Ship this after QA review."},
        ),
        (
            "PUT",
            "/traces/trace_sdk/feedback",
            {
                "rating": "up",
                "comment": "Approved",
                "correction_output": {"answer": "Approved answer"},
            },
        ),
        (
            "PATCH",
            "/traces/trace_sdk/collaboration",
            {"is_bookmarked": True, "is_published": None},
        ),
        (
            "PATCH",
            "/traces/trace_sdk/collaboration",
            {"is_bookmarked": None, "is_published": True},
        ),
    ]


def test_observability_client_query_helpers_follow_backend_contract():
    request_calls: list[tuple[str, str, dict | None]] = []

    def request_sender(method: str, path: str, payload: dict | None):
        request_calls.append((method, path, payload))
        if method == "GET" and path.startswith("/traces?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "trace_sdk_query",
                            "name": "query-trace",
                            "status": "running",
                            "session_id": "session_sdk_query",
                            "user_id": "sdk-user",
                            "environment": "production",
                            "release": "2026.03.10",
                            "tags": ["demo"],
                            "observation_count": 2,
                            "root_observation_count": 1,
                            "total_input_tokens": 10,
                            "total_output_tokens": 5,
                            "total_tokens": 15,
                            "total_cost_usd": 0.002,
                            "total_latency_ms": 250,
                        }
                    ],
                    "pagination": {
                        "page": 2,
                        "per_page": 10,
                        "total": 1,
                        "total_pages": 1,
                        "has_next": False,
                        "has_prev": True,
                    },
                }
            }
        if method == "GET" and path == "/traces/trace_sdk_query":
            return {
                "data": {
                    "id": "trace_sdk_query",
                    "name": "query-trace",
                    "status": "completed",
                    "session_id": "session_sdk_query",
                    "user_id": "sdk-user",
                    "environment": "production",
                    "session": {
                        "id": "session_sdk_query",
                        "trace_count": 1,
                        "observation_count": 2,
                        "total_tokens": 15,
                    },
                    "collaboration": {
                        "is_bookmarked": True,
                        "comment_count": 1,
                        "feedback_summary": {"up_count": 1, "down_count": 0},
                    },
                }
            }
        if method == "GET" and path == "/traces/trace_sdk_query/observations":
            return {
                "data": {
                    "trace_id": "trace_sdk_query",
                    "observation_count": 1,
                    "items": [
                        {
                            "id": "obs_sdk_root",
                            "trace_id": "trace_sdk_query",
                            "type": "span",
                            "name": "root",
                            "status": "completed",
                            "depth": 0,
                            "sequence_number": 0,
                            "latency_ms": 250,
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "cost_usd": 0.002,
                            "children": [],
                        }
                    ],
                }
            }
        if method == "GET" and path.startswith("/sessions?"):
            return {
                "data": {
                    "items": [
                        {
                            "id": "session_sdk_query",
                            "user_id": "sdk-user",
                            "environment": "production",
                            "trace_count": 1,
                            "observation_count": 2,
                            "total_tokens": 15,
                            "total_cost_usd": 0.002,
                            "ended_at": "2026-03-10T14:00:00+00:00",
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
        if method == "GET" and path == "/sessions/session_sdk_query":
            return {
                "data": {
                    "id": "session_sdk_query",
                    "user_id": "sdk-user",
                    "environment": "production",
                    "trace_count": 1,
                    "observation_count": 2,
                    "total_tokens": 15,
                    "traces": [
                        {
                            "id": "trace_sdk_query",
                            "name": "query-trace",
                            "status": "completed",
                            "observation_count": 2,
                            "root_observation_count": 1,
                            "total_tokens": 15,
                            "total_cost_usd": 0.002,
                            "total_latency_ms": 250,
                        }
                    ],
                }
            }
        raise AssertionError(f"Unexpected SDK request: {method} {path}")

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=request_sender,
    )

    traces = client.list_traces(
        page=2,
        per_page=10,
        environment="production",
        tags=["demo"],
        start_time_from=datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC),
    )
    trace = client.get_trace("trace_sdk_query")
    observations = client.get_trace_observations("trace_sdk_query")
    sessions = client.list_sessions(search="session_sdk_query", release="2026.03.10")
    session = client.get_session("session_sdk_query")
    client.close()

    assert traces.pagination.page == 2
    assert traces.items[0].status == "running"
    assert trace.collaboration is not None
    assert trace.collaboration.is_bookmarked is True
    assert observations.items[0].type.value == "span"
    assert sessions.items[0].id == "session_sdk_query"
    assert session.traces[0].id == "trace_sdk_query"
    assert request_calls[0][1].startswith(
        "/traces?page=2&per_page=10&environment=production&tags=demo&start_time_from="
    )


def test_observability_client_rejects_collaboration_requests_after_close():
    request_calls: list[tuple[str, str, dict | None]] = []

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=lambda method, path, payload: request_calls.append(
            (method, path, payload)
        )
        or {"data": {}},
    )
    client.close()

    with pytest.raises(ClientError, match="closed"):
        client.list_comments("trace_sdk")

    assert request_calls == []


def test_observability_client_validates_feedback_correction_output():
    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=lambda method, path, payload: {"data": {}},
    )

    with pytest.raises(ClientError, match="correction_output"):
        client.submit_feedback(
            "trace_sdk", ThumbRating.UP, correction_output={"bad": {1, 2, 3}}
        )

    client.close()


def test_observability_client_surfaces_collaboration_error_paths():
    def missing_trace_request_sender(method: str, path: str, payload: dict | None):
        raise ClientError(
            "Observability request failed with status 404",
            status_code=404,
            details={"body": '{"error":"not found"}'},
        )

    missing_trace_client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=missing_trace_request_sender,
    )

    with pytest.raises(ClientError, match="404"):
        missing_trace_client.add_comment("missing_trace", "Comment")

    missing_trace_client.close()

    def forbidden_request_sender(method: str, path: str, payload: dict | None):
        raise AuthenticationError("Observability request rejected with status 403")

    forbidden_client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
        request_sender=forbidden_request_sender,
    )

    with pytest.raises(AuthenticationError, match="403"):
        forbidden_client.submit_feedback("trace_sdk", ThumbRating.UP)

    forbidden_client.close()


def test_observability_client_logs_ingest_warnings(monkeypatch, caplog):
    class _FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return (
                b'{"data":{"warnings":["Prompt reference could not be resolved for trace '
                b"'trace_warn': support/missing (label=latest)\"]}}"
            )

    client = ObservabilityClient(
        ObservabilityConfig(
            backend_origin="http://localhost:5000",
            api_key="test-key",  # pragma: allowlist secret
            batch_size=10,
            max_buffer_age=0.1,
            max_queue_size=10,
        ),
        sender=lambda traces: None,
    )

    monkeypatch.setattr(
        "traigent.observability.client.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(),
    )

    with caplog.at_level(logging.WARNING):
        client._post_batch_sync(
            [{"id": "trace_warn", "name": "warn-trace", "observations": []}]
        )

    client.close()

    assert "Observability ingest warning" in caplog.text
