"""Sync-friendly observability client built on a buffered batch transport."""

from __future__ import annotations

import atexit
import copy
import io
import json
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, cast
from urllib import error, request
from urllib.parse import urlencode

from traigent.cloud.async_batch_transport import BatchFlushResult
from traigent.observability.config import ObservabilityConfig
from traigent.observability.dtos import (
    CorrelationIds,
    FlushResult,
    ObservationDTO,
    ObservationType,
    PromptReferenceDTO,
    SessionDTO,
    SessionListResponse,
    SessionRecord,
    ThumbRating,
    TraceCollaborationState,
    TraceCommentRecord,
    TraceCommentsResponse,
    TraceDTO,
    TraceFeedbackResponse,
    TraceListResponse,
    TraceObservationsResponse,
    TraceRecord,
    utc_now,
)
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)
from traigent.utils.logging import get_logger
from traigent.utils.retry import CLOUD_API_RETRY_CONFIG, RetryHandler

logger = get_logger(__name__)


def _new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def _new_observation_id() -> str:
    return f"obs_{uuid.uuid4().hex}"


@dataclass
class _TraceState:
    trace: TraceDTO
    observations: dict[str, ObservationDTO] = field(default_factory=dict)
    observation_order: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        ordered: dict[str, ObservationDTO] = {}
        for observation_id in self.observation_order:
            observation = self.observations[observation_id]
            ordered[observation_id] = replace(observation, children=[])

        roots: list[ObservationDTO] = []
        for observation_id in self.observation_order:
            observation = ordered[observation_id]
            parent_id = observation.parent_observation_id
            if parent_id and parent_id in ordered:
                ordered[parent_id].children.append(observation)
            else:
                roots.append(observation)

        payload_trace = replace(self.trace, observations=roots)
        return payload_trace.to_dict()


class _SyncBatchTransport:
    """Thread-safe batch transport that avoids cross-thread asyncio coordination."""

    def __init__(
        self,
        sender: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
        *,
        batch_size: int,
        max_buffer_age: float,
        max_queue_size: int,
        max_batch_bytes: int,
    ) -> None:
        self._sender = sender
        self.batch_size = batch_size
        self.max_buffer_age = max_buffer_age
        self.max_queue_size = max_queue_size
        self.max_batch_bytes = max_batch_bytes
        self._retry_handler = RetryHandler(CLOUD_API_RETRY_CONFIG)
        self._buffer: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._closed = False
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._stats: dict[str, int] = {
            "submitted_items": 0,
            "sent_items": 0,
            "dropped_items": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "pending_items": 0,
        }

    def submit(self, item_id: str, payload: dict[str, Any]) -> bool:
        send_now = False
        with self._lock:
            if self._closed:
                return False

            if item_id in self._buffer:
                self._buffer.pop(item_id, None)
            elif len(self._buffer) >= self.max_queue_size:
                self._stats["dropped_items"] += 1
                self._append_error(
                    f"transport queue full; dropped payload for item '{item_id}'"
                )
                return False

            self._buffer[item_id] = copy.deepcopy(payload)
            self._stats["submitted_items"] += 1
            self._stats["pending_items"] = len(self._buffer)

            if len(self._buffer) >= self.batch_size:
                self._cancel_timer_locked()
                send_now = True
            else:
                self._ensure_timer_locked()

        if send_now:
            self._send_available()
        return True

    def flush(self) -> BatchFlushResult:
        self._cancel_timer()
        self._send_available()
        return self._build_result()

    def close(self) -> BatchFlushResult:
        with self._lock:
            self._closed = True
        return self.flush()

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            snapshot: dict[str, Any] = dict(self._stats)
        snapshot["errors"] = list(self._errors)
        snapshot["warnings"] = list(self._warnings)
        return snapshot

    def _ensure_timer_locked(self) -> None:
        if self._timer is not None and self._timer.is_alive():
            return
        self._timer = threading.Timer(self.max_buffer_age, self._flush_from_timer)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer_locked(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()

    def _cancel_timer(self) -> None:
        with self._lock:
            self._cancel_timer_locked()

    def _flush_from_timer(self) -> None:
        try:
            self.flush()
        except Exception as exc:
            self._append_error(f"timer flush failed: {exc}")

    def _send_available(self) -> None:
        while True:
            with self._lock:
                if not self._buffer:
                    self._stats["pending_items"] = 0
                    return

                batch_items: list[tuple[str, dict[str, Any]]] = []
                batch_payloads: list[dict[str, Any]] = []
                for _ in range(min(self.batch_size, len(self._buffer))):
                    item_id, payload = next(iter(self._buffer.items()))
                    item_bytes = self._batch_payload_size([payload])
                    if item_bytes > self.max_batch_bytes:
                        self._buffer.popitem(last=False)
                        self._stats["dropped_items"] += 1
                        self._append_error(
                            "observability payload for item "
                            f"'{item_id}' is {item_bytes} bytes, exceeding "
                            f"max_batch_bytes={self.max_batch_bytes}; dropped"
                        )
                        logger.warning(
                            "Observability transport dropped payload '%s' because "
                            "its encoded size %d exceeds max_batch_bytes=%d",
                            item_id,
                            item_bytes,
                            self.max_batch_bytes,
                        )
                        continue

                    projected_bytes = self._batch_payload_size(
                        [*batch_payloads, payload]
                    )
                    if batch_items and projected_bytes > self.max_batch_bytes:
                        break
                    batch_items.append(self._buffer.popitem(last=False))
                    batch_payloads.append(payload)
                self._stats["pending_items"] = len(self._buffer)

            if not batch_items:
                continue
            self._send_batch(batch_items)

    @staticmethod
    def _batch_payload_size(payloads: list[dict[str, Any]]) -> int:
        return len(json.dumps({"traces": payloads}, default=str).encode("utf-8"))

    def _send_batch(self, batch_items: list[tuple[str, dict[str, Any]]]) -> None:
        payloads = [payload for _, payload in batch_items]
        result = self._retry_handler.execute_with_result(self._sender, payloads)

        if result.success:
            if isinstance(result.result, dict):
                for warning in result.result.get("warnings") or []:
                    self._append_warning(str(warning))
            with self._lock:
                self._stats["successful_batches"] += 1
                self._stats["sent_items"] += len(payloads)
            return

        with self._lock:
            self._stats["failed_batches"] += 1
            self._stats["dropped_items"] += len(payloads)
        self._append_error(str(result.error or "batch delivery failed"))
        logger.warning(
            "Observability transport dropped %d payloads after retries: %s",
            len(payloads),
            result.error,
        )

    def _build_result(self) -> BatchFlushResult:
        with self._lock:
            return BatchFlushResult(
                success=self._stats["failed_batches"] == 0,
                items_sent=self._stats["sent_items"],
                items_pending=self._stats["pending_items"],
                items_dropped=self._stats["dropped_items"],
                successful_batches=self._stats["successful_batches"],
                failed_batches=self._stats["failed_batches"],
                errors=list(self._errors),
                warnings=list(self._warnings),
            )

    def _append_error(self, message: str) -> None:
        self._errors.append(message)
        if len(self._errors) > 20:
            self._errors = self._errors[-20:]

    def _append_warning(self, message: str) -> None:
        self._warnings.append(message)
        if len(self._warnings) > 50:
            self._warnings = self._warnings[-50:]


class ObservabilityClient:
    """Client for capturing and shipping generic application traces.

    Delivery happens on a background daemon thread. Normal interpreter shutdown
    will trigger a best-effort atexit flush, but abrupt termination (for example
    `SIGKILL` or `os._exit()`) can still drop buffered payloads. Short-lived or
    serverless processes should call `flush()` or `close()` explicitly.
    """

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        *,
        sender: Callable[[list[dict[str, Any]]], None] | None = None,
        request_sender: (
            Callable[[str, str, dict[str, Any] | None], dict[str, Any]] | None
        ) = None,
    ) -> None:
        self.config = config or ObservabilityConfig()
        self._sender_override = sender
        self._request_sender_override = request_sender
        self._trace_states: dict[str, _TraceState] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._transport = self._create_transport()

        if self.config.enable_atexit_flush:
            atexit.register(self._atexit_close)

    def start_trace(
        self,
        name: str,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
        release: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        input_data: Any = None,
        output_data: Any = None,
        custom_trace_id: str | None = None,
        started_at: datetime | None = None,
        session: SessionDTO | dict[str, Any] | None = None,
        correlation_ids: CorrelationIds | dict[str, str] | None = None,
        prompt_reference: PromptReferenceDTO | dict[str, Any] | None = None,
        status: str = "running",
    ) -> str:
        trace_id = trace_id or _new_trace_id()
        session_dto = self._coerce_session(session)
        if session_dto is not None and session_id is None:
            session_id = session_dto.id

        trace = TraceDTO(
            id=trace_id,
            name=name,
            status=status,
            session_id=session_id,
            user_id=user_id,
            environment=environment or self.config.default_environment,
            release=release or self.config.default_release,
            custom_trace_id=custom_trace_id,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
            input_data=input_data,
            output_data=output_data,
            started_at=started_at or utc_now(),
            session=session_dto,
            correlation_ids=self._coerce_correlation_ids(correlation_ids),
            prompt_reference=self._coerce_prompt_reference(prompt_reference),
        )

        with self._lock:
            self._trace_states[trace_id] = _TraceState(trace=trace)

        self._queue_trace_snapshot(trace_id)
        return trace_id

    def record_observation(
        self,
        trace_id: str,
        *,
        name: str,
        observation_type: ObservationType | str = ObservationType.SPAN,
        observation_id: str | None = None,
        parent_observation_id: str | None = None,
        status: str = "running",
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        model_name: str | None = None,
        tool_name: str | None = None,
        input_data: Any = None,
        output_data: Any = None,
        metadata: dict[str, Any] | None = None,
        correlation_ids: CorrelationIds | dict[str, str] | None = None,
        prompt_reference: PromptReferenceDTO | dict[str, Any] | None = None,
    ) -> str:
        if isinstance(observation_type, str):
            observation_type = ObservationType(observation_type)

        observation_id = observation_id or _new_observation_id()
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None:
                raise ValueError(f"Unknown trace_id '{trace_id}'")

            existing = state.observations.get(observation_id)
            if existing is None:
                observation = ObservationDTO(
                    id=observation_id,
                    type=observation_type,
                    name=name,
                    parent_observation_id=parent_observation_id,
                    status=status,
                    started_at=started_at or utc_now(),
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens if input_tokens is not None else 0,
                    output_tokens=output_tokens if output_tokens is not None else 0,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd if cost_usd is not None else 0.0,
                    model_name=model_name,
                    tool_name=tool_name,
                    input_data=input_data,
                    output_data=output_data,
                    metadata=dict(metadata or {}),
                    correlation_ids=self._coerce_correlation_ids(correlation_ids),
                    prompt_reference=self._coerce_prompt_reference(prompt_reference),
                )
                state.observation_order.append(observation_id)
            else:
                merged_metadata = dict(existing.metadata)
                if metadata:
                    merged_metadata.update(metadata)
                observation = ObservationDTO(
                    id=observation_id,
                    type=observation_type,
                    name=name,
                    parent_observation_id=(
                        parent_observation_id
                        if parent_observation_id is not None
                        else existing.parent_observation_id
                    ),
                    status=status,
                    started_at=started_at or existing.started_at,
                    ended_at=ended_at or existing.ended_at,
                    latency_ms=(
                        latency_ms if latency_ms is not None else existing.latency_ms
                    ),
                    input_tokens=(
                        input_tokens
                        if input_tokens is not None
                        else existing.input_tokens
                    ),
                    output_tokens=(
                        output_tokens
                        if output_tokens is not None
                        else existing.output_tokens
                    ),
                    total_tokens=(
                        total_tokens
                        if total_tokens is not None
                        else existing.total_tokens
                    ),
                    cost_usd=cost_usd if cost_usd is not None else existing.cost_usd,
                    model_name=model_name or existing.model_name,
                    tool_name=tool_name or existing.tool_name,
                    input_data=(
                        input_data if input_data is not None else existing.input_data
                    ),
                    output_data=(
                        output_data if output_data is not None else existing.output_data
                    ),
                    metadata=merged_metadata,
                    correlation_ids=(
                        self._coerce_correlation_ids(correlation_ids)
                        or existing.correlation_ids
                    ),
                    prompt_reference=(
                        self._coerce_prompt_reference(prompt_reference)
                        or existing.prompt_reference
                    ),
                )

            state.observations[observation_id] = observation

        self._queue_trace_snapshot(trace_id)
        return observation_id

    def end_trace(
        self,
        trace_id: str,
        *,
        status: str = "completed",
        output_data: Any = None,
        ended_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None:
                raise ValueError(f"Unknown trace_id '{trace_id}'")
            state.trace.status = status
            state.trace.ended_at = ended_at or utc_now()
            if output_data is not None:
                state.trace.output_data = output_data
            if metadata:
                merged = dict(state.trace.metadata)
                merged.update(metadata)
                state.trace.metadata = merged

        self._queue_trace_snapshot(trace_id)

    def flush(self, timeout: float | None = None) -> FlushResult:
        timeout = timeout or self.config.flush_timeout
        with self._lock:
            trace_ids = list(self._trace_states.keys())
        for trace_id in trace_ids:
            self._queue_trace_snapshot(trace_id)

        result = self._transport.flush()
        return FlushResult(
            success=result.success,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=result.warnings,
        )

    def get_stats(self) -> dict[str, Any]:
        stats = cast(dict[str, Any], self._transport.get_stats())
        with self._lock:
            stats["active_trace_count"] = len(self._trace_states)
        return stats

    def close(self, timeout: float | None = None) -> FlushResult:
        if self._closed:
            return FlushResult(
                success=True,
                items_sent=0,
                items_pending=0,
                items_dropped=0,
                successful_batches=0,
                failed_batches=0,
                errors=[],
                warnings=[],
            )

        self._closed = True
        result = self._transport.close()
        return FlushResult(
            success=result.success,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=result.warnings,
        )

    def list_comments(self, trace_id: str) -> TraceCommentsResponse:
        payload = self._request_json("GET", f"/traces/{trace_id}/comments")
        return TraceCommentsResponse.from_dict(
            self._unwrap_data(payload, "trace comments")
        )

    def add_comment(self, trace_id: str, content: str) -> TraceCommentRecord:
        payload = self._request_json(
            "POST",
            f"/traces/{trace_id}/comments",
            {"content": content},
        )
        return TraceCommentRecord.from_dict(self._unwrap_data(payload, "trace comment"))

    def submit_feedback(
        self,
        trace_id: str,
        rating: ThumbRating | str,
        *,
        comment: str | None = None,
        correction_output: Any = None,
    ) -> TraceFeedbackResponse:
        if isinstance(rating, str):
            rating = ThumbRating(rating)
        correction_output = self._ensure_json_serializable(
            correction_output,
            field_name="correction_output",
        )
        payload = self._request_json(
            "PUT",
            f"/traces/{trace_id}/feedback",
            {
                "rating": rating.value,
                "comment": comment,
                "correction_output": correction_output,
            },
        )
        return TraceFeedbackResponse.from_dict(
            self._unwrap_data(payload, "trace feedback")
        )

    def update_collaboration(
        self,
        trace_id: str,
        *,
        is_bookmarked: bool | None = None,
        is_published: bool | None = None,
    ) -> TraceCollaborationState:
        payload = self._request_json(
            "PATCH",
            f"/traces/{trace_id}/collaboration",
            {
                "is_bookmarked": is_bookmarked,
                "is_published": is_published,
            },
        )
        return TraceCollaborationState.from_dict(
            self._unwrap_data(payload, "trace collaboration")
        )

    def set_bookmarked(
        self, trace_id: str, is_bookmarked: bool
    ) -> TraceCollaborationState:
        return self.update_collaboration(trace_id, is_bookmarked=is_bookmarked)

    def set_published(
        self, trace_id: str, is_published: bool
    ) -> TraceCollaborationState:
        return self.update_collaboration(trace_id, is_published=is_published)

    def list_traces(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        environment: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        model: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> TraceListResponse:
        query = self._build_query_string(
            page=page,
            per_page=per_page,
            search=search,
            environment=environment,
            status=status,
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            release=release,
            model=model,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )
        payload = self._request_json("GET", f"/traces{query}")
        return TraceListResponse.from_dict(self._unwrap_data(payload, "trace list"))

    def get_trace(self, trace_id: str) -> TraceRecord:
        payload = self._request_json("GET", f"/traces/{trace_id}")
        return TraceRecord.from_dict(self._unwrap_data(payload, "trace detail"))

    def get_trace_observations(self, trace_id: str) -> TraceObservationsResponse:
        payload = self._request_json("GET", f"/traces/{trace_id}/observations")
        return TraceObservationsResponse.from_dict(
            self._unwrap_data(payload, "trace observations")
        )

    def list_sessions(
        self,
        *,
        page: int = 1,
        per_page: int = 20,
        search: str | None = None,
        environment: str | None = None,
        user_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> SessionListResponse:
        query = self._build_query_string(
            page=page,
            per_page=per_page,
            search=search,
            environment=environment,
            user_id=user_id,
            tags=tags,
            release=release,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )
        payload = self._request_json("GET", f"/sessions{query}")
        return SessionListResponse.from_dict(self._unwrap_data(payload, "session list"))

    def get_session(self, session_id: str) -> SessionRecord:
        payload = self._request_json("GET", f"/sessions/{session_id}")
        return SessionRecord.from_dict(self._unwrap_data(payload, "session detail"))

    def _create_transport(self) -> _SyncBatchTransport:
        return _SyncBatchTransport(
            self._send_payload_batch,
            batch_size=self.config.batch_size,
            max_buffer_age=self.config.max_buffer_age,
            max_queue_size=self.config.max_queue_size,
            max_batch_bytes=self.config.max_batch_bytes,
        )

    def _send_payload_batch(
        self, traces: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if self._sender_override is not None:
            return self._sender_override(traces)

        return self._post_batch_sync(traces)

    def _post_batch_sync(self, traces: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload = json.dumps({"traces": traces}).encode("utf-8")
        http_request = request.Request(
            self.config.ingest_url,
            data=payload,
            headers=self.config.build_headers(),
            method="POST",
        )
        try:
            with request.urlopen(  # nosec B310 - backend_origin is caller-configured API endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                if status_code >= 400:
                    raise ClientError(
                        f"Observability ingest failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                    )
                return self._parse_ingest_response(body)
        except error.HTTPError as exc:
            body = self._read_http_error_body(exc)
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Observability ingest rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Observability ingest failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to observability backend at {self.config.ingest_url}"
            ) from exc

    def _log_ingest_warnings(self, body: str) -> None:
        payload = self._decode_ingest_response(body)
        self._log_ingest_warnings_from_data(payload)

    def _log_ingest_warnings_from_data(self, payload: dict[str, Any] | None) -> None:
        warnings = (payload or {}).get("warnings")
        if not isinstance(warnings, list) or not warnings:
            return

        for warning in warnings:
            logger.warning("Observability ingest warning: %s", warning)

    def _parse_ingest_response(self, body: str) -> dict[str, Any] | None:
        data = self._decode_ingest_response(body)
        self._log_ingest_warnings_from_data(data)
        return data

    def _decode_ingest_response(self, body: str) -> dict[str, Any] | None:
        if not body:
            return None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return None
        data = parsed.get("data")
        if not isinstance(data, dict):
            return None
        return data

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ClientError("Observability client is closed")
        if self._request_sender_override is not None:
            return cast(
                dict[str, Any], self._request_sender_override(method, path, payload)
            )
        return self._request_json_sync(method, path, payload)

    def _request_json_sync(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded_payload = None
        if payload is not None:
            encoded_payload = json.dumps(payload).encode("utf-8")

        http_request = request.Request(
            f"{self.config.backend_origin}{self.config.api_path}{path}",
            data=encoded_payload,
            headers=self.config.build_headers(),
            method=method,
        )
        try:
            with request.urlopen(  # nosec B310 - backend_origin is caller-configured API endpoint
                http_request, timeout=self.config.request_timeout
            ) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise ClientError(
                        f"Observability request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                    )
                return parsed
        except error.HTTPError as exc:
            body = self._read_http_error_body(exc)
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Observability request rejected with status {exc.code}"
                ) from exc
            raise ClientError(
                f"Observability request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
            ) from exc
        except error.URLError as exc:
            raise TraigentConnectionError(
                f"Failed to connect to observability backend at {self.config.backend_origin}"
            ) from exc

    def _unwrap_data(self, payload: dict[str, Any], label: str) -> dict[str, Any]:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ClientError(f"Unexpected response structure for {label}")
        return data

    def _queue_trace_snapshot(self, trace_id: str) -> None:
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None or self._closed:
                return
            payload = state.to_payload()
        self._transport.submit(trace_id, payload)

    def _build_query_string(self, **params: Any) -> str:
        serialized: dict[str, str] = {}
        for key, value in params.items():
            if value is None or value == "":
                continue
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, list):
                cleaned = [str(item).strip() for item in value if str(item).strip()]
                if cleaned:
                    serialized[key] = ",".join(cleaned)
            else:
                serialized[key] = str(value)

        if not serialized:
            return ""
        return "?" + urlencode(serialized)

    def _ensure_json_serializable(self, value: Any, *, field_name: str) -> Any:
        if value is None:
            return None
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ClientError(f"{field_name} must be JSON serializable") from exc
        return value

    def _read_http_error_body(self, exc: error.HTTPError) -> str:
        try:
            if exc.fp is None:
                return ""
            body = exc.read()
            if isinstance(body, bytes):
                return body.decode("utf-8")
            if isinstance(body, io.BytesIO):
                return body.getvalue().decode("utf-8")
            return str(body)
        finally:
            exc.close()

    def _coerce_session(
        self, session: SessionDTO | dict[str, Any] | None
    ) -> SessionDTO | None:
        if session is None:
            return None
        if isinstance(session, SessionDTO):
            return session
        return SessionDTO(**session)

    def _coerce_correlation_ids(
        self,
        correlation_ids: CorrelationIds | dict[str, str] | None,
    ) -> CorrelationIds | None:
        if correlation_ids is None:
            return None
        if isinstance(correlation_ids, CorrelationIds):
            return correlation_ids
        return CorrelationIds(**correlation_ids)

    def _coerce_prompt_reference(
        self,
        prompt_reference: PromptReferenceDTO | dict[str, Any] | None,
    ) -> PromptReferenceDTO | None:
        if prompt_reference is None:
            return None
        if isinstance(prompt_reference, PromptReferenceDTO):
            return prompt_reference
        return PromptReferenceDTO(**prompt_reference)

    def _atexit_close(self) -> None:
        if self._closed:
            return
        try:
            self.close(timeout=self.config.flush_timeout)
        except Exception as exc:
            logger.warning("Observability atexit flush failed: %s", exc)
