"""Sync-friendly observability client built on a buffered batch transport."""

from __future__ import annotations

import atexit
import copy
import io
import json
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast
from urllib import error, request
from urllib.parse import urlencode

from traigent.cloud.async_batch_transport import BatchFlushResult
from traigent.observability.config import ObservabilityConfig
from traigent.observability.dtos import (
    OBSERVABILITY_STATUSES,
    CorrelationIds,
    ExecutionContextDTO,
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
from traigent.security.redaction import redact_sensitive_data, redact_sensitive_text
from traigent.utils.exceptions import (
    AuthenticationError,
    ClientError,
    TraigentConnectionError,
)
from traigent.utils.logging import get_logger
from traigent.utils.pagination import iter_pages
from traigent.utils.retry import CLOUD_API_RETRY_CONFIG, RetryHandler

logger = get_logger(__name__)

_TRACE_BATCH_PREFIX_BYTES = len(b'{"traces": [')
_TRACE_BATCH_SUFFIX_BYTES = len(b"]}")
_TRACE_BATCH_ITEM_SEPARATOR_BYTES = len(b", ")
_HEALTH_DISPATCH_STOP = object()


class _NoRedirectHandler(request.HTTPRedirectHandler):
    """Prevent credentialed observability requests from replaying headers elsewhere."""

    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _new_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex}"


def _new_observation_id() -> str:
    return f"obs_{uuid.uuid4().hex}"


def _parse_retry_after(headers: Any) -> float | None:
    if not headers:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


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
        return cast(dict[str, Any], payload_trace.to_dict())


@dataclass(frozen=True)
class _BufferedPayload:
    payload: dict[str, Any]
    payload_bytes: int


class _SyncBatchTransport:
    """Thread-safe batch transport that avoids cross-thread asyncio coordination.

    ``health_callback`` receives snapshots after transport state has been
    updated and after transport locks have been released. A dedicated daemon
    dispatcher delivers snapshots in state-update order. Callback exceptions are
    swallowed so observability delivery is never disrupted, and user callbacks
    never extend a caller's flush or close deadline. Close drains snapshots
    accepted before dispatcher shutdown; later snapshots increment
    ``dropped_health_events``.
    """

    def __init__(
        self,
        sender: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
        *,
        batch_size: int,
        max_buffer_age: float,
        max_queue_size: int,
        max_batch_bytes: int,
        health_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._sender = sender
        self._health_callback = health_callback
        self.batch_size = batch_size
        self.max_buffer_age = max_buffer_age
        self.max_queue_size = max_queue_size
        self.max_batch_bytes = max_batch_bytes
        self._retry_handler = RetryHandler(
            replace(CLOUD_API_RETRY_CONFIG, callback_on_retry=self._record_retry)
        )
        self._buffer: OrderedDict[str, _BufferedPayload] = OrderedDict()
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._health_event_queue: deque[
            tuple[int, str, dict[str, Any]] | threading.Event | object
        ] = deque()
        self._health_event_condition = threading.Condition(self._lock)
        self._next_health_event_sequence = 0
        self._dropped_health_events = 0
        self._health_dispatch_closed = False
        self._timer: threading.Timer | None = None
        self._flush_thread: threading.Thread | None = None
        self._closed = False
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._inflight_items = 0
        self._oldest_inflight_started_at: float | None = None
        self._active_send_completion: threading.Event | None = None
        self._stats: dict[str, int] = {
            "submitted_items": 0,
            "sent_items": 0,
            "dropped_items": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "pending_items": 0,
            "background_flushes": 0,
            "retry_attempts": 0,
        }
        self._dropped_by_reason: dict[str, int] = {}
        self._health_dispatcher_thread: threading.Thread | None = None
        if self._health_callback is not None:
            self._health_dispatcher_thread = threading.Thread(
                target=self._dispatch_health_events,
                name="traigent-observability-health-dispatcher",
                daemon=True,
            )
            self._health_dispatcher_thread.start()

    def submit(
        self,
        item_id: str,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> bool:
        try:
            buffered_payload = self._prepare_payload(payload)
        except TypeError as exc:
            message = (
                "observability payload for item "
                f"'{item_id}' is not JSON serializable: {exc}; dropped"
            )
            self._record_drop("payload_not_json_serializable", message, item_id=item_id)
            logger.warning(
                "Observability transport dropped payload '%s' because "
                "it is not JSON serializable: %s",
                item_id,
                exc,
            )
            return False

        item_bytes = self._batch_payload_size_from_body(buffered_payload.payload_bytes)
        if item_bytes > self.max_batch_bytes:
            message = (
                "observability payload for item "
                f"'{item_id}' is {item_bytes} bytes, exceeding "
                f"max_batch_bytes={self.max_batch_bytes}; dropped"
            )
            self._record_drop(
                "payload_too_large",
                message,
                item_id=item_id,
                item_bytes=item_bytes,
                max_batch_bytes=self.max_batch_bytes,
            )
            logger.warning(
                "Observability transport dropped payload '%s' because "
                "its encoded size %d exceeds max_batch_bytes=%d",
                item_id,
                item_bytes,
                self.max_batch_bytes,
            )
            return False

        drop_event: tuple[str, dict[str, Any]] | None = None
        if not self._acquire_lock_until(deadline):
            return False
        try:
            if self._closed:
                drop_event = self._record_drop_locked(
                    "transport_closed",
                    f"transport closed; dropped payload for item '{item_id}'",
                    item_id=item_id,
                )
            elif (
                item_id not in self._buffer and len(self._buffer) >= self.max_queue_size
            ):
                drop_event = self._record_drop_locked(
                    "queue_full",
                    f"transport queue full; dropped payload for item '{item_id}'",
                    item_id=item_id,
                )
            else:
                self._buffer.pop(item_id, None)
                self._buffer[item_id] = buffered_payload
                self._stats["submitted_items"] += 1
                self._stats["pending_items"] = len(self._buffer)

                if len(self._buffer) >= self.batch_size:
                    self._ensure_flush_thread_locked()
                # Always keep an age timer armed while the buffer is non-empty. On the
                # batch-size path this is a backstop: if the flush thread is between its
                # buffer-empty check and exit when this item arrives, is_alive() still
                # reads True so _ensure_flush_thread_locked spawns nothing — the timer
                # then guarantees the tail item is not stranded until the next
                # submit/close. When the thread does drain everything, the leftover
                # timer fires once, finds an empty buffer, and returns harmlessly.
                self._ensure_timer_locked()
        finally:
            self._lock.release()

        if drop_event is not None:
            self._emit_health_events([drop_event], deadline=deadline)
            return False

        return True

    def flush(
        self,
        timeout: float | None = None,
        *,
        warn_on_deadline_expiry: bool | None = None,
    ) -> BatchFlushResult:
        """Flush queued payloads, waiting no longer than ``timeout`` seconds."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if warn_on_deadline_expiry is None:
            warn_on_deadline_expiry = timeout != 0
        if not self._cancel_timer(deadline):
            return self._deadline_expiry_result(
                "Observability transport flush deadline exceeded while acquiring "
                "the timer state lock"
            )
        completed = self._send_available(
            deadline=deadline, warn_on_deadline_expiry=warn_on_deadline_expiry
        )
        return self._build_result(
            deadline,
            deadline_expired=not completed,
            warn_on_deadline_expiry=warn_on_deadline_expiry,
        )

    def close(
        self,
        timeout: float | None = None,
        *,
        warn_on_deadline_expiry: bool | None = None,
    ) -> BatchFlushResult:
        """Prevent new submissions, flush, and stop health-event dispatch."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if not self._acquire_lock_until(deadline):
            return self._deadline_expiry_result(
                "Observability transport close deadline exceeded while acquiring "
                "the transport state lock"
            )
        try:
            self._closed = True
        finally:
            self._lock.release()
        result = self.flush(
            timeout=self._remaining_time(deadline),
            warn_on_deadline_expiry=warn_on_deadline_expiry,
        )
        self._stop_health_dispatcher(deadline)
        return result

    def get_stats(self, timeout: float | None = None) -> dict[str, Any]:
        """Return a thread-safe transport health snapshot.

        ``dropped_by_reason`` identifies local data loss,
        ``dropped_health_events`` counts callback snapshots discarded because
        the dispatch queue overflowed or had stopped, ``queue_depth`` is the
        number of buffered payloads, and ``inflight_items`` covers payloads that
        have left the queue but whose sender has not completed.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if not self._acquire_lock_until(deadline):
            return self._stats_snapshot_without_lock()
        try:
            snapshot: dict[str, Any] = dict(self._stats)
            snapshot["queue_depth"] = len(self._buffer)
            snapshot["inflight_items"] = self._inflight_items
            snapshot["send_in_progress"] = (
                self._active_send_completion is not None or self._send_lock.locked()
            )
            snapshot["oldest_inflight_age_seconds"] = (
                None
                if self._oldest_inflight_started_at is None
                else max(0.0, time.monotonic() - self._oldest_inflight_started_at)
            )
            snapshot["dropped_by_reason"] = dict(self._dropped_by_reason)
            snapshot["dropped_health_events"] = self._dropped_health_events
            snapshot["errors"] = list(self._errors)
            snapshot["warnings"] = list(self._warnings)
            return snapshot
        finally:
            self._lock.release()

    def _ensure_timer_locked(self) -> None:
        if self._timer is not None and self._timer.is_alive():
            return
        self._timer = threading.Timer(self.max_buffer_age, self._flush_from_timer)
        self._timer.daemon = True
        self._timer.start()

    def _ensure_flush_thread_locked(self) -> None:
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(
            target=self._flush_from_worker,
            name="traigent-observability-flush",
            daemon=True,
        )
        self._stats["background_flushes"] += 1
        self._flush_thread.start()

    def _cancel_timer_locked(self) -> None:
        timer = self._timer
        self._timer = None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()

    def _cancel_timer(self, deadline: float | None = None) -> bool:
        if not self._acquire_lock_until(deadline):
            return False
        try:
            self._cancel_timer_locked()
            return True
        finally:
            self._lock.release()

    def _flush_from_timer(self) -> None:
        try:
            self.flush()
        except Exception as exc:
            self._append_error(f"timer flush failed: {exc}")

    def _flush_from_worker(self) -> None:
        try:
            self._send_available()
        except Exception as exc:
            self._append_error(f"background flush failed: {exc}")

    def _send_available(
        self,
        *,
        deadline: float | None = None,
        warn_on_deadline_expiry: bool = True,
    ) -> bool:
        events: list[tuple[str, dict[str, Any]]] = []
        caller_released = threading.Event()
        if deadline is None:
            acquired = self._send_lock.acquire()
        else:
            acquired = self._send_lock.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            )
        if not acquired:
            if warn_on_deadline_expiry:
                event = self._record_deadline_expiry(deadline)
                if event is not None:
                    events.append(event)
            self._emit_health_events(events, deadline=deadline)
            return False

        try:
            while True:
                if not self._wait_for_active_send(deadline):
                    if warn_on_deadline_expiry:
                        event = self._record_deadline_expiry(deadline)
                        if event is not None:
                            events.append(event)
                    return False
                if deadline is not None and time.monotonic() >= deadline:
                    if warn_on_deadline_expiry:
                        event = self._record_deadline_expiry(deadline)
                        if event is not None:
                            events.append(event)
                    return False
                if not self._acquire_lock_until(deadline):
                    return False
                try:
                    if not self._buffer:
                        self._stats["pending_items"] = self._inflight_items
                        return True

                    batch_items: list[tuple[str, dict[str, Any]]] = []
                    batch_body_bytes = 0
                    for _ in range(min(self.batch_size, len(self._buffer))):
                        item_id, buffered = next(iter(self._buffer.items()))
                        projected_body_bytes = batch_body_bytes + buffered.payload_bytes
                        if batch_items:
                            projected_body_bytes += _TRACE_BATCH_ITEM_SEPARATOR_BYTES
                        projected_bytes = self._batch_payload_size_from_body(
                            projected_body_bytes
                        )
                        if batch_items and projected_bytes > self.max_batch_bytes:
                            break
                        self._buffer.popitem(last=False)
                        batch_items.append((item_id, buffered.payload))
                        batch_body_bytes = projected_body_bytes
                    self._inflight_items += len(batch_items)
                    if self._oldest_inflight_started_at is None:
                        self._oldest_inflight_started_at = time.monotonic()
                    self._stats["pending_items"] = (
                        len(self._buffer) + self._inflight_items
                    )
                finally:
                    self._lock.release()

                if not batch_items:
                    continue
                if deadline is None:
                    try:
                        events.extend(self._send_batch(batch_items))
                    finally:
                        self._complete_inflight_batch(len(batch_items))
                else:
                    batch_events = self._send_batch_until(
                        batch_items, deadline, caller_released
                    )
                    if batch_events is None:
                        if warn_on_deadline_expiry:
                            event = self._record_deadline_expiry(deadline)
                            if event is not None:
                                events.append(event)
                        return False
                    events.extend(batch_events)
        finally:
            self._send_lock.release()
            caller_released.set()
            self._emit_health_events(events, deadline=deadline)

    def _send_batch_until(
        self,
        batch_items: list[tuple[str, dict[str, Any]]],
        deadline: float,
        caller_released: threading.Event,
    ) -> list[tuple[str, dict[str, Any]]] | None:
        completed = threading.Event()
        events: list[tuple[str, dict[str, Any]]] = []

        if not self._acquire_lock_until(deadline):
            return None
        try:
            self._active_send_completion = completed
        finally:
            self._lock.release()

        def send() -> None:
            try:
                events.extend(self._send_batch(batch_items))
            except Exception as exc:
                # Flush and close are best effort; background sender failures must
                # be visible in the result but must never escape user code.
                self._append_error(f"batch delivery failed unexpectedly: {exc}")
            finally:
                self._complete_inflight_batch(len(batch_items), completed)
                completed.set()

        threading.Thread(
            target=send,
            name="traigent-observability-send",
            daemon=True,
        ).start()
        if completed.wait(timeout=max(0.0, deadline - time.monotonic())):
            return events

        threading.Thread(
            target=self._emit_orphan_health_events,
            args=(completed, caller_released, events),
            name="traigent-observability-send-events",
            daemon=True,
        ).start()
        return None

    def _wait_for_active_send(self, deadline: float | None) -> bool:
        """Wait for a timed-out sender without ever starting a concurrent sender."""
        while True:
            if not self._acquire_lock_until(deadline):
                return False
            try:
                completion = self._active_send_completion
            finally:
                self._lock.release()
            if completion is None:
                return True
            if deadline is None:
                completion.wait()
                continue
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0 or not completion.wait(timeout=remaining):
                return False

    def _emit_orphan_health_events(
        self,
        completed: threading.Event,
        caller_released: threading.Event,
        events: list[tuple[str, dict[str, Any]]],
    ) -> None:
        completed.wait()
        caller_released.wait()
        self._emit_health_events(events)

    def _complete_inflight_batch(
        self, item_count: int, completion: threading.Event | None = None
    ) -> None:
        with self._lock:
            self._inflight_items -= item_count
            if self._inflight_items == 0:
                self._oldest_inflight_started_at = None
            self._stats["pending_items"] = len(self._buffer) + self._inflight_items
            if completion is not None and self._active_send_completion is completion:
                self._active_send_completion = None

    def _record_deadline_expiry(
        self, deadline: float | None
    ) -> tuple[str, dict[str, Any]] | None:
        if not self._acquire_lock_until(deadline):
            return None
        try:
            remaining = len(self._buffer) + self._inflight_items
            self._stats["pending_items"] = remaining
            message = (
                "Observability transport flush deadline exceeded with "
                f"{remaining} items remaining"
            )
            self._warnings.append(message)
            if len(self._warnings) > 50:
                self._warnings = self._warnings[-50:]
            event = self._queue_health_event_locked("warning", {"message": message})
        finally:
            self._lock.release()
        logger.warning(message)
        return event

    def _prepare_payload(self, payload: dict[str, Any]) -> _BufferedPayload:
        # Direct transport callers bypass ObservabilityClient's trace redaction.
        # Preserve credential-key-name redaction (c4d874c3): this is a confirmed
        # egress path carrying arbitrary user keys, so opt in as the other two
        # egress call sites do.
        redacted_payload = cast(
            dict[str, Any],
            redact_sensitive_data(payload, redact_credential_keys=True),
        )
        copied_payload = copy.deepcopy(redacted_payload)
        return _BufferedPayload(
            payload=copied_payload,
            payload_bytes=self._payload_json_size(copied_payload),
        )

    @staticmethod
    def _payload_json_size(payload: dict[str, Any]) -> int:
        return len(json.dumps(payload).encode("utf-8"))

    @staticmethod
    def _batch_payload_size_from_body(body_bytes: int) -> int:
        return _TRACE_BATCH_PREFIX_BYTES + body_bytes + _TRACE_BATCH_SUFFIX_BYTES

    def _send_batch(
        self, batch_items: list[tuple[str, dict[str, Any]]]
    ) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        payloads = [payload for _, payload in batch_items]
        result = self._retry_handler.execute_with_result(self._sender, payloads)

        if result.success:
            if isinstance(result.result, dict):
                for warning in result.result.get("warnings") or []:
                    events.append(self._append_warning(str(warning)))
            with self._lock:
                self._stats["successful_batches"] += 1
                self._stats["sent_items"] += len(payloads)
            return events

        with self._lock:
            self._stats["failed_batches"] += 1
            self._stats["dropped_items"] += len(payloads)
            self._dropped_by_reason["batch_delivery_failed"] = (
                self._dropped_by_reason.get("batch_delivery_failed", 0) + len(payloads)
            )
            dropped_items = self._stats["dropped_items"]
            queue_depth = len(self._buffer)
            event = self._queue_health_event_locked(
                "batch_delivery_failed",
                {
                    "drop_reason": "batch_delivery_failed",
                    "dropped_items": dropped_items,
                    "queue_depth": queue_depth,
                    "message": str(result.error or "batch delivery failed"),
                    "item_count": len(payloads),
                    "trace_ids": [item_id for item_id, _ in batch_items[:20]],
                    "trace_ids_truncated": len(batch_items) > 20,
                },
            )
        self._append_error(str(result.error or "batch delivery failed"))
        events.append(event)
        logger.warning(
            "Observability transport dropped %d payloads after retries: %s",
            len(payloads),
            result.error,
        )
        return events

    def _build_result(
        self,
        deadline: float | None = None,
        *,
        deadline_expired: bool = False,
        warn_on_deadline_expiry: bool = True,
    ) -> BatchFlushResult:
        if not self._acquire_lock_until(deadline):
            return self._deadline_expiry_result(
                "Observability transport flush deadline exceeded while acquiring "
                "the transport state lock"
            )
        try:
            remaining = len(self._buffer) + self._inflight_items
            self._stats["pending_items"] = remaining
            warnings = list(self._warnings)
            if (
                warn_on_deadline_expiry
                and deadline_expired
                and not any(
                    "flush deadline exceeded" in warning for warning in warnings
                )
            ):
                warnings.append(
                    "Observability transport flush deadline exceeded before "
                    "delivery completed"
                )
            return BatchFlushResult(
                success=self._stats["failed_batches"] == 0 and remaining == 0,
                items_sent=self._stats["sent_items"],
                items_pending=remaining,
                items_dropped=self._stats["dropped_items"],
                successful_batches=self._stats["successful_batches"],
                failed_batches=self._stats["failed_batches"],
                errors=list(self._errors),
                warnings=warnings,
            )
        finally:
            self._lock.release()

    def _append_error(self, message: str) -> None:
        with self._lock:
            self._errors.append(message)
            if len(self._errors) > 20:
                self._errors = self._errors[-20:]

    def _append_warning(self, message: str) -> tuple[str, dict[str, Any]]:
        with self._lock:
            self._warnings.append(message)
            if len(self._warnings) > 50:
                self._warnings = self._warnings[-50:]
            return self._queue_health_event_locked("warning", {"message": message})

    def _record_drop(self, event_type: str, message: str, **details: Any) -> None:
        with self._lock:
            event = self._record_drop_locked(event_type, message, **details)
        self._emit_health_events([event])

    def _record_drop_locked(
        self, event_type: str, message: str, **details: Any
    ) -> tuple[str, dict[str, Any]]:
        self._stats["dropped_items"] += 1
        self._dropped_by_reason[event_type] = (
            self._dropped_by_reason.get(event_type, 0) + 1
        )
        self._errors.append(message)
        if len(self._errors) > 20:
            self._errors = self._errors[-20:]
        return self._queue_health_event_locked(
            event_type,
            {
                "drop_reason": event_type,
                "dropped_items": self._stats["dropped_items"],
                "queue_depth": len(self._buffer),
                "message": message,
                **details,
            },
        )

    def _record_retry(self, error: Exception, attempt: int) -> None:
        del error, attempt
        with self._lock:
            self._stats["retry_attempts"] += 1

    def _queue_health_event_locked(
        self, event_type: str, payload: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Queue a state snapshot while holding ``_lock`` for ordered dispatch."""
        event = (event_type, payload)
        if self._health_callback is None:
            return event
        if self._health_dispatch_closed:
            self._dropped_health_events += 1
            return event
        self._next_health_event_sequence += 1
        if len(self._health_event_queue) >= 256:
            self._health_event_queue.popleft()
            self._dropped_health_events += 1
        self._health_event_queue.append(
            (self._next_health_event_sequence, event_type, payload)
        )
        return event

    def _emit_health_events(
        self,
        events: list[tuple[str, dict[str, Any]]],
        *,
        deadline: float | None = None,
    ) -> None:
        del events
        if self._health_callback is None:
            return
        if deadline is None:
            acquired = self._health_event_condition.acquire()
        else:
            acquired = self._health_event_condition.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            )
        if not acquired:
            return
        try:
            self._health_event_condition.notify()
        finally:
            self._health_event_condition.release()

    def _dispatch_health_events(self) -> None:
        """Deliver health events from one daemon thread, preserving queue order."""
        while True:
            with self._health_event_condition:
                while not self._health_event_queue:
                    self._health_event_condition.wait()
                queued = self._health_event_queue.popleft()
            if queued is _HEALTH_DISPATCH_STOP:
                return
            if isinstance(queued, threading.Event):
                queued.set()
                continue
            _, event_type, payload = queued
            callback = self._health_callback
            if callback is None:
                continue
            try:
                callback(event_type, payload)
            except Exception:
                logger.debug("Observability health callback failed", exc_info=True)

    def _stop_health_dispatcher(self, deadline: float | None) -> None:
        """Drain accepted events, then stop the per-transport dispatcher."""
        dispatcher = self._health_dispatcher_thread
        if dispatcher is None:
            return
        if not self._health_event_condition.acquire(
            timeout=(-1 if deadline is None else max(0.0, deadline - time.monotonic()))
        ):
            return
        try:
            if not self._health_dispatch_closed:
                self._health_dispatch_closed = True
                final_drain = threading.Event()
                self._health_event_queue.append(final_drain)
                self._health_event_queue.append(_HEALTH_DISPATCH_STOP)
                self._health_event_condition.notify_all()
        finally:
            self._health_event_condition.release()
        if dispatcher is not threading.current_thread():
            dispatcher.join(timeout=self._remaining_time(deadline))

    @staticmethod
    def _remaining_time(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _acquire_lock_until(self, deadline: float | None) -> bool:
        if deadline is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

    def _stats_snapshot_without_lock(self) -> dict[str, Any]:
        """Return the last state values when a bounded caller cannot lock them.

        All writers update these values while holding ``_lock``. A caller reaches
        this fallback only because another thread still owns that lock, so the
        values are a stable, if necessarily partial, snapshot for the deadline
        result rather than a fabricated success.
        """
        return {
            **self._stats,
            "queue_depth": len(self._buffer),
            "inflight_items": self._inflight_items,
            "send_in_progress": (
                self._active_send_completion is not None or self._send_lock.locked()
            ),
            "oldest_inflight_age_seconds": None,
            "dropped_by_reason": dict(self._dropped_by_reason),
            "dropped_health_events": self._dropped_health_events,
            "errors": list(self._errors),
            "warnings": list(self._warnings),
        }

    def _deadline_expiry_result(self, warning: str) -> BatchFlushResult:
        stats = self._stats_snapshot_without_lock()
        return BatchFlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )


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
        self._snapshot_submission_complete = threading.Event()
        self._snapshot_submission_complete.set()
        self._inflight_snapshot_submissions = 0
        self._closing = False
        # A close may stop new snapshot submissions before the transport can
        # acquire its own state lock. Keep that recoverable state separate
        # from a completed close so a later close() can retry transport.close().
        self._close_pending = False
        self._close_complete = threading.Event()
        self._close_complete.set()
        self._closed = False
        self._offline_notice_logged = False
        self._missing_generation_usage_notice_logged = False
        self._http_opener = request.build_opener(_NoRedirectHandler)
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
        execution_context: ExecutionContextDTO | dict[str, Any] | None = None,
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
            execution_context=self._coerce_execution_context(execution_context),
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
        observation_type: ObservationType | str | None = None,
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
        requested_type = (
            ObservationType(observation_type)
            if isinstance(observation_type, str)
            else observation_type
        )

        observation_id = observation_id or _new_observation_id()
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None:
                raise ValueError(f"Unknown trace_id '{trace_id}'")
            existing = state.observations.get(observation_id)
            effective_type = requested_type or (
                existing.type if existing is not None else ObservationType.SPAN
            )
            if tool_name is None and effective_type in {
                ObservationType.TOOL,
                ObservationType.TOOL_CALL,
            }:
                tool_name = name
            self._validate_observation_relationship(
                state,
                observation_id=observation_id,
                observation_type=effective_type,
                parent_observation_id=parent_observation_id,
            )

            if existing is None:
                observation = ObservationDTO(
                    id=observation_id,
                    type=effective_type,
                    name=name,
                    parent_observation_id=parent_observation_id,
                    status=status,
                    started_at=started_at or utc_now(),
                    ended_at=ended_at,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
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
                    type=effective_type,
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

            if (
                observation.type is ObservationType.GENERATION
                and observation.status in {"completed", "failed"}
                and observation.input_tokens is None
                and observation.output_tokens is None
                and observation.total_tokens is None
                and observation.cost_usd is None
                and not self._missing_generation_usage_notice_logged
            ):
                self._missing_generation_usage_notice_logged = True
                logger.warning(
                    "Generation observation '%s' has no token/cost usage fields; "
                    "usage will be reported as unknown. Pass input_tokens, "
                    "output_tokens, total_tokens, or cost_usd from your provider "
                    "response when recording generation spans.",
                    observation.name,
                )

        self._queue_trace_snapshot(trace_id)
        return observation_id

    def end_trace(
        self,
        trace_id: str,
        *,
        status: str = "completed",
        output_data: Any = None,
        ended_at: datetime | None = None,
        started_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if status not in OBSERVABILITY_STATUSES:
            allowed = ", ".join(sorted(OBSERVABILITY_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None:
                raise ValueError(f"Unknown trace_id '{trace_id}'")
            state.trace.status = status
            if started_at is not None:
                state.trace.started_at = started_at
            state.trace.ended_at = ended_at or utc_now()
            if output_data is not None:
                state.trace.output_data = output_data
            if metadata:
                merged = dict(state.trace.metadata)
                merged.update(metadata)
                state.trace.metadata = merged

        self._queue_trace_snapshot(trace_id)

    def flush(self, timeout: float | None = None) -> FlushResult:
        """Flush queued traces.

        Supplying ``timeout`` bounds delivery with a monotonic deadline. Omitting
        it preserves the historical synchronous flush behavior; ``timeout=0`` is
        an immediate poll that does not emit a deadline warning.
        Health callbacks are asynchronous, so this can return before callbacks
        for delivery events from this flush have run.
        """
        deadline = self._flush_deadline(timeout)
        if self.config.offline_mode and not self._has_custom_trace_sender:
            self._log_offline_mode_once()
            return self._offline_flush_result()
        if not self._acquire_client_lock_until(deadline):
            return self._flush_lock_timeout_result(deadline)
        try:
            trace_payloads = [
                (
                    trace_id,
                    cast(
                        dict[str, Any],
                        redact_sensitive_data(
                            state.to_payload(), redact_credential_keys=True
                        ),
                    ),
                )
                for trace_id, state in self._trace_states.items()
            ]
        finally:
            self._lock.release()

        for trace_id, payload in trace_payloads:
            self._submit_trace_snapshot(trace_id, payload, deadline=deadline)
        return self._flush_transport_until(deadline)

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of transport delivery and local-drop health.

        The snapshot includes ``dropped_by_reason``, ``dropped_health_events``,
        ``queue_depth``, ``inflight_items``, ``oldest_inflight_age_seconds``,
        and ``retry_attempts``. Configure
        :attr:`ObservabilityConfig.health_callback` to receive structured
        batch-level drop events. They run on one daemon dispatcher thread after
        internal transport locks are released; a bounded flush or close can
        return before callbacks for its events have run.
        """
        stats = cast(dict[str, Any], self._transport.get_stats())
        with self._lock:
            stats["active_trace_count"] = len(self._trace_states)
        return stats

    def close(self, timeout: float | None = None) -> FlushResult:
        """Stop submissions and flush queued traces.

        An explicit ``timeout`` bounds delivery; ``timeout=0`` is an immediate,
        warning-free poll. The atexit handler explicitly uses
        :attr:`ObservabilityConfig.flush_timeout` so interpreter shutdown stays
        bounded even though a normal no-argument close remains synchronous.
        Queued health callbacks accepted before an unbounded close are drained
        before it returns. A bounded close stops waiting at its deadline. Health
        events produced after dispatcher shutdown are counted in
        ``dropped_health_events``.
        """
        deadline = self._flush_deadline(timeout)
        if not self._acquire_client_lock_until(deadline):
            return self._close_lock_timeout_result(deadline)
        try:
            if self._closing:
                close_in_progress = True
                already_closed = False
                trace_payloads: list[tuple[str, dict[str, Any]]] = []
                inflight_snapshot_submissions = 0
                offline_close = False
            elif self._closed:
                close_in_progress = False
                already_closed = True
                trace_payloads = []
                inflight_snapshot_submissions = 0
                offline_close = False
            elif self._close_pending:
                close_in_progress = False
                already_closed = False
                self._closing = True
                self._close_complete.clear()
                offline_close = (
                    self.config.offline_mode and not self._has_custom_trace_sender
                )
                trace_payloads = []
                inflight_snapshot_submissions = 0
            else:
                close_in_progress = False
                already_closed = False
                self._closing = True
                self._close_pending = True
                self._close_complete.clear()
                offline_close = (
                    self.config.offline_mode and not self._has_custom_trace_sender
                )
                if offline_close:
                    trace_payloads = []
                    inflight_snapshot_submissions = 0
                else:
                    trace_payloads = [
                        (
                            trace_id,
                            cast(
                                dict[str, Any],
                                redact_sensitive_data(
                                    state.to_payload(), redact_credential_keys=True
                                ),
                            ),
                        )
                        for trace_id, state in self._trace_states.items()
                    ]
                    inflight_snapshot_submissions = self._inflight_snapshot_submissions
        finally:
            self._lock.release()
        if close_in_progress:
            return self._wait_for_concurrent_close(deadline)
        if already_closed:
            return self._flush_transport_until(deadline)

        result: FlushResult | None = None
        transport_close_completed = False
        finalizer_completed = False
        try:
            if offline_close:
                self._log_offline_mode_once()
                transport_result = self._transport.close(
                    timeout=self._remaining_flush_time(deadline),
                    warn_on_deadline_expiry=timeout != 0,
                )
                result = self._flush_result_from_transport(transport_result)
            else:
                if inflight_snapshot_submissions:
                    remaining = self._remaining_flush_time(deadline)
                    completed = (
                        self._snapshot_submission_complete.wait()
                        if remaining is None
                        else self._snapshot_submission_complete.wait(timeout=remaining)
                    )
                    if not completed:
                        logger.warning(
                            "Observability close deadline expired while waiting for "
                            "%d snapshot submissions",
                            inflight_snapshot_submissions,
                        )

                for trace_id, payload in trace_payloads:
                    self._submit_trace_snapshot(trace_id, payload, deadline=deadline)
                transport_result = self._transport.close(
                    timeout=self._remaining_flush_time(deadline)
                )
                result = self._flush_result_from_transport(transport_result)
            transport_close_completed = self._transport_close_completed(
                transport_result
            )
            if not transport_close_completed:
                result = self._transport_close_pending_result(result)
        finally:
            if self._acquire_client_lock_until(deadline):
                try:
                    self._finalize_close_state(transport_close_completed)
                    finalizer_completed = True
                finally:
                    self._lock.release()
            else:
                logger.warning(
                    "Observability close deadline exceeded while reacquiring the "
                    "client state lock for finalization"
                )
                threading.Thread(
                    target=self._finalize_close_when_client_lock_releases,
                    args=(transport_close_completed,),
                    name="traigent-observability-close-finalizer",
                    daemon=True,
                ).start()
        if not finalizer_completed:
            return self._close_finalizer_lock_timeout_result(deadline)
        if result is None:  # defensive: finalizer_completed implies a result was set
            return self._close_finalizer_lock_timeout_result(deadline)
        return result

    @staticmethod
    def _flush_result_from_transport(result: BatchFlushResult) -> FlushResult:
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

    def _flush_transport_until(self, deadline: float | None) -> FlushResult:
        """Return the current transport truth without waiting past ``deadline``."""
        result = self._transport.flush(timeout=self._remaining_flush_time(deadline))
        return self._flush_result_from_transport(result)

    def _transport_close_completed(self, result: BatchFlushResult) -> bool:
        """Whether transport close finished delivery and dispatcher teardown."""
        if not result.success:
            return False
        if not getattr(self._transport, "_closed", True):
            return False
        dispatcher = getattr(self._transport, "_health_dispatcher_thread", None)
        return dispatcher is None or not dispatcher.is_alive()

    @staticmethod
    def _transport_close_pending_result(result: FlushResult) -> FlushResult:
        """Keep a partial transport close visibly retryable to callers."""
        warning = (
            "Observability transport close is pending; the client remains closed "
            "to new snapshots until a later close completes teardown"
        )
        return FlushResult(
            success=False,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=[*result.warnings, warning],
        )

    def _flush_lock_timeout_result(self, deadline: float | None) -> FlushResult:
        """Report that flush never inspected client state or submitted snapshots."""
        stats = self._transport.get_stats(timeout=self._remaining_flush_time(deadline))
        warning = (
            "Observability flush deadline exceeded while acquiring the client "
            "state lock; no client snapshots were submitted"
        )
        return FlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )

    def _close_lock_timeout_result(self, deadline: float | None) -> FlushResult:
        """Report that close never acquired client state and changed nothing."""
        stats = self._transport.get_stats(timeout=self._remaining_flush_time(deadline))
        warning = (
            "Observability close deadline exceeded while acquiring the client "
            "state lock; client and transport remain open"
        )
        return FlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )

    def _close_finalizer_lock_timeout_result(
        self, deadline: float | None
    ) -> FlushResult:
        """Report a close whose transport work finished but state finalization did not."""
        stats = self._transport.get_stats(timeout=self._remaining_flush_time(deadline))
        warning = (
            "Observability close deadline exceeded while reacquiring the client "
            "state lock for finalization; close state remains in progress"
        )
        return FlushResult(
            success=False,
            items_sent=int(stats["sent_items"]),
            items_pending=int(stats["pending_items"]),
            items_dropped=int(stats["dropped_items"]),
            successful_batches=int(stats["successful_batches"]),
            failed_batches=int(stats["failed_batches"]),
            errors=list(stats["errors"]),
            warnings=[*stats["warnings"], warning],
        )

    def _finalize_close_state(self, transport_close_completed: bool) -> None:
        """Record whether transport teardown completed while holding client state."""
        self._closing = False
        self._close_pending = not transport_close_completed
        self._closed = transport_close_completed
        self._close_complete.set()

    def _finalize_close_when_client_lock_releases(
        self, transport_close_completed: bool
    ) -> None:
        """Complete state cleanup after a bounded close returned its partial result."""
        with self._lock:
            self._finalize_close_state(transport_close_completed)

    def _wait_for_concurrent_close(self, deadline: float | None) -> FlushResult:
        remaining = self._remaining_flush_time(deadline)
        completed = (
            self._close_complete.wait()
            if remaining is None
            else self._close_complete.wait(timeout=remaining)
        )
        result = self._flush_transport_until(deadline)
        if completed:
            return result
        warning = (
            "Observability close deadline exceeded while another close was running"
        )
        return FlushResult(
            success=False,
            items_sent=result.items_sent,
            items_pending=result.items_pending,
            items_dropped=result.items_dropped,
            successful_batches=result.successful_batches,
            failed_batches=result.failed_batches,
            errors=result.errors,
            warnings=[*result.warnings, warning],
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

    def iter_traces(
        self,
        *,
        per_page: int = 100,
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
    ) -> Iterator[TraceRecord]:
        """Iterate over *all* traces across pages, fetching lazily.

        Filter arguments are forwarded unchanged on every page request.

        Parameters
        ----------
        per_page:
            Items per page.  Defaults to 100 to minimise round-trips.
        search, environment, status, user_id, session_id, tags, release,
        model, start_time_from, start_time_to:
            Same filter semantics as :meth:`list_traces`.
        """
        yield from iter_pages(
            self.list_traces,
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

    def iter_sessions(
        self,
        *,
        per_page: int = 100,
        search: str | None = None,
        environment: str | None = None,
        user_id: str | None = None,
        tags: list[str] | str | None = None,
        release: str | None = None,
        start_time_from: datetime | None = None,
        start_time_to: datetime | None = None,
    ) -> Iterator[SessionRecord]:
        """Iterate over *all* sessions across pages, fetching lazily.

        Filter arguments are forwarded unchanged on every page request.

        Parameters
        ----------
        per_page:
            Items per page.  Defaults to 100 to minimise round-trips.
        search, environment, user_id, tags, release, start_time_from,
        start_time_to:
            Same filter semantics as :meth:`list_sessions`.
        """
        yield from iter_pages(
            self.list_sessions,
            per_page=per_page,
            search=search,
            environment=environment,
            user_id=user_id,
            tags=tags,
            release=release,
            start_time_from=start_time_from,
            start_time_to=start_time_to,
        )

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
            health_callback=self.config.health_callback,
        )

    def _send_payload_batch(
        self, traces: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if self._sender_override is not None:
            return self._sender_override(traces)

        return self._post_batch_sync(traces)

    @property
    def _has_custom_trace_sender(self) -> bool:
        return self._sender_override is not None

    def _post_batch_sync(self, traces: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.config.offline_mode:
            self._log_offline_mode_once()
            return None

        payload = json.dumps({"traces": traces}).encode("utf-8")
        http_request = request.Request(
            self.config.ingest_url,
            data=payload,
            headers=self.config.build_headers(),
            method="POST",
        )
        try:
            with self._open_http_request(http_request) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                if status_code >= 400:
                    raise self._client_error_with_retry_after(
                        f"Observability ingest failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                        headers=getattr(response, "headers", None),
                    )
                return self._parse_ingest_response(body)
        except error.HTTPError as exc:
            body = self._read_http_error_body(exc)
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Observability ingest rejected with status {exc.code}"
                ) from exc
            raise self._client_error_with_retry_after(
                f"Observability ingest failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
                headers=getattr(exc, "headers", None),
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
        with self._lock:
            if self._closed or self._close_pending:
                raise ClientError("Observability client is closed")
        if self.config.offline_mode:
            self._log_offline_mode_once()
            raise ClientError(
                "Observability backend request skipped because TRAIGENT_OFFLINE_MODE=true"
            )
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
            with self._open_http_request(http_request) as response:
                status_code = getattr(response, "status", 200)
                body = response.read().decode("utf-8") if response else ""
                parsed = json.loads(body) if body else {}
                if status_code >= 400:
                    raise self._client_error_with_retry_after(
                        f"Observability request failed with status {status_code}",
                        status_code=status_code,
                        details={"body": body},
                        headers=getattr(response, "headers", None),
                    )
                return parsed
        except error.HTTPError as exc:
            body = self._read_http_error_body(exc)
            if exc.code in {401, 403}:
                raise AuthenticationError(
                    f"Observability request rejected with status {exc.code}"
                ) from exc
            raise self._client_error_with_retry_after(
                f"Observability request failed with status {exc.code}",
                status_code=exc.code,
                details={"body": body},
                headers=getattr(exc, "headers", None),
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

    def _open_http_request(self, http_request: request.Request) -> Any:
        return self._http_opener.open(
            http_request,
            timeout=self.config.request_timeout,
        )

    @staticmethod
    def _validate_observation_relationship(
        state: _TraceState,
        *,
        observation_id: str,
        observation_type: ObservationType,
        parent_observation_id: str | None,
    ) -> None:
        if parent_observation_id is not None:
            parent = state.observations.get(parent_observation_id)
            if parent is not None and parent.type is ObservationType.EVENT:
                raise ValueError("event observations cannot have children")

        if observation_type is ObservationType.EVENT:
            has_children = any(
                existing_id != observation_id
                and observation.parent_observation_id == observation_id
                for existing_id, observation in state.observations.items()
            )
            if has_children:
                raise ValueError("event observations cannot have children")

    def _queue_trace_snapshot(self, trace_id: str) -> None:
        if self.config.offline_mode and not self._has_custom_trace_sender:
            return
        with self._lock:
            state = self._trace_states.get(trace_id)
            if state is None or self._closed or self._close_pending:
                return
            payload = cast(
                dict[str, Any],
                redact_sensitive_data(state.to_payload(), redact_credential_keys=True),
            )
            self._inflight_snapshot_submissions += 1
            self._snapshot_submission_complete.clear()
        try:
            self._submit_trace_snapshot(trace_id, payload)
        finally:
            with self._lock:
                self._inflight_snapshot_submissions -= 1
                if self._inflight_snapshot_submissions == 0:
                    self._snapshot_submission_complete.set()

    def _submit_trace_snapshot(
        self,
        trace_id: str,
        payload: dict[str, Any],
        *,
        deadline: float | None = None,
    ) -> None:
        submitted = (
            self._transport.submit(trace_id, payload)
            if deadline is None
            else self._transport.submit(trace_id, payload, deadline=deadline)
        )
        if not submitted:
            stats = (
                self._transport.get_stats()
                if deadline is None
                else self._transport.get_stats(
                    timeout=self._remaining_flush_time(deadline)
                )
            )
            errors = stats.get("errors") or []
            reason = errors[-1] if errors else "transport rejected trace snapshot"
            reason = redact_sensitive_text(str(reason))
            logger.warning(
                "Observability trace snapshot for %s was not queued: %s",
                trace_id,
                reason,
            )

    def _log_offline_mode_once(self) -> None:
        if self._offline_notice_logged:
            return
        self._offline_notice_logged = True
        logger.info("Observability transport in offline mode; traces will not be sent")

    @staticmethod
    def _offline_flush_result() -> FlushResult:
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

    @staticmethod
    def _client_error_with_retry_after(
        message: str,
        *,
        status_code: int,
        details: dict[str, Any],
        headers: Any,
    ) -> ClientError:
        exc = ClientError(message, status_code=status_code, details=details)
        retry_after = _parse_retry_after(headers)
        if retry_after is not None:
            exc.retry_after = retry_after
        return exc

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

    def _coerce_execution_context(
        self,
        execution_context: ExecutionContextDTO | dict[str, Any] | None,
    ) -> ExecutionContextDTO | None:
        defaults = dict(self.config.default_execution_context)
        if isinstance(execution_context, ExecutionContextDTO):
            explicit = execution_context.to_dict()
        else:
            explicit = dict(execution_context or {})
        merged = {**defaults, **explicit}
        if not merged:
            return None
        merged.setdefault("schema_version", "1.0")
        return ExecutionContextDTO.from_dict(merged)

    def _atexit_close(self) -> None:
        with self._lock:
            if self._closed:
                return
        try:
            self.close(timeout=self.config.flush_timeout)
        except Exception as exc:
            logger.warning("Observability atexit flush failed: %s", exc)

    def _flush_deadline(self, timeout: float | None) -> float | None:
        if timeout is None:
            return None
        return time.monotonic() + max(0.0, timeout)

    def _acquire_client_lock_until(self, deadline: float | None) -> bool:
        """Acquire client state without allowing a bounded call to exceed its budget."""
        if deadline is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _remaining_flush_time(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
