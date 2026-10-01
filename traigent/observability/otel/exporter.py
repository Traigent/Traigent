"""OTLP span exporter: Traigent content policy + official protobuf encoders.

Retry contract (PLAN-v2, from the public OTLP/HTTP specification):

* retried: HTTP 429, 502, 503, 504 and network errors;
* ``Retry-After`` honoured, capped at ``retry_after_cap`` seconds;
* full-jitter exponential backoff, at most ``max_attempts`` attempts, a batch
  is abandoned once older than ``max_batch_age`` seconds;
* 413 splits the batch in two ONCE; a second 413 drops the part;
* 200 with ``partial_success`` is final (never retried); a non-empty 200 body
  that is not a valid ``ExportTraceServiceResponse`` is a protocol failure, not
  a success (an empty body is success);
* a single span whose encoded request exceeds ``max_batch_bytes`` is dropped
  and counted before transport (no request ever exceeds the cap);
* a shut-down exporter never starts another POST;
* every other status (400/401/403/404/408/500/501 ...) is dropped and counted.

Every wait is bounded by an optional absolute ``deadline`` that is independent
of the retry budget (used by ``flush`` and exit hooks).  The deadline may be a
callable so an export already in flight sees a deadline set LATER by ``flush``.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from traigent.observability.otel.policy import ContentPolicy
from traigent.observability.otel.transport import (
    Transport,
    TransportError,
    TransportResponse,
)

RETRYABLE_STATUS = frozenset({429, 502, 503, 504})

#: An absolute deadline, or a callable returning the CURRENT one (or ``None``).
Deadline = float | Callable[[], float | None] | None


@dataclass
class ExportOutcome:
    exported: int = 0
    rejected_by_server: int = 0
    dropped_retry_exhausted: int = 0
    dropped_non_retryable: int = 0
    dropped_deadline: int = 0
    retries: int = 0
    stripped_attrs: int = 0
    protocol_failures: int = 0

    def add(self, other: ExportOutcome) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


class TraigentOTLPExporter(SpanExporter):
    def __init__(
        self,
        transport: Transport,
        *,
        content_mode: str = "metadata",
        allowed_span_names: Collection[str] | None = None,
        max_batch_bytes: int = 4 * 1024 * 1024,
        export_timeout: float = 10.0,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        retry_after_cap: float = 60.0,
        max_batch_age: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._transport = transport
        self.policy = ContentPolicy(content_mode, allowed_span_names)
        self._max_batch_bytes = max_batch_bytes
        self._timeout = export_timeout
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._retry_after_cap = retry_after_cap
        self._max_batch_age = max_batch_age
        self._clock = clock
        self._stop = threading.Event()
        self._wake = threading.Event()  # set by shutdown() and by wake()
        self._injected_sleep = sleep
        self._rng = rng

    @property
    def content_mode(self) -> str:
        return self.policy.mode

    def wake(self) -> None:
        """Ask a backoff wait to re-read its deadline (a flush just arrived)."""
        self._wake.set()

    def _backoff_wait(self, delay: float, deadline: Deadline) -> None:
        """Wait ``delay`` seconds, ending early on shutdown or at the live deadline.

        The deadline is re-read on every wake, so a flush that lands DURING the
        wait bounds it: a 60 s Retry-After cannot outlive a 1 s flush.
        """
        end = self._clock() + delay
        while not self._stop.is_set():
            left = end - self._clock()
            remaining = self._remaining(deadline)
            if remaining is not None:
                left = min(left, remaining)
            if left <= 0:
                return
            self._wake.wait(left)
            self._wake.clear()

    # -- public API -----------------------------------------------------
    def export_batch(
        self, spans: Sequence[ReadableSpan], deadline: Deadline = None
    ) -> ExportOutcome:
        outcome = ExportOutcome()
        if not spans:
            return outcome
        sanitized = [self.policy.sanitize(s) for s in spans]
        for span in sanitized:
            outcome.stripped_attrs += int(
                (span.attributes or {}).get("traigent.content.dropped_attrs", 0)
            )
        self._send_sized(sanitized, deadline, outcome)
        return outcome

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        outcome = self.export_batch(spans)
        lost = (
            outcome.dropped_retry_exhausted
            + outcome.dropped_non_retryable
            + outcome.dropped_deadline
            + outcome.protocol_failures
        )
        return SpanExportResult.FAILURE if lost else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    # -- internals ------------------------------------------------------
    def _send_sized(
        self,
        spans: list[ReadableSpan],
        deadline: Deadline,
        outcome: ExportOutcome,
    ) -> None:
        """Encode; halve until each request fits ``max_batch_bytes``."""
        body = encode_spans(spans).SerializeToString()
        if len(body) > self._max_batch_bytes:
            if len(spans) > 1:
                mid = len(spans) // 2
                self._send_sized(spans[:mid], deadline, outcome)
                self._send_sized(spans[mid:], deadline, outcome)
                return
            # one span alone is over the cap: never send it (the cap is a hard
            # bound on every request), count it as a non-retryable loss
            outcome.dropped_non_retryable += 1
            return
        self._send_with_retry(spans, body, deadline, outcome, may_split=True)

    def _remaining(self, deadline: Deadline) -> float | None:
        current = deadline() if callable(deadline) else deadline
        return None if current is None else current - self._clock()

    def _send_with_retry(
        self,
        spans: list[ReadableSpan],
        body: bytes,
        deadline: Deadline,
        outcome: ExportOutcome,
        *,
        may_split: bool,
    ) -> None:
        started = self._clock()
        count = len(spans)
        for attempt in range(self._max_attempts):
            if self._stop.is_set():  # shut down: no further POST, ever
                outcome.dropped_deadline += count
                return
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                outcome.dropped_deadline += count
                return
            timeout = (
                self._timeout if remaining is None else min(self._timeout, remaining)
            )
            try:
                resp = self._transport.post(body, timeout=timeout)
            except TransportError:
                resp = None
            if resp is not None:
                if resp.status == 200:
                    self._record_success(resp, count, outcome)
                    return
                if resp.status == 413:
                    if may_split and count > 1:
                        mid = count // 2
                        for part in (spans[:mid], spans[mid:]):
                            self._send_with_retry(
                                part,
                                encode_spans(part).SerializeToString(),
                                deadline,
                                outcome,
                                may_split=False,
                            )
                    else:
                        outcome.dropped_non_retryable += count
                    return
                if resp.status not in RETRYABLE_STATUS:
                    outcome.dropped_non_retryable += count
                    return
            if attempt + 1 >= self._max_attempts:
                break
            delay = self._delay(attempt, resp)
            remaining = self._remaining(deadline)
            if remaining is not None and delay >= remaining:
                outcome.dropped_deadline += count
                return
            if self._clock() + delay - started > self._max_batch_age:
                outcome.dropped_retry_exhausted += count
                return
            outcome.retries += 1
            if self._injected_sleep is not None:
                self._injected_sleep(delay)
            else:
                self._backoff_wait(delay, deadline)
        outcome.dropped_retry_exhausted += count

    def _delay(self, attempt: int, resp: TransportResponse | None) -> float:
        if resp is not None and resp.retry_after is not None:
            return min(max(resp.retry_after, 0.0), self._retry_after_cap)
        ceiling = min(self._backoff_max, self._backoff_base * (2**attempt))
        return float(ceiling * self._rng())

    @staticmethod
    def _record_success(
        resp: TransportResponse, count: int, outcome: ExportOutcome
    ) -> None:
        rejected = 0
        if resp.body:
            try:
                parsed = ExportTraceServiceResponse()
                parsed.ParseFromString(resp.body)
                rejected = max(0, int(parsed.partial_success.rejected_spans))
            except Exception:
                # a garbled acknowledgement proves nothing was accepted
                outcome.protocol_failures += count
                return
        rejected = min(rejected, count)
        outcome.rejected_by_server += rejected
        outcome.exported += count - rejected
