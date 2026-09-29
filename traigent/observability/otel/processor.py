"""Bounded, best-effort span processor with a single in-flight export.

Behaviour (all covered by tests):

* bounded queue: when full, NEW spans are dropped and counted (never blocks
  application code, never grows without bound);
* one worker thread, one export in flight; batches leave on a timer, when
  ``max_batch_spans`` or the estimated byte budget is reached, on ``flush`` or
  at shutdown;
* ``flush(timeout)`` has a hard deadline that does not depend on retry
  budgets: the exporter stops retrying at the deadline and the caller returns;
* process exit: an ``atexit`` hook flushes with ``exit_flush_timeout``;
* ``fork``: the child discards the parent's queue and lock state and starts a
  fresh worker lazily (the parent still exports what it queued);
* spans whose trace was not sampled (e.g. a foreign sampler's record-only
  spans) are never queued.

Delivery is best-effort; there is no durable outbox.  Losses are visible in
``stats()`` and via ``on_drop`` (process-local: a crash loses the counters too).
"""

from __future__ import annotations

import atexit
import os
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from traigent.observability.otel.exporter import ExportOutcome, TraigentOTLPExporter
from traigent.observability.otel.lineage import stamp_span
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FlushOutcome:
    flushed: bool
    timed_out: bool
    remaining_spans: int


_STAT_KEYS = (
    "queued",
    "exported",
    "dropped_queue_full",
    "dropped_retry_exhausted",
    "dropped_non_retryable",
    "dropped_deadline",
    "rejected_by_server",
    "sampled_out",
    "content_attrs_stripped",
    "retries",
)


def _estimate_bytes(span: ReadableSpan) -> int:
    return 200 + len(span.name or "") + 48 * len(span.attributes or {})


class TraigentSpanProcessor(SpanProcessor):
    def __init__(
        self,
        exporter: TraigentOTLPExporter,
        *,
        max_queue_spans: int = 10_000,
        schedule_delay_s: float = 5.0,
        max_batch_spans: int = 512,
        max_batch_bytes: int = 4 * 1024 * 1024,
        exit_flush: bool = True,
        exit_flush_timeout_s: float = 5.0,
        on_drop: Callable[[str, int], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_queue_spans <= 0 or max_batch_spans <= 0 or schedule_delay_s <= 0:
            raise ValueError("queue, batch and delay bounds must be positive")
        self._exporter = exporter
        self._max_queue = max_queue_spans
        self._delay = schedule_delay_s
        self._max_batch = max_batch_spans
        self._max_bytes = max_batch_bytes
        self._exit_timeout = exit_flush_timeout_s
        self._on_drop = on_drop
        self._clock = clock
        self._shutdown = False
        self._exit_flush = exit_flush
        self._init_state()
        self._pid = os.getpid()
        if hasattr(os, "register_at_fork"):
            ref = weakref.ref(self)

            def _after_fork() -> None:
                inst = ref()
                if inst is not None:
                    inst._after_fork_child()

            os.register_at_fork(after_in_child=_after_fork)
        if exit_flush:
            self._atexit = _make_exit_hook(weakref.ref(self))
            atexit.register(self._atexit)
        else:
            self._atexit = None

    # -- state ----------------------------------------------------------
    def _init_state(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._queue: deque[ReadableSpan] = deque()
        self._queued_bytes = 0
        self._first_queued_at: float | None = None
        self._in_flight = False
        self._flush_deadline: float | None = None
        self._worker: threading.Thread | None = None
        self._stats = dict.fromkeys(_STAT_KEYS, 0)

    def _after_fork_child(self) -> None:
        # The parent's lock may have been held by another thread at fork time
        # and its worker thread does not exist here: rebuild both.
        self._init_state()
        self._pid = os.getpid()

    # -- SpanProcessor API ---------------------------------------------
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        try:
            stamp_span(span, metadata_mode=self._exporter.content_mode)
        except Exception:  # never let stamping break the application
            logger.debug("lineage stamping failed", exc_info=True)

    def on_end(self, span: ReadableSpan) -> None:
        if self._shutdown:
            return
        ctx = span.context
        if ctx is None or not ctx.trace_flags.sampled:
            self._bump("sampled_out")
            return
        dropped = False
        with self._cond:
            if len(self._queue) >= self._max_queue:
                self._stats["dropped_queue_full"] += 1
                dropped = True
            else:
                if not self._queue:
                    self._first_queued_at = self._clock()
                self._queue.append(span)
                self._queued_bytes += _estimate_bytes(span)
                self._stats["queued"] += 1
                self._ensure_worker_locked()
                if (
                    len(self._queue) >= self._max_batch
                    or self._queued_bytes >= self._max_bytes
                ):
                    self._cond.notify_all()
        if dropped:
            self._notify_drop("queue_full", 1)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self.flush(self._exit_timeout)
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()
            worker = self._worker
        self._exporter.shutdown()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        if self._atexit is not None:
            atexit.unregister(self._atexit)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.flush(timeout_millis / 1000.0).flushed

    # -- public ---------------------------------------------------------
    def flush(self, timeout: float = 5.0) -> FlushOutcome:
        deadline = self._clock() + max(0.0, timeout)
        with self._cond:
            if self._pid != os.getpid():  # pragma: no cover - fork hook covers it
                self._after_fork_child()
            before = self._stats["dropped_deadline"]
            if self._queue or self._in_flight:
                self._flush_deadline = max(deadline, self._flush_deadline or 0.0)
                self._ensure_worker_locked()
                self._cond.notify_all()
            while self._queue or self._in_flight:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return FlushOutcome(False, True, len(self._queue))
                self._cond.wait(min(remaining, 0.05))
            if self._stats["dropped_deadline"] > before:
                # the export was abandoned at the deadline: nothing was flushed
                return FlushOutcome(False, True, 0)
            return FlushOutcome(True, False, 0)

    def stats(self) -> dict[str, Any]:
        with self._cond:
            snapshot: dict[str, Any] = dict(self._stats)
            snapshot["queue_depth"] = len(self._queue)
            snapshot["in_flight"] = self._in_flight
        return snapshot

    # -- worker ---------------------------------------------------------
    def _ensure_worker_locked(self) -> None:
        if self._pid != os.getpid():
            self._after_fork_child()
        if self._worker is None or not self._worker.is_alive():
            worker = threading.Thread(
                target=self._run, name="traigent-otel-export", daemon=True
            )
            self._worker = worker
            worker.start()

    def _should_export(self, now: float) -> bool:
        if not self._queue:
            return False
        if self._flush_deadline is not None:
            return True
        if len(self._queue) >= self._max_batch or self._queued_bytes >= self._max_bytes:
            return True
        return (
            self._first_queued_at is not None
            and now - self._first_queued_at >= self._delay
        )

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._shutdown and not self._should_export(self._clock()):
                    wait = self._delay
                    if self._first_queued_at is not None:
                        wait = max(
                            0.01, self._delay - (self._clock() - self._first_queued_at)
                        )
                    self._cond.wait(min(wait, self._delay))
                if self._shutdown and not self._queue:
                    return
                batch: list[ReadableSpan] = []
                size = 0
                while self._queue and len(batch) < self._max_batch:
                    item = self._queue.popleft()
                    batch.append(item)
                    size += _estimate_bytes(item)
                    if size >= self._max_bytes:
                        break
                self._queued_bytes = sum(_estimate_bytes(s) for s in self._queue)
                self._first_queued_at = self._clock() if self._queue else None
                deadline = self._flush_deadline
                self._in_flight = True
            outcome = ExportOutcome()
            try:
                outcome = self._exporter.export_batch(batch, deadline=deadline)
            except Exception:  # exporter bug must not kill the worker
                logger.debug("otel export failed unexpectedly", exc_info=True)
                outcome.dropped_non_retryable += len(batch)
            self._account(outcome)
            with self._cond:
                self._in_flight = False
                if not self._queue:
                    self._flush_deadline = None
                self._cond.notify_all()
                if self._shutdown and not self._queue:
                    return

    def _account(self, outcome: ExportOutcome) -> None:
        lost = {
            "retry_exhausted": outcome.dropped_retry_exhausted,
            "non_retryable": outcome.dropped_non_retryable,
            "deadline": outcome.dropped_deadline,
            "rejected_by_server": outcome.rejected_by_server,
        }
        with self._cond:
            s = self._stats
            s["exported"] += outcome.exported
            s["dropped_retry_exhausted"] += outcome.dropped_retry_exhausted
            s["dropped_non_retryable"] += outcome.dropped_non_retryable
            s["dropped_deadline"] += outcome.dropped_deadline
            s["rejected_by_server"] += outcome.rejected_by_server
            s["content_attrs_stripped"] += outcome.stripped_attrs
            s["retries"] += outcome.retries
        for reason, count in lost.items():
            if count:
                self._notify_drop(reason, count)

    def _bump(self, key: str) -> None:
        with self._cond:
            self._stats[key] += 1

    def _notify_drop(self, reason: str, count: int) -> None:
        if self._on_drop is None:
            return
        try:
            self._on_drop(reason, count)
        except Exception:  # user callback must never break delivery
            logger.debug("on_drop callback raised", exc_info=True)


def _make_exit_hook(
    ref: weakref.ReferenceType[TraigentSpanProcessor],
) -> Callable[[], None]:
    def _hook() -> None:
        proc = ref()
        if proc is not None and not proc._shutdown:
            try:
                proc.flush(proc._exit_timeout)
            except Exception:  # pragma: no cover - exit path
                pass

    return _hook
