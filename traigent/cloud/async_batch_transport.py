"""Reusable async batch transport for background backend delivery."""

from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from traigent.utils.logging import get_logger
from traigent.utils.retry import CLOUD_API_RETRY_CONFIG, RetryConfig, RetryHandler
from traigent.utils.validation import CoreValidators, validate_or_raise

logger = get_logger(__name__)


@dataclass
class BatchFlushResult:
    """Structured flush result surfaced to callers."""

    success: bool
    items_sent: int
    items_pending: int
    items_dropped: int
    successful_batches: int
    failed_batches: int
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AsyncBatchTransport:
    """Async transport that coalesces payloads and delivers them in batches."""

    def __init__(
        self,
        sender,
        *,
        batch_size: int = 100,
        max_buffer_age: float = 5.0,
        max_queue_size: int = 10_000,
        retry_config: RetryConfig | None = None,
    ) -> None:
        validate_or_raise(
            CoreValidators.validate_positive_int(batch_size, "batch_size")
        )
        validate_or_raise(
            CoreValidators.validate_positive_int(max_queue_size, "max_queue_size")
        )
        validate_or_raise(
            CoreValidators.validate_number(
                max_buffer_age, "max_buffer_age", min_value=0.0
            )
        )
        if max_buffer_age <= 0:
            raise ValueError("max_buffer_age must be greater than 0")

        self._sender = sender
        self.batch_size = batch_size
        self.max_buffer_age = max_buffer_age
        self.max_queue_size = max_queue_size

        self._retry_handler = RetryHandler(retry_config or CLOUD_API_RETRY_CONFIG)
        self._buffer: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats_lock = Lock()
        self._flush_timer_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
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

    async def submit(self, item_id: str, payload: dict[str, Any]) -> bool:
        """Queue or replace a payload for delivery."""
        async with self._lock:
            if self._closed:
                return False

            if item_id in self._buffer:
                self._buffer.pop(item_id, None)
            elif len(self._buffer) >= self.max_queue_size:
                with self._stats_lock:
                    self._stats["dropped_items"] += 1
                self._append_error(
                    f"transport queue full; dropped payload for item '{item_id}'"
                )
                return False

            self._buffer[item_id] = copy.deepcopy(payload)
            with self._stats_lock:
                self._stats["submitted_items"] += 1
                self._stats["pending_items"] = len(self._buffer)

            if len(self._buffer) >= self.batch_size:
                self._cancel_timer_locked()
                self._ensure_send_task_locked()
            elif self._flush_timer_task is None or self._flush_timer_task.done():
                self._flush_timer_task = asyncio.create_task(self._flush_after_delay())

        return True

    async def flush(self) -> BatchFlushResult:
        """Flush buffered payloads and await any in-flight send."""
        while True:
            async with self._lock:
                self._cancel_timer_locked()
                if self._send_task is None or self._send_task.done():
                    if not self._buffer:
                        return self._build_result()
                    self._ensure_send_task_locked()
                send_task = self._send_task

            if send_task is not None:
                await send_task

    async def close(self) -> BatchFlushResult:
        """Flush remaining payloads and stop the transport."""
        async with self._lock:
            self._closed = True
            self._cancel_timer_locked()

        return await self.flush()

    def get_stats(self) -> dict[str, Any]:
        """Return a thread-safe stats snapshot."""
        with self._stats_lock:
            snapshot: dict[str, Any] = dict(self._stats)
        snapshot["errors"] = list(self._errors)
        snapshot["warnings"] = list(self._warnings)
        return snapshot

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self.max_buffer_age)
            async with self._lock:
                if self._closed or not self._buffer:
                    return
                self._ensure_send_task_locked()
                send_task = self._send_task
            if send_task is not None:
                await send_task
        except asyncio.CancelledError:
            return

    async def _send_available(self) -> None:
        try:
            while True:
                async with self._lock:
                    if not self._buffer:
                        self._send_task = None
                        return

                    batch_items = []
                    for _ in range(min(self.batch_size, len(self._buffer))):
                        item_id, payload = self._buffer.popitem(last=False)
                        batch_items.append((item_id, payload))
                    with self._stats_lock:
                        self._stats["pending_items"] = len(self._buffer)

                await self._send_batch(batch_items)
        finally:
            async with self._lock:
                if self._send_task is not None and self._send_task.done():
                    self._send_task = None

    async def _send_batch(self, batch_items: list[tuple[str, dict[str, Any]]]) -> None:
        payloads = [payload for _, payload in batch_items]
        result = await self._retry_handler.execute_async_with_result(
            self._sender,
            payloads,
        )

        if result.success:
            if isinstance(result.result, dict):
                for warning in result.result.get("warnings") or []:
                    self._append_warning(str(warning))
            with self._stats_lock:
                self._stats["successful_batches"] += 1
                self._stats["sent_items"] += len(payloads)
            return

        with self._stats_lock:
            self._stats["failed_batches"] += 1
            self._stats["dropped_items"] += len(payloads)
        self._append_error(str(result.error or "batch delivery failed"))
        logger.warning(
            "AsyncBatchTransport dropped %d payloads after retries: %s",
            len(payloads),
            result.error,
        )

    def _ensure_send_task_locked(self) -> None:
        if self._send_task is None or self._send_task.done():
            self._send_task = asyncio.create_task(self._send_available())

    def _cancel_timer_locked(self) -> None:
        current = asyncio.current_task()
        if (
            self._flush_timer_task is not None
            and not self._flush_timer_task.done()
            and self._flush_timer_task is not current
        ):
            self._flush_timer_task.cancel()
        self._flush_timer_task = None

    def _build_result(self) -> BatchFlushResult:
        with self._stats_lock:
            pending_items = self._stats["pending_items"]
            sent_items = self._stats["sent_items"]
            dropped_items = self._stats["dropped_items"]
            successful_batches = self._stats["successful_batches"]
            failed_batches = self._stats["failed_batches"]
        return BatchFlushResult(
            success=failed_batches == 0,
            items_sent=sent_items,
            items_pending=pending_items,
            items_dropped=dropped_items,
            successful_batches=successful_batches,
            failed_batches=failed_batches,
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
