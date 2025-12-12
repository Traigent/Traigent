"""Backend Synchronization for TraiGent SDK.

Focused component responsible for synchronizing session and trial states with backend,
extracted from the SessionLifecycleManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, cast

from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.retry import CLOUD_API_RETRY_CONFIG, RetryHandler
from traigent.utils.validation import CoreValidators, validate_or_raise

logger = get_logger(__name__)


class SyncStatus(Enum):
    """Synchronization status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SYNCED = "synced"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    success: bool
    session_id: str
    sync_type: str  # "session", "trial", "batch"
    sync_time: float
    error_message: str | None = None
    retries: int = 0
    items_synced: int = 0


class BackendSynchronizer:
    """Handles synchronization of session and trial states with backend services."""

    def __init__(
        self,
        max_concurrent_syncs: int = 5,
        sync_interval: float = 10.0,
        enable_auto_sync: bool = True,
        batch_size: int = 50,
    ) -> None:
        """Initialize backend synchronizer.

        Args:
            max_concurrent_syncs: Maximum concurrent sync operations
            sync_interval: Interval for automatic sync (seconds)
            enable_auto_sync: Enable automatic synchronization
            batch_size: Number of items to sync in each batch
        """
        validate_or_raise(
            CoreValidators.validate_positive_int(
                max_concurrent_syncs, "max_concurrent_syncs"
            )
        )
        validate_or_raise(
            CoreValidators.validate_positive_int(batch_size, "batch_size")
        )
        validate_or_raise(
            CoreValidators.validate_number(
                sync_interval, "sync_interval", min_value=0.0
            )
        )
        if sync_interval <= 0:
            raise ValidationException("sync_interval must be greater than 0")

        self.max_concurrent_syncs = max_concurrent_syncs
        self.sync_interval = sync_interval
        self.enable_auto_sync = enable_auto_sync
        self.batch_size = batch_size

        # Synchronization state
        self._sync_semaphore = asyncio.Semaphore(max_concurrent_syncs)
        self._sync_tasks: dict[str, asyncio.Task[SyncResult]] = {}
        self._sync_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_syncs: set[str] = set()
        self._session_snapshots: dict[str, dict[str, Any]] = {}

        # Retry handler for sync operations
        self._retry_handler = RetryHandler(CLOUD_API_RETRY_CONFIG)

        # Statistics (protected by _stats_lock for thread-safe access)
        self._stats: dict[str, Any] = {
            "total_sync_attempts": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "retried_syncs": 0,
            "items_synced": 0,
            "sync_duration_total": 0.0,
            "concurrent_syncs_peak": 0,
        }
        self._stats_lock = Lock()  # Protects _stats dict for thread-safe read/write

        # Background sync task
        self._background_sync_task: asyncio.Task[None] | None = None

        if self.enable_auto_sync:
            self._start_background_sync()

    async def sync_session_state(
        self, session_id: str, session_data: dict[str, Any], force: bool = False
    ) -> SyncResult:
        """Synchronize session state with backend.

        Ensures the background sync loop is running when automatic sync is enabled.

        Args:
            session_id: Session ID
            session_data: Session state data
            force: Force sync even if not dirty

        Returns:
            Synchronization result
        """
        if self.enable_auto_sync and (
            self._background_sync_task is None or self._background_sync_task.done()
        ):
            self._start_background_sync()
        # Validate inputs
        validate_or_raise(
            CoreValidators.validate_string_non_empty(session_id, "session_id")
        )
        validate_or_raise(
            CoreValidators.validate_type(session_data, dict, "session_data")
        )

        session_payload = self._clone_payload(session_data)
        self._session_snapshots[session_id] = session_payload

        # Check if already syncing
        if session_id in self._pending_syncs and not force:
            logger.debug(f"Session {session_id} sync already pending")
            return SyncResult(
                success=False,
                session_id=session_id,
                sync_type="session",
                sync_time=time.time(),
                error_message="Sync already in progress",
            )

        async with self._sync_semaphore:
            self._pending_syncs.add(session_id)
            start_time = time.time()

            try:
                # Update statistics (thread-safe)
                with self._stats_lock:
                    self._stats["total_sync_attempts"] += 1
                    current_concurrent = len(self._pending_syncs)
                    self._stats["concurrent_syncs_peak"] = max(
                        self._stats["concurrent_syncs_peak"], current_concurrent
                    )

                # Perform sync with retry
                result = await self._retry_handler.execute_async_with_result(
                    self._sync_session_to_backend,
                    session_id,
                    session_payload,
                    operation_id=f"sync_session_{session_id}",
                )

                if result.success:
                    sync_result = SyncResult(
                        success=True,
                        session_id=session_id,
                        sync_type="session",
                        sync_time=time.time(),
                        retries=result.attempts - 1,
                        items_synced=1,
                    )

                    with self._stats_lock:
                        self._stats["successful_syncs"] += 1
                        self._stats["items_synced"] += 1

                    logger.debug(f"Successfully synced session {session_id}")

                else:
                    sync_result = SyncResult(
                        success=False,
                        session_id=session_id,
                        sync_type="session",
                        sync_time=time.time(),
                        error_message=str(result.error),
                        retries=result.attempts - 1,
                    )

                    with self._stats_lock:
                        self._stats["failed_syncs"] += 1

                    logger.error(f"Failed to sync session {session_id}: {result.error}")

                # Update duration stats
                duration = time.time() - start_time
                with self._stats_lock:
                    self._stats["sync_duration_total"] += duration

                return sync_result

            finally:
                self._pending_syncs.discard(session_id)

    async def sync_trial_states(
        self, session_id: str, trial_data: list[dict[str, Any]]
    ) -> SyncResult:
        """Synchronize trial states with backend.

        Args:
            session_id: Session ID
            trial_data: List of trial state data

        Returns:
            Synchronization result
        """
        # Validate inputs
        validate_or_raise(
            CoreValidators.validate_string_non_empty(session_id, "session_id")
        )
        validate_or_raise(CoreValidators.validate_type(trial_data, list, "trial_data"))

        if not trial_data:
            return SyncResult(
                success=True,
                session_id=session_id,
                sync_type="trial",
                sync_time=time.time(),
                items_synced=0,
            )

        async with self._sync_semaphore:
            start_time = time.time()

            try:
                with self._stats_lock:
                    self._stats["total_sync_attempts"] += 1

                # Sync trials in batches
                total_synced = 0
                all_successful = True
                error_messages = []
                total_retries = 0

                for i in range(0, len(trial_data), self.batch_size):
                    batch = trial_data[i : i + self.batch_size]

                    result = await self._retry_handler.execute_async_with_result(
                        self._sync_trials_to_backend,
                        session_id,
                        batch,
                        operation_id=f"sync_trials_{session_id}_{i}",
                    )

                    if result.success:
                        total_synced += len(batch)
                        total_retries += result.attempts - 1
                    else:
                        all_successful = False
                        error_messages.append(str(result.error))
                        total_retries += result.attempts - 1

                # Create result
                sync_result = SyncResult(
                    success=all_successful,
                    session_id=session_id,
                    sync_type="trial",
                    sync_time=time.time(),
                    error_message="; ".join(error_messages) if error_messages else None,
                    retries=total_retries,
                    items_synced=total_synced,
                )

                # Update statistics (thread-safe)
                duration = time.time() - start_time
                with self._stats_lock:
                    if all_successful:
                        self._stats["successful_syncs"] += 1
                    else:
                        self._stats["failed_syncs"] += 1
                    self._stats["items_synced"] += total_synced
                    self._stats["sync_duration_total"] += duration

                logger.debug(
                    f"Synced {total_synced}/{len(trial_data)} trials for session {session_id}"
                )

                return sync_result

            except Exception as e:
                with self._stats_lock:
                    self._stats["failed_syncs"] += 1
                logger.error(f"Failed to sync trials for session {session_id}: {e}")

                return SyncResult(
                    success=False,
                    session_id=session_id,
                    sync_type="trial",
                    sync_time=time.time(),
                    error_message=str(e),
                )

    async def sync_batch_sessions(
        self, session_sync_data: dict[str, dict[str, Any]]
    ) -> list[SyncResult]:
        """Synchronize multiple sessions in batch.

        Args:
            session_sync_data: Dictionary mapping session IDs to sync data

        Returns:
            List of synchronization results
        """
        if not session_sync_data:
            return []

        logger.info(f"Starting batch sync for {len(session_sync_data)} sessions")

        # Create sync tasks for concurrent execution
        sync_tasks = []
        for session_id, data in session_sync_data.items():
            task = asyncio.create_task(
                self.sync_session_state(session_id, data), name=f"sync_{session_id}"
            )
            sync_tasks.append(task)

        # Wait for all syncs to complete
        results = await asyncio.gather(*sync_tasks, return_exceptions=True)

        # Process results
        sync_results = []
        for i, result in enumerate(results):
            session_id = list(session_sync_data.keys())[i]

            if isinstance(result, Exception):
                sync_results.append(
                    SyncResult(
                        success=False,
                        session_id=session_id,
                        sync_type="batch",
                        sync_time=time.time(),
                        error_message=str(result),
                    )
                )
            else:
                # Result should be a SyncResult since we handled exceptions above
                sync_results.append(result)  # type: ignore[arg-type]

        # Update batch statistics
        successful_count = sum(1 for r in sync_results if r.success)
        logger.info(
            f"Batch sync completed: {successful_count}/{len(sync_results)} successful"
        )

        return sync_results

    async def _sync_session_to_backend(
        self, session_id: str, session_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Internal method to sync session to backend.

        This is a placeholder implementation. In production, this would
        make actual API calls to the backend service.
        """
        # Simulate API call delay
        await asyncio.sleep(0.1)

        # Validate required fields
        required_fields = ["status", "function_name", "objectives"]
        for field in required_fields:
            if field not in session_data:
                raise ValueError(f"Missing required field: {field}") from None

        # Simulate success/failure based on session data
        if session_data.get("status") == "invalid":
            raise RuntimeError("Invalid session status")

        logger.debug(f"Synced session {session_id} to backend")
        return {"status": "synced", "backend_id": f"backend_{session_id}"}

    async def _sync_trials_to_backend(
        self, session_id: str, trial_batch: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Internal method to sync trials to backend.

        This is a placeholder implementation. In production, this would
        make actual API calls to the backend service.
        """
        # Simulate API call delay
        await asyncio.sleep(0.05 * len(trial_batch))

        # Validate trial data
        for trial in trial_batch:
            if "trial_id" not in trial:
                raise ValueError("Trial missing trial_id")
            if "status" not in trial:
                raise ValueError("Trial missing status")

        logger.debug(
            f"Synced {len(trial_batch)} trials for session {session_id} to backend"
        )
        return {"status": "synced", "trials_synced": len(trial_batch)}

    @staticmethod
    def _clone_payload(data: dict[str, Any]) -> dict[str, Any]:
        """Safely clone payload dictionaries to avoid external mutation."""
        try:
            return copy.deepcopy(data)
        except Exception as e:
            logger.debug(f"Could not deepcopy payload, using shallow copy: {e}")
            return dict(data)

    def _start_background_sync(self) -> None:
        """Start background synchronization task."""
        try:
            loop = asyncio.get_running_loop()
            if self._background_sync_task is None or self._background_sync_task.done():
                self._background_sync_task = loop.create_task(
                    self._background_sync_loop(), name="background_sync"
                )
                logger.info("Started background synchronization")
        except RuntimeError:
            # No running event loop, skip background sync
            logger.debug(
                "No running event loop, background sync will start when in async context"
            )

    async def _background_sync_loop(self) -> None:
        """Background synchronization loop."""
        while self.enable_auto_sync:
            try:
                # Wait for sync interval
                await asyncio.sleep(self.sync_interval)

                # Process any queued sync operations
                await self._process_sync_queue()

            except asyncio.CancelledError:
                logger.info("Background sync cancelled")
                break
            except Exception as e:
                logger.error(f"Background sync error: {e}")
                # Continue loop despite errors

    async def _process_sync_queue(self) -> None:
        """Process queued sync operations."""
        drained_items: list[dict[str, Any]] = []
        while True:
            try:
                item = self._sync_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                drained_items.append(item)

        if not drained_items:
            return

        def should_replace(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
            cand_priority = candidate.get("priority", 0)
            curr_priority = current.get("priority", 0)
            if cand_priority < curr_priority:
                return True
            if cand_priority > curr_priority:
                return False
            return cast(
                bool, candidate.get("queued_at", 0.0) > current.get("queued_at", 0.0)
            )

        session_items: dict[str, dict[str, Any]] = {}

        for item in drained_items:
            item_type = item.get("type")
            if item_type == "session":
                session_id = item.get("session_id")
                if not session_id:
                    logger.warning("Sync queue item missing session_id: %s", item)
                    continue

                existing = session_items.get(session_id)
                if existing is None or should_replace(item, existing):
                    session_items[session_id] = item
            else:
                logger.debug(
                    "Unsupported sync queue item type '%s'; skipping", item_type
                )

        for _ in drained_items:
            self._sync_queue.task_done()

        if not session_items:
            return

        ordered_items = sorted(
            session_items.values(),
            key=lambda data: (data.get("priority", 0), data.get("queued_at", 0.0)),
        )

        requeue_items: list[dict[str, Any]] = []
        tasks: list[asyncio.Task[Any]] = []
        task_sessions: list[str] = []
        task_payloads: list[dict[str, Any]] = []  # S4 fix: Track payloads for requeue

        for item in ordered_items:
            session_id = item["session_id"]

            if session_id in self._pending_syncs:
                item["queued_at"] = time.time()
                requeue_items.append(item)
                continue

            existing_task = self._sync_tasks.get(session_id)
            if existing_task is not None:
                if existing_task.done():
                    self._sync_tasks.pop(session_id, None)
                else:
                    item["queued_at"] = time.time()
                    requeue_items.append(item)
                    continue

            payload = item.get("session_data")
            if payload is None:
                payload = self._session_snapshots.get(session_id)

            if payload is None:
                # S2 fix: Don't silently drop syncs - this indicates data loss
                logger.warning(
                    "SYNC DATA LOSS: No cached session data available for %s. "
                    "The sync was queued but neither item nor snapshot contained data. "
                    "This may indicate a bug in snapshot management.",
                    session_id,
                )
                # Track the dropped sync for debugging
                with self._stats_lock:
                    self._stats["dropped_syncs"] = (
                        self._stats.get("dropped_syncs", 0) + 1
                    )
                continue

            task = asyncio.create_task(
                self.sync_session_state(
                    session_id, self._clone_payload(payload), force=True
                ),
                name=f"queued_sync_{session_id}",
            )
            self._sync_tasks[session_id] = task
            tasks.append(task)
            task_sessions.append(session_id)
            task_payloads.append(payload)  # S4 fix: Track payload for potential requeue

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # S4 fix: Include payload in zip to preserve data for requeue
            for session_id, result, payload in zip(
                task_sessions, results, task_payloads, strict=False
            ):
                self._sync_tasks.pop(session_id, None)

                if isinstance(result, Exception):
                    logger.error(
                        "Queued session sync for %s failed with error: %s",
                        session_id,
                        result,
                    )
                    # S4 fix: Include session_data to prevent data loss on retry
                    requeue_items.append(
                        {
                            "type": "session",
                            "session_id": session_id,
                            "priority": 0,
                            "queued_at": time.time(),
                            "session_data": payload,
                        }
                    )
                elif isinstance(result, SyncResult):
                    if not result.success:
                        logger.warning(
                            "Queued session sync for %s did not succeed: %s",
                            session_id,
                            result.error_message,
                        )
                        # Use tracked payload instead of potentially stale snapshot
                        requeue_items.append(
                            {
                                "type": "session",
                                "session_id": session_id,
                                "priority": 0,
                                "queued_at": time.time(),
                                "session_data": payload,
                            }
                        )

        for item in requeue_items:
            item.setdefault("queued_at", time.time())
            try:
                self._sync_queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.warning(
                    "Sync queue full, dropping requeued session %s",
                    item.get("session_id"),
                )

    def queue_session_sync(
        self,
        session_id: str,
        priority: int = 0,
        session_data: dict[str, Any] | None = None,
    ) -> None:
        """Queue a session for synchronization.

        Args:
            session_id: Session ID to sync
            priority: Sync priority (lower = higher priority)
            session_data: Optional snapshot of session state to sync
        """
        try:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(session_id, "session_id")
            )
            if priority < 0:
                raise ValidationException("priority must be non-negative")
            session_payload = None
            if session_data is not None:
                validate_or_raise(
                    CoreValidators.validate_type(session_data, dict, "session_data")
                )
                session_payload = self._clone_payload(session_data)
                self._session_snapshots[session_id] = session_payload

            sync_item = {
                "type": "session",
                "session_id": session_id,
                "priority": priority,
                "queued_at": time.time(),
            }
            if session_payload is not None:
                sync_item["session_data"] = session_payload

            self._sync_queue.put_nowait(sync_item)
            logger.debug(f"Queued session {session_id} for sync")

        except asyncio.QueueFull:
            logger.warning(f"Sync queue full, dropping session {session_id}")

    def stop_background_sync(self) -> None:
        """Stop background synchronization."""
        self.enable_auto_sync = False

        if self._background_sync_task and not self._background_sync_task.done():
            self._background_sync_task.cancel()
            logger.info("Stopped background synchronization")

    async def wait_for_pending_syncs(self, timeout: float = 30.0) -> bool:
        """Wait for all pending synchronizations to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all syncs completed, False if timeout
        """
        start_time = time.time()

        while self._pending_syncs and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        success = len(self._pending_syncs) == 0

        if not success:
            logger.warning(
                f"Timeout waiting for syncs: {len(self._pending_syncs)} pending"
            )

        return success

    def get_sync_status(self) -> dict[str, Any]:
        """Get current synchronization status."""
        # Take a snapshot of stats under lock to ensure consistency
        with self._stats_lock:
            stats_snapshot = self._stats.copy()

        # Compute derived values outside lock
        total_attempts = stats_snapshot["total_sync_attempts"]
        avg_duration = (
            stats_snapshot["sync_duration_total"] / total_attempts
            if total_attempts > 0
            else 0
        )
        success_rate = (
            stats_snapshot["successful_syncs"] / total_attempts
            if total_attempts > 0
            else 0
        )

        return {
            "pending_syncs": len(self._pending_syncs),
            "queue_size": self._sync_queue.qsize(),
            "background_sync_active": (
                self._background_sync_task and not self._background_sync_task.done()
            ),
            "statistics": {
                **stats_snapshot,
                "success_rate": success_rate,
                "average_sync_duration": avg_duration,
            },
        }

    def cleanup(self) -> None:
        """Clean up synchronizer resources."""
        self.stop_background_sync()

        # Cancel all pending sync tasks
        for task in self._sync_tasks.values():
            if not task.done():
                task.cancel()

        self._sync_tasks.clear()
        self._pending_syncs.clear()
        self._session_snapshots.clear()

        logger.info("Backend synchronizer cleaned up")
