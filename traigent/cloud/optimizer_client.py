"""Direct client for Traigent Optimizer metric submission in hybrid mode."""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import asyncio
import logging
from typing import Any, cast

import backoff

from traigent.cloud._aiohttp_compat import AIOHTTP_AVAILABLE, aiohttp

logger = logging.getLogger(__name__)

# Error message for uninitialized session
_SESSION_NOT_INITIALIZED = "Session not initialized"


class OptimizerDirectClient:
    """Direct client for optimizer metric submission in hybrid mode."""

    def __init__(self, endpoint: str, token: str) -> None:
        """Initialize optimizer direct client.

        Args:
            endpoint: Optimizer endpoint URL
            token: JWT token for authentication
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for OptimizerDirectClient. Install the optional dependency."
            )

        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("Optimizer endpoint must be a non-empty string")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Optimizer token must be a non-empty string")

        self.endpoint = endpoint.strip().rstrip("/")
        self.token = token.strip()
        self.session: aiohttp.ClientSession | None = None
        self._metric_buffer: list[Any] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_interval = 5.0  # seconds
        self._batch_size = 100
        self._flush_task: asyncio.Task[None] | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )
        # Start background flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining metrics
        await self.flush()

        # Close session
        if self.session:
            await self.session.close()

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30,
    )
    async def submit_metrics(
        self,
        session_id: str,
        trial_id: str,
        metrics: dict[str, float],
        execution_time: float,
        metadata: dict[str, Any | None] = None,
    ) -> dict[str, Any]:
        """Submit metrics to optimizer.

        Uses batching and retry logic for reliability.

        Args:
            session_id: Hybrid session ID
            trial_id: Trial/configuration ID
            metrics: Evaluation metrics
            execution_time: Execution time in seconds
            metadata: Additional metadata

        Returns:
            Submission response
        """
        submission = {
            "trial_id": trial_id,
            "metrics": metrics,
            "execution_time": execution_time,
            "metadata": metadata or {},
        }

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(trial_id, str) or not trial_id.strip():
            raise ValueError("trial_id must be a non-empty string")
        session_id = session_id.strip()
        trial_id = trial_id.strip()

        # Add to buffer
        async with self._buffer_lock:
            self._metric_buffer.append((session_id, submission))
            buffer_size = len(self._metric_buffer)

            # Flush if buffer is full or this is the first item
            if buffer_size >= self._batch_size or buffer_size == 1:
                return await self._flush_buffer()

        return {
            "status": "buffered",
            "message": "Metric buffered for batch submission",
            "buffer_size": buffer_size,
        }

    async def get_next_configuration(self, session_id: str) -> dict[str, Any]:
        """Get next configuration to execute.

        Args:
            session_id: Hybrid session ID

        Returns:
            Next configuration or completion status
        """
        if self.session is None:
            raise RuntimeError(_SESSION_NOT_INITIALIZED)
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        session_id = session_id.strip()
        try:
            async with self.session.get(
                f"{self.endpoint}/{session_id}/next"
            ) as response:
                if response.status == 403:
                    error_text = await response.text()
                    excerpt = self._first_error_line(error_text)
                    raise ValueError(
                        f"Requesting next configuration for session {session_id} returned 403 Forbidden. "
                        "Hybrid optimizer endpoints now require an owner or admin token. "
                        f"Backend response: {excerpt}"
                    )
                response.raise_for_status()
                return cast(dict[str, Any], await response.json())

        except aiohttp.ClientError as e:
            logger.error(f"Failed to get next configuration: {str(e)}")
            raise

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get metrics submission status for session.

        Args:
            session_id: Hybrid session ID

        Returns:
            Session status
        """
        if self.session is None:
            raise RuntimeError(_SESSION_NOT_INITIALIZED)
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        session_id = session_id.strip()
        try:
            async with self.session.get(
                f"{self.endpoint}/{session_id}/status"
            ) as response:
                if response.status == 403:
                    error_text = await response.text()
                    excerpt = self._first_error_line(error_text)
                    raise ValueError(
                        f"Fetching session status for {session_id} returned 403 Forbidden. "
                        "Use the session owner's token or an admin role when polling status. "
                        f"Backend response: {excerpt}"
                    )
                response.raise_for_status()
                return cast(dict[str, Any], await response.json())

        except aiohttp.ClientError as e:
            logger.error(f"Failed to get session status: {str(e)}")
            raise

    async def flush(self) -> None:
        """Force flush all buffered metrics."""
        async with self._buffer_lock:
            if self._metric_buffer:
                await self._flush_buffer()

    async def _flush_buffer(self) -> dict[str, Any]:
        """Flush metric buffer to optimizer.

        Returns:
            Flush response
        """
        if not self._metric_buffer:
            return {"status": "no_metrics", "message": "No metrics to flush"}

        # Group by session
        sessions: dict[str, list[Any]] = {}
        for session_id, submission in self._metric_buffer:
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(submission)

        # Submit each session's metrics
        results = []
        errors = []

        for session_id, submissions in sessions.items():
            try:
                if len(submissions) == 1:
                    # Single metric submission
                    result = await self._submit_single(session_id, submissions[0])
                else:
                    # Batch submission
                    result = await self._submit_batch(session_id, submissions)
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Failed to submit metrics for session {session_id}: {str(e)}"
                )
                errors.append(
                    {
                        "session_id": session_id,
                        "error": str(e),
                        "metrics_count": len(submissions),
                    }
                )

        # Clear buffer
        self._metric_buffer.clear()

        # Return appropriate response
        if not errors:
            if len(results) == 1:
                return results[0]
            else:
                return {
                    "status": "batch_flushed",
                    "sessions_processed": len(results),
                    "results": results,
                }
        else:
            return {
                "status": "partial_success",
                "successful": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors,
            }

    async def _submit_single(
        self, session_id: str, submission: dict[str, Any]
    ) -> dict[str, Any]:
        """Submit single metric.

        Args:
            session_id: Session ID
            submission: Metric submission

        Returns:
            Submission response
        """
        if self.session is None:
            raise RuntimeError(_SESSION_NOT_INITIALIZED)
        try:
            async with self.session.post(
                f"{self.endpoint}/{session_id}", json=submission
            ) as response:
                if response.status == 403:
                    error_text = await response.text()
                    excerpt = self._first_error_line(error_text)
                    raise ValueError(
                        f"Submitting metrics for session {session_id} returned 403 Forbidden. "
                        "Ensure the optimizer token belongs to the session owner or has admin scope. "
                        f"Backend response: {excerpt}"
                    )
                response.raise_for_status()
                return cast(dict[str, Any], await response.json())

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                raise ValueError(
                    "Authentication failed - token may be expired"
                ) from None
            elif e.status == 403:
                raise ValueError("Session ID mismatch or access denied") from e
            else:
                raise

    async def _submit_batch(
        self, session_id: str, submissions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Submit batch of metrics.

        Args:
            session_id: Session ID
            submissions: List of metric submissions

        Returns:
            Batch response
        """
        if self.session is None:
            raise RuntimeError(_SESSION_NOT_INITIALIZED)
        try:
            async with self.session.post(
                f"{self.endpoint}/{session_id}/batch", json={"submissions": submissions}
            ) as response:
                if response.status == 403:
                    error_text = await response.text()
                    excerpt = self._first_error_line(error_text)
                    raise ValueError(
                        f"Submitting metric batch for session {session_id} returned 403 Forbidden. "
                        "Provide session-owner credentials or admin privileges. "
                        f"Backend response: {excerpt}"
                    )
                response.raise_for_status()
                return cast(dict[str, Any], await response.json())

        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                raise ValueError(
                    "Authentication failed - token may be expired"
                ) from None
            elif e.status == 403:
                raise ValueError("Session ID mismatch or access denied") from e
            else:
                raise

    @staticmethod
    def _first_error_line(error_text: str | None) -> str:
        """Return a safe, trimmed first line of an error response."""

        if not error_text:
            return ""

        lines = error_text.strip().splitlines()
        if not lines:
            return ""

        excerpt = lines[0].strip()
        if len(excerpt) > 200:
            return f"{excerpt[:197]}..."
        return excerpt

    async def _periodic_flush(self) -> None:
        """Background task to periodically flush metrics."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)

                # Check if we have metrics to flush
                async with self._buffer_lock:
                    if self._metric_buffer:
                        logger.debug(
                            f"Periodic flush: {len(self._metric_buffer)} metrics"
                        )

                await self.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush failed: {str(e)}")
                # Continue running even if flush fails

    def set_batch_size(self, size: int) -> None:
        """Set the batch size for metric submission.

        Args:
            size: New batch size (must be > 0)
        """
        if size <= 0:
            raise ValueError("Batch size must be positive") from None
        self._batch_size = size

    def set_flush_interval(self, interval: float) -> None:
        """Set the flush interval in seconds.

        Args:
            interval: Flush interval in seconds (must be > 0)
        """
        if interval <= 0:
            raise ValueError("Flush interval must be positive")
        self._flush_interval = interval
