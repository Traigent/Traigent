"""Process pool for concurrent JS trial execution.

This module provides JSProcessPool, which manages a pool of JSBridge workers
for parallel JavaScript trial execution. Each worker is a separate Node.js
subprocess that can execute one trial at a time.

Key features:
- Acquire/release pattern for worker management
- Health checks on release to detect dead workers
- Automatic worker replacement on failure
- Graceful shutdown with drain logic
- Custom exceptions for capacity issues

Usage:
    config = JSProcessPoolConfig(
        max_workers=4,
        module_path="./dist/agent.js",
    )
    async with JSProcessPool(config) as pool:
        result = await pool.run_trial(trial_config)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from traigent.bridges.js_bridge import (
    JSBridge,
    JSBridgeConfig,
    JSBridgeError,
    JSTrialResult,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Note: We intentionally do NOT use atexit handlers for pool cleanup.
# See js_bridge.py for the rationale - atexit handlers with os.killpg()
# are dangerous due to PID reuse race conditions.
# Pools should be explicitly shut down via shutdown() or context manager.

# Default timeout values
DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 60.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class PoolCapacityError(JSBridgeError):
    """Raised when no workers available within timeout."""


class PoolShutdownError(JSBridgeError):
    """Raised when pool is shutting down and cannot accept new work."""


@dataclass
class JSProcessPoolConfig:
    """Configuration for JS process pool.

    Attributes:
        max_workers: Maximum number of worker processes.
        module_path: Path to the JS module containing the trial function.
        function_name: Name of the exported function to call.
        trial_timeout: Timeout for trial execution in seconds.
        startup_timeout: Timeout for starting individual workers.
        acquire_timeout: Timeout for acquiring a worker from the pool.
        shutdown_timeout: Timeout for graceful shutdown.
        use_npx: Whether to use npx to invoke traigent-js.
        node_executable: Path to Node.js executable.
        env: Additional environment variables for workers.
        cwd: Working directory for workers.
    """

    max_workers: int = 4
    module_path: str = ""
    function_name: str = "runTrial"
    trial_timeout: float = 300.0
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS
    acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT_SECONDS
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    use_npx: bool = True
    node_executable: str = "node"
    env: dict[str, str] | None = None
    cwd: str | None = None


class JSProcessPool:
    """Pool of JSBridge workers for parallel JS trial execution.

    This class manages a pool of Node.js subprocesses for concurrent trial
    execution. Workers are acquired from the pool, used for a single trial,
    and then released back to the pool.

    Thread-safety note: This class uses asyncio primitives only and must be
    used from a single event loop. All methods are async and should be awaited.

    Features:
    - Lazy initialization: Workers are started on first acquire
    - Health checks: Workers are pinged on release to detect failures
    - Auto-replacement: Dead workers are replaced automatically
    - Graceful shutdown: All workers are properly terminated

    Usage:
        async with JSProcessPool(config) as pool:
            # Run multiple trials concurrently
            results = await asyncio.gather(
                pool.run_trial(config1),
                pool.run_trial(config2),
            )
    """

    def __init__(self, config: JSProcessPoolConfig) -> None:
        """Initialize JSProcessPool with configuration.

        Args:
            config: Pool configuration including worker count and bridge settings.
        """
        self._config = config
        self._workers: list[JSBridge] = []
        self._available: asyncio.Queue[JSBridge] = asyncio.Queue()
        self._started = False
        self._shutdown = False
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Check if the pool is running and accepting work."""
        return self._started and not self._shutdown

    @property
    def worker_count(self) -> int:
        """Get the current number of workers in the pool."""
        return len(self._workers)

    @property
    def available_count(self) -> int:
        """Get the approximate number of available workers."""
        return self._available.qsize()

    def _create_bridge_config(self) -> JSBridgeConfig:
        """Create a JSBridgeConfig from pool configuration."""
        return JSBridgeConfig(
            module_path=self._config.module_path,
            function_name=self._config.function_name,
            trial_timeout_seconds=self._config.trial_timeout,
            command_timeout_seconds=self._config.startup_timeout,
            use_npx=self._config.use_npx,
            node_executable=self._config.node_executable,
            env=self._config.env,
            cwd=self._config.cwd,
        )

    async def _start_workers(self) -> None:
        """Start all workers in the pool.

        This is called internally with the lock held.

        Raises:
            JSProcessError: If any worker fails to start.
        """
        if self._started:
            return

        bridge_config = self._create_bridge_config()

        logger.info(
            "Starting JS process pool with %d workers for %s",
            self._config.max_workers,
            self._config.module_path,
        )

        # Start workers in parallel
        start_tasks = []
        for _ in range(self._config.max_workers):
            bridge = JSBridge(bridge_config)
            self._workers.append(bridge)
            start_tasks.append(bridge.start())

        # Wait for all workers to start
        results = await asyncio.gather(*start_tasks, return_exceptions=True)

        # Check for failures
        failed = [i for i, r in enumerate(results) if isinstance(r, BaseException)]
        if failed:
            # Clean up any successfully started workers
            for worker in self._workers:
                if worker.is_running:
                    try:
                        await worker.stop()
                    except Exception:
                        pass
            self._workers.clear()
            first_error = results[failed[0]]
            if isinstance(first_error, BaseException):
                raise first_error
            raise RuntimeError(f"Worker startup failed: {first_error}")

        # All workers started successfully - add to available queue
        for worker in self._workers:
            await self._available.put(worker)

        self._started = True
        logger.info(
            "JS process pool started successfully with %d workers", len(self._workers)
        )

    async def start(self) -> None:
        """Start the pool and all workers.

        This is typically called automatically on first acquire,
        but can be called explicitly for eager initialization.

        Raises:
            JSProcessError: If workers fail to start.
        """
        async with self._lock:
            if self._shutdown:
                raise PoolShutdownError("Pool is shutting down")
            await self._start_workers()

    async def acquire(self, timeout: float | None = None) -> JSBridge:
        """Acquire an available worker from the pool.

        This method blocks until a worker is available or times out.
        The returned worker must be released back to the pool after use.

        Args:
            timeout: Timeout in seconds (default: acquire_timeout from config).

        Returns:
            An available JSBridge worker.

        Raises:
            PoolCapacityError: If no worker available within timeout.
            PoolShutdownError: If pool is shutting down.
        """
        if self._shutdown:
            raise PoolShutdownError("Pool is shutting down")

        # Start workers if not already started (with lock protection)
        async with self._lock:
            if self._shutdown:
                raise PoolShutdownError("Pool is shutting down")
            if not self._started:
                await self._start_workers()

        # Acquire from queue (outside lock to allow concurrent acquires)
        acquire_timeout = (
            timeout if timeout is not None else self._config.acquire_timeout
        )
        try:
            worker = await asyncio.wait_for(
                self._available.get(),
                timeout=acquire_timeout,
            )
            logger.debug(
                "Acquired worker from pool (%d available)", self._available.qsize()
            )
            return worker
        except TimeoutError:
            raise PoolCapacityError(
                f"No worker available within {acquire_timeout}s "
                f"(pool size: {self._config.max_workers}, "
                f"available: {self._available.qsize()})"
            ) from None

    async def release(self, worker: JSBridge) -> None:
        """Return a worker to the pool.

        If the worker has died, it will be replaced with a new worker.

        Args:
            worker: The worker to return to the pool.
        """
        if self._shutdown:
            # During shutdown, just stop the worker, don't re-queue
            logger.debug("Pool shutting down, stopping worker instead of releasing")
            try:
                await worker.stop()
            except Exception as e:
                logger.debug("Error stopping worker during shutdown: %s", e)
            return

        # Check if worker is healthy
        if worker.is_running:
            # Try a health check (ping)
            try:
                await asyncio.wait_for(
                    worker.ping(),
                    timeout=5.0,  # Quick health check timeout
                )
                # Worker is healthy, return to pool
                await self._available.put(worker)
                logger.debug(
                    "Released worker to pool (%d available)", self._available.qsize()
                )
                return
            except Exception as e:
                logger.warning("Worker failed health check, replacing: %s", e)

        # Worker is dead or failed health check - replace it
        await self._replace_dead_worker(worker)

    async def _replace_dead_worker(self, dead_worker: JSBridge) -> None:
        """Replace a dead worker with a new one.

        Args:
            dead_worker: The worker that died and needs replacement.
        """
        # Remove from workers list
        try:
            self._workers.remove(dead_worker)
        except ValueError:
            pass  # Already removed

        # Try to stop the dead worker (in case it's just unresponsive)
        try:
            await dead_worker.stop()
        except Exception:
            pass

        # Spawn replacement
        logger.info(
            "Spawning replacement worker (current count: %d)", len(self._workers)
        )
        try:
            bridge_config = self._create_bridge_config()
            new_worker = JSBridge(bridge_config)
            await new_worker.start()
            self._workers.append(new_worker)
            await self._available.put(new_worker)
            logger.info("Replacement worker started successfully")
        except Exception as e:
            logger.error("Failed to spawn replacement worker: %s", e)
            # Don't put anything back in the queue - pool capacity is reduced
            # Future acquires will get the remaining workers

    async def run_trial(
        self,
        trial_config: dict[str, Any],
        timeout: float | None = None,
    ) -> JSTrialResult:
        """Run a trial on an available worker.

        This is a convenience method that acquires a worker, runs the trial,
        and releases the worker back to the pool.

        Args:
            trial_config: Trial configuration to pass to the worker.
            timeout: Timeout for trial execution (overrides config default).

        Returns:
            Trial result from the worker.

        Raises:
            PoolCapacityError: If no worker available.
            PoolShutdownError: If pool is shutting down.
            JSTrialTimeoutError: If the trial times out.
            JSProtocolError: If protocol communication fails.
        """
        worker = await self.acquire()
        try:
            return await worker.run_trial(trial_config, timeout)
        finally:
            await self.release(worker)

    async def shutdown(self, timeout: float | None = None) -> None:
        """Gracefully shutdown all workers.

        This stops accepting new work, drains the queue, and terminates
        all workers with a timeout.

        Args:
            timeout: Timeout for graceful shutdown (default: shutdown_timeout from config).
        """
        shutdown_timeout = (
            timeout if timeout is not None else self._config.shutdown_timeout
        )

        async with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

        logger.info("Shutting down JS process pool (%d workers)", len(self._workers))

        # Drain the available queue (discard workers)
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Stop all workers with timeout
        if self._workers:
            cancel_tasks = [worker.cancel_active_trial() for worker in self._workers]
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
            shutdown_tasks = [worker.stop() for worker in self._workers]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=shutdown_timeout,
                )
            except TimeoutError:
                logger.warning("Graceful shutdown timed out, force terminating workers")
                # Force terminate remaining workers
                for worker in self._workers:
                    if worker.is_running:
                        try:
                            await worker._terminate()
                        except Exception as e:
                            logger.debug("Error force-terminating worker: %s", e)

        self._workers.clear()
        self._started = False
        logger.info("JS process pool shutdown complete")

    async def __aenter__(self) -> JSProcessPool:
        """Start the pool (context manager entry)."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Shutdown the pool (context manager exit)."""
        await self.shutdown()
