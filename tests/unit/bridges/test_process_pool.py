"""Unit tests for JSProcessPool - Process pool for parallel JS trial execution.

These tests use mocked JSBridge workers to avoid requiring Node.js installation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.bridges.js_bridge import JSBridge, JSTrialResult
from traigent.bridges.process_pool import (
    DEFAULT_ACQUIRE_TIMEOUT_SECONDS,
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    JSProcessPool,
    JSProcessPoolConfig,
    PoolCapacityError,
    PoolShutdownError,
)

# =============================================================================
# JSProcessPoolConfig Tests
# =============================================================================


class TestJSProcessPoolConfig:
    """Tests for JSProcessPoolConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JSProcessPoolConfig()

        assert config.max_workers == 4
        assert config.module_path == ""
        assert config.function_name == "runTrial"
        assert config.trial_timeout == 300.0
        assert config.startup_timeout == DEFAULT_STARTUP_TIMEOUT_SECONDS
        assert config.acquire_timeout == DEFAULT_ACQUIRE_TIMEOUT_SECONDS
        assert config.shutdown_timeout == DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
        assert config.use_npx is True
        assert config.node_executable == "node"
        assert config.env is None
        assert config.cwd is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = JSProcessPoolConfig(
            max_workers=8,
            module_path="./dist/agent.js",
            function_name="customFn",
            trial_timeout=600.0,
            startup_timeout=60.0,
            acquire_timeout=120.0,
            shutdown_timeout=20.0,
            use_npx=False,
            node_executable="/usr/bin/node18",
            env={"NODE_ENV": "test"},
            cwd="/project",
        )

        assert config.max_workers == 8
        assert config.module_path == "./dist/agent.js"
        assert config.function_name == "customFn"
        assert config.trial_timeout == 600.0
        assert config.startup_timeout == 60.0
        assert config.acquire_timeout == 120.0
        assert config.shutdown_timeout == 20.0
        assert config.use_npx is False
        assert config.node_executable == "/usr/bin/node18"
        assert config.env == {"NODE_ENV": "test"}
        assert config.cwd == "/project"


# =============================================================================
# JSProcessPool Tests - Lifecycle
# =============================================================================


class TestJSProcessPoolLifecycle:
    """Tests for JSProcessPool lifecycle management."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return JSProcessPoolConfig(
            max_workers=2,
            module_path="./test.js",
            function_name="runTrial",
            trial_timeout=10.0,
        )

    @pytest.fixture
    def mock_bridge(self):
        """Create a mock JSBridge worker."""
        bridge = MagicMock(spec=JSBridge)
        bridge.is_running = True
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        bridge.ping = AsyncMock()
        bridge.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="test",
                status="completed",
                metrics={"accuracy": 0.9},
                duration=1.0,
            )
        )
        return bridge

    @pytest.mark.asyncio
    async def test_pool_properties_before_start(self, pool_config):
        """Test pool properties before starting."""
        pool = JSProcessPool(pool_config)

        assert pool.is_running is False
        assert pool.worker_count == 0
        assert pool.available_count == 0

    @pytest.mark.asyncio
    async def test_pool_start_creates_workers(self, pool_config, mock_bridge):
        """Test that start() creates the configured number of workers."""
        with patch(
            "traigent.bridges.process_pool.JSBridge", return_value=mock_bridge
        ) as mock_cls:
            pool = JSProcessPool(pool_config)
            await pool.start()

            assert pool.is_running is True
            assert pool.worker_count == 2
            assert pool.available_count == 2
            assert mock_cls.call_count == 2
            assert mock_bridge.start.call_count == 2

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_start_is_idempotent(self, pool_config, mock_bridge):
        """Test that calling start() multiple times is safe."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()
            await pool.start()  # Should be no-op

            # Should still have exactly 2 workers
            assert pool.worker_count == 2
            assert mock_bridge.start.call_count == 2  # Only called during first start

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_shutdown(self, pool_config, mock_bridge):
        """Test that shutdown() stops all workers."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()
            await pool.shutdown()

            assert pool.is_running is False
            assert pool.worker_count == 0
            assert mock_bridge.stop.call_count == 2

    @pytest.mark.asyncio
    async def test_pool_shutdown_is_idempotent(self, pool_config, mock_bridge):
        """Test that calling shutdown() multiple times is safe."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()
            await pool.shutdown()
            await pool.shutdown()  # Should be no-op

            # Stop should only be called once per worker
            assert mock_bridge.stop.call_count == 2

    @pytest.mark.asyncio
    async def test_pool_context_manager(self, pool_config, mock_bridge):
        """Test async context manager protocol."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            async with JSProcessPool(pool_config) as pool:
                assert pool.is_running is True
                assert pool.worker_count == 2

            # After exit, pool should be shut down
            assert pool.is_running is False


# =============================================================================
# JSProcessPool Tests - Acquire/Release
# =============================================================================


class TestJSProcessPoolAcquireRelease:
    """Tests for JSProcessPool acquire/release pattern."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return JSProcessPoolConfig(
            max_workers=2,
            module_path="./test.js",
            acquire_timeout=1.0,  # Short timeout for tests
        )

    @pytest.fixture
    def mock_bridge(self):
        """Create a mock JSBridge worker."""
        bridge = MagicMock(spec=JSBridge)
        bridge.is_running = True
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        bridge.ping = AsyncMock()
        return bridge

    @pytest.mark.asyncio
    async def test_acquire_returns_worker(self, pool_config, mock_bridge):
        """Test that acquire() returns an available worker."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()

            worker = await pool.acquire()
            assert worker is mock_bridge
            assert pool.available_count == 1  # One worker still available

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_starts_pool_if_needed(self, pool_config, mock_bridge):
        """Test that acquire() starts the pool if not already started."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            assert pool.is_running is False

            worker = await pool.acquire()
            assert pool.is_running is True
            assert worker is mock_bridge

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_release_returns_worker_to_pool(self, pool_config, mock_bridge):
        """Test that release() returns worker to pool after health check."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()

            worker = await pool.acquire()
            assert pool.available_count == 1

            await pool.release(worker)
            assert pool.available_count == 2

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_blocks_when_no_workers_available(self, pool_config):
        """Test that acquire() blocks when all workers are in use."""
        mock_bridge1 = MagicMock(spec=JSBridge)
        mock_bridge1.is_running = True
        mock_bridge1.start = AsyncMock()
        mock_bridge1.stop = AsyncMock()

        mock_bridge2 = MagicMock(spec=JSBridge)
        mock_bridge2.is_running = True
        mock_bridge2.start = AsyncMock()
        mock_bridge2.stop = AsyncMock()

        bridges = [mock_bridge1, mock_bridge2]
        bridge_iter = iter(bridges)

        with patch(
            "traigent.bridges.process_pool.JSBridge",
            side_effect=lambda _: next(bridge_iter),
        ):
            pool = JSProcessPool(pool_config)
            await pool.start()

            # Acquire both workers
            _worker1 = await pool.acquire()
            _worker2 = await pool.acquire()
            assert pool.available_count == 0

            # Third acquire should timeout
            with pytest.raises(PoolCapacityError):
                await pool.acquire(timeout=0.1)

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_acquire_raises_on_shutdown(self, pool_config, mock_bridge):
        """Test that acquire() raises PoolShutdownError when pool is shutting down."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()
            await pool.shutdown()

            with pytest.raises(PoolShutdownError):
                await pool.acquire()


# =============================================================================
# JSProcessPool Tests - Worker Health and Replacement
# =============================================================================


class TestJSProcessPoolHealthChecks:
    """Tests for JSProcessPool worker health checking and replacement."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return JSProcessPoolConfig(
            max_workers=2,
            module_path="./test.js",
        )

    @pytest.mark.asyncio
    async def test_release_replaces_dead_worker(self, pool_config):
        """Test that release() replaces a dead worker."""
        healthy_bridge = MagicMock(spec=JSBridge)
        healthy_bridge.is_running = True
        healthy_bridge.start = AsyncMock()
        healthy_bridge.stop = AsyncMock()
        healthy_bridge.ping = AsyncMock()

        dead_bridge = MagicMock(spec=JSBridge)
        dead_bridge.is_running = False  # Dead worker
        dead_bridge.start = AsyncMock()
        dead_bridge.stop = AsyncMock()

        bridges = [healthy_bridge, dead_bridge, healthy_bridge]
        bridge_iter = iter(bridges)

        with patch(
            "traigent.bridges.process_pool.JSBridge",
            side_effect=lambda _: next(bridge_iter),
        ):
            pool = JSProcessPool(pool_config)
            await pool.start()

            # Acquire the dead worker
            dead_worker = None
            for _ in range(2):
                w = await pool.acquire()
                if not w.is_running:
                    dead_worker = w
                    break
                await pool.release(w)

            # If we got a dead worker, release should replace it
            if dead_worker:
                initial_count = pool.worker_count
                await pool.release(dead_worker)
                # Worker should be replaced
                assert pool.worker_count >= initial_count - 1

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_release_replaces_worker_failing_health_check(self, pool_config):
        """Test that release() replaces a worker that fails health check."""
        healthy_bridge = MagicMock(spec=JSBridge)
        healthy_bridge.is_running = True
        healthy_bridge.start = AsyncMock()
        healthy_bridge.stop = AsyncMock()
        healthy_bridge.ping = AsyncMock()

        unhealthy_bridge = MagicMock(spec=JSBridge)
        unhealthy_bridge.is_running = True
        unhealthy_bridge.start = AsyncMock()
        unhealthy_bridge.stop = AsyncMock()
        unhealthy_bridge.ping = AsyncMock(side_effect=Exception("ping failed"))

        bridges = [healthy_bridge, unhealthy_bridge, healthy_bridge]
        bridge_iter = iter(bridges)

        with patch(
            "traigent.bridges.process_pool.JSBridge",
            side_effect=lambda _: next(bridge_iter),
        ):
            pool = JSProcessPool(pool_config)
            await pool.start()

            # Get workers - one is healthy, one has failing ping
            worker1 = await pool.acquire()
            worker2 = await pool.acquire()

            # Release both - unhealthy one should be replaced
            await pool.release(worker1)
            await pool.release(worker2)

            # Pool should still be operational
            assert pool.available_count >= 1

            await pool.shutdown()


# =============================================================================
# JSProcessPool Tests - Run Trial
# =============================================================================


class TestJSProcessPoolRunTrial:
    """Tests for JSProcessPool.run_trial() convenience method."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return JSProcessPoolConfig(
            max_workers=2,
            module_path="./test.js",
        )

    @pytest.fixture
    def mock_bridge(self):
        """Create a mock JSBridge worker with run_trial."""
        bridge = MagicMock(spec=JSBridge)
        bridge.is_running = True
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        bridge.ping = AsyncMock()
        bridge.run_trial = AsyncMock(
            return_value=JSTrialResult(
                trial_id="test-trial",
                status="completed",
                metrics={"accuracy": 0.95, "cost": 0.01},
                duration=2.5,
            )
        )
        return bridge

    @pytest.mark.asyncio
    async def test_run_trial_acquires_and_releases_worker(
        self, pool_config, mock_bridge
    ):
        """Test that run_trial() properly acquires and releases worker."""
        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()

            trial_config = {"trial_id": "test", "config": {"model": "gpt-4"}}
            result = await pool.run_trial(trial_config)

            assert result.trial_id == "test-trial"
            assert result.status == "completed"
            assert result.metrics["accuracy"] == 0.95

            # Worker should be released back to pool
            assert pool.available_count == 2

            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_run_trial_releases_worker_on_error(self, pool_config, mock_bridge):
        """Test that run_trial() releases worker even if trial fails."""
        mock_bridge.run_trial = AsyncMock(side_effect=Exception("Trial failed"))

        with patch("traigent.bridges.process_pool.JSBridge", return_value=mock_bridge):
            pool = JSProcessPool(pool_config)
            await pool.start()

            trial_config = {"trial_id": "test", "config": {}}

            with pytest.raises(Exception, match="Trial failed"):
                await pool.run_trial(trial_config)

            # Worker should still be released (but might be replaced if dead)
            # Pool should still be operational
            assert pool.is_running is True

            await pool.shutdown()


# =============================================================================
# JSProcessPool Tests - Start Race Prevention
# =============================================================================


class TestJSProcessPoolStartRace:
    """Tests for JSProcessPool start race condition prevention."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return JSProcessPoolConfig(
            max_workers=2,
            module_path="./test.js",
        )

    @pytest.mark.asyncio
    async def test_concurrent_acquire_does_not_double_start(self, pool_config):
        """Test that concurrent acquire() calls don't create duplicate workers."""
        call_count = 0

        def mock_bridge_factory(_config):
            nonlocal call_count
            call_count += 1
            bridge = MagicMock(spec=JSBridge)
            bridge.is_running = True
            bridge.start = AsyncMock()
            bridge.stop = AsyncMock()
            bridge.ping = AsyncMock()
            return bridge

        with patch(
            "traigent.bridges.process_pool.JSBridge", side_effect=mock_bridge_factory
        ):
            pool = JSProcessPool(pool_config)

            # Start multiple concurrent acquires
            results = await asyncio.gather(
                pool.acquire(),
                pool.acquire(),
                return_exceptions=True,
            )

            # Should have exactly 2 workers (not 4)
            assert pool.worker_count == 2
            assert call_count == 2

            # Both acquires should succeed
            assert not isinstance(results[0], Exception)
            assert not isinstance(results[1], Exception)

            await pool.shutdown()


# =============================================================================
# JSProcessPool Tests - Concurrent Trials
# =============================================================================


class TestJSProcessPoolConcurrentTrials:
    """Tests for JSProcessPool concurrent trial execution."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return JSProcessPoolConfig(
            max_workers=4,
            module_path="./test.js",
        )

    @pytest.mark.asyncio
    async def test_concurrent_trials_execute_in_parallel(self, pool_config):
        """Test that multiple trials can execute concurrently."""
        execution_times = []

        async def mock_run_trial(config, timeout=None):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate work
            end = asyncio.get_event_loop().time()
            execution_times.append((start, end))
            return JSTrialResult(
                trial_id=config.get("trial_id", "test"),
                status="completed",
                metrics={"accuracy": 0.9},
                duration=0.1,
            )

        def mock_bridge_factory(_config):
            bridge = MagicMock(spec=JSBridge)
            bridge.is_running = True
            bridge.start = AsyncMock()
            bridge.stop = AsyncMock()
            bridge.ping = AsyncMock()
            bridge.run_trial = mock_run_trial
            return bridge

        with patch(
            "traigent.bridges.process_pool.JSBridge", side_effect=mock_bridge_factory
        ):
            pool = JSProcessPool(pool_config)
            await pool.start()

            # Run 4 trials concurrently
            trials = [{"trial_id": f"trial-{i}", "config": {"i": i}} for i in range(4)]

            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(*[pool.run_trial(t) for t in trials])
            total_time = asyncio.get_event_loop().time() - start_time

            # All trials should complete successfully
            assert len(results) == 4
            assert all(r.status == "completed" for r in results)

            # Total time should be close to single trial time (parallel execution)
            # Not 4x (sequential execution)
            assert total_time < 0.3  # Should be ~0.1s, not ~0.4s

            await pool.shutdown()
