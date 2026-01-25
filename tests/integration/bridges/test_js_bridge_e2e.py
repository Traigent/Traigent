"""End-to-end integration tests for JS Bridge.

These tests verify the full Python-to-JS trial execution flow using
the actual Node.js runtime and JS SDK demo app.

Requirements:
- Node.js installed and available in PATH
- traigent-js demo built (npm run build in traigent-js/demos/agent-app)

Tests are skipped if:
- Node.js is not available
- The JS demo app is not built
"""

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from traigent.bridges.js_bridge import (
    JSBridge,
    JSBridgeConfig,
    JSBridgeError,
    JSTrialResult,
)
from traigent.bridges.process_pool import (
    JSProcessPool,
    JSProcessPoolConfig,
    PoolCapacityError,
)

# Path to the JS demo app
JS_DEMO_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "traigent-js"
    / "demos"
    / "agent-app"
)
JS_DEMO_DIST = JS_DEMO_DIR / "dist"
JS_TRIAL_MODULE = JS_DEMO_DIST / "trial.js"


def node_available() -> bool:
    """Check if Node.js is available."""
    return shutil.which("node") is not None


def js_demo_built() -> bool:
    """Check if the JS demo app is built."""
    return JS_TRIAL_MODULE.exists()


skip_no_node = pytest.mark.skipif(
    not node_available(), reason="Node.js not available in PATH"
)

skip_no_demo = pytest.mark.skipif(
    not js_demo_built(), reason=f"JS demo not built at {JS_DEMO_DIST}"
)

requires_js_runtime = pytest.mark.skipif(
    not (node_available() and js_demo_built()),
    reason="Node.js or JS demo not available",
)


@requires_js_runtime
class TestJSBridgeE2E:
    """End-to-end tests for JSBridge with real Node.js process."""

    @pytest.fixture
    def bridge_config(self) -> JSBridgeConfig:
        """Create bridge config for the demo app."""
        return JSBridgeConfig(
            module_path=str(JS_TRIAL_MODULE),
            function_name="runTrial",
            trial_timeout_seconds=60.0,
            use_npx=False,  # Use direct node invocation
            cwd=str(JS_DEMO_DIR),
        )

    @pytest.mark.asyncio
    async def test_bridge_lifecycle(self, bridge_config):
        """Test starting and stopping the JS bridge."""
        bridge = JSBridge(bridge_config)

        assert not bridge.is_running

        await bridge.start()
        assert bridge.is_running

        await bridge.stop()
        assert not bridge.is_running

    @pytest.mark.asyncio
    async def test_ping(self, bridge_config):
        """Test ping/health check."""
        bridge = JSBridge(bridge_config)
        await bridge.start()

        try:
            result = await bridge.ping()
            assert "timestamp" in result
            assert "uptime_ms" in result
            assert result["uptime_ms"] >= 0
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_run_trial_success(self, bridge_config):
        """Test running a trial successfully."""
        bridge = JSBridge(bridge_config)
        await bridge.start()

        try:
            trial_config = {
                "trial_id": "test-e2e-001",
                "trial_number": 1,
                "experiment_run_id": "e2e-test-run",
                "config": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "system_prompt": "concise",
                },
                "dataset_subset": {
                    "indices": [0, 1, 2],
                    "total": 10,
                },
            }

            result = await bridge.run_trial(trial_config)

            assert isinstance(result, JSTrialResult)
            assert result.trial_id == "test-e2e-001"
            assert result.status == "completed"
            assert "accuracy" in result.metrics
            assert isinstance(result.metrics["accuracy"], (int, float))
            assert result.duration > 0
        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_cancel_trial(self, bridge_config):
        """Test cancelling an in-flight trial.

        Note: In demo mode, trials complete very quickly. This test verifies
        that cancel works without causing errors, but the trial may complete
        before the cancel is processed.
        """
        bridge = JSBridge(bridge_config)
        await bridge.start()

        try:
            # Use a larger dataset to give more time for cancel
            trial_config = {
                "trial_id": "test-cancel-001",
                "trial_number": 1,
                "experiment_run_id": "e2e-test-run",
                "config": {"model": "gpt-3.5-turbo"},
                "dataset_subset": {"indices": list(range(20)), "total": 100},
            }

            # Start trial in background
            trial_task = asyncio.create_task(bridge.run_trial(trial_config))

            # Wait a bit for trial to start
            await asyncio.sleep(0.05)

            # Try to cancel - may or may not succeed depending on timing
            try:
                cancel_result = await asyncio.wait_for(
                    bridge.cancel("test-cancel-001"), timeout=2.0
                )
                cancel_succeeded = cancel_result.get("cancelled", False)
            except (asyncio.TimeoutError, Exception):
                # Cancel failed (trial already done, or timing issue)
                cancel_succeeded = False

            # Wait for trial result
            result = await trial_task

            # Trial should complete successfully (or be cancelled if cancel worked)
            if cancel_succeeded:
                assert result.status in ("cancelled", "completed")
            else:
                # Trial completed before/during cancel attempt
                assert result.status == "completed"
        finally:
            await bridge.stop()


@requires_js_runtime
class TestJSProcessPoolE2E:
    """End-to-end tests for JSProcessPool with real Node.js processes."""

    @pytest.fixture
    def pool_config(self) -> JSProcessPoolConfig:
        """Create pool config for the demo app."""
        return JSProcessPoolConfig(
            max_workers=2,
            module_path=str(JS_TRIAL_MODULE),
            function_name="runTrial",
            trial_timeout=60.0,
            use_npx=False,
            cwd=str(JS_DEMO_DIR),
        )

    @pytest.mark.asyncio
    async def test_pool_lifecycle(self, pool_config):
        """Test pool start and shutdown."""
        pool = JSProcessPool(pool_config)

        assert not pool.is_running

        await pool.start()
        assert pool.is_running
        assert pool.worker_count == 2

        await pool.shutdown()
        assert not pool.is_running

    @pytest.mark.asyncio
    async def test_pool_context_manager(self, pool_config):
        """Test pool as async context manager."""
        async with JSProcessPool(pool_config) as pool:
            assert pool.is_running
            assert pool.worker_count == 2

        assert not pool.is_running

    @pytest.mark.asyncio
    async def test_pool_run_trial(self, pool_config):
        """Test running a single trial via pool."""
        async with JSProcessPool(pool_config) as pool:
            trial_config = {
                "trial_id": "pool-e2e-001",
                "trial_number": 1,
                "experiment_run_id": "pool-e2e-test",
                "config": {"model": "gpt-3.5-turbo", "temperature": 0.5},
                "dataset_subset": {"indices": [0, 1], "total": 10},
            }

            result = await pool.run_trial(trial_config)

            assert result.trial_id == "pool-e2e-001"
            assert result.status == "completed"
            assert "accuracy" in result.metrics

    @pytest.mark.asyncio
    async def test_pool_concurrent_trials(self, pool_config):
        """Test running concurrent trials via pool."""
        async with JSProcessPool(pool_config) as pool:
            # Create multiple trial configs
            trial_configs = [
                {
                    "trial_id": f"concurrent-{i}",
                    "trial_number": i,
                    "experiment_run_id": "concurrent-test",
                    "config": {"model": "gpt-3.5-turbo", "temperature": 0.1 * i},
                    "dataset_subset": {"indices": [i], "total": 10},
                }
                for i in range(4)  # More trials than workers
            ]

            # Run concurrently
            results = await asyncio.gather(
                *[pool.run_trial(cfg) for cfg in trial_configs]
            )

            # All should complete successfully
            assert len(results) == 4
            for i, result in enumerate(results):
                assert result.trial_id == f"concurrent-{i}"
                assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_pool_worker_reuse(self, pool_config):
        """Test that workers are reused across trials."""
        pool_config.max_workers = 1  # Single worker to verify reuse

        async with JSProcessPool(pool_config) as pool:
            # Run multiple trials sequentially on same worker
            for i in range(3):
                trial_config = {
                    "trial_id": f"reuse-{i}",
                    "trial_number": i,
                    "experiment_run_id": "reuse-test",
                    "config": {"model": "gpt-3.5-turbo"},
                    "dataset_subset": {"indices": [i], "total": 10},
                }

                result = await pool.run_trial(trial_config)
                assert result.status == "completed"

            # Should still have only 1 worker
            assert pool.worker_count == 1
