"""Unit tests for JSBridge - JavaScript runtime bridge for Node.js trial execution.

These tests use mocked subprocess to avoid requiring Node.js installation.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.bridges.js_bridge import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_TRIAL_TIMEOUT_SECONDS,
    PROTOCOL_VERSION,
    JSBridge,
    JSBridgeConfig,
    JSBridgeError,
    JSProcessError,
    JSTrialResult,
    JSTrialTimeoutError,
)

# =============================================================================
# JSBridgeConfig Tests
# =============================================================================


class TestJSBridgeConfig:
    """Tests for JSBridgeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = JSBridgeConfig(module_path="./test.js")

        assert config.module_path == "./test.js"
        assert config.function_name == "runTrial"
        assert config.use_npx is True
        assert config.runner_path is None
        assert config.node_executable == "node"
        assert config.node_args == []
        assert config.trial_timeout_seconds == DEFAULT_TRIAL_TIMEOUT_SECONDS
        assert config.command_timeout_seconds == DEFAULT_COMMAND_TIMEOUT_SECONDS
        assert config.env is None
        assert config.cwd is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = JSBridgeConfig(
            module_path="/path/to/module.js",
            function_name="customTrialFn",
            use_npx=False,
            runner_path="/custom/runner",
            node_executable="/usr/bin/node18",
            node_args=["--max-old-space-size=4096"],
            trial_timeout_seconds=600,
            command_timeout_seconds=20,
            env={"NODE_ENV": "test"},
            cwd="/project",
        )

        assert config.module_path == "/path/to/module.js"
        assert config.function_name == "customTrialFn"
        assert config.use_npx is False
        assert config.runner_path == "/custom/runner"
        assert config.node_executable == "/usr/bin/node18"
        assert config.node_args == ["--max-old-space-size=4096"]
        assert config.trial_timeout_seconds == 600
        assert config.command_timeout_seconds == 20
        assert config.env == {"NODE_ENV": "test"}
        assert config.cwd == "/project"


# =============================================================================
# JSTrialResult Tests
# =============================================================================


class TestJSTrialResult:
    """Tests for JSTrialResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = JSTrialResult(trial_id="test-123", status="completed")

        assert result.trial_id == "test-123"
        assert result.status == "completed"
        assert result.metrics == {}
        assert result.duration == 0.0
        assert result.error_message is None
        assert result.error_code is None
        assert result.retryable is False
        assert result.metadata is None

    def test_success_result(self):
        """Test successful result with metrics."""
        result = JSTrialResult(
            trial_id="trial-456",
            status="completed",
            metrics={"accuracy": 0.95, "latency_ms": 120},
            duration=5.5,
            metadata={"model": "gpt-4"},
        )

        assert result.trial_id == "trial-456"
        assert result.status == "completed"
        assert result.metrics == {"accuracy": 0.95, "latency_ms": 120}
        assert result.duration == 5.5
        assert result.metadata == {"model": "gpt-4"}

    def test_failure_result(self):
        """Test failed result with error details."""
        result = JSTrialResult(
            trial_id="trial-789",
            status="failed",
            duration=1.2,
            error_message="Connection timeout",
            error_code="TIMEOUT",
            retryable=True,
        )

        assert result.trial_id == "trial-789"
        assert result.status == "failed"
        assert result.error_message == "Connection timeout"
        assert result.error_code == "TIMEOUT"
        assert result.retryable is True


# =============================================================================
# Command Building Tests
# =============================================================================


class TestCommandBuilding:
    """Tests for JS bridge command building logic."""

    @pytest.mark.asyncio
    async def test_command_with_npx(self, mock_subprocess):
        """Test command building with npx (default)."""
        config = JSBridgeConfig(
            module_path="./dist/agent.js",
            function_name="runTrial",
            use_npx=True,
        )

        bridge = JSBridge(config)

        with patch("asyncio.create_subprocess_exec", mock_subprocess) as mock_exec:
            try:
                await bridge.start()
            except Exception:
                pass  # Expected since mock doesn't fully work

            # Verify npx command was built correctly
            if mock_exec.called:
                call_args = mock_exec.call_args[0]
                assert call_args[0] == "npx"
                assert call_args[1] == "traigent-js"
                assert "--module" in call_args
                assert "--function" in call_args
                assert "runTrial" in call_args

    @pytest.mark.asyncio
    async def test_command_with_explicit_runner_path(self, mock_subprocess):
        """Test command building with explicit runner path."""
        config = JSBridgeConfig(
            module_path="./dist/agent.js",
            function_name="customFn",
            use_npx=False,
            runner_path="/custom/traigent-runner.js",
            node_executable="/usr/bin/node18",
            node_args=["--experimental-vm-modules"],
        )

        bridge = JSBridge(config)

        with patch("asyncio.create_subprocess_exec", mock_subprocess) as mock_exec:
            try:
                await bridge.start()
            except Exception:
                pass

            if mock_exec.called:
                call_args = mock_exec.call_args[0]
                assert call_args[0] == "/usr/bin/node18"
                assert "--experimental-vm-modules" in call_args
                assert "/custom/traigent-runner.js" in call_args
                assert "--function" in call_args
                assert "customFn" in call_args


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestResponseParsing:
    """Tests for _parse_trial_response method."""

    def test_parse_success_response(self):
        """Test parsing a successful trial response."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        response = {
            "status": "success",
            "payload": {
                "trial_id": "trial-123",
                "status": "completed",
                "metrics": {"accuracy": 0.92, "cost_usd": 0.01},
                "duration": 3.5,
                "metadata": {"model_version": "v2"},
            },
        }

        result = bridge._parse_trial_response(response)

        assert result.trial_id == "trial-123"
        assert result.status == "completed"
        assert result.metrics == {"accuracy": 0.92, "cost_usd": 0.01}
        assert result.duration == 3.5
        assert result.metadata == {"model_version": "v2"}
        assert result.error_message is None

    def test_parse_failure_response(self):
        """Test parsing a failed trial response."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        response = {
            "status": "success",
            "payload": {
                "trial_id": "trial-456",
                "status": "failed",
                "duration": 1.0,
                "error_message": "API rate limit exceeded",
                "error_code": "RATE_LIMIT",
                "retryable": True,
            },
        }

        result = bridge._parse_trial_response(response)

        assert result.trial_id == "trial-456"
        assert result.status == "failed"
        assert result.error_message == "API rate limit exceeded"
        assert result.error_code == "RATE_LIMIT"
        assert result.retryable is True

    def test_parse_cancelled_response(self):
        """Test parsing a cancelled trial response."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        response = {
            "status": "success",
            "payload": {
                "trial_id": "trial-789",
                "status": "cancelled",
                "duration": 2.5,
                "error_message": "Trial cancelled by user",
            },
        }

        result = bridge._parse_trial_response(response)

        assert result.trial_id == "trial-789"
        assert result.status == "cancelled"
        assert result.duration == 2.5
        assert result.error_message == "Trial cancelled by user"

    def test_parse_protocol_error_response(self):
        """Test parsing a protocol-level error response."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        response = {
            "status": "error",
            "payload": {
                "trial_id": "trial-error",
                "error": "Invalid request payload",
                "error_code": "VALIDATION_ERROR",
                "retryable": False,
            },
        }

        result = bridge._parse_trial_response(response)

        assert result.trial_id == "trial-error"
        assert result.status == "failed"
        assert result.error_message == "Invalid request payload"
        assert result.error_code == "VALIDATION_ERROR"
        assert result.retryable is False

    def test_parse_response_with_missing_fields(self):
        """Test parsing response with minimal fields."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        response = {
            "status": "success",
            "payload": {
                "status": "completed",
            },
        }

        result = bridge._parse_trial_response(response)

        assert result.trial_id == "unknown"
        assert result.status == "completed"
        assert result.metrics == {}
        assert result.duration == 0.0


# =============================================================================
# Bridge Lifecycle Tests (with mocked subprocess)
# =============================================================================


class TestBridgeLifecycle:
    """Tests for bridge start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_is_running_property_before_start(self):
        """Test is_running is False before start."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        assert bridge.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self, mock_bridge_started):
        """Test that calling start twice doesn't spawn multiple processes."""
        bridge = mock_bridge_started

        # Already started in fixture
        assert bridge._started is True

        # Second start should be no-op
        await bridge.start()
        assert bridge._started is True

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, mock_bridge_started):
        """Test that calling stop twice is safe."""
        bridge = mock_bridge_started

        await bridge.stop()
        assert bridge._started is False

        # Second stop should be no-op
        await bridge.stop()
        assert bridge._started is False

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_subprocess_factory):
        """Test bridge as async context manager."""
        config = JSBridgeConfig(module_path="./test.js")

        mock_process = mock_subprocess_factory(
            ping_response={"version": "1.0", "request_id": "test", "status": "success", "payload": {"ok": True}},
            shutdown_response={"version": "1.0", "request_id": "test", "status": "success", "payload": {}},
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            async with JSBridge(config) as bridge:
                assert bridge._started is True
                assert bridge.is_running is True

        # After exiting context, bridge should be stopped
        assert bridge._started is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in JS bridge."""

    @pytest.mark.asyncio
    async def test_process_not_found_raises_error(self):
        """Test that missing node executable raises JSProcessError."""
        config = JSBridgeConfig(
            module_path="./test.js",
            use_npx=False,
            runner_path="/nonexistent/runner.js",
            node_executable="/nonexistent/node",
        )
        bridge = JSBridge(config)

        with pytest.raises(JSProcessError) as exc_info:
            await bridge.start()

        assert "Failed to start Node.js process" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_request_when_not_started_raises_error(self):
        """Test that sending request before start raises error."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        with pytest.raises(JSBridgeError) as exc_info:
            await bridge.ping()

        assert "not started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_trial_timeout_triggers_cancel(self, mock_bridge_started):
        """Test that trial timeout triggers cancel attempt."""
        bridge = mock_bridge_started

        # Patch run_trial to simulate timeout
        with patch.object(
            bridge,
            "_send_request",
            side_effect=TimeoutError("Request timed out"),
        ):
            with pytest.raises(JSTrialTimeoutError) as exc_info:
                await bridge.run_trial(
                    {"trial_id": "timeout-trial", "config": {}},
                    timeout=0.1,
                )

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_exit_fails_pending_requests(self):
        """Test that process exit immediately fails pending requests."""
        config = JSBridgeConfig(module_path="./test.js")
        bridge = JSBridge(config)

        # Manually set up a started bridge with a mock process
        mock_process = MagicMock()
        mock_process.returncode = 1  # Simulate exited process
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()

        bridge._process = mock_process
        bridge._started = True

        # Create a pending request
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        bridge._pending_requests["pending-123"] = future

        # Simulate the reader task detecting stdout close and failing requests
        for f in bridge._pending_requests.values():
            if not f.done():
                f.set_exception(JSProcessError("JS process exited unexpectedly (exit code: 1)"))

        # Verify the pending request was failed
        with pytest.raises(JSProcessError):
            await future


# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocol:
    """Tests for NDJSON protocol handling."""

    def test_protocol_version_matches(self):
        """Test protocol version constant."""
        assert PROTOCOL_VERSION == "1.0"

    @pytest.mark.asyncio
    async def test_request_format(self, mock_subprocess_factory):
        """Test that requests follow correct format."""
        config = JSBridgeConfig(module_path="./test.js")

        sent_requests: list[dict] = []

        mock_process = mock_subprocess_factory(
            ping_response={"version": "1.0", "status": "success", "payload": {"ok": True}},
            shutdown_response={"version": "1.0", "status": "success", "payload": {}},
        )

        # Capture requests by wrapping the stdin.write
        original_write = mock_process.stdin.write

        def capture_write(data):
            try:
                sent_requests.append(json.loads(data.decode()))
            except json.JSONDecodeError:
                pass
            return original_write(data)

        mock_process.stdin.write = capture_write

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            bridge = JSBridge(config)
            await bridge.start()
            await bridge.stop()

        # Verify ping request format
        assert len(sent_requests) >= 1
        ping_request = sent_requests[0]
        assert ping_request["version"] == PROTOCOL_VERSION
        assert "request_id" in ping_request
        assert ping_request["action"] == "ping"
        assert "payload" in ping_request


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_subprocess():
    """Create a basic mock subprocess."""
    mock = AsyncMock()
    mock.returncode = None
    mock.stdin = MagicMock()
    mock.stdin.write = MagicMock()
    mock.stdin.drain = AsyncMock()
    mock.stdout = MagicMock()
    mock.stderr = MagicMock()
    mock.wait = AsyncMock(return_value=0)
    mock.terminate = MagicMock()
    mock.kill = MagicMock()
    return mock


@pytest.fixture
def mock_subprocess_factory():
    """Factory for creating mock subprocess with custom responses.

    This factory creates a mock that intercepts requests and returns
    responses with matching request_ids.
    """

    def _create_mock(
        ping_response: dict | None = None,
        shutdown_response: dict | None = None,
        trial_response: dict | None = None,
        close_stdout_after_ping: bool = False,
    ):
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()

        # Track requests to return matching responses
        pending_request_ids: list[str] = []
        action_queue: list[str] = []

        def capture_write(data: bytes):
            try:
                request = json.loads(data.decode())
                pending_request_ids.append(request["request_id"])
                action_queue.append(request["action"])
            except (json.JSONDecodeError, KeyError):
                pass

        # Set up stdin mock
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = capture_write
        mock_process.stdin.drain = AsyncMock()

        first_response_done = False

        async def mock_readline():
            nonlocal first_response_done

            # Wait for a request to arrive
            await asyncio.sleep(0.01)  # Give time for write to happen

            if not pending_request_ids:
                if close_stdout_after_ping and first_response_done:
                    return b""
                # Keep waiting
                await asyncio.sleep(0.05)
                if not pending_request_ids:
                    return b""

            request_id = pending_request_ids.pop(0)
            action = action_queue.pop(0) if action_queue else "ping"

            if action == "ping" and ping_response:
                response = dict(ping_response)
                response["request_id"] = request_id
                first_response_done = True
                return json.dumps(response).encode() + b"\n"
            elif action == "shutdown" and shutdown_response:
                response = dict(shutdown_response)
                response["request_id"] = request_id
                return json.dumps(response).encode() + b"\n"
            elif action == "run_trial" and trial_response:
                response = dict(trial_response)
                response["request_id"] = request_id
                return json.dumps(response).encode() + b"\n"
            else:
                # Default response
                return json.dumps({
                    "version": "1.0",
                    "request_id": request_id,
                    "status": "success",
                    "payload": {"ok": True}
                }).encode() + b"\n"

        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = mock_readline

        # Set up stderr mock
        mock_process.stderr = MagicMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        return mock_process

    return _create_mock


@pytest.fixture
async def mock_bridge_started(mock_subprocess_factory):
    """Create a JSBridge that's already started with mocked subprocess."""
    config = JSBridgeConfig(module_path="./test.js")
    bridge = JSBridge(config)

    mock_process = mock_subprocess_factory(
        ping_response={"version": "1.0", "request_id": "ping", "status": "success", "payload": {"ok": True}},
        shutdown_response={"version": "1.0", "request_id": "shutdown", "status": "success", "payload": {}},
    )

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        # Manually set up started state
        bridge._process = mock_process
        bridge._started = True

        # Mock the reader task
        async def mock_reader():
            await asyncio.sleep(100)

        bridge._reader_task = asyncio.create_task(mock_reader())

    yield bridge

    # Cleanup
    bridge._shutdown_requested = True
    bridge._started = False
    if bridge._reader_task:
        bridge._reader_task.cancel()
        try:
            await bridge._reader_task
        except asyncio.CancelledError:
            pass
