"""JavaScript runtime bridge for Node.js trial execution.

This module provides JSBridge, which manages a Node.js subprocess for executing
optimization trials in JavaScript. It communicates via NDJSON (newline-delimited
JSON) protocol over stdin/stdout.

Protocol:
    - Requests are JSON objects written to Node's stdin (one per line)
    - Responses are JSON objects read from Node's stdout (one per line)
    - Logs and debug output go to Node's stderr
    - Each request has a unique request_id for correlation

Usage:
    async with JSBridge(config) as bridge:
        result = await bridge.run_trial(trial_config)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Protocol version must match traigent-js/src/cli/protocol.ts
PROTOCOL_VERSION = "1.0"

# Default timeout for trial execution (5 minutes)
DEFAULT_TRIAL_TIMEOUT_SECONDS = 300

# Default timeout for ping/shutdown commands (10 seconds)
DEFAULT_COMMAND_TIMEOUT_SECONDS = 10


class JSBridgeError(Exception):
    """Base exception for JS bridge errors."""


class JSProcessError(JSBridgeError):
    """Exception raised when Node.js process fails to start or crashes."""


class JSProtocolError(JSBridgeError):
    """Exception raised when protocol communication fails."""


class JSTrialTimeoutError(JSBridgeError):
    """Exception raised when a trial times out."""


@dataclass
class JSBridgeConfig:
    """Configuration for JSBridge.

    Attributes:
        module_path: Path to the JS module containing the trial function.
            Can be absolute or relative to the current working directory.
        function_name: Name of the exported function to call (default: "runTrial").
        use_npx: Whether to use npx to invoke traigent-js (default: True).
            Set to False if you want to specify a custom runner_path.
        runner_path: Path to the traigent-js CLI runner (optional).
            If not specified and use_npx=False, will look for traigent-js in node_modules.
        node_executable: Path to Node.js executable (default: "node").
            Only used if use_npx=False and runner_path is specified.
        node_args: Additional arguments to pass to Node.js.
        trial_timeout_seconds: Timeout for trial execution in seconds.
        command_timeout_seconds: Timeout for ping/shutdown commands.
        env: Additional environment variables for the Node.js process.
        cwd: Working directory for the Node.js process.
    """

    module_path: str
    function_name: str = "runTrial"
    use_npx: bool = True
    runner_path: str | None = None
    node_executable: str = "node"
    node_args: list[str] = field(default_factory=list)
    trial_timeout_seconds: float = DEFAULT_TRIAL_TIMEOUT_SECONDS
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    env: dict[str, str] | None = None
    cwd: str | None = None


@dataclass
class JSTrialResult:
    """Result from a JS trial execution.

    Attributes:
        trial_id: The trial identifier.
        status: Trial status (completed, failed, cancelled).
        metrics: Metrics returned by the trial function.
        duration: Trial duration in seconds.
        error_message: Error message if the trial failed.
        error_code: Error code for classification (VALIDATION_ERROR, USER_FUNCTION_ERROR, etc.)
        retryable: Whether the error is retryable.
        metadata: Additional metadata from the trial.
    """

    trial_id: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error_message: str | None = None
    error_code: str | None = None
    retryable: bool = False
    metadata: dict[str, Any] | None = None


class JSBridge:
    """Manages a Node.js subprocess for JavaScript trial execution.

    The bridge spawns a single Node.js process and communicates with it via
    NDJSON protocol. The process is reused across multiple trials.

    Usage:
        config = JSBridgeConfig(module_path="./dist/my-trial.js")
        async with JSBridge(config) as bridge:
            result = await bridge.run_trial({
                "trial_id": "abc123",
                "trial_number": 1,
                "config": {"temperature": 0.7},
                ...
            })
    """

    def __init__(self, config: JSBridgeConfig) -> None:
        """Initialize JSBridge with configuration.

        Args:
            config: Bridge configuration.
        """
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._started = False
        self._shutdown_requested = False
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> JSBridge:
        """Start the bridge (spawn Node.js process)."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop the bridge (shutdown Node.js process)."""
        await self.stop()

    async def start(self) -> None:
        """Start the Node.js process.

        Raises:
            JSProcessError: If the process fails to start.
        """
        async with self._lock:
            if self._started:
                return

            # Resolve module path
            module_path = self._config.module_path
            if not os.path.isabs(module_path):
                cwd = self._config.cwd or os.getcwd()
                module_path = os.path.join(cwd, module_path)

            # Build command - use traigent-js CLI runner, not the module directly
            if self._config.use_npx:
                # Use npx to find traigent-js (works with local or global install)
                cmd = [
                    "npx",
                    "traigent-js",
                    "--module",
                    module_path,
                    "--function",
                    self._config.function_name,
                ]
            elif self._config.runner_path:
                # Use explicit runner path
                cmd = [
                    self._config.node_executable,
                    *self._config.node_args,
                    self._config.runner_path,
                    "--module",
                    module_path,
                    "--function",
                    self._config.function_name,
                ]
            else:
                # Try to find runner in node_modules
                cwd = self._config.cwd or os.getcwd()
                runner_path = os.path.join(
                    cwd, "node_modules", ".bin", "traigent-js"
                )
                if not os.path.exists(runner_path):
                    raise JSProcessError(
                        "traigent-js not found. Install it with 'npm install traigent-js' "
                        "or set use_npx=True or provide runner_path."
                    )
                cmd = [
                    self._config.node_executable,
                    *self._config.node_args,
                    runner_path,
                    "--module",
                    module_path,
                    "--function",
                    self._config.function_name,
                ]

            # Set up environment
            env = os.environ.copy()
            if self._config.env:
                env.update(self._config.env)

            logger.info("Starting JS bridge: %s", " ".join(cmd))

            try:
                self._process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=self._config.cwd,
                )
            except FileNotFoundError as e:
                raise JSProcessError(
                    f"Failed to start Node.js process: {e}. "
                    f"Is Node.js installed and the module path correct?"
                ) from e
            except OSError as e:
                raise JSProcessError(f"Failed to start Node.js process: {e}") from e

            self._started = True
            self._shutdown_requested = False

            # Start background reader task for stdout
            self._reader_task = asyncio.create_task(self._read_responses())

            # Start background task to log stderr
            asyncio.create_task(self._log_stderr())

            # Verify the process started successfully with a ping
            try:
                await self.ping(timeout=self._config.command_timeout_seconds)
                logger.info("JS bridge started successfully")
            except Exception as e:
                await self._terminate()
                raise JSProcessError(f"JS bridge failed health check: {e}") from e

    async def stop(self, timeout: float | None = None) -> None:
        """Stop the Node.js process gracefully.

        Args:
            timeout: Timeout for graceful shutdown (default: command_timeout_seconds).
        """
        async with self._lock:
            if not self._started or self._shutdown_requested:
                return

            self._shutdown_requested = True
            timeout = timeout or self._config.command_timeout_seconds

            try:
                # Send shutdown command
                await self._send_request(
                    action="shutdown",
                    payload={},
                    timeout=timeout,
                )
            except Exception as e:
                logger.warning("Shutdown command failed, terminating: %s", e)
            finally:
                await self._terminate()

    async def _terminate(self) -> None:
        """Force terminate the Node.js process."""
        if self._process is not None:
            try:
                self._process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning("JS process did not terminate, killing")
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Already dead

        # Cancel pending requests
        for _request_id, future in self._pending_requests.items():
            if not future.done():
                future.set_exception(JSBridgeError("Bridge shutdown"))
        self._pending_requests.clear()

        # Cancel reader task
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        self._process = None
        self._reader_task = None
        self._started = False

    async def ping(self, timeout: float | None = None) -> dict[str, Any]:
        """Ping the Node.js process to check health.

        Args:
            timeout: Timeout in seconds (default: command_timeout_seconds).

        Returns:
            Ping response with timestamp and uptime.

        Raises:
            JSProtocolError: If ping fails.
        """
        timeout = timeout or self._config.command_timeout_seconds
        response = await self._send_request(
            action="ping",
            payload={},
            timeout=timeout,
        )
        return response["payload"]

    async def cancel(self, trial_id: str | None = None) -> dict[str, Any]:
        """Cancel an in-flight trial.

        Args:
            trial_id: Specific trial ID to cancel (or None for current trial).

        Returns:
            Cancel response with status.
        """
        response = await self._send_request(
            action="cancel",
            payload={"trial_id": trial_id} if trial_id else {},
            timeout=self._config.command_timeout_seconds,
        )
        return response["payload"]

    async def run_trial(
        self,
        trial_config: dict[str, Any],
        timeout: float | None = None,
    ) -> JSTrialResult:
        """Run a trial in the Node.js process.

        Args:
            trial_config: Trial configuration to pass to the JS function.
                Must include: trial_id, trial_number, experiment_run_id, config, dataset_subset.
            timeout: Timeout in seconds (default: trial_timeout_seconds).

        Returns:
            JSTrialResult with trial outcome.

        Raises:
            JSTrialTimeoutError: If the trial times out.
            JSProtocolError: If protocol communication fails.
        """
        timeout = timeout or self._config.trial_timeout_seconds

        # Add timeout to payload so JS can also enforce it
        payload = dict(trial_config)
        payload["timeout_ms"] = int(timeout * 1000)

        try:
            response = await self._send_request(
                action="run_trial",
                payload=payload,
                timeout=timeout + 5,  # Give a bit of buffer for protocol overhead
            )
        except TimeoutError:
            # Try to cancel the trial
            try:
                await self.cancel(trial_config.get("trial_id"))
            except Exception:
                pass
            raise JSTrialTimeoutError(
                f"Trial {trial_config.get('trial_id')} timed out after {timeout}s"
            ) from None

        return self._parse_trial_response(response)

    def _parse_trial_response(self, response: dict[str, Any]) -> JSTrialResult:
        """Parse a trial response into JSTrialResult.

        Args:
            response: Raw response from Node.js.

        Returns:
            Parsed JSTrialResult.
        """
        payload = response.get("payload", {})

        if response.get("status") == "error":
            # Protocol-level error
            return JSTrialResult(
                trial_id=payload.get("trial_id", "unknown"),
                status="failed",
                error_message=payload.get("error", "Unknown error"),
                error_code=payload.get("error_code", "PROTOCOL_ERROR"),
                retryable=payload.get("retryable", False),
            )

        # Success response - payload contains trial result
        status = payload.get("status", "completed")
        trial_id = payload.get("trial_id", "unknown")

        if status in ("completed", "success"):
            return JSTrialResult(
                trial_id=trial_id,
                status="completed",
                metrics=payload.get("metrics", {}),
                duration=payload.get("duration", 0.0),
                metadata=payload.get("metadata"),
            )
        elif status == "cancelled":
            return JSTrialResult(
                trial_id=trial_id,
                status="cancelled",
                duration=payload.get("duration", 0.0),
                error_message=payload.get("error_message", "Trial cancelled"),
            )
        else:
            # Failed trial
            return JSTrialResult(
                trial_id=trial_id,
                status="failed",
                duration=payload.get("duration", 0.0),
                error_message=payload.get("error_message", "Unknown error"),
                error_code=payload.get("error_code"),
                retryable=payload.get("retryable", False),
            )

    async def _send_request(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Send a request to the Node.js process and wait for response.

        Args:
            action: Action type (run_trial, ping, shutdown, cancel).
            payload: Request payload.
            timeout: Response timeout in seconds.

        Returns:
            Response dictionary.

        Raises:
            JSProtocolError: If communication fails.
            asyncio.TimeoutError: If response times out.
        """
        if not self._started or self._process is None:
            raise JSBridgeError("JS bridge not started")

        if self._process.stdin is None:
            raise JSBridgeError("JS process stdin not available")

        request_id = str(uuid.uuid4())
        request = {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "action": action,
            "payload": payload,
        }

        # Create future for response
        response_future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = response_future

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self._process.stdin.write(request_line.encode())
            await self._process.stdin.drain()
            logger.debug("Sent request: %s", action)

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except TimeoutError:
            logger.warning("Request %s timed out after %ss", request_id, timeout)
            raise
        except Exception as e:
            logger.error("Failed to send request: %s", e)
            raise JSProtocolError(f"Failed to send request: {e}") from e
        finally:
            self._pending_requests.pop(request_id, None)

    async def _read_responses(self) -> None:
        """Background task to read responses from Node.js stdout."""
        if self._process is None or self._process.stdout is None:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    logger.info("JS process stdout closed")
                    # Process exited - fail all pending requests immediately
                    # instead of letting them wait for timeout
                    exit_code = (
                        self._process.returncode if self._process else None
                    )
                    error_msg = f"JS process exited unexpectedly (exit code: {exit_code})"
                    for future in self._pending_requests.values():
                        if not future.done():
                            future.set_exception(JSProcessError(error_msg))
                    break

                try:
                    response = json.loads(line.decode())
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON from JS process: %s", e)
                    continue

                request_id = response.get("request_id")
                if request_id and request_id in self._pending_requests:
                    future = self._pending_requests[request_id]
                    if not future.done():
                        future.set_result(response)
                else:
                    logger.warning("Received response for unknown request: %s", request_id)

        except asyncio.CancelledError:
            # Re-raise to signal clean task completion
            raise
        except Exception as e:
            logger.error("Error reading from JS process: %s", e)
            # Fail all pending requests
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(JSProtocolError(f"Read error: {e}"))

    async def _log_stderr(self) -> None:
        """Background task to log stderr output from Node.js."""
        if self._process is None or self._process.stderr is None:
            return

        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                # Log as debug since JS runner prefixes user logs
                logger.debug("[JS] %s", line.decode().rstrip())
        except asyncio.CancelledError:
            # Re-raise to signal clean task completion
            raise
        except Exception as e:
            logger.warning("Error reading JS stderr: %s", e)

    @property
    def is_running(self) -> bool:
        """Check if the bridge is running."""
        return self._started and self._process is not None and self._process.returncode is None
