"""Progress tracking and callback system for Traigent optimization."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Observability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..api.types import OptimizationResult, TrialResult

logger = get_logger(__name__)

CallbackInvocationKey = tuple[int, str]


def _shutdown_callback_executor(
    executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    """Shut down a shared callback executor without blocking process exit."""
    executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class ProgressInfo:
    """Information about optimization progress."""

    current_trial: int
    total_trials: int
    completed_trials: int
    successful_trials: int
    failed_trials: int
    best_score: float | None
    best_config: dict[str, Any] | None
    elapsed_time: float
    estimated_remaining: float | None
    current_algorithm: str

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_trials == 0:
            return 0.0
        return (self.completed_trials / self.total_trials) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.completed_trials == 0:
            return 0.0
        return (self.successful_trials / self.completed_trials) * 100


class OptimizationCallback(ABC):
    """Abstract base class for optimization callbacks."""

    @abstractmethod
    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        raise NotImplementedError

    @abstractmethod
    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts."""
        raise NotImplementedError

    @abstractmethod
    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        raise NotImplementedError

    @abstractmethod
    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        raise NotImplementedError


class ProgressBarCallback(OptimizationCallback):
    """Progress bar callback using simple text output."""

    def __init__(self, width: int = 50, update_interval: float = 1.0) -> None:
        """Initialize progress bar callback.

        Args:
            width: Width of progress bar in characters
            update_interval: Minimum seconds between updates
        """
        self.width = width
        self.update_interval = update_interval
        self.last_update = 0.0
        self.start_time = 0.0
        # Track last progress for final update
        self._last_progress: ProgressInfo | None = None
        self._config_space: dict[str, Any] = {}
        self._objectives: list[str] = []

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        self.start_time = time.time()
        self._config_space = dict(config_space)
        self._objectives = list(objectives)
        # Note: ProgressBarCallback uses print for interactive console output
        # This is intentional for user-facing progress display
        print(f"🚀 Starting optimization with {algorithm}")
        print(f"📊 Objectives: {', '.join(objectives)}")
        print(f"⚙️  Configuration space: {len(config_space)} parameters")
        print()

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts."""
        return None  # Update occurs on completion for cleaner output

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        # Always store last progress for final update
        self._last_progress = progress

        current_time = time.time()

        # Throttle updates
        if current_time - self.last_update < self.update_interval:
            return

        self.last_update = current_time

        # Create progress bar
        filled_width = int(self.width * progress.progress_percent / 100)
        bar = "█" * filled_width + "░" * (self.width - filled_width)

        # Format time
        elapsed = time.strftime("%M:%S", time.gmtime(progress.elapsed_time))

        # Format best score - handle None case properly
        best_score_str = (
            f"{progress.best_score:.3f}" if progress.best_score is not None else "N/A"
        )

        # Print progress line - using print for interactive console output
        print(
            f"\r[{bar}] {progress.progress_percent:5.1f}% "
            f"({progress.completed_trials}/{progress.total_trials}) "
            f"✅ {progress.successful_trials} "
            f"❌ {progress.failed_trials} "
            f"⏱️  {elapsed} "
            f"🏆 {best_score_str}",
            end="",
            flush=True,
        )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        # Draw final 100% progress bar (throttle may have skipped the last update)
        if self._last_progress is not None:
            bar = "█" * self.width
            elapsed = time.strftime(
                "%M:%S", time.gmtime(self._last_progress.elapsed_time)
            )
            best_score_str = (
                f"{self._last_progress.best_score:.3f}"
                if self._last_progress.best_score is not None
                else "N/A"
            )
            total = self._last_progress.total_trials
            # Using print for final progress display output
            print(
                f"\r[{bar}] 100.0% "
                f"({total}/{total}) "
                f"✅ {self._last_progress.successful_trials} "
                f"❌ {self._last_progress.failed_trials} "
                f"⏱️  {elapsed} "
                f"🏆 {best_score_str}",
                flush=True,
            )

        print("\n")
        if result.stop_reason == "timeout" or result.status == "cancelled":
            timeout_value = None
            if isinstance(result.metadata, dict):
                timeout_value = result.metadata.get("timeout")
            timeout_hint = f" ({timeout_value}s)" if timeout_value else ""
            print(f"⚠️ Optimization stopped early: timeout reached{timeout_hint}.")
        else:
            print("✅ Optimization complete!")
        best_score_str = (
            f"{result.best_score:.3f}" if result.best_score is not None else "N/A"
        )
        print(f"🏆 Best score: {best_score_str}")
        print(f"⏱️  Total time: {result.duration:.1f}s")
        print(f"📈 Success rate: {result.success_rate:.1%}")

        if result.stop_reason and result.stop_reason != "timeout":
            print(f"🛑 Stop reason: {result.stop_reason}")

        if result.run_label:
            print(f"🔖 Run: {result.run_label}")
        if result.cloud_url:
            print(f"🔗 View: {result.cloud_url}")
        elif (
            result.run_label
            and isinstance(result.metadata, dict)
            and result.metadata.get("offline_mode")
        ):
            print("   Run locally — sync to cloud with `traigent sync`")

        self._print_results_table(result)

    def _print_results_table(self, result: OptimizationResult) -> None:
        """Render the canonical optimization results table when enough context exists."""
        if not getattr(result, "trials", None):
            return
        if not self._config_space or not self._objectives:
            return

        try:
            from traigent.utils.results_table import print_results_table

            print_results_table(result, self._config_space, self._objectives)
        except Exception as exc:
            logger.warning("Failed to render results table: %s", exc)


class ResultsTableCallback(OptimizationCallback):
    """Lightweight callback that renders the results table after optimization.

    Unlike ``ProgressBarCallback``, this callback produces *no* output during
    optimization and only prints the rich results table once optimization
    completes.  It is auto-injected by ``_resolve_callbacks`` so that users
    always see a results summary — even in non-interactive terminals where the
    progress bar is suppressed.
    """

    def __init__(self) -> None:
        self._config_space: dict[str, Any] = {}
        self._objectives: list[str] = []

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        self._config_space = dict(config_space)
        self._objectives = list(objectives)

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        return None

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        return None

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        if not getattr(result, "trials", None):
            return
        if not self._config_space or not self._objectives:
            return
        try:
            from traigent.utils.results_table import print_results_table

            print_results_table(result, self._config_space, self._objectives)
        except Exception as exc:
            logger.warning("Failed to render results table: %s", exc)


class LoggingCallback(OptimizationCallback):
    """Callback that logs optimization progress."""

    def __init__(self, logger: Any | None = None, log_level: str = "INFO") -> None:
        """Initialize logging callback.

        Args:
            logger: Logger instance (uses print if None)
            log_level: Logging level
        """
        self.logger = logger
        self.log_level = log_level

    def _log(self, message: str) -> None:
        """Log message."""
        if self.logger:
            getattr(self.logger, self.log_level.lower())(message)
        else:
            # Fallback to module logger if no logger provided
            getattr(logger, self.log_level.lower())(message)

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        self._log(
            f"Starting optimization: algorithm={algorithm}, objectives={objectives}"
        )

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts."""
        self._log(f"Trial {trial_number} started: config={config}")

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        self._log(
            f"Trial {progress.current_trial} complete: "
            f"status={trial.status}, "
            f"metrics={trial.metrics}, "
            f"progress={progress.progress_percent:.1f}%"
        )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        best_score_str = (
            f"{result.best_score:.3f}" if result.best_score is not None else "N/A"
        )
        self._log(
            f"Optimization complete: "
            f"best_score={best_score_str}, "
            f"success_rate={result.success_rate:.1%}, "
            f"duration={result.duration:.1f}s"
        )


class StatisticsCallback(OptimizationCallback):
    """Callback that collects optimization statistics."""

    def __init__(self) -> None:
        """Initialize statistics callback."""
        self.stats: dict[str, Any] = {
            "trial_times": [],
            "scores_by_trial": [],
            "configs_tried": [],
            "failure_reasons": [],
            "parameter_values": {},
        }

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        self.stats["algorithm"] = algorithm
        self.stats["objectives"] = objectives
        self.stats["config_space"] = config_space

        # Initialize parameter tracking
        for param in config_space.keys():
            self.stats["parameter_values"][param] = []

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts."""
        self.stats["configs_tried"].append(config)

        # Track parameter values
        for param, value in config.items():
            if param in self.stats["parameter_values"]:
                self.stats["parameter_values"][param].append(value)

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        self.stats["trial_times"].append(trial.duration)

        if trial.status == "completed":
            # Extract primary objective score
            score = trial.metrics.get("accuracy", 0.0)  # Default to accuracy
            self.stats["scores_by_trial"].append(score)
        else:
            self.stats["scores_by_trial"].append(None)
            self.stats["failure_reasons"].append(trial.error_message or "Unknown error")

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        self.stats["total_duration"] = result.duration
        self.stats["best_score"] = result.best_score
        self.stats["best_config"] = result.best_config

    def get_parameter_importance(self) -> dict[str, float]:
        """Calculate parameter importance based on score variance.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        importance = {}

        for param, values in self.stats["parameter_values"].items():
            if len(values) < 2:
                importance[param] = 0.0
                continue

            # Group scores by parameter value
            value_scores: dict[Any, list[float]] = {}
            for i, value in enumerate(values):
                if (
                    i < len(self.stats["scores_by_trial"])
                    and self.stats["scores_by_trial"][i] is not None
                ):
                    if value not in value_scores:
                        value_scores[value] = []
                    value_scores[value].append(self.stats["scores_by_trial"][i])

            if len(value_scores) < 2:
                importance[param] = 0.0
                continue

            # Calculate variance between different parameter values
            group_means = [
                sum(scores) / len(scores) for scores in value_scores.values() if scores
            ]
            if len(group_means) >= 2:
                overall_mean = sum(group_means) / len(group_means)
                variance = sum(
                    (mean - overall_mean) ** 2 for mean in group_means
                ) / len(group_means)
                importance[param] = variance
            else:
                importance[param] = 0.0

        # Normalize importance scores
        max_importance = max(importance.values()) if importance.values() else 1.0
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}

        return importance


class SimpleProgressCallback(OptimizationCallback):
    """Simple progress callback for basic console output.

    This callback provides simple progress tracking suitable for demos
    and quick prototyping, with configurable output destinations.
    """

    def __init__(self, output: str | Any = "print", show_details: bool = True) -> None:
        """Initialize simple progress callback.

        Args:
            output: Output destination - "print" for console, "log" for logger,
                   or a callable that accepts strings
            show_details: Whether to show detailed progress information
        """
        self.output = output
        self.show_details = show_details
        self.total_trials = 0
        self.current_trial = 0
        self.best_score: float | None = None
        self._output_handler = self._get_output_handler(output)

    def _get_output_handler(self, output: str | Any) -> Any:
        """Get the appropriate output handler."""
        if output == "print":
            return print
        elif output == "log":
            return logger.info
        elif callable(output):
            return output
        else:
            # Default to print if invalid output specified
            logger.warning(f"Invalid output type {output}, defaulting to print")
            return print

    def _output(self, message: str) -> None:
        """Output a message using the configured handler."""
        self._output_handler(message)

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        # Calculate total configurations
        if config_space:
            total_configs = 1
            for _, values in config_space.items():
                if isinstance(values, list):
                    total_configs *= len(values)
            self.total_trials = total_configs

        self._output(
            f"\n🔄 Starting {algorithm} optimization with {self.total_trials} configurations..."
        )
        if self.show_details:
            self._output(f"📊 Objectives: {', '.join(objectives)}")

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts."""
        self.current_trial = trial_number + 1

        if self.show_details:
            # Build a descriptive string of the configuration
            config_parts = []

            # Prioritize showing model/approach/method first
            primary_param = (
                config.get("model") or config.get("approach") or config.get("method")
            )
            if primary_param:
                config_parts.append(str(primary_param))

            # Add other important parameters
            for key, value in config.items():
                if key not in ["model", "approach", "method"] and value is not None:
                    # Format the value nicely
                    if isinstance(value, float):
                        config_parts.append(f"{key}={value:.1f}")
                    else:
                        config_parts.append(f"{key}={value}")

            config_str = ", ".join(config_parts) if config_parts else "config"
            self._output(
                f"[{self.current_trial}/{self.total_trials}] Testing: {config_str}"
            )

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        if progress.best_score is not None:
            self.best_score = progress.best_score

        if self.show_details and trial.status == "completed":
            # Extract primary metric
            score = None
            if trial.metrics:
                score = trial.metrics.get("accuracy") or trial.metrics.get(
                    "score", None
                )

            if score is not None:
                self._output(
                    f"[{self.current_trial}/{self.total_trials}] "
                    f"Completed. Score: {score:.1%} | Best so far: {self.best_score:.1%}"
                )
            elif self.best_score is not None:
                self._output(
                    f"[{self.current_trial}/{self.total_trials}] "
                    f"Completed. Best score so far: {self.best_score:.1%}"
                )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        self._output("\n✨ Optimization complete!")
        if self.show_details and result.best_score is not None:
            self._output(f"🏆 Best score: {result.best_score:.3f}")
            if result.best_config:
                self._output(f"⚙️  Best config: {result.best_config}")


@dataclass
class CallbackFailure:
    """Records a callback failure for reporting.

    Attributes:
        callback_name: Name of the callback class that failed
        method: The callback method that failed (e.g., 'on_trial_complete')
        error_type: Type of error ('timeout' or 'exception')
        error_message: Human-readable error description
        timestamp: When the failure occurred
    """

    callback_name: str
    method: str
    error_type: str  # 'timeout' or 'exception'
    error_message: str
    timestamp: float = field(default_factory=time.time)


class CallbackManager:
    """Manages multiple optimization callbacks with timeout and exception isolation.

    This class ensures that callback failures (exceptions or timeouts) do not
    block or crash the optimization process. All failures are logged and tracked
    for reporting.

    Timeout semantics are intentionally conservative:

    - the optimization loop stops waiting once the timeout expires
    - a timed-out callback may continue running in its worker thread
    - ``future.cancel()`` only prevents execution if the callback has not started yet

    For callbacks with side effects, prefer idempotent operations and avoid holding
    locks or scarce resources for long periods. If a callback must complete before
    optimization proceeds, disable timeout protection for that manager instance.

    Attributes:
        callbacks: List of registered callbacks
        timeout: Maximum seconds to wait for a callback (default 5.0)
        max_callback_threads: Maximum concurrent callback worker threads
        callback_failures: List of recorded callback failures
    """

    DEFAULT_TIMEOUT: float = 5.0
    DEFAULT_MAX_CALLBACK_THREADS: int = 4

    def __init__(
        self,
        callbacks: list[OptimizationCallback] | None = None,
        timeout: float | None = None,
        max_callback_threads: int = DEFAULT_MAX_CALLBACK_THREADS,
    ) -> None:
        """Initialize callback manager.

        Args:
            callbacks: List of callback instances
            timeout: Maximum seconds to wait for a callback (default 5.0).
                Set to 0 to disable timeout protection; passing None uses the
                default timeout (5.0 seconds). When a timeout occurs,
                optimization continues immediately, but the callback thread may
                still run to completion in the background.
            max_callback_threads: Maximum number of shared callback worker
                threads used when timeout protection is enabled.
        """
        if max_callback_threads <= 0:
            raise ValueError("max_callback_threads must be greater than 0")

        self.callbacks = callbacks or []
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.max_callback_threads = max_callback_threads
        self.callback_failures: list[CallbackFailure] = []
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._executor_lock = threading.Lock()
        self._executor_finalizer: weakref.finalize | None = None
        self._running_callbacks: dict[
            CallbackInvocationKey, concurrent.futures.Future[Any]
        ] = {}
        self._running_callbacks_lock = threading.Lock()

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create or return the shared callback executor."""
        executor = self._executor
        if executor is not None:
            return executor

        with self._executor_lock:
            executor = self._executor
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_callback_threads,
                    thread_name_prefix="callback",
                )
                self._executor = executor
                self._executor_finalizer = weakref.finalize(
                    self,
                    _shutdown_callback_executor,
                    executor,
                )
            return executor

    def close(self) -> None:
        """Release callback worker threads owned by this manager."""
        with self._executor_lock:
            finalizer = self._executor_finalizer
            self._executor = None
            self._executor_finalizer = None

        with self._running_callbacks_lock:
            self._running_callbacks.clear()

        if finalizer is not None and finalizer.alive:
            finalizer()

    def _clear_running_callback(
        self,
        key: CallbackInvocationKey,
        future: concurrent.futures.Future[Any],
    ) -> None:
        """Remove a completed callback from the running-callback registry."""
        with self._running_callbacks_lock:
            current_future = self._running_callbacks.get(key)
            if current_future is future:
                self._running_callbacks.pop(key, None)

    def _submit_callback(
        self,
        callback_name: str,
        callback: OptimizationCallback,
        method_name: str,
        method_func: Callable[..., Any],
        *args: Any,
    ) -> concurrent.futures.Future[Any] | None:
        """Submit a callback unless the same method is already running."""
        executor = self._ensure_executor()
        key = (id(callback), method_name)

        with self._running_callbacks_lock:
            current_future = self._running_callbacks.get(key)
            if current_future is not None:
                if current_future.done():
                    self._running_callbacks.pop(key, None)
                else:
                    logger.warning(
                        f"Callback {callback_name}.{method_name} is still running "
                        "from a previous invocation - skipping"
                    )
                    return None

            # Start callbacks in a fresh context so pooled worker threads do not
            # retain or inherit caller-specific contextvars across invocations.
            future = executor.submit(contextvars.Context().run, method_func, *args)
            self._running_callbacks[key] = future

        self_ref = weakref.ref(self)

        def _cleanup(done_future: concurrent.futures.Future[Any]) -> None:
            manager = self_ref()
            if manager is not None:
                manager._clear_running_callback(key, done_future)

        future.add_done_callback(_cleanup)
        return future

    def _invoke_callback_safe(
        self,
        callback: OptimizationCallback,
        method_name: str,
        method_func: Callable[..., Any],
        *args: Any,
    ) -> None:
        """Safely invoke a callback method with timeout and exception handling.

        Args:
            callback: The callback instance
            method_name: Name of the method being called (for logging)
            method_func: The actual method to call
            *args: Arguments to pass to the method

        Notes:
            Timeout protection is best-effort only. A timeout stops waiting on
            the callback future but does not forcibly terminate a callback that
            is already running.
        """
        callback_name = callback.__class__.__name__

        # If timeout is disabled, invoke directly with exception handling only
        if not self.timeout or self.timeout <= 0:
            try:
                method_func(*args)
            except Exception as e:
                self._record_failure(callback_name, method_name, "exception", str(e))
                logger.warning(
                    f"Callback {callback_name}.{method_name} failed: {e} "
                    "- continuing optimization"
                )
            return

        # Use a shared ThreadPoolExecutor for timeout protection to avoid
        # creating a new thread pool per callback invocation.
        future = self._submit_callback(
            callback_name,
            callback,
            method_name,
            method_func,
            *args,
        )
        if future is None:
            return

        try:
            future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            self._record_failure(
                callback_name,
                method_name,
                "timeout",
                f"Timed out after {self.timeout}s",
            )
            logger.warning(
                f"Callback {callback_name}.{method_name} timed out "
                f"after {self.timeout}s - continuing optimization"
            )
            # Cancel if possible (may not stop already-running threads)
            future.cancel()
        except Exception as e:
            self._record_failure(callback_name, method_name, "exception", str(e))
            logger.warning(
                f"Callback {callback_name}.{method_name} failed: {e} "
                "- continuing optimization"
            )

    def _record_failure(
        self, callback_name: str, method: str, error_type: str, error_message: str
    ) -> None:
        """Record a callback failure for later reporting."""
        self.callback_failures.append(
            CallbackFailure(
                callback_name=callback_name,
                method=method,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def get_failure_summary(self) -> dict[str, Any]:
        """Get a summary of all callback failures.

        Returns:
            Dictionary with failure counts and details
        """
        if not self.callback_failures:
            return {"total_failures": 0, "failures": []}

        return {
            "total_failures": len(self.callback_failures),
            "timeout_count": sum(
                1 for f in self.callback_failures if f.error_type == "timeout"
            ),
            "exception_count": sum(
                1 for f in self.callback_failures if f.error_type == "exception"
            ),
            "failures": [
                {
                    "callback": f.callback_name,
                    "method": f.method,
                    "type": f.error_type,
                    "message": f.error_message,
                }
                for f in self.callback_failures
            ],
        }

    def add_callback(self, callback: OptimizationCallback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: OptimizationCallback) -> None:
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Notify all callbacks of optimization start."""
        logger.debug(
            f"on_optimization_start called with {len(self.callbacks)} callbacks"
        )
        for i, callback in enumerate(self.callbacks):
            logger.debug(f"Calling callback {i}: {callback.__class__.__name__}")
            self._invoke_callback_safe(
                callback,
                "on_optimization_start",
                callback.on_optimization_start,
                config_space,
                objectives,
                algorithm,
            )

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Notify all callbacks of trial start."""
        for callback in self.callbacks:
            self._invoke_callback_safe(
                callback,
                "on_trial_start",
                callback.on_trial_start,
                trial_number,
                config,
            )

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Notify all callbacks of trial completion."""
        for callback in self.callbacks:
            self._invoke_callback_safe(
                callback,
                "on_trial_complete",
                callback.on_trial_complete,
                trial,
                progress,
            )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Notify all callbacks of optimization completion."""
        try:
            for callback in self.callbacks:
                self._invoke_callback_safe(
                    callback,
                    "on_optimization_complete",
                    callback.on_optimization_complete,
                    result,
                )
        finally:
            self.close()


class DetailedProgressCallback(OptimizationCallback):
    """Detailed progress callback that shows comprehensive optimization information.

    This callback provides detailed progress tracking with:
    - Configuration space overview
    - Trial-by-trial configuration details
    - Metrics for each trial
    - Visual progress bar
    - Running best score tracking
    """

    def __init__(
        self, show_config_details: bool = True, show_metrics: bool = True
    ) -> None:
        """Initialize detailed progress callback.

        Args:
            show_config_details: Whether to show configuration details for each trial
            show_metrics: Whether to show metrics after each trial
        """
        self.show_config_details = show_config_details
        self.show_metrics = show_metrics
        self.trial_count = 0
        self.total_trials = 0
        self.configs_tested: list[Any] = []

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts."""
        # Calculate total number of configurations
        total = 1
        for _param, values in config_space.items():
            if isinstance(values, list):
                total *= len(values)
        if total <= 0:
            logger.warning(
                "Calculated configuration combinations is zero; detailed progress "
                "will fall back to runtime counts."
            )
            self.total_trials = 0
        else:
            self.total_trials = total

        print("\n" + "=" * 60)
        print("🚀 OPTIMIZATION STARTING")
        print("=" * 60)
        print(f"📊 Algorithm: {algorithm}")
        print(f"🎯 Objectives: {', '.join(objectives)}")
        print("🔧 Configuration Space:")

        for param, values in config_space.items():
            print(f"   • {param}: {values}")

        if self.total_trials > 0:
            print(f"\n📈 Total configurations to test: {self.total_trials}")
        else:
            print(
                "\n📈 Total configurations to test: unknown (will infer from observed trials)"
            )
        print("-" * 60 + "\n")

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts."""
        self.trial_count = trial_number + 1
        self.configs_tested.append(config)

        total_display = self.total_trials if self.total_trials > 0 else "?"
        print(f"▶️  Trial {self.trial_count}/{total_display} starting...")

        if self.show_config_details:
            print("   Configuration:")
            for key, value in config.items():
                if isinstance(value, float):
                    print(f"     • {key}: {value:.2f}")
                else:
                    print(f"     • {key}: {value}")

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Called when a trial completes."""
        status_icon = "✅" if trial.status == "completed" else "❌"

        total_display = self.total_trials or progress.total_trials
        if not total_display or total_display <= 0:
            total_display_str = "?"
        else:
            total_display_str = str(total_display)

        print(f"{status_icon} Trial {self.trial_count}/{total_display_str} completed")

        if self.show_metrics and trial.metrics:
            print("   Metrics:")
            for metric, value in trial.metrics.items():
                if isinstance(value, float):
                    print(f"     • {metric}: {value:.3f}")
                else:
                    print(f"     • {metric}: {value}")

        if progress.best_score is not None:
            print(f"   🏆 Best score so far: {progress.best_score:.3f}")

        # Progress bar
        denominator_candidates = [
            progress.total_trials,
            self.total_trials,
        ]
        denominator = next(
            (
                value
                for value in denominator_candidates
                if isinstance(value, (int, float)) and value > 0
            ),
            None,
        )

        if denominator is None:
            denominator = max(progress.completed_trials, self.trial_count, 1)

        completed_count = max(progress.completed_trials, self.trial_count)
        completed_ratio = min(completed_count / max(denominator, 1), 1.0)
        percent = completed_ratio * 100

        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"   Progress: [{bar}] {percent:.0f}%")
        print()

    def _print_header(self, result: OptimizationResult) -> None:
        """Print the completion header."""
        print("\n" + "=" * 60)
        is_early_stop = result.stop_reason == "timeout" or result.status == "cancelled"
        if is_early_stop:
            timeout_value = (
                result.metadata.get("timeout")
                if isinstance(result.metadata, dict)
                else None
            )
            timeout_hint = f" ({timeout_value}s)" if timeout_value else ""
            print(f"⚠️ OPTIMIZATION STOPPED EARLY: TIMEOUT REACHED{timeout_hint}")
        else:
            print("✨ OPTIMIZATION COMPLETE!")
        print("=" * 60)

    def _print_config(self, config: dict[str, Any]) -> None:
        """Print the best configuration."""
        print("⚙️  Best Configuration:")
        for key, value in config.items():
            formatted = f"{value:.2f}" if isinstance(value, float) else str(value)
            print(f"   • {key}: {formatted}")

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes."""
        self._print_header(result)

        if result.best_score is not None:
            print(f"🏆 Best Score: {result.best_score:.3f}")

        if result.best_config:
            self._print_config(result.best_config)

        if result.duration:
            print(f"⏱️  Total Time: {result.duration:.1f} seconds")

        if result.success_rate:
            print(f"📊 Success Rate: {result.success_rate:.1%}")

        if result.stop_reason and result.stop_reason != "timeout":
            print(f"🛑 Stop reason: {result.stop_reason}")

        if result.run_label:
            print(f"🔖 Run: {result.run_label}")
        if result.cloud_url:
            print(f"🔗 View: {result.cloud_url}")
        elif (
            result.run_label
            and isinstance(result.metadata, dict)
            and result.metadata.get("offline_mode")
        ):
            print("   Run locally — sync to cloud with `traigent sync`")

        print("=" * 60 + "\n")


# Convenience functions for common callback combinations
def get_default_callbacks() -> list[OptimizationCallback]:
    """Get default callbacks for optimization."""
    return [ProgressBarCallback(), StatisticsCallback()]


def get_verbose_callbacks() -> list[OptimizationCallback]:
    """Get verbose callbacks for detailed logging."""
    return [ProgressBarCallback(), LoggingCallback(), StatisticsCallback()]


def get_detailed_callbacks() -> list[OptimizationCallback]:
    """Get detailed callbacks with comprehensive progress tracking."""
    return [DetailedProgressCallback(), StatisticsCallback()]
