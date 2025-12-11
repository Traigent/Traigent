"""Coordinator and execution helpers for distributed Optuna optimisation."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from traigent.optimizers.optuna_utils import (
    config_space_to_distributions,
    infer_directions,
)
from traigent.optimizers.pruners import CeilingPruner
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "OptunaCoordinator",
    "RateLimiter",
    "RateLimitedOptimizer",
    "BatchOptimizer",
    "EdgeExecutor",
    "ResilientCoordinator",
]

try:  # pragma: no cover - executed only if Optuna missing
    import optuna
except ImportError as exc:  # pragma: no cover
    optuna = None
    OPTUNA_IMPORT_ERROR: ImportError | None = exc
else:
    OPTUNA_IMPORT_ERROR = None

_OPTUNA_SEED_ENV = "TRAIGENT_OPTUNA_SEED"


def _require_optuna() -> None:
    if optuna is None:  # pragma: no cover
        raise OptimizationError(
            "Optuna is required for the distributed coordination utilities."
        ) from OPTUNA_IMPORT_ERROR


def _resolve_default_sampler() -> optuna.samplers.BaseSampler:
    """Return a deterministic default sampler with optional user overrides."""

    seed_value: int | None = 0
    env_value = os.getenv(_OPTUNA_SEED_ENV)
    if env_value is not None:
        normalised = env_value.strip().lower()
        if normalised in {"", "none", "null"}:
            seed_value = None
        elif normalised in {"random", "time"}:
            seed_value = None
        else:
            try:
                seed_value = int(env_value)
            except ValueError:
                logger.warning(
                    "Invalid %s value '%s'; falling back to seed=0",
                    _OPTUNA_SEED_ENV,
                    env_value,
                )
                seed_value = 0

    return optuna.samplers.TPESampler(seed=seed_value)


class OptunaCoordinator:
    """Coordinate distributed execution using Optuna's ask/tell interface."""

    def __init__(
        self,
        *,
        config_space: Mapping[str, Any] | None = None,
        objectives: Iterable[str] | None = None,
        storage: str | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        directions: Iterable[str] | None = None,
        study_name: str | None = None,
        search_space: Mapping[str, Any] | None = None,
    ) -> None:
        _require_optuna()

        resolved_space = config_space or search_space
        if resolved_space is None:
            raise TypeError("config_space or search_space must be provided")

        objective_list = list(objectives) if objectives is not None else []
        if not objective_list and directions is None:
            raise OptimizationError("Either objectives or directions must be supplied")

        self._config_space = dict(resolved_space)
        self._distributions = config_space_to_distributions(self._config_space)
        self._directions = (
            list(directions) if directions else infer_directions(objective_list)
        )

        if pruner is None:
            if len(self._directions) > 1:
                pruner = optuna.pruners.SuccessiveHalvingPruner()
            else:
                pruner = CeilingPruner(
                    min_completed_trials=2,
                    warmup_steps=2,
                    epsilon=1e-6,
                )

        create_kwargs = {
            "directions": self._directions,
            "sampler": sampler or _resolve_default_sampler(),
            "pruner": pruner,
            "storage": storage,
            "study_name": study_name,
            "load_if_exists": False,
        }

        try:
            self._study = optuna.create_study(**create_kwargs)
        except optuna.exceptions.DuplicatedStudyError:
            if storage and study_name:
                optuna.delete_study(study_name=study_name, storage=storage)
                self._study = optuna.create_study(**create_kwargs)
            else:
                raise

        self._pending_trials: dict[int, optuna.trial.Trial] = {}
        self._pending_configs: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def study(self) -> optuna.study.Study:
        return self._study

    @property
    def pending_trials(self) -> dict[int, optuna.trial.Trial]:
        with self._lock:
            return dict(self._pending_trials)

    @property
    def pending_configs(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {key: dict(value) for key, value in self._pending_configs.items()}

    def ask_batch(
        self, n_suggestions: int = 1
    ) -> tuple[list[dict[str, Any]], list[optuna.trial.Trial]]:
        configs: list[dict[str, Any]] = []
        trials: list[optuna.trial.Trial] = []
        with self._lock:
            for _ in range(n_suggestions):
                trial = self._study.ask()
                config = self._trial_to_config(trial)
                config["_trial_id"] = trial.number
                configs.append(config)
                trials.append(trial)
                self._pending_trials[trial.number] = trial
                self._pending_configs[trial.number] = config

        logger.debug("Coordinator generated %s configs", len(configs))
        return configs, trials

    def report_intermediate(self, trial_id: int, step: int, value: float) -> bool:
        with self._lock:
            trial = self._pending_trials.get(trial_id)
            if trial is None:
                logger.warning(
                    "Intermediate report received for unknown trial %s", trial_id
                )
                return False

            trial.report(value, step)
            should_prune = bool(trial.should_prune())
        logger.debug(
            "Intermediate value for trial=%s step=%s value=%s prune=%s",
            trial_id,
            step,
            value,
            should_prune,
        )
        return should_prune

    def tell_result(
        self,
        trial_id: int,
        values: float | Iterable[float] | None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        trial = self._pop_trial(trial_id)
        if trial is None:
            return

        state_override = (metadata or {}).get("state") if metadata else None

        if values is None:
            trial_state = None
            if state_override:
                lowered = str(state_override).lower()
                if lowered == "pruned":
                    trial_state = optuna.trial.TrialState.PRUNED
                elif lowered == "failed":
                    trial_state = optuna.trial.TrialState.FAIL
                elif lowered == "completed":
                    trial_state = optuna.trial.TrialState.COMPLETE

            if trial_state is None:
                trial_state = optuna.trial.TrialState.COMPLETE

            self._study.tell(trial, values=None, state=trial_state)
            logger.info(
                "Trial %s recorded with explicit state %s", trial_id, trial_state
            )
            return

        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            ordered_list = list(values)
            self._study.tell(trial, values=ordered_list)
        else:
            single_val = float(values)
            self._study.tell(trial, single_val)

        logger.info("Trial %s completed", trial_id)

    def tell_pruned(
        self, trial_id: int, *, step: int, metadata: dict[str, Any] | None = None
    ) -> None:
        trial = self._pop_trial(trial_id)
        if trial is None:
            return

        logger.info("Trial %s pruned at step %s", trial_id, step)
        self._study.tell(
            trial,
            values=None,
            state=optuna.trial.TrialState.PRUNED,
        )

    def tell_failure(self, trial_id: int, error_message: str) -> None:
        trial = self._pop_trial(trial_id)
        if trial is None:
            return

        logger.error("Trial %s failed: %s", trial_id, error_message)
        self._study.tell(
            trial,
            values=None,
            state=optuna.trial.TrialState.FAIL,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _pop_trial(self, trial_id: int) -> optuna.trial.Trial | None:
        with self._lock:
            trial = self._pending_trials.pop(trial_id, None)
            self._pending_configs.pop(trial_id, None)
        if trial is None:
            logger.warning("No pending trial found for id=%s", trial_id)
        return trial

    def _trial_to_config(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        config: dict[str, Any] = {}
        from traigent.optimizers.optuna_utils import suggest_from_definition

        for name, definition in self._config_space.items():
            suggestion = suggest_from_definition(trial, name, definition, config)
            if suggestion is not None:
                config[name] = suggestion
        return config


class RateLimiter:
    """Simple token-bucket rate limiter used by RateLimitedOptimizer."""

    def __init__(self, max_calls_per_minute: int) -> None:
        if max_calls_per_minute <= 0:
            raise OptimizationError("max_calls_per_minute must be positive")

        self._interval = 60.0 / max_calls_per_minute
        self._last_call = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._last_call >= self._interval:
            self._last_call = now
            return True
        return False


class RateLimitedOptimizer:
    """Wrapper that rate limits Optuna coordinator suggestions."""

    def __init__(
        self,
        coordinator: OptunaCoordinator,
        *,
        max_trials_per_minute: int = 10,
        pending_queue_size: int = 100,
    ) -> None:
        self.coordinator = coordinator
        self.rate_limiter = RateLimiter(max_trials_per_minute)
        self.pending_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=pending_queue_size
        )

    def suggest_next(self) -> dict[str, Any] | None:
        if not self.rate_limiter.allow():
            if not self.pending_queue.empty():
                return self.pending_queue.get()
            return None

        configs, _ = self.coordinator.ask_batch(1)
        return configs[0] if configs else None

    def queue(self, config: dict[str, Any]) -> None:
        try:
            self.pending_queue.put_nowait(config)
        except queue.Full:  # pragma: no cover - defensive
            logger.warning("Pending queue full; dropping configuration")


class BatchOptimizer:
    """Manage batches of suggestions for distributed execution."""

    def __init__(
        self,
        *,
        config_space: dict[str, Any],
        objectives: Iterable[str],
        n_workers: int,
        batch_size: int | None = None,
        worker_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        coordinator_kwargs: dict[str, Any] | None = None,
        trial_timeout: float = 300.0,
    ) -> None:
        if n_workers <= 0:
            raise OptimizationError("n_workers must be positive")

        coordinator_kwargs = coordinator_kwargs or {}
        self.coordinator = OptunaCoordinator(
            config_space=config_space,
            objectives=list(objectives),
            **coordinator_kwargs,
        )
        self.n_workers = n_workers
        self.batch_size = batch_size or n_workers
        self.worker_fn = worker_fn
        self.trial_timeout = trial_timeout

    def dispatch_to_worker(self, config: dict[str, Any]):
        if self.worker_fn is None:
            raise OptimizationError("worker_fn must be provided for BatchOptimizer")
        return self.worker_fn(config)

    def optimize_batch(self, n_trials: int) -> None:
        completed = 0
        while completed < n_trials:
            batch = min(self.batch_size, n_trials - completed)
            configs, _trials = self.coordinator.ask_batch(batch)

            for config in configs:
                trial_id = config["_trial_id"]
                try:
                    raw_result = self.dispatch_to_worker(config)
                    resolved = self._resolve_worker_result(raw_result)
                except TimeoutError:
                    message = f"Worker timed out after {self.trial_timeout} seconds"
                    logger.error("Trial %s timed out", trial_id)
                    self.coordinator.tell_failure(trial_id, message)
                except Exception as exc:  # pragma: no cover - exercised via tests
                    logger.exception("Worker execution failed for trial %s", trial_id)
                    message = str(exc) or exc.__class__.__name__
                    self.coordinator.tell_failure(trial_id, message)
                else:
                    self._handle_worker_result(trial_id, resolved)

                completed += 1

    def _resolve_worker_result(self, result: Any) -> Any:
        if isinstance(result, Mapping):
            return dict(result)

        get_method = getattr(result, "get", None)
        if callable(get_method):
            return get_method(timeout=self.trial_timeout)

        return result

    def _handle_worker_result(self, trial_id: int, result: Any) -> None:
        if result is None:
            self.coordinator.tell_failure(trial_id, "Worker returned no result")
            return

        if isinstance(result, Mapping):
            status = result.get("status")
            if status == "failed":
                error_message = result.get("error", "Worker reported failure")
                self.coordinator.tell_failure(trial_id, str(error_message))
                return
            if status == "pruned":
                step = int(result.get("step", 0))
                self.coordinator.tell_pruned(trial_id, step=step)
                return

            values = result.get("values")
            if values is None and "value" in result:
                values = result["value"]
        else:
            values = result

        self.coordinator.tell_result(trial_id, values)


class EdgeExecutor:
    """Simple executor that buffers step outputs while offline."""

    def __init__(
        self,
        coordinator_url: str,
        device_id: str,
        *,
        n_steps: int = 1,
        queue_limit: int = 256,
        run_step: Callable[[dict[str, Any], int], Any] | None = None,
        compute_final: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.coordinator_url = coordinator_url
        self.device_id = device_id
        self.n_steps = n_steps
        self.queue_limit = max(1, queue_limit)
        self.run_step = run_step
        self.compute_final = compute_final
        self.local_queue: list[dict[str, Any]] = []

    def is_online(self) -> bool:
        return True

    def report_to_coordinator(self, payload: Mapping[str, Any]) -> None:
        logger.debug("Reporting payload for trial %s", payload.get("trial_id"))

    def report_final(self, trial_id: Any, result: Any) -> None:
        payload = {"trial_id": trial_id, "value": result, "final": True}
        if self.is_online():
            self.report_to_coordinator(payload)
        else:
            self._queue_payload(payload)

    def execute_trial(self, config: Mapping[str, Any]) -> Any:
        trial_id = config.get("_trial_id")
        if trial_id is None:
            raise OptimizationError(
                "Configuration missing '_trial_id' for edge execution"
            )

        for step in range(self.n_steps):
            if self.run_step is None:
                break

            value = self.run_step(dict(config), step)
            payload = {
                "trial_id": trial_id,
                "step": step,
                "value": value,
                "timestamp": time.time(),
            }
            if self.is_online():
                self.report_to_coordinator(payload)
            else:
                self._queue_payload(payload)

        final_result = None
        if self.compute_final is not None:
            final_result = self.compute_final(dict(config))
            self.report_final(trial_id, final_result)

        return final_result

    def sync_offline_results(self) -> None:
        if not self.is_online():
            return

        while self.local_queue:
            payload = self.local_queue.pop(0)
            self.report_to_coordinator(payload)

    def _queue_payload(self, payload: dict[str, Any]) -> None:
        if len(self.local_queue) >= self.queue_limit:
            logger.warning(
                "EdgeExecutor queue full on %s; dropping oldest payload", self.device_id
            )
            self.local_queue.pop(0)
        self.local_queue.append(payload)


class ResilientCoordinator:
    """Utility that retries result reporting and queues payloads when offline."""

    def __init__(
        self,
        *,
        retry_policy: Mapping[str, Any] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.retry_policy = dict(self._default_retry_policy())
        if retry_policy:
            self.retry_policy.update(retry_policy)
        self.offline_queue: deque[dict[str, Any]] = deque()
        self._sleep = sleep_fn or time.sleep

    @staticmethod
    def _default_retry_policy() -> dict[str, Any]:
        return {
            "max_retries": 3,
            "backoff_base": 0.1,
            "max_backoff": 2.0,
        }

    def is_online(self) -> bool:
        return True

    def send_result(self, trial_id: int | str, value: Any) -> bool:
        raise NotImplementedError("send_result must be implemented by subclasses")

    def flush_offline_queue(self) -> None:
        if not self.is_online():
            return

        remaining: deque[dict[str, Any]] = deque()
        while self.offline_queue:
            payload = self.offline_queue.popleft()
            try:
                self.send_result(payload["trial_id"], payload["value"])
            except ConnectionError:
                remaining.appendleft(payload)
                break
        while remaining:
            self.offline_queue.appendleft(remaining.pop())

    def report_with_retry(
        self,
        *,
        trial_id: int | str,
        value: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        attempts = self.retry_policy.get("max_retries", 3)
        backoff_base = float(self.retry_policy.get("backoff_base", 0.1))
        max_backoff = float(self.retry_policy.get("max_backoff", 2.0))

        for attempt in range(max(1, attempts)):
            try:
                if not self.is_online():
                    raise ConnectionError("offline")
                self.send_result(trial_id, value)
                self.flush_offline_queue()
                return True
            except ConnectionError:
                wait_time = min(backoff_base * (2**attempt), max_backoff)
                self._sleep(wait_time)

        payload = {
            "trial_id": trial_id,
            "value": value,
            "timestamp": time.time(),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        self.offline_queue.append(payload)
        logger.debug("Queued trial %s result for later delivery", trial_id)
        return False
