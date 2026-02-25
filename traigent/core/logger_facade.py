"""Helper facade for orchestration logging."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from traigent.utils.logging import get_logger
from traigent.utils.optimization_logger import OptimizationLogger

logger = get_logger(__name__)
LOGGER_UNAVAILABLE_REASON = "logger is unavailable"


class LoggerFacade:
    """Wraps :class:`OptimizationLogger` with safety guards and conveniences."""

    def __init__(
        self,
        experiment_name: str | None = None,
        *,
        session_id: str | None = None,
        execution_mode: str | None = None,
        logger_instance: Any | None = None,
        **logger_kwargs: Any,
    ) -> None:
        self._logger: Any | None = None
        self._logging_expected: bool = bool(logger_instance is not None)
        self._unavailable_warning_emitted: bool = False
        self._failure_warning_emitted: bool = False

        if logger_instance is not None:
            self._logger = logger_instance
            return

        # Allow session_id=None for local-only modes (edge_analytics)
        if experiment_name is None or execution_mode is None:
            return
        self._logging_expected = True

        try:
            # `OptimizationLogger` is accessed from module scope so tests can patch it.
            # Session ID defaults to a placeholder for edge_analytics mode
            effective_session_id = session_id or "local-session"
            self._logger = OptimizationLogger(
                experiment_name=experiment_name,
                session_id=effective_session_id,
                execution_mode=execution_mode,
                **logger_kwargs,
            )
        except Exception:
            logger.exception("Failed to initialize optimization logger")
            self._logger = None
            self._warn_unavailable_once("initialization failed")

    def attach(self, logger_instance: Any | None) -> None:
        """Attach a pre-built optimization logger instance."""
        self._logger = logger_instance
        self._logging_expected = self._logging_expected or logger_instance is not None

    def log_session_start(self, **kwargs: Any) -> None:
        """Log session start while enriching dataset metadata when available."""
        if not self._logger:
            self._warn_unavailable_once(LOGGER_UNAVAILABLE_REASON)
            return

        payload = dict(kwargs)
        dataset = payload.pop("dataset", None)
        dataset_info = payload.get("dataset_info")

        if dataset_info is None and dataset is not None:
            dataset_info = self._build_dataset_info(dataset)
            payload["dataset_info"] = dataset_info
        elif dataset_info is not None and not isinstance(dataset_info, Mapping):
            payload["dataset_info"] = dict(dataset_info)

        try:
            self._logger.log_session_start(**payload)
        except Exception:
            logger.exception("Failed to log session start")
            self._warn_failure_once("session start")

    def log_trial(self, trial_result: Any) -> None:
        """Log a trial result using the underlying optimization logger."""
        if not self._logger:
            self._warn_unavailable_once(LOGGER_UNAVAILABLE_REASON)
            return
        try:
            self._logger.log_trial_result(trial_result)
        except Exception:
            logger.exception(
                "Failed to log trial result for %s",
                getattr(trial_result, "trial_id", "unknown"),
            )
            self._warn_failure_once("trial result")

    def log_trial_result(self, trial_result: Any) -> None:
        """Backward compatible alias for :meth:`log_trial`."""
        self.log_trial(trial_result)

    def log_checkpoint(self, **kwargs: Any) -> None:
        """Persist checkpoint information if the logger exposes the capability."""
        if not self._logger:
            self._warn_unavailable_once(LOGGER_UNAVAILABLE_REASON)
            return
        if not hasattr(self._logger, "save_checkpoint"):
            return

        try:
            self._logger.save_checkpoint(**kwargs)
        except Exception:
            logger.exception("Failed to save optimization checkpoint")
            self._warn_failure_once("checkpoint save")

    def save_checkpoint(self, **kwargs: Any) -> None:
        """Alias for :meth:`log_checkpoint` for backward compatibility."""
        self.log_checkpoint(**kwargs)

    def log_session_end(self, **kwargs: Any) -> None:
        if not self._logger:
            self._warn_unavailable_once(LOGGER_UNAVAILABLE_REASON)
            return
        try:
            self._logger.log_session_end(**kwargs)
        except Exception:
            logger.exception("Failed to log session end")
            self._warn_failure_once("session end")

    @property
    def logger(self) -> Any | None:
        return self._logger

    @staticmethod
    def _build_dataset_info(dataset: Any) -> dict[str, Any]:
        """Derive dataset metadata for logging without being overly strict."""
        try:
            size = len(dataset)
        except TypeError:
            # Dataset doesn't support len()
            size = None

        name = getattr(dataset, "name", "unknown")
        if not name:
            name = "unknown"

        return {"size": size, "name": name}

    def _warn_unavailable_once(self, reason: str) -> None:
        """Emit one visible warning when logging is expected but unavailable."""
        if not self._logging_expected or self._unavailable_warning_emitted:
            return
        logger.warning(
            "Optimization logging is unavailable (%s). "
            "Telemetry/checkpoints may be incomplete.",
            reason,
        )
        self._unavailable_warning_emitted = True

    def _warn_failure_once(self, operation: str) -> None:
        """Emit one visible warning when runtime logging calls fail."""
        if self._failure_warning_emitted:
            return
        logger.warning(
            "Optimization logging failed during %s. "
            "Telemetry/checkpoints may be incomplete.",
            operation,
        )
        self._failure_warning_emitted = True


__all__ = ["LoggerFacade", "OptimizationLogger"]
