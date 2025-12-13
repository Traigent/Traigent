"""Telemetry helpers for Optuna-backed optimizers."""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Reliability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from traigent.security.enterprise import MetricsCollector
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config without private/internal keys."""

    return {k: v for k, v in config.items() if not k.startswith("_")}


class OptunaMetricsEmitter:
    """Fan-out telemetry emitter for Optuna trial lifecycle events."""

    def __init__(
        self,
        *,
        metrics_collector: MetricsCollector | None = None,
        listeners: Iterable[Callable[[dict[str, Any]], None]] | None = None,
    ) -> None:
        self._metrics_collector = metrics_collector
        self._listeners = list(listeners or [])
        self._lock = threading.RLock()
        self._disabled = os.getenv("TRAIGENT_DISABLE_TELEMETRY", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit_trial_update(
        self,
        *,
        event: str,
        trial_id: int | str | None,
        study_name: str | None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit a structured telemetry payload to collectors and listeners."""

        if self._disabled:
            return {}

        message: dict[str, Any] = {
            "event": event,
            "trial_id": trial_id,
            "study_name": study_name,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if payload:
            message["payload"] = payload

        if self._metrics_collector:
            try:
                self._metrics_collector.record_optuna_event(event, message)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to record Optuna event via MetricsCollector")

        with self._lock:
            for listener in list(self._listeners):
                try:
                    listener(message)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Optuna telemetry listener raised an exception")

        return message
