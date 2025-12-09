"""Adapter bridging Optuna trial configurations into seamless injection."""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

from traigent.config.context import TrialContext
from traigent.config.runtime_injector import create_runtime_shim
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove internal bookkeeping keys from configuration payloads."""

    return {key: value for key, value in config.items() if not key.startswith("_")}


class SeamlessOptunaAdapter:
    """Inject Optuna trial configurations via the seamless runtime shim."""

    def __init__(
        self,
        *,
        telemetry_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._telemetry_hook = telemetry_hook

    def _emit_telemetry(self, payload: dict[str, Any]) -> None:
        if not self._telemetry_hook:
            return
        try:
            self._telemetry_hook(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Telemetry hook raised an exception: %s",
                exc,
                exc_info=False,
            )

    def inject(
        self,
        target_fn: Callable[..., Any],
        trial_config: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Return a wrapped callable with configuration injected."""

        if "_optuna_trial_id" not in trial_config:
            raise ValueError("trial_config missing '_optuna_trial_id'")

        trial_id = trial_config["_optuna_trial_id"]
        sanitized_config = _sanitize_config(trial_config)

        metadata = metadata.copy() if metadata else {}
        metadata.setdefault("config_snapshot", dict(sanitized_config))

        shim = create_runtime_shim(target_fn, sanitized_config)

        def _clone_metadata() -> dict[str, Any]:
            payload = metadata.copy()
            snapshot = payload.get("config_snapshot")
            if isinstance(snapshot, dict):
                payload["config_snapshot"] = snapshot.copy()
            return payload

        if inspect.iscoroutinefunction(shim):

            @wraps(target_fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                payload = _clone_metadata()
                self._emit_telemetry(
                    {"event": "trial_call_async_started", "trial_id": trial_id}
                )
                async with TrialContext(trial_id=trial_id, metadata=payload):
                    result = await shim(*args, **kwargs)
                self._emit_telemetry(
                    {"event": "trial_call_async_completed", "trial_id": trial_id}
                )
                return result

            return async_wrapper

        @wraps(target_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = _clone_metadata()
            self._emit_telemetry({"event": "trial_call_started", "trial_id": trial_id})
            with TrialContext(trial_id=trial_id, metadata=payload):
                result = shim(*args, **kwargs)
            self._emit_telemetry(
                {"event": "trial_call_completed", "trial_id": trial_id}
            )
            return result

        return wrapper


__all__ = ["SeamlessOptunaAdapter"]
