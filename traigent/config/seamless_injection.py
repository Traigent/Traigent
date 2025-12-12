"""Seamless configuration injection helpers for Optuna-backed flows."""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

from traigent.config.runtime_injector import create_runtime_shim
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class _StateCache(dict[Any, Any]):
    """Dictionary wrapper that preserves existing entries when reassigning."""

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self and isinstance(self[key], dict) and isinstance(value, Mapping):
            existing = self[key]
            for sub_key, sub_value in value.items():
                existing.setdefault(sub_key, sub_value)
            return
        if key in self:
            return
        super().__setitem__(key, value)


class RuntimeShim:
    """Lightweight runtime shim that exposes an isolated configuration mapping."""

    _state_cache: _StateCache = _StateCache()
    _cache_lock = Lock()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cache = cls.__dict__.get("_state_cache")
        if cache is not None and not isinstance(cache, _StateCache):
            cls._state_cache = _StateCache(cache)

    def __init__(self, config: Mapping[str, Any]) -> None:
        # Copy so trial metadata owned by orchestrators is not mutated in-place.
        self.config: dict[str, Any] = dict(config)
        self._access_log: list[tuple[str, Any]] = []

        trial_id = self.config.get("_optuna_trial_id")
        if trial_id is not None:
            with self.__class__._cache_lock:
                cache = self.__class__._state_cache
                cache.setdefault(trial_id, {})

    def get_value(self, key: str, default: Any | None = None) -> Any:
        value = self.config.get(key, default)
        self._access_log.append((key, value))
        return value

    def apply_value(self, key: str, value: Any) -> Any:
        self.config[key] = value
        return value

    def update(self, overrides: Mapping[str, Any]) -> None:
        for key, value in overrides.items():
            self.apply_value(key, value)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.config)

    @property
    def access_log(self) -> Iterable[tuple[str, Any]]:
        return tuple(self._access_log)


@dataclass(slots=True)
class InjectionMetadata:
    """Metadata recorded during seamless injection."""

    trial_id: str | int
    config_snapshot: dict[str, Any]


class SeamlessInjectionConfigurator:
    """Factory that prepares runtime shims and wrappers for target callables."""

    def __init__(
        self,
        *,
        runtime_shim_factory: Callable[[Mapping[str, Any]], RuntimeShim] | None = None,
    ) -> None:
        self._shim_factory = runtime_shim_factory or RuntimeShim

    def inject(
        self,
        target_fn: Callable[..., Any],
        trial_config: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        if "_optuna_trial_id" not in trial_config:
            raise ValueError("trial_config must include '_optuna_trial_id'")

        trial_id = trial_config["_optuna_trial_id"]
        shim = self._shim_factory(trial_config)
        sanitized_config = {
            key: value for key, value in trial_config.items() if not key.startswith("_")
        }
        wrapper_metadata = InjectionMetadata(
            trial_id=trial_id,
            config_snapshot=dict(sanitized_config),
        )
        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault(
            "config_snapshot", wrapper_metadata.config_snapshot
        )

        runtime_shim = create_runtime_shim(target_fn, sanitized_config)

        if inspect.iscoroutinefunction(runtime_shim):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await runtime_shim(*args, **kwargs)

            wrapper: Callable[..., Any] = async_wrapper
        else:

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return runtime_shim(*args, **kwargs)

        wrapper.__traigent_trial_id__ = trial_id  # type: ignore[attr-defined]
        wrapper.__runtime_shim__ = shim  # type: ignore[attr-defined]
        wrapper.__seamless_metadata__ = combined_metadata  # type: ignore[attr-defined]

        return wrapper


__all__ = [
    "RuntimeShim",
    "SeamlessInjectionConfigurator",
    "InjectionMetadata",
]
