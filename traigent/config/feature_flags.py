"""Feature flag management for TraiGent.

Provides a lightweight registry for runtime-togglable flags with support for
environment variables, configuration mappings, and test overrides.
"""

# Traceability: CONC-ConfigInjection CONC-OptimizerEngine FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Iterable, Mapping

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

_TRUE_LITERALS = {"1", "true", "yes", "on", "enable", "enabled"}
_FALSE_LITERALS = {"0", "false", "no", "off", "disable", "disabled"}


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a dynamic value into a boolean if possible."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_LITERALS:
            return True
        if lowered in _FALSE_LITERALS:
            return False

    return None


@dataclass(slots=True)
class Flag:
    """Represents a feature flag definition."""

    name: str
    default: bool = False
    env_var: str | None = None
    description: str = ""
    config_path: str | None = None  # dotted path within configuration mapping


class FlagRegistry:
    """Runtime registry for feature flags with override support."""

    def __init__(self) -> None:
        self._flags: dict[str, Flag] = {}
        self._overrides: dict[str, bool] = {}
        self._config_values: dict[str, bool] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration & configuration
    # ------------------------------------------------------------------
    def register(self, flag: Flag) -> None:
        """Register a new feature flag."""

        with self._lock:
            if flag.name in self._flags:
                raise ValueError(f"Flag '{flag.name}' already registered")
            self._flags[flag.name] = flag
            logger.debug("Registered feature flag %s", flag.name)

    def apply_config(self, mapping: Mapping[str, Any] | None) -> None:
        """Apply configuration mapping to populate flag defaults."""

        if not mapping:
            return

        with self._lock:
            self._config_values.clear()
            for flag in self._flags.values():
                if not flag.config_path:
                    continue
                value = self._lookup_config_value(mapping, flag.config_path.split("."))
                coerced = _coerce_bool(value)
                if coerced is not None:
                    self._config_values[flag.name] = coerced
                    logger.debug(
                        "Config override for flag %s -> %s", flag.name, coerced
                    )

    def _lookup_config_value(
        self, mapping: Mapping[str, Any], path: Iterable[str]
    ) -> Any:
        current: Any = mapping
        for key in path:
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            else:
                return None
        return current

    # ------------------------------------------------------------------
    # Overrides & evaluation
    # ------------------------------------------------------------------
    def set_override(self, name: str, value: bool) -> None:
        with self._lock:
            self._ensure_flag(name)
            self._overrides[name] = bool(value)
            logger.debug("Set override for flag %s -> %s", name, value)

    def clear_override(self, name: str) -> None:
        with self._lock:
            self._overrides.pop(name, None)
            logger.debug("Cleared override for flag %s", name)

    @contextmanager
    def override(self, name: str, value: bool) -> Generator[None, None, None]:
        """Context manager for temporary flag overrides."""

        self.set_override(name, value)
        try:
            yield
        finally:
            self.clear_override(name)

    def is_enabled(self, name: str) -> bool:
        """Return the evaluated boolean state for a flag."""

        with self._lock:
            flag = self._ensure_flag(name)

            if name in self._overrides:
                return self._overrides[name]

            if flag.env_var:
                env_value = os.getenv(flag.env_var)
                coerced = _coerce_bool(env_value)
                if coerced is not None:
                    return coerced

            if name in self._config_values:
                return self._config_values[name]

            return flag.default

    def reset(self) -> None:
        """Reset overrides and config-derived values (retain registrations)."""

        with self._lock:
            self._overrides.clear()
            self._config_values.clear()

    def snapshot(self) -> dict[str, bool]:
        """Return a snapshot of all flags and their evaluated values."""

        with self._lock:
            return {name: self.is_enabled(name) for name in self._flags}

    def _ensure_flag(self, name: str) -> Flag:
        if name not in self._flags:
            raise KeyError(f"Unknown flag '{name}'")
        return self._flags[name]


class FlagNames:
    """Canonical flag identifiers used across the codebase."""

    OPTUNA_ROLLOUT = "optimizers.optuna.enabled"


# Global registry instance
flag_registry = FlagRegistry()

# Register built-in flags
flag_registry.register(
    Flag(
        name=FlagNames.OPTUNA_ROLLOUT,
        default=False,
        env_var="TRAIGENT_OPTUNA_ENABLED",
        description="Toggle Optuna-backed optimizers globally.",
        config_path="optimizers.optuna.enabled",
    )
)


def is_optuna_enabled() -> bool:
    """Convenience accessor for the Optuna rollout flag."""

    return flag_registry.is_enabled(FlagNames.OPTUNA_ROLLOUT)


__all__ = [
    "Flag",
    "FlagNames",
    "FlagRegistry",
    "flag_registry",
    "is_optuna_enabled",
]
