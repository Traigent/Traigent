"""Plugin registry for optional Traigent constraint validators."""

from __future__ import annotations

import importlib.metadata
import logging
import threading
from collections.abc import Callable
from typing import Any

from traigent_validation.validators import (
    PythonConstraintValidator,
    SATConstraintValidator,
)

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "traigent.validators"

_registry_lock = threading.RLock()
_validator_factories: dict[str, Callable[[], Any]] = {}
_entry_points_loaded = False


def _ensure_builtin_validators() -> None:
    _validator_factories.setdefault("python", PythonConstraintValidator)
    _validator_factories.setdefault("sat", SATConstraintValidator)


def _coerce_factory(candidate: Any) -> Callable[[], Any] | None:
    """Normalize entry-point objects to zero-arg factories."""
    if isinstance(candidate, type):
        return candidate

    if callable(candidate):
        return candidate

    logger.warning(
        "Ignoring validator plugin %r because it is not callable", candidate
    )
    return None


def load_entry_point_validators() -> None:
    """Load validators discovered through ``traigent.validators`` entry points."""
    global _entry_points_loaded

    with _registry_lock:
        if _entry_points_loaded:
            return

        _ensure_builtin_validators()

        try:
            eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.get(ENTRY_POINT_GROUP, [])
        except Exception as exc:
            logger.warning("Failed to inspect validator entry points: %s", exc)
            _entry_points_loaded = True
            return

        for ep in eps:
            try:
                loaded = ep.load()
            except Exception as exc:
                logger.warning(
                    "Failed to load validator entry point '%s': %s", ep.name, exc
                )
                continue

            factory = _coerce_factory(loaded)
            if factory is None:
                continue

            if ep.name in _validator_factories:
                logger.debug(
                    "Validator entry point '%s' ignored; name already registered",
                    ep.name,
                )
                continue

            _validator_factories[ep.name] = factory
            logger.info("Registered validator plugin: %s", ep.name)

        _entry_points_loaded = True


def register_validator(
    name: str,
    factory: Callable[[], Any],
    *,
    overwrite: bool = False,
) -> None:
    """Register a validator factory under a symbolic name."""
    if not name:
        raise ValueError("Validator name cannot be empty")

    with _registry_lock:
        _ensure_builtin_validators()
        if name in _validator_factories and not overwrite:
            raise ValueError(
                f"Validator '{name}' is already registered; pass overwrite=True to replace it"
            )
        _validator_factories[name] = factory


def get_validator(name: str = "python") -> Any | None:
    """Instantiate a validator by name, loading plugins lazily."""
    load_entry_point_validators()

    factory = _validator_factories.get(name)
    if factory is None:
        return None

    try:
        return factory()
    except Exception as exc:
        logger.warning("Failed to instantiate validator '%s': %s", name, exc)
        return None


def list_validators() -> list[str]:
    """List registered validator names."""
    load_entry_point_validators()
    return sorted(_validator_factories)


def _reset_registry_for_testing() -> None:
    """Reset plugin registry state for isolated unit tests."""
    global _entry_points_loaded
    with _registry_lock:
        _validator_factories.clear()
        _entry_points_loaded = False


__all__ = [
    "ENTRY_POINT_GROUP",
    "get_validator",
    "list_validators",
    "load_entry_point_validators",
    "register_validator",
]
