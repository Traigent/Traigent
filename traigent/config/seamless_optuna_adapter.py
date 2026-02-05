"""Adapter bridging Optuna trial configurations into seamless injection."""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, Union, get_origin, get_type_hints

from traigent.config.context import TrialContext
from traigent.config.runtime_injector import create_runtime_shim
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class TypeValidationError(TypeError):
    """Exception raised when config values don't match expected types.

    Attributes:
        errors: List of validation error messages
        config: The config that failed validation
    """

    def __init__(self, errors: list[str], config: dict[str, Any]) -> None:
        self.errors = errors
        self.config = config
        message = "Config type validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


class TypedConfigInjector:
    """Validates types before injecting config into function.

    This class extracts type hints from a function signature and validates
    that config values match the expected types before injection. This catches
    type mismatches early, providing clear error messages rather than cryptic
    errors deep in user code.

    Example:
        >>> def my_func(temperature: float, model: str) -> str:
        ...     return f"{model}: {temperature}"
        >>>
        >>> injector = TypedConfigInjector(my_func)
        >>> errors = injector.validate_config({"temperature": "hot", "model": "gpt-4"})
        >>> print(errors)
        ["Parameter 'temperature': expected float, got str"]
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        strict: bool = False,
    ) -> None:
        """Initialize the type validator.

        Args:
            func: The function whose signature will be used for type validation
            strict: If True, raise error for params without type hints in config
        """
        self._func = func
        self._strict = strict
        self._type_hints = self._get_type_hints_safe(func)

    def _get_type_hints_safe(self, func: Callable[..., Any]) -> dict[str, type]:
        """Get type hints, handling common edge cases."""
        try:
            # get_type_hints can fail on some decorated functions
            return get_type_hints(func)
        except Exception:  # noqa: BLE001
            # Fall back to __annotations__ if get_type_hints fails
            return getattr(func, "__annotations__", {})

    def _is_union_type(self, tp: Any) -> bool:
        """Check if type is a Union (including Optional which is Union[T, None])."""
        origin = get_origin(tp)
        # Python 3.10+ has types.UnionType for X | Y syntax
        if origin is types.UnionType:
            return True
        return origin is Union

    def _get_union_args(self, tp: Any) -> tuple[Any, ...]:
        """Get the type arguments from a Union type."""
        # For Union[X, Y], __args__ is (X, Y)
        return getattr(tp, "__args__", ())

    def _check_type_match(self, value: Any, expected_type: Any) -> bool:
        """Check if value matches expected type, handling complex types."""
        # None check for Optional types
        if value is None:
            return (
                type(None) in self._get_union_args(expected_type)
                if self._is_union_type(expected_type)
                else False
            )

        # Handle Union types (including Optional)
        if self._is_union_type(expected_type):
            args = self._get_union_args(expected_type)
            return any(self._check_type_match(value, arg) for arg in args)

        # Handle generic types like list[int], dict[str, Any]
        origin = get_origin(expected_type)
        if origin is not None:
            # For generic types, just check the origin (list, dict, etc.)
            return isinstance(value, origin)

        # Basic type check
        # Special case: accept int for float (common in numeric configs)
        if expected_type is float and isinstance(value, int):
            return True

        return isinstance(value, expected_type)

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate config types against function signature.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        for name, value in config.items():
            expected_type = self._type_hints.get(name)

            if expected_type is None:
                if self._strict:
                    errors.append(
                        f"Parameter '{name}': no type hint found (strict mode enabled)"
                    )
                continue

            if not self._check_type_match(value, expected_type):
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                actual_name = type(value).__name__
                errors.append(
                    f"Parameter '{name}': expected {expected_name}, got {actual_name}"
                )

        return errors

    def inject(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate and return config for injection.

        Args:
            config: Configuration to validate

        Returns:
            The validated config (unchanged if valid)

        Raises:
            TypeValidationError: If config contains type mismatches
        """
        errors = self.validate_config(config)
        if errors:
            raise TypeValidationError(errors, config)
        return config


def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove internal bookkeeping keys from configuration payloads."""

    return {key: value for key, value in config.items() if not key.startswith("_")}


class SeamlessOptunaAdapter:
    """Inject Optuna trial configurations via the seamless runtime shim.

    Args:
        telemetry_hook: Optional callback for telemetry events
        validate_types: Whether to validate config types against function signature
            (default: True). Set to False to skip type validation.
        strict_types: If True, require type hints for all config parameters
            (default: False). Only used when validate_types is True.
    """

    def __init__(
        self,
        *,
        telemetry_hook: Callable[[dict[str, Any]], None] | None = None,
        validate_types: bool = True,
        strict_types: bool = False,
    ) -> None:
        self._telemetry_hook = telemetry_hook
        self._validate_types = validate_types
        self._strict_types = strict_types
        # Cache for TypedConfigInjector instances per function
        self._type_validators: dict[int, TypedConfigInjector] = {}

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

    def _get_type_validator(self, func: Callable[..., Any]) -> TypedConfigInjector:
        """Get or create a TypedConfigInjector for the given function."""
        func_id = id(func)
        if func_id not in self._type_validators:
            self._type_validators[func_id] = TypedConfigInjector(
                func, strict=self._strict_types
            )
        return self._type_validators[func_id]

    def inject(
        self,
        target_fn: Callable[..., Any],
        trial_config: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Return a wrapped callable with configuration injected.

        Args:
            target_fn: The function to wrap with config injection
            trial_config: Configuration including '_optuna_trial_id'
            metadata: Optional metadata to include in trial context

        Returns:
            Wrapped function with configuration injected

        Raises:
            ValueError: If trial_config is missing '_optuna_trial_id'
            TypeValidationError: If validate_types is True and config has type mismatches
        """
        if "_optuna_trial_id" not in trial_config:
            raise ValueError("trial_config missing '_optuna_trial_id'")

        trial_id = trial_config["_optuna_trial_id"]
        sanitized_config = _sanitize_config(trial_config)

        # Validate types before injection if enabled
        if self._validate_types:
            validator = self._get_type_validator(target_fn)
            validator.inject(sanitized_config)  # Raises TypeValidationError on failure

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


__all__ = ["SeamlessOptunaAdapter", "TypedConfigInjector", "TypeValidationError"]
