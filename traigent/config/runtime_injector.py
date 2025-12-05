"""Runtime helpers for seamless configuration injection."""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any, Callable

__all__ = ["create_runtime_shim"]


def _prepare_arguments(
    func: Callable[..., Any],
    signature: inspect.Signature,
    config_snapshot: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Prepare arguments with configuration overrides applied.

    Args:
        func: Original function being wrapped (unused, kept for clarity)
        signature: Cached function signature
        config_snapshot: Frozen configuration for this trial
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Tuple containing positional args and keyword args to use when invoking
        the original function.
    """

    bound = signature.bind_partial(*args, **kwargs)
    provided_arguments = set(bound.arguments.keys())

    bound.apply_defaults()

    for name, parameter in signature.parameters.items():
        if name not in config_snapshot:
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in kwargs:
            continue

        if name in provided_arguments:
            continue

        bound.arguments[name] = config_snapshot[name]

    return tuple(bound.args), dict(bound.kwargs)


def create_runtime_shim(
    func: Callable[..., Any],
    config: dict[str, Any],
    signature: inspect.Signature | None = None,
) -> Callable[..., Any]:
    """Create a runtime shim that injects configuration into function parameters.

    Args:
        func: Function to wrap
        config: Configuration values for this trial (will be copied)
        signature: Optional cached signature to avoid recomputation

    Returns:
        Wrapped function applying configuration overrides at call time.
    """

    signature = signature or inspect.signature(func)
    config_snapshot = dict(config)

    def _apply(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return _prepare_arguments(func, signature, config_snapshot, *args, **kwargs)

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_shimmed(*args: Any, **kwargs: Any) -> Any:
            new_args, new_kwargs = _apply(*args, **kwargs)
            return await func(*new_args, **new_kwargs)

        return async_shimmed

    @wraps(func)
    def shimmed(*args: Any, **kwargs: Any) -> Any:
        new_args, new_kwargs = _apply(*args, **kwargs)
        return func(*new_args, **new_kwargs)

    return shimmed
