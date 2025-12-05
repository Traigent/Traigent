"""Context-based configuration management."""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any, Literal

logger = logging.getLogger(__name__)

from traigent.config.types import TraigentConfig

# Global context variable for configuration
# Default to empty TraigentConfig for thread safety (avoids LookupError in new threads)
config_context: ContextVar[TraigentConfig | dict[str, Any]] = ContextVar(
    "traigent_config", default=TraigentConfig()
)

# Global context variable for active configuration space during optimization
config_space_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "traigent_config_space", default=None
)
trial_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "traigent_trial_context", default=None
)


def get_config() -> TraigentConfig | dict[str, Any]:
    """Get current configuration from context.

    Returns:
        Current configuration (TraigentConfig or dict)

    Example:
        >>> set_config({"model": "GPT-4o", "temperature": 0.7})
        >>> config = get_config()
        >>> print(config)
        {'model': 'gpt-4', 'temperature': 0.7}
    """
    try:
        return config_context.get()
    except LookupError:
        # If context is not set, return default TraigentConfig
        return TraigentConfig()


def get_config_space() -> dict[str, Any] | None:
    """Get current configuration space from context.

    Returns:
        Current configuration space being optimized, or None if not in optimization
    """
    try:
        return config_space_context.get()
    except LookupError:
        return None


def set_config_space(
    config_space: dict[str, Any] | None,
) -> Token[dict[str, Any] | None]:
    """Set configuration space in context.

    Args:
        config_space: Configuration space being optimized

    Returns:
        Token that can be used to reset configuration space
    """
    return config_space_context.set(config_space)


def set_config(
    config: TraigentConfig | dict[str, Any],
) -> Token[TraigentConfig | dict[str, Any]]:
    """Set configuration in context.

    Args:
        config: Configuration to set (TraigentConfig or dict)

    Returns:
        Token that can be used to reset configuration

    Example:
        >>> token = set_config({"model": "GPT-4o"})
        >>> # Use configuration
        >>> config_context.reset(token)  # Reset to previous
    """
    return config_context.set(config)


class ConfigurationSpaceContext:
    """Context manager for configuration space during optimization."""

    def __init__(self, config_space: dict[str, Any]) -> None:
        """Initialize configuration space context.

        Args:
            config_space: Configuration space being optimized
        """
        self.config_space = config_space
        self._token: Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> dict[str, Any]:
        """Enter configuration space context."""
        self._token = set_config_space(self.config_space)
        return self.config_space

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit configuration space context and restore previous."""
        if self._token is not None:
            config_space_context.reset(self._token)
        return False  # Don't suppress exceptions


class ConfigurationContext:
    """Context manager for temporary configuration changes.

    Example:
        >>> with ConfigurationContext({"model": "GPT-4o"}):
        ...     config = get_config()  # Returns {"model": "GPT-4o"}
        >>> config = get_config()  # Returns previous configuration
    """

    def __init__(self, config: TraigentConfig | dict[str, Any]) -> None:
        """Initialize context manager.

        Args:
            config: Configuration to use in context
        """
        self.config = config
        self._token: Token[TraigentConfig | dict[str, Any]] | None = None

    def __enter__(self) -> TraigentConfig | dict[str, Any]:
        """Enter configuration context."""
        self._token = set_config(self.config)
        return self.config

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit configuration context and restore previous."""
        if self._token is not None:
            config_context.reset(self._token)
        return False  # Don't suppress exceptions


def merge_with_context(
    override_config: dict[str, Any] | None = None,
) -> TraigentConfig | dict[str, Any]:
    """Merge override configuration with context configuration.

    Context configuration takes precedence over override_config for any
    keys present in both. Override_config provides default values for
    keys not in context. This is critical for the optimization loop where
    trial configurations (set in context) must override default configurations.

    Args:
        override_config: Default/fallback configuration values. Context values
            will take precedence over these.

    Returns:
        Merged configuration where context values take precedence

    Example:
        >>> set_config({"model": "gpt-4o-mini", "temperature": 0.5})
        >>> merged = merge_with_context({"temperature": 0.8, "max_tokens": 1000})
        >>> print(merged)
        {'model': 'gpt-4o-mini', 'temperature': 0.5, 'max_tokens': 1000}
    """
    context_config = get_config()

    if override_config is None:
        return context_config

    # Handle different configuration types
    # Context config takes precedence over override config (important for optimization loop!)
    # This ensures trial configurations from the optimizer override default/static configurations
    if isinstance(context_config, TraigentConfig):
        # Start with override, then apply context on top (context wins)
        context_dict = context_config.to_dict()
        context_dict.update(context_config.custom_params)
        merged_dict = override_config.copy()
        merged_dict.update({k: v for k, v in context_dict.items() if v is not None})
        return TraigentConfig.from_dict(merged_dict)
    elif isinstance(context_config, dict):
        # Start with override, then apply context on top (context wins)
        merged = override_config.copy()
        merged.update(context_config)
        return merged
    else:
        # This should never happen based on get_config() return type
        # but we need to handle it for type checker
        return override_config


def get_trial_context() -> dict[str, Any] | None:
    """Return the active trial context if one is set."""

    try:
        return trial_context.get()
    except LookupError:
        return None


def set_trial_context(payload: dict[str, Any] | None) -> Token[dict[str, Any] | None]:
    """Set the active trial context payload."""

    return trial_context.set(payload)


class ContextSnapshot:
    """Snapshot of all TraiGent context variables for thread propagation.

    Python's contextvars don't automatically propagate to ThreadPoolExecutor
    workers. This class captures all context for manual propagation.

    Example:
        >>> import concurrent.futures
        >>>
        >>> def worker_func(snapshot, data):
        ...     # Restore all context in worker thread
        ...     with snapshot.restore():
        ...         config = traigent.get_trial_config()
        ...         return process(data, config)
        >>>
        >>> # Capture context before submitting to executor
        >>> snapshot = copy_context_to_thread()
        >>> with concurrent.futures.ThreadPoolExecutor() as executor:
        ...     futures = [executor.submit(worker_func, snapshot, d) for d in data]
    """

    def __init__(
        self,
        trial_ctx: dict[str, Any] | None,
        config: TraigentConfig | dict[str, Any] | None,
        config_space: dict[str, Any] | None,
    ) -> None:
        """Initialize context snapshot.

        Args:
            trial_ctx: Trial context dict or None
            config: Configuration (TraigentConfig or dict) or None
            config_space: Configuration space dict or None
        """
        self.trial_context = trial_ctx
        self.config = config
        self.config_space = config_space

    def restore(self) -> ContextRestorer:
        """Return a context manager that restores this snapshot.

        Returns:
            Context manager that sets all context variables
        """
        return ContextRestorer(self)

    # For backward compatibility: allow dict-like access for trial context
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from trial context (for backward compatibility)."""
        if self.trial_context:
            return self.trial_context.get(key, default)
        return default

    def __bool__(self) -> bool:
        """Return True if any context is set."""
        return bool(self.trial_context or self.config or self.config_space)


class ContextRestorer:
    """Context manager for restoring a ContextSnapshot in a worker thread."""

    def __init__(self, snapshot: ContextSnapshot) -> None:
        self.snapshot = snapshot
        self._tokens: list[tuple[ContextVar[Any], Token[Any]]] = []

    def __enter__(self) -> ContextSnapshot:
        """Enter context and restore all context variables."""
        if self.snapshot.trial_context:
            token = trial_context.set(self.snapshot.trial_context)
            self._tokens.append((trial_context, token))

        if self.snapshot.config is not None:
            token = config_context.set(self.snapshot.config)
            self._tokens.append((config_context, token))

        if self.snapshot.config_space is not None:
            token = config_space_context.set(self.snapshot.config_space)
            self._tokens.append((config_space_context, token))

        return self.snapshot

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Exit context and reset all context variables."""
        # Reset in reverse order
        for ctx_var, token in reversed(self._tokens):
            try:
                ctx_var.reset(token)
            except ValueError:
                # Token is stale - already reset or from different context.
                # This indicates a bug elsewhere (double __exit__ or mismatched context),
                # but we should not crash here. Log warning for debugging.
                logger.warning(
                    "Stale context token detected during ContextRestorer cleanup. "
                    "This may indicate a double-exit or context mismatch."
                )
        self._tokens.clear()
        return False


def copy_context_to_thread() -> ContextSnapshot:
    """Capture all TraiGent context variables for propagation to worker threads.

    Python's contextvars don't automatically propagate to ThreadPoolExecutor
    workers. Use this function to capture all context, then restore it in the
    worker thread using the returned snapshot's restore() method.

    Returns:
        ContextSnapshot containing all current context variables.

    Example:
        >>> import concurrent.futures
        >>>
        >>> def worker_func(snapshot, data):
        ...     # Restore all context in worker thread
        ...     with snapshot.restore():
        ...         config = traigent.get_trial_config()
        ...         return process(data, config)
        >>>
        >>> # Capture context before submitting to executor
        >>> snapshot = copy_context_to_thread()
        >>> with concurrent.futures.ThreadPoolExecutor() as executor:
        ...     futures = [executor.submit(worker_func, snapshot, d) for d in data]

    Note:
        TraiGent's built-in parallel evaluator already handles context
        propagation. This helper is only needed when you create your own
        ThreadPoolExecutor within an optimized function.

        For asyncio tasks spawned with asyncio.create_task(), context is
        automatically inherited - no manual propagation is needed.
    """
    return ContextSnapshot(
        trial_ctx=get_trial_context(),
        config=get_config(),
        config_space=get_config_space(),
    )


class TrialContext:
    """Context manager to expose active trial metadata.

    This context manager sets the trial context so that get_trial_config()
    can return the current configuration during optimization trials.

    Important: Python's contextvars don't automatically propagate to
    ThreadPoolExecutor workers. If you spawn threads inside an optimized
    function, use copy_context_to_thread() to capture the context before
    spawning, then restore it in each worker using TrialContext.
    """

    def __init__(
        self, trial_id: str | int | None, metadata: dict[str, Any] | None = None
    ) -> None:
        self.trial_id = trial_id
        self.metadata = metadata or {}
        self._token: Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> dict[str, Any]:
        payload = {"trial_id": self.trial_id, **self.metadata}
        self._token = set_trial_context(payload)
        return payload

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if self._token is not None:
            try:
                trial_context.reset(self._token)
            except ValueError:
                # Token is stale - already reset or from different context.
                # This indicates a bug elsewhere (double __exit__ or mismatched context),
                # but we should not crash here. Log warning for debugging.
                logger.warning(
                    "Stale trial context token detected during TrialContext cleanup. "
                    "This may indicate a double-exit or context mismatch."
                )
            finally:
                # Clear token to prevent double-reset attempts
                self._token = None
        return False

    async def __aenter__(self) -> dict[str, Any]:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        return self.__exit__(exc_type, exc_val, exc_tb)
