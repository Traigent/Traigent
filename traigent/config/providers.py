"""Safe configuration injection providers without exec().

This module provides secure alternatives to the original providers,
particularly for seamless injection which previously used exec().
"""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow CONC-Layer-Core

from __future__ import annotations

import ast
import inspect
import textwrap
import types
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from threading import Lock
from typing import Any

from traigent.config.ast_transformer import ConfigTransformer, SafeASTCompiler
from traigent.config.context import ConfigurationContext, get_config, merge_with_context
from traigent.config.runtime_injector import create_runtime_shim
from traigent.config.types import TraigentConfig
from traigent.utils.exceptions import ConfigurationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigurationProvider:
    """Abstract base class for configuration injection strategies."""

    def inject_config(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        config_param: str | None = None,
    ) -> Callable[..., Any]:
        """Create configured version of function.

        Args:
            func: Function to inject configuration into
            config: Configuration to inject
            config_param: Optional parameter name for configuration

        Returns:
            Wrapped function with configuration injected
        """
        raise NotImplementedError

    def extract_config(self, func: Callable[..., Any]) -> dict[str, Any] | None:
        """Extract current configuration from function.

        Args:
            func: Function to extract configuration from

        Returns:
            Current configuration or None
        """
        raise NotImplementedError

    def get_config(self) -> dict[str, Any] | None:
        """Get the current active configuration.

        This provides a unified interface for getting configuration
        regardless of the injection mode. Defaults to using context-based
        configuration retrieval.

        Returns:
            Current configuration or None if no configuration is set
        """
        from traigent.config.context import get_config as ctx_get_config
        from traigent.config.types import TraigentConfig

        config = ctx_get_config()
        if isinstance(config, TraigentConfig):
            return config.to_dict()
        return config if config else None

    def supports_function(self, func: Callable[..., Any]) -> bool:
        """Check if provider can handle this function.

        Args:
            func: Function to check

        Returns:
            True if provider can handle function
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clean up any resources held by this provider.

        Called when an OptimizedFunction is cleaned up or reset.
        Subclasses should override to clean up any state they maintain.

        Default implementation does nothing.
        """
        pass


class ContextBasedProvider(ConfigurationProvider):
    """Provider that uses context variables for configuration injection.

    This provider uses Python's contextvars to inject configuration,
    allowing functions to access configuration through get_config().

    Example:
        >>> @traigent.optimize(injection_mode="context")
        ... def my_function(query: str) -> str:
        ...     config = get_config()
        ...     return f"Using model: {config.get('model')}"

    Thread Safety and Limitations:
        - **Thread Isolation**: Each thread maintains its own configuration context.
          Configurations are NOT shared between threads (not singleton-style).
          This ensures thread safety but means child threads won't inherit parent
          thread's configuration unless explicitly passed.

        - **Async Safety**: Async tasks (coroutines) properly inherit context from
          their parent task, making this provider safe for async/await patterns.

        - **Process Boundaries**: Context variables don't cross process boundaries.
          In multiprocessing scenarios, each process starts with a clean context.

        - **Nested Contexts**: Inner context configurations override outer ones.
          When contexts are nested, the innermost configuration takes precedence.

        - **Performance**: Minimal overhead as contextvars are implemented in C
          and optimized for performance in Python 3.7+.

    Common Pitfalls:
        - **Thread Pools**: When using thread pools (e.g., ThreadPoolExecutor),
          each worker thread starts with an empty context. You must explicitly
          set configuration in each thread:

          >>> with ConfigurationContext(config):
          ...     # This config is NOT automatically available in thread pool workers
          ...     with ThreadPoolExecutor() as executor:
          ...         future = executor.submit(my_function)  # Won't have config

          Solution: Pass configuration explicitly or use run_in_executor with
          proper context copying.

        - **Callbacks**: Callbacks executed in different threads/contexts won't
          have access to the original configuration unless explicitly provided.

        - **Global State**: While contextvars are thread-safe, they're still a
          form of implicit state. For better testability and clarity, consider
          using parameter-based injection for simple cases.
    """

    def inject_config(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        config_param: str | None = None,
    ) -> Callable[..., Any]:
        """Inject configuration using context variables."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Merge with existing context
            merged_config = merge_with_context(config)

            # Set context for function execution
            from traigent.config.context import ConfigurationContext

            with ConfigurationContext(merged_config):
                return func(*args, **kwargs)

        # For async functions
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                merged_config = merge_with_context(config)
                from traigent.config.context import ConfigurationContext

                with ConfigurationContext(merged_config):
                    return await func(*args, **kwargs)

            return async_wrapper

        return wrapper

    def extract_config(self, func: Callable[..., Any]) -> dict[str, Any] | None:
        """Extract configuration from context."""
        config: TraigentConfig | dict[str, Any] = get_config()
        if isinstance(config, TraigentConfig):
            return config.to_dict()
        # config is dict[str, Any] after isinstance narrowing
        return config

    def supports_function(self, func: Callable[..., Any]) -> bool:
        """Context injection works with any function."""
        return True


class ParameterBasedProvider(ConfigurationProvider):
    """Provider that injects configuration as a function parameter.

    This provider adds configuration as an explicit parameter to the function,
    providing type safety and clear dependency injection.

    Example:
        >>> @traigent.optimize(
        ...     injection_mode="parameter",
        ...     config_param="config"
        ... )
        ... def my_function(query: str, config: TraigentConfig) -> str:
        ...     return f"Using model: {config.model}"
    """

    def __init__(self, default_param_name: str = "config") -> None:
        """Initialize parameter-based provider.

        Args:
            default_param_name: Default parameter name for configuration
        """
        self.default_param_name = default_param_name

    def inject_config(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        config_param: str | None = None,
    ) -> Callable[..., Any]:
        """Inject configuration as function parameter."""
        param_name = config_param or self.default_param_name

        # Get function signature
        sig = inspect.signature(func)
        params = sig.parameters

        # Check if function already has config parameter
        if param_name not in params:
            raise ConfigurationError(
                f"Function {func.__name__} does not have parameter '{param_name}'. "
                f"Available parameters: {list(params.keys())}"
            )

        # Create TraigentConfig from dict if needed
        config_obj = TraigentConfig.from_dict(config)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Only inject configuration if not already provided
            if param_name not in kwargs:
                kwargs[param_name] = config_obj
            with ConfigurationContext(config_obj):
                return func(*args, **kwargs)

        # For async functions
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Only inject configuration if not already provided
                if param_name not in kwargs:
                    kwargs[param_name] = config_obj
                with ConfigurationContext(config_obj):
                    return await func(*args, **kwargs)

            return async_wrapper

        return wrapper

    def extract_config(self, func: Callable[..., Any]) -> dict[str, Any] | None:
        """Cannot extract config from parameter injection."""
        return None

    def supports_function(self, func: Callable[..., Any]) -> bool:
        """Check if function has config parameter."""
        sig = inspect.signature(func)
        return self.default_param_name in sig.parameters


class AttributeBasedProvider(ConfigurationProvider):
    """Provider that stores configuration as function attribute.

    This provider attaches configuration to the function object,
    allowing access through function attributes. Configuration is set
    on both the wrapper and original function to support accessing
    config from inside the function body.

    Example:
        >>> @traigent.optimize(injection_mode="attribute")
        ... def my_function(query: str) -> str:
        ...     config = my_function.current_config
        ...     return f"Using model: {config['model']}"
    """

    def __init__(self, attribute_name: str = "current_config") -> None:
        """Initialize attribute-based provider.

        Args:
            attribute_name: Attribute name for storing configuration.
                Defaults to "current_config" for backward compatibility.
        """
        self.attribute_name = attribute_name
        # Track wrappers and originals for cleanup
        self._wrappers: list[Callable[..., Any]] = []
        self._originals: list[Callable[..., Any]] = []

    def inject_config(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        config_param: str | None = None,
    ) -> Callable[..., Any]:
        """Inject configuration as function attribute.

        Configuration is set on the wrapper only, not the original function.
        This avoids polluting the original function with unexpected attributes.

        To access config inside your function, use traigent.get_config() which
        works both during optimization trials and after apply_best_config().
        For post-optimization access, you can also use func.current_config.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Store current config on wrapper only (not original)
            setattr(wrapper, self.attribute_name, config)
            with ConfigurationContext(config):
                return func(*args, **kwargs)

        # For async functions
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Set config on wrapper only (not original)
                setattr(async_wrapper, self.attribute_name, config)
                with ConfigurationContext(config):
                    return await func(*args, **kwargs)

            # Set initial config on wrapper for immediate visibility after inject_config()
            # The inner setattr ensures config stays up-to-date on each call
            setattr(async_wrapper, self.attribute_name, config)
            self._wrappers.append(async_wrapper)
            self._originals.append(func)
            return async_wrapper

        # Set initial config on wrapper for immediate visibility after inject_config()
        # The inner setattr ensures config stays up-to-date on each call
        setattr(wrapper, self.attribute_name, config)
        self._wrappers.append(wrapper)
        self._originals.append(func)
        return wrapper

    def extract_config(self, func: Callable[..., Any]) -> dict[str, Any] | None:
        """Extract configuration from function attribute."""
        return getattr(func, self.attribute_name, None)

    def supports_function(self, func: Callable[..., Any]) -> bool:
        """Attribute injection works with any function."""
        return True

    def cleanup(self) -> None:
        """Clean up configuration attributes from tracked wrappers.

        Note: We only clean wrappers, not originals, since we no longer
        set attributes on original functions (to avoid pollution).
        """
        for wrapper in self._wrappers:
            if hasattr(wrapper, self.attribute_name):
                try:
                    delattr(wrapper, self.attribute_name)
                except AttributeError:
                    pass  # Already removed
        self._wrappers.clear()
        self._originals.clear()


class SeamlessParameterProvider(ConfigurationProvider):
    """Safe provider that seamlessly injects parameters using AST transformation.

    This provider achieves the same seamless parameter injection as the original
    but without using exec(), eliminating security risks while maintaining
    full functionality.

    Example:
        >>> @traigent.optimize(
        ...     injection_mode="seamless",
        ...     configuration_space={"model": ["gpt-3.5", "gpt-4"]}
        ... )
        ... def my_function():
        ...     model = "gpt-3.5"  # TraiGent will override this
        ...     return f"Using {model}"

    Security Improvements:
        - No exec() usage - eliminates arbitrary code execution risk
        - AST validation - ensures only safe operations are performed
        - Type-safe value injection - only simple types are injected
        - Compilation caching - improves performance
        - Full audit trail - all transformations are logged
    """

    def __init__(self, max_cache_size: int = 100) -> None:
        """Initialize the safe seamless provider.

        Args:
            max_cache_size: Maximum number of cached transformations
        """
        # Cache for compiled functions to improve performance
        self._compiled_cache: dict[str, Callable[..., Any]] = {}
        self._max_cache_size = max_cache_size
        self._cache_access_count: dict[str, int] = {}  # Track cache usage for LRU
        self._cache_lock = Lock()  # Protect _compiled_cache and _cache_access_count
        self._signature_cache: dict[Callable[..., Any], inspect.Signature] = {}
        self._signature_cache_lock = Lock()
        self._max_signature_cache_size = max_cache_size
        self._stats: dict[str, Any] = {
            "ast_rewrites": 0,
            "runtime_shims": 0,
            "fallback_triggers": defaultdict(list),
        }
        self._stats_lock = Lock()

    def inject_config(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        config_param: str | None = None,
    ) -> Callable[..., Any]:
        """Inject configuration by safely transforming the function's AST.

        Args:
            func: Function to inject configuration into
            config: Configuration to inject
            config_param: Unused for seamless mode

        Returns:
            Wrapped function with configuration injected

        Raises:
            ConfigurationError: If function cannot be safely transformed
        """

        @wraps(func)
        def seamless_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get current TraiGent configuration
            current_config = get_config()

            # Start with the passed config (this is the primary config we want to use)
            active_config = config.copy() if config else {}

            # Merge with context config if available (context takes precedence)
            if isinstance(current_config, TraigentConfig):
                context_dict = current_config.to_dict()
                # Only update if context_dict has actual values
                if context_dict:
                    # Also include custom_params
                    context_dict.update(current_config.custom_params)
                    # Update active_config with context values
                    active_config.update(context_dict)
            elif isinstance(current_config, dict) and current_config:
                # Update active_config with context values
                active_config.update(current_config)

            # If no config at all, just call original function
            if not active_config:
                logger.debug(f"No configuration for {func.__name__}, using original")
                return func(*args, **kwargs)

            # Check cache for already transformed function
            cache_key = self._get_cache_key(func, active_config)
            cached_func = None
            with self._cache_lock:
                if cache_key in self._compiled_cache:
                    cached_func = self._compiled_cache[cache_key]
                    self._cache_access_count[cache_key] = (
                        self._cache_access_count.get(cache_key, 0) + 1
                    )
                    logger.debug(f"Using cached transformation for {func.__name__}")
                else:
                    # Enforce cache size limit (LRU eviction) only when adding new entries
                    if len(self._compiled_cache) >= self._max_cache_size:
                        # Remove least recently used item
                        if self._cache_access_count:  # Guard against empty dict
                            lru_key = min(
                                self._cache_access_count,
                                key=lambda k: self._cache_access_count[k],
                            )
                            del self._compiled_cache[lru_key]
                            del self._cache_access_count[lru_key]
                            logger.debug(f"Evicted {lru_key} from cache (LRU)")

            # Execute cached function OUTSIDE the lock to avoid holding it during execution
            if cached_func is not None:
                return cached_func(*args, **kwargs)

            def _handle_injection_error(exc: Exception) -> None:
                error_text = str(exc).lower()
                model_value = active_config.get("model")
                config_error_markers = [
                    "model_not_found",
                    "model not found",
                    "invalid model",
                    "does not exist",
                ]
                if isinstance(exc, ConfigurationError) or any(
                    marker in error_text for marker in config_error_markers
                ):
                    raise ConfigurationError(
                        "Failed to execute with injected configuration"
                        + (
                            f" (model='{model_value}')"
                            if model_value is not None
                            else ""
                        )
                        + f": {exc}"
                    ) from exc
                raise

            # Transform the function
            try:
                # Don't log sensitive config values
                config_keys = list(active_config.keys()) if active_config else []
                logger.debug(
                    f"Transforming function {func.__name__} with config keys: {config_keys}"
                )
                transformed_func, modified_vars = self._transform_function(
                    func, active_config
                )

                if not callable(transformed_func):
                    raise ConfigurationError("Transformed function must be callable")

                should_shim, signature = self._should_apply_runtime_shim(
                    func, active_config, modified_vars
                )

                if should_shim:
                    shimmed = create_runtime_shim(
                        func, active_config, signature=signature
                    )
                    try:
                        result = shimmed(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        _handle_injection_error(exc)
                    else:
                        with self._cache_lock:
                            self._compiled_cache[cache_key] = shimmed
                            self._cache_access_count[cache_key] = 1
                        with self._stats_lock:
                            self._stats["runtime_shims"] += 1
                        logger.debug(
                            f"Seamless provider using runtime shim for {func.__name__}"
                        )
                        return result
                if modified_vars:
                    try:
                        result = transformed_func(*args, **kwargs)
                    except Exception as exc:  # noqa: BLE001
                        _handle_injection_error(exc)
                    else:
                        with self._cache_lock:
                            self._compiled_cache[cache_key] = transformed_func
                            self._cache_access_count[cache_key] = 1
                        with self._stats_lock:
                            self._stats["ast_rewrites"] += 1
                        logger.debug(
                            f"Cached and executing transformed {func.__name__}"
                        )
                        return result

                # No assignments were modified and no shim required
                with self._cache_lock:
                    self._compiled_cache[cache_key] = func
                    self._cache_access_count[cache_key] = 1
                with self._stats_lock:
                    self._stats["fallback_triggers"]["no_injection"].append(
                        sorted(active_config.keys())
                    )
                logger.debug(
                    f"Seamless provider found no injectable targets for {func.__name__}; "
                    "configuration keys were %s",
                    sorted(active_config.keys()),
                )
                return func(*args, **kwargs)

            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Failed to transform function %s. Attempting runtime shim fallback.",
                    func.__name__,
                    exc_info=False,
                )
                import traceback

                logger.debug("Traceback: %s", traceback.format_exc())
                with self._stats_lock:
                    self._stats["fallback_triggers"]["transform_failure"].append(str(e))
                try:
                    signature = self._get_signature(func)
                    shimmed = create_runtime_shim(
                        func, active_config, signature=signature
                    )
                except Exception as shim_build_exc:  # noqa: BLE001
                    raise ConfigurationError(
                        "Failed to inject configuration via seamless provider"
                    ) from shim_build_exc

                try:
                    result = shimmed(*args, **kwargs)
                except Exception as shim_exc:  # noqa: BLE001
                    _handle_injection_error(shim_exc)
                else:
                    with self._cache_lock:
                        self._compiled_cache[cache_key] = shimmed
                        self._cache_access_count[cache_key] = 1
                    with self._stats_lock:
                        self._stats["runtime_shims"] += 1
                    return result

        # Handle async functions
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_seamless_wrapper(*args: Any, **kwargs: Any) -> Any:
                # For async functions, we need to handle the coroutine
                result = seamless_wrapper(*args, **kwargs)
                if inspect.iscoroutine(result):
                    return await result
                return result

            return async_seamless_wrapper

        return seamless_wrapper

    def _transform_function(
        self, func: Callable[..., Any], config: dict[str, Any]
    ) -> tuple[Callable[..., Any] | None, set[str]]:
        """Transform a function by replacing variable assignments with config values.

        Args:
            func: The function to transform
            config: The configuration values to inject

        Returns:
            New function with transformed code

        Raises:
            ConfigurationError: If transformation fails
        """
        # Validate function before transformation
        if not self._is_safe_function(func):
            raise ConfigurationError(
                f"Function {func.__name__} contains unsafe patterns"
            )

        # Get function source
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError) as e:
            raise ConfigurationError(
                f"Cannot get source for function {func.__name__}: {e}"
            ) from e

        # Check source length to prevent excessive memory usage
        if len(source) > 100000:  # 100KB limit
            raise ConfigurationError(
                f"Function {func.__name__} source too large for transformation"
            )

        # Remove any indentation (for methods and nested functions)
        source = textwrap.dedent(source)

        # Parse source to AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ConfigurationError(
                f"Cannot parse function {func.__name__}: {e}"
            ) from None

        # Validate config before transformation
        self._validate_config(config)

        # Transform the AST with security checks
        transformer = ConfigTransformer(config, max_depth=10)
        modified_tree = transformer.visit(tree)

        # Log what variables were modified for debugging
        modified_vars = transformer.get_modified_variables()
        if modified_vars:
            logger.debug(f"Modified variables in {func.__name__}: {modified_vars}")

        # Validate the transformed AST
        if not SafeASTCompiler.validate_ast(modified_tree):
            raise ConfigurationError(
                f"Transformed AST for {func.__name__} contains unsafe operations"
            )

        # Compile the transformed AST
        try:
            code_obj = SafeASTCompiler.compile_ast_safe(
                modified_tree, filename=func.__code__.co_filename or "<ast>"
            )
        except (ValueError, SyntaxError) as e:
            raise ConfigurationError(
                f"Cannot compile transformed function {func.__name__}: {e}"
            ) from e

        if code_obj is None:
            raise ConfigurationError(f"Compilation returned None for {func.__name__}")

        # Extract the function code object from the module code
        # The compiled code is a module, we need the function inside it
        for const in code_obj.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == func.__name__:
                func_code = const
                break
        else:
            raise ConfigurationError(
                f"Cannot find function {func.__name__} in compiled code"
            )

        # Create new function with the transformed code
        # Note: When functions are defined inside other functions or methods,
        # they might have closures. The compiled code might not expect the same
        # closure, so we need to handle this carefully.
        try:
            new_func = types.FunctionType(
                func_code,
                func.__globals__,  # Use original globals
                func.__name__,  # Keep same name
                func.__defaults__,  # Keep default arguments
                func.__closure__,  # Keep closure
            )
        except ValueError as e:
            # If there's a closure mismatch, try without closure
            if "closure" in str(e):
                new_func = types.FunctionType(
                    func_code,
                    func.__globals__,
                    func.__name__,
                    func.__defaults__,
                    None,  # No closure
                )
            else:
                raise

        # Copy function attributes
        new_func.__dict__.update(func.__dict__)
        new_func.__doc__ = func.__doc__
        new_func.__annotations__ = func.__annotations__
        new_func.__module__ = func.__module__

        return new_func, transformer.get_modified_variables()

    def _get_cache_key(self, func: Callable[..., Any], config: dict[str, Any]) -> str:
        """Generate a cache key for a function and configuration.

        Args:
            func: The function
            config: The configuration

        Returns:
            Cache key string
        """
        import hashlib

        # Create a stable key from function and config
        func_id = f"{func.__module__}.{func.__name__}"

        # Sort config items for stable key and use secure hash
        config_items = sorted(config.items())
        config_str = str(config_items)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return f"{func_id}:{config_hash}"

    def _get_signature(self, func: Callable[..., Any]) -> inspect.Signature:
        """Return cached function signature (compute if missing)."""

        with self._signature_cache_lock:
            signature = self._signature_cache.get(func)
            if signature is None:
                signature = inspect.signature(func)
                if len(self._signature_cache) >= self._max_signature_cache_size:
                    oldest_key = next(iter(self._signature_cache))
                    self._signature_cache.pop(oldest_key)
                self._signature_cache[func] = signature
            return signature

    def _should_apply_runtime_shim(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        modified_vars: set[str],
    ) -> tuple[bool, inspect.Signature | None]:
        """Determine whether runtime shim is required for this function."""

        if modified_vars:
            return False, None
        if not config:
            return False, None

        signature = self._get_signature(func)
        param_names = set(signature.parameters.keys())
        config_keys = set(config.keys())

        if not (param_names & config_keys):
            return False, signature

        return True, signature

    def get_stats(self) -> dict[str, Any]:
        """Return a snapshot of injection statistics."""

        with self._stats_lock:
            return {
                "ast_rewrites": self._stats["ast_rewrites"],
                "runtime_shims": self._stats["runtime_shims"],
                "fallback_triggers": {
                    reason: list(values)
                    for reason, values in self._stats["fallback_triggers"].items()
                },
            }

    def _is_safe_function(self, func: Callable[..., Any]) -> bool:
        """Check if a function is safe to transform.

        Args:
            func: Function to check

        Returns:
            True if function appears safe
        """
        # Check function name doesn't contain dangerous patterns
        dangerous_patterns = ["exec", "eval", "__", "compile"]
        func_name = func.__name__.lower()

        for pattern in dangerous_patterns:
            if pattern in func_name:
                logger.warning(
                    f"Function {func.__name__} contains dangerous pattern: {pattern}"
                )
                return False

        # Check module isn't from dangerous sources
        if func.__module__:
            # Allow __main__ for testing and interactive use
            dangerous_modules = ["builtins", "os", "sys", "subprocess"]
            for mod in dangerous_modules:
                if mod in func.__module__:
                    logger.warning(
                        f"Function {func.__name__} from dangerous module: {func.__module__}"
                    )
                    return False

        return True

    def _validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration for safety.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If config contains unsafe values
        """
        if not config:
            return

        # Check config size
        if len(config) > 1000:
            raise ConfigurationError(
                f"Configuration too large: {len(config)} items"
            ) from None

        # Check for dangerous keys
        # These are exact matches or patterns that should be blocked
        dangerous_exact_keys = {
            "exec",
            "eval",
            "compile",
            "open",
            "file",
            "__builtins__",
            "__import__",
        }
        for key in config:
            if not isinstance(key, str):
                raise ConfigurationError(f"Config key must be string: {type(key)}")

            # Check for exact dangerous keys
            if key in dangerous_exact_keys:
                raise ConfigurationError(f"Dangerous config key: {key}")

            # Check for double underscore (magic methods/attributes)
            if key.startswith("__") and key.endswith("__"):
                raise ConfigurationError(f"Dangerous config key: {key}")

        # Recursively check values
        def check_value(value, depth=0) -> None:
            if depth > 10:
                raise ConfigurationError("Config nesting too deep")

            if isinstance(value, (list, tuple)):
                if len(value) > 1000:
                    raise ConfigurationError(
                        f"Config collection too large: {len(value)} items"
                    )
                for item in value:
                    check_value(item, depth + 1)
            elif isinstance(value, dict):
                if len(value) > 1000:
                    raise ConfigurationError(
                        f"Config dict too large: {len(value)} items"
                    )
                for v in value.values():
                    check_value(v, depth + 1)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                raise ConfigurationError(f"Unsafe config value type: {type(value)}")

        for value in config.values():
            check_value(value)

    def extract_config(self, func: Callable[..., Any]) -> dict[str, Any] | None:
        """Extract current configuration from context.

        Args:
            func: The function (unused for seamless mode)

        Returns:
            Current configuration from context or None
        """
        config: TraigentConfig | dict[str, Any] = get_config()
        if isinstance(config, TraigentConfig):
            return config.to_dict()
        return config

    def supports_function(self, func: Callable[..., Any]) -> bool:
        """Check if function source is available for transformation.

        Args:
            func: The function to check

        Returns:
            True if function can be transformed, False otherwise
        """
        try:
            inspect.getsource(func)
            return True
        except (OSError, TypeError):
            return False

    def cleanup(self) -> None:
        """Clean up all caches and resources held by this provider.

        This clears:
        - Compiled function cache
        - Cache access counts (for LRU tracking)
        - Signature cache
        - Statistics (stats are reset, not preserved)
        """
        with self._cache_lock:
            self._compiled_cache.clear()
            self._cache_access_count.clear()

        with self._signature_cache_lock:
            self._signature_cache.clear()

        with self._stats_lock:
            self._stats = {
                "ast_rewrites": 0,
                "runtime_shims": 0,
                "fallback_triggers": defaultdict(list),
            }


# Provider registry
_PROVIDERS: dict[str, type[ConfigurationProvider]] = {
    "context": ContextBasedProvider,
    "parameter": ParameterBasedProvider,
    "attribute": AttributeBasedProvider,
    "seamless": SeamlessParameterProvider,
}


def get_provider(injection_mode: str, **kwargs: Any) -> ConfigurationProvider:
    """Get configuration provider by injection mode.

    Args:
        injection_mode: One of "context", "parameter", "attribute", "seamless"
                       ("decorator" is deprecated but still supported)
        **kwargs: Additional arguments for provider initialization

    Returns:
        ConfigurationProvider instance

    Raises:
        ConfigurationError: If injection mode is not supported
    """
    # Handle backward compatibility
    if injection_mode == "decorator":
        injection_mode = "attribute"

    if injection_mode not in _PROVIDERS:
        raise ConfigurationError(
            f"Unknown injection mode: {injection_mode}. "
            f"Available modes: {list(_PROVIDERS.keys())}"
        )

    provider_class = _PROVIDERS[injection_mode]

    # Handle provider-specific initialization
    if injection_mode == "parameter":
        config_param = (
            kwargs.get("config_param") or "config"
        )  # Default to "config" if None or missing
        return ParameterBasedProvider(default_param_name=config_param)
    elif injection_mode == "attribute":
        attribute_name = kwargs.get(
            "attribute_name", "current_config"
        )  # Default attribute name
        return AttributeBasedProvider(attribute_name=attribute_name)
    else:
        return provider_class()
