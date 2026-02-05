"""Plugin registry system for Traigent SDK.

This module provides the unified plugin discovery and registration system.
Plugins can register via:
1. Entry points in pyproject.toml: [project.entry-points."traigent.plugins"]
2. Manual registration via register_plugin()
3. Auto-discovery from standard directories

The registry provides feature flag capabilities for graceful degradation
when optional plugins are not installed.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..utils.exceptions import PluginVersionError
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Handles versions like "1.0.0", "v1.2.3", "1.0.0-beta", "1.0.0.dev1".
    Leading/trailing whitespace, 'v' prefix, and non-numeric suffixes are stripped.
    Constraint operators (>=, ~=, etc.) are NOT supported and will
    result in partial parsing.

    Args:
        version_str: Version string to parse

    Returns:
        Tuple of version components as integers. Returns (0,) for
        unparseable strings.

    Examples:
        >>> _parse_version("1.2.3")
        (1, 2, 3)
        >>> _parse_version("v1.2.3")
        (1, 2, 3)
        >>> _parse_version("1.0.0-beta")
        (1, 0, 0)
        >>> _parse_version("0.1.0.dev1")
        (0, 1, 0)
        >>> _parse_version("  1.0.0  ")
        (1, 0, 0)
    """
    # Strip whitespace first
    version_str = version_str.strip()

    # Strip leading 'v' or 'V' prefix (common convention)
    if version_str.startswith(("v", "V")):
        version_str = version_str[1:]

    # Strip common suffixes: -beta, +local, etc.
    version_str = version_str.split("-")[0].split("+")[0]
    parts = []
    for part in version_str.split("."):
        # Handle cases like "0.dev1" by extracting leading digits
        numeric = ""
        for char in part:
            if char.isdigit():
                numeric += char
            else:
                break
        if numeric:
            parts.append(int(numeric))
    return tuple(parts) if parts else (0,)


def _is_version_compatible(
    required_version: str, current_version: str, strict: bool = False
) -> bool:
    """Check if current version meets the required minimum version.

    Args:
        required_version: Minimum version required
        current_version: Current installed version
        strict: If True, require exact major version match (for breaking changes)

    Returns:
        True if current version is >= required version
    """
    required = _parse_version(required_version)
    current = _parse_version(current_version)

    if strict and required and current:
        # Strict mode: major version must match
        if required[0] != current[0]:
            return False

    # Pad shorter tuple with zeros for comparison
    max_len = max(len(required), len(current))
    required = required + (0,) * (max_len - len(required))
    current = current + (0,) * (max_len - len(current))

    return current >= required


def _get_traigent_version() -> str:
    """Get the current Traigent version.

    Uses a fallback chain to handle various installation scenarios:
    1. Try traigent._version (canonical source)
    2. Try traigent.__version__ (package-level export)
    3. Fall back to "0.0.0" if neither is available

    Returns:
        Current Traigent version string
    """
    # Try canonical version module first
    try:
        from traigent._version import __version__

        return str(__version__)
    except ImportError:
        pass

    # Fallback to package-level __version__ (editable installs)
    try:
        import traigent

        version = getattr(traigent, "__version__", None)
        if version is not None:
            return str(version)
    except ImportError:
        pass

    # Last resort fallback
    return "0.0.0"


# Feature flag constants for plugin capabilities
FEATURE_PARALLEL = "parallel"
FEATURE_MULTI_OBJECTIVE = "multi_objective"
FEATURE_SEAMLESS = "seamless"
FEATURE_CLOUD = "cloud"
FEATURE_ADVANCED_ALGORITHMS = "advanced_algorithms"
FEATURE_TVL = "tvl"
FEATURE_TRACING = "tracing"
FEATURE_ANALYTICS = "analytics"
FEATURE_INTEGRATIONS = "integrations"
FEATURE_SECURITY = "security"
FEATURE_HOOKS = "hooks"
FEATURE_EVALUATION = "evaluation"
FEATURE_UI = "ui"
FEATURE_EXPERIMENT_TRACKING = "experiment_tracking"


class TraigentPlugin(ABC):
    """Abstract base class for Traigent plugins.

    Plugins must implement:
    - name: Unique plugin identifier
    - version: Semantic version string
    - description: Human-readable description
    - initialize(): Setup method called on registration
    - provides_features(): List of feature flags this plugin enables

    Optional:
    - author: Plugin author
    - dependencies: Required Python packages
    - traigent_version: Minimum Traigent version
    - cleanup(): Teardown method
    - get_feature_impl(): Get implementation for a feature
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        raise NotImplementedError

    @property
    def author(self) -> str:
        """Plugin author."""
        return "Unknown"

    @property
    def dependencies(self) -> list[str]:
        """List of required dependencies."""
        return []

    @property
    def traigent_version(self) -> str:
        """Minimum Traigent version required."""
        return "0.1.0"

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the plugin."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Cleanup when plugin is unloaded."""
        return None

    def provides_features(self) -> list[str]:
        """Return list of feature flags this plugin provides.

        Override this method to declare which features your plugin enables.
        Use the FEATURE_* constants defined in this module.

        Returns:
            List of feature flag strings (e.g., ["parallel", "batch_invoker"])
        """
        return []

    def get_feature_impl(self, feature: str) -> Any | None:
        """Get implementation for a specific feature.

        Override this to provide feature-specific implementations that
        other parts of the system can retrieve.

        Args:
            feature: Feature name to get implementation for

        Returns:
            Implementation object or None if not provided
        """
        return None


class OptimizerPlugin(TraigentPlugin):
    """Base class for optimizer plugins."""

    @abstractmethod
    def create_optimizer(self, **kwargs) -> Any:
        """Create optimizer instance.

        Returns:
            Optimizer instance that implements BaseOptimizer interface
        """
        raise NotImplementedError

    @property
    def optimizer_name(self) -> str:
        """Name to register optimizer under."""
        return self.name.lower().replace(" ", "_")


class EvaluatorPlugin(TraigentPlugin):
    """Base class for evaluator plugins."""

    @abstractmethod
    def create_evaluator(self, **kwargs) -> Any:
        """Create evaluator instance.

        Returns:
            Evaluator instance that implements BaseEvaluator interface
        """
        raise NotImplementedError

    @property
    def evaluator_name(self) -> str:
        """Name to register evaluator under."""
        return self.name.lower().replace(" ", "_")


class MetricPlugin(TraigentPlugin):
    """Base class for custom metric plugins."""

    @abstractmethod
    def calculate_metric(
        self, actual: Any, expected: Any, context: dict[str, Any] | None = None
    ) -> float:
        """Calculate custom metric.

        Args:
            actual: Actual output from function
            expected: Expected output
            context: Optional context information

        Returns:
            Metric value (higher is better)
        """
        raise NotImplementedError

    @property
    def metric_name(self) -> str:
        """Name to register metric under."""
        return self.name.lower().replace(" ", "_")


class IntegrationPlugin(TraigentPlugin):
    """Base class for framework integration plugins."""

    @abstractmethod
    def get_integration_functions(self) -> dict[str, Callable[..., Any]]:
        """Get integration functions to expose.

        Returns:
            Dict mapping function names to callables
        """
        raise NotImplementedError

    @property
    def integration_name(self) -> str:
        """Name to register integration under."""
        return self.name.lower().replace(" ", "_")


class FeaturePlugin(TraigentPlugin):
    """Base class for feature plugins that enable optional capabilities.

    Use this for plugins that add features like parallel execution,
    multi-objective optimization, seamless injection, etc.
    """

    def provides_features(self) -> list[str]:
        """Must be overridden to declare provided features."""
        raise NotImplementedError("FeaturePlugin must declare provides_features()")


class PluginRegistry:
    """Registry for managing Traigent plugins.

    The registry provides:
    - Plugin registration and lifecycle management
    - Feature flag querying (has_feature, get_feature_impl)
    - Entry point discovery for automatic plugin loading
    - Specialized sub-registries for optimizers, evaluators, metrics, integrations
    """

    # Entry point group for plugin discovery
    ENTRY_POINT_GROUP = "traigent.plugins"

    def __init__(self) -> None:
        """Initialize plugin registry."""
        self._plugins: dict[str, TraigentPlugin] = {}
        self._optimizers: dict[str, Callable[..., Any]] = {}
        self._evaluators: dict[str, Callable[..., Any]] = {}
        self._metrics: dict[str, Callable[..., Any]] = {}
        self._integrations: dict[str, dict[str, Callable[..., Any]]] = {}
        self._features: dict[str, list[str]] = {}  # feature -> list of plugin names
        self._entry_points_loaded = False
        self._discovery_thread_id: int | None = (
            None  # Thread doing discovery (for re-entrancy)
        )
        # Use RLock to allow reentrant calls (e.g., if plugin.initialize() calls registry methods)
        self._discovery_lock = threading.RLock()
        # Event to signal discovery completion - concurrent threads wait on this
        self._discovery_complete = threading.Event()
        self._discovery_complete.set()  # Initially set (no discovery in progress)

    def register_plugin(self, plugin: TraigentPlugin) -> None:
        """Register a plugin.

        If a plugin with the same name is already registered, it will be
        unregistered first to prevent stale registry entries.

        Args:
            plugin: Plugin to register
        """
        plugin_name = plugin.name

        if plugin_name in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' already registered, replacing")
            # Clean up existing plugin before re-registering to prevent stale entries
            self.unregister_plugin(plugin_name)

        try:
            # Check version compatibility
            self._check_version_compatibility(plugin)

            # Check dependencies
            self._check_dependencies(plugin)

            # Initialize plugin
            plugin.initialize()

            # Register plugin
            self._plugins[plugin_name] = plugin

            # Register specific capabilities
            if isinstance(plugin, OptimizerPlugin):
                self._optimizers[plugin.optimizer_name] = plugin.create_optimizer
                logger.info(f"Registered optimizer plugin: {plugin.optimizer_name}")

            if isinstance(plugin, EvaluatorPlugin):
                self._evaluators[plugin.evaluator_name] = plugin.create_evaluator
                logger.info(f"Registered evaluator plugin: {plugin.evaluator_name}")

            if isinstance(plugin, MetricPlugin):
                self._metrics[plugin.metric_name] = plugin.calculate_metric
                logger.info(f"Registered metric plugin: {plugin.metric_name}")

            if isinstance(plugin, IntegrationPlugin):
                self._integrations[plugin.integration_name] = (
                    plugin.get_integration_functions()
                )
                logger.info(f"Registered integration plugin: {plugin.integration_name}")

            # Register feature flags
            for feature in plugin.provides_features():
                if feature not in self._features:
                    self._features[feature] = []
                self._features[feature].append(plugin_name)
                logger.debug(f"Plugin '{plugin_name}' provides feature: {feature}")

            logger.info(
                f"Successfully registered plugin: {plugin_name} v{plugin.version}"
            )

        except Exception as e:
            logger.error(f"Failed to register plugin '{plugin_name}': {e}")
            raise

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin.

        Always removes registry entries even if cleanup fails to prevent
        stale data. Cleanup errors are logged but don't prevent removal.

        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' not found")
            return

        plugin = self._plugins[plugin_name]

        # Try cleanup but don't let it prevent removal
        try:
            plugin.cleanup()
        except Exception as e:
            logger.warning(f"Plugin '{plugin_name}' cleanup failed: {e}")

        # Always remove from registries regardless of cleanup outcome
        if isinstance(plugin, OptimizerPlugin):
            self._optimizers.pop(plugin.optimizer_name, None)

        if isinstance(plugin, EvaluatorPlugin):
            self._evaluators.pop(plugin.evaluator_name, None)

        if isinstance(plugin, MetricPlugin):
            self._metrics.pop(plugin.metric_name, None)

        if isinstance(plugin, IntegrationPlugin):
            self._integrations.pop(plugin.integration_name, None)

        # Remove feature flags
        for feature in plugin.provides_features():
            if feature in self._features:
                self._features[feature] = [
                    p for p in self._features[feature] if p != plugin_name
                ]
                if not self._features[feature]:
                    del self._features[feature]

        # Remove plugin
        del self._plugins[plugin_name]

        logger.info(f"Unregistered plugin: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> TraigentPlugin | None:
        """Get plugin by name.

        Args:
            plugin_name: Name of plugin

        Returns:
            Plugin instance or None
        """
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all registered plugins.

        Returns:
            List of plugin information
        """
        self._ensure_entry_points_loaded()
        plugins_info = []
        for name, plugin in self._plugins.items():
            plugins_info.append(
                {
                    "name": name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "author": plugin.author,
                    "type": type(plugin).__name__,
                    "dependencies": plugin.dependencies,
                }
            )
        return plugins_info

    def get_optimizers(self) -> dict[str, Callable[..., Any]]:
        """Get registered optimizers.

        Returns:
            Dict mapping optimizer names to factory functions
        """
        self._ensure_entry_points_loaded()
        return self._optimizers.copy()

    def get_evaluators(self) -> dict[str, Callable[..., Any]]:
        """Get registered evaluators.

        Returns:
            Dict mapping evaluator names to factory functions
        """
        self._ensure_entry_points_loaded()
        return self._evaluators.copy()

    def get_metrics(self) -> dict[str, Callable[..., Any]]:
        """Get registered metrics.

        Returns:
            Dict mapping metric names to calculation functions
        """
        self._ensure_entry_points_loaded()
        return self._metrics.copy()

    def get_integrations(self) -> dict[str, dict[str, Callable[..., Any]]]:
        """Get registered integrations.

        Returns:
            Dict mapping integration names to their functions
        """
        self._ensure_entry_points_loaded()
        return self._integrations.copy()

    def has_feature(self, feature: str) -> bool:
        """Check if a feature is available via any installed plugin.

        This is the primary method for checking feature availability before
        using optional functionality.

        Args:
            feature: Feature flag name (use FEATURE_* constants)

        Returns:
            True if feature is available, False otherwise

        Example:
            if not registry.has_feature(FEATURE_PARALLEL):
                raise FeatureNotAvailableError(
                    "Parallel execution",
                    plugin_name="traigent-parallel",
                    install_hint="pip install traigent[ml]"
                )
        """
        # Ensure entry points are loaded
        self._ensure_entry_points_loaded()
        return feature in self._features and len(self._features[feature]) > 0

    def get_feature_impl(self, feature: str) -> Any | None:
        """Get the implementation of a feature if available.

        Queries all plugins that provide the feature and returns the first
        implementation found.

        Args:
            feature: Feature flag name

        Returns:
            Implementation object or None if not available
        """
        self._ensure_entry_points_loaded()

        if feature not in self._features:
            return None

        for plugin_name in self._features[feature]:
            plugin = self._plugins.get(plugin_name)
            if plugin:
                impl = plugin.get_feature_impl(feature)
                if impl is not None:
                    return impl

        return None

    def get_available_features(self) -> dict[str, list[str]]:
        """Get all available features and their providing plugins.

        Returns:
            Dict mapping feature names to list of plugin names providing them
        """
        self._ensure_entry_points_loaded()
        return self._features.copy()

    def discover_entry_points(self) -> None:
        """Discover and load plugins from entry points.

        Loads plugins registered via pyproject.toml entry points:
        [project.entry-points."traigent.plugins"]
        parallel = "traigent_parallel:ParallelPlugin"

        Note:
            This method is called automatically by _ensure_entry_points_loaded()
            which handles thread synchronization. Direct calls to this method
            bypass the synchronization guard and may result in partial registry
            state if called concurrently. For public access, use has_feature()
            or get_plugin() which trigger discovery through the guard.
        """
        # Entry point discovery with cross-version compatibility
        eps: Any  # EntryPoints or list depending on Python version
        try:
            # Python 3.10+ API with group parameter
            eps = importlib.metadata.entry_points(group=self.ENTRY_POINT_GROUP)
        except TypeError:
            # Python 3.9 fallback - entry_points() returns dict-like SelectableGroups
            all_eps = importlib.metadata.entry_points()
            if hasattr(all_eps, "select"):
                # Python 3.10+ SelectableGroups
                eps = all_eps.select(group=self.ENTRY_POINT_GROUP)
            elif hasattr(all_eps, "get"):
                # Python 3.9 dict-like interface
                eps = all_eps.get(self.ENTRY_POINT_GROUP, [])
            else:
                eps = []

        for ep in eps:
            try:
                plugin_class = ep.load()
                if inspect.isclass(plugin_class) and issubclass(
                    plugin_class, TraigentPlugin
                ):
                    plugin = plugin_class()
                    self.register_plugin(plugin)
                    logger.info(f"Loaded plugin from entry point: {ep.name}")
                else:
                    logger.warning(
                        f"Entry point {ep.name} did not return a TraigentPlugin class"
                    )
            except Exception as e:
                logger.warning(f"Failed to load plugin from entry point {ep.name}: {e}")

        self._entry_points_loaded = True

    def _ensure_entry_points_loaded(self) -> None:
        """Ensure entry points have been discovered.

        Uses double-checked locking pattern for thread safety. Concurrent
        callers wait on an Event until discovery completes, ensuring they
        see the full registry state. Same-thread re-entrancy (e.g., plugin
        initialize() calling has_feature()) is allowed via thread ID check.
        """
        # Fast path: already loaded
        if self._entry_points_loaded:
            return

        current_thread = threading.get_ident()

        # Check if this is a re-entrant call from the discovery thread
        # This allows plugins to call registry methods during initialize()
        if self._discovery_thread_id == current_thread:
            return

        # Wait for any in-progress discovery to complete (blocks other threads)
        # This ensures concurrent callers see complete registry state
        self._discovery_complete.wait()

        # Check again after waiting
        if self._entry_points_loaded:
            return

        # Slow path: acquire lock and check again
        with self._discovery_lock:
            if self._entry_points_loaded:
                return

            # We're the discovery thread now
            self._discovery_thread_id = current_thread
            self._discovery_complete.clear()  # Signal discovery in progress
            try:
                self.discover_entry_points()
            finally:
                self._discovery_thread_id = None
                self._discovery_complete.set()  # Signal discovery complete

    def load_plugin_from_module(self, module_path: str, plugin_class_name: str) -> None:
        """Load plugin from module.

        Args:
            module_path: Path to module (e.g., 'my_package.my_plugin')
            plugin_class_name: Name of plugin class in module
        """
        # Validate module path to prevent arbitrary imports
        if not self._is_safe_module_path(module_path):
            raise ValueError(f"Module path '{module_path}' is not allowed") from None

        # Validate class name
        if not self._is_valid_identifier(plugin_class_name):
            raise ValueError(f"Invalid plugin class name: {plugin_class_name}")

        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_class_name)

            if not issubclass(plugin_class, TraigentPlugin):
                raise ValueError(f"Class {plugin_class_name} is not a TraigentPlugin")

            plugin = plugin_class()
            self.register_plugin(plugin)

        except Exception as e:
            logger.error(
                f"Failed to load plugin from {module_path}.{plugin_class_name}: {e}"
            )
            raise

    def load_plugins_from_directory(self, directory: str | Path) -> None:
        """Load all plugins from directory.

        Args:
            directory: Directory containing plugin files
        """
        directory = Path(directory).resolve()  # Resolve to absolute path

        # Validate directory path
        if not self._is_safe_directory(directory):
            raise ValueError(f"Directory '{directory}' is not in allowed locations")

        if not directory.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return

        # Add directory to Python path temporarily
        sys.path.insert(0, str(directory))

        try:
            for plugin_file in directory.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue

                try:
                    module_name = plugin_file.stem
                    module = importlib.import_module(module_name)

                    # Find TraigentPlugin classes in module
                    for _name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, TraigentPlugin)
                            and obj != TraigentPlugin
                            and not inspect.isabstract(obj)
                        ):
                            plugin = obj()
                            self.register_plugin(plugin)

                except Exception as e:
                    logger.warning(f"Failed to load plugin from {plugin_file}: {e}")

        finally:
            # Remove directory from Python path
            if str(directory) in sys.path:
                sys.path.remove(str(directory))

    def auto_discover_plugins(self) -> None:
        """Automatically discover and load plugins from standard locations."""
        # Standard plugin directories
        plugin_dirs = [
            Path.home() / ".traigent" / "plugins",
            Path.cwd() / "traigent_plugins",
            Path(__file__).parent / "contrib",
        ]

        for plugin_dir in plugin_dirs:
            if plugin_dir.exists():
                logger.info(f"Discovering plugins in: {plugin_dir}")
                self.load_plugins_from_directory(plugin_dir)

    def _check_version_compatibility(self, plugin: TraigentPlugin) -> None:
        """Check if plugin is compatible with current Traigent version.

        Args:
            plugin: Plugin to check

        Raises:
            PluginVersionError: If plugin requires a newer Traigent version
        """
        required_version = plugin.traigent_version
        current_version = _get_traigent_version()

        if not _is_version_compatible(required_version, current_version):
            raise PluginVersionError(
                plugin_name=plugin.name,
                plugin_version=plugin.version,
                required_traigent_version=required_version,
                current_traigent_version=current_version,
            )

        logger.debug(
            f"Plugin '{plugin.name}' version check passed "
            f"(requires traigent>={required_version}, have {current_version})"
        )

    def _check_dependencies(self, plugin: TraigentPlugin) -> None:
        """Check if plugin dependencies are available.

        Args:
            plugin: Plugin to check

        Raises:
            ImportError: If dependencies are missing
        """
        missing_deps = []

        for dep in plugin.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            raise ImportError(
                f"Plugin '{plugin.name}' missing dependencies: {', '.join(missing_deps)}"
            )

    def _is_safe_module_path(self, module_path: str) -> bool:
        """Validate module path for safety.

        Args:
            module_path: Module path to validate

        Returns:
            True if module path is safe, False otherwise
        """
        # Disallow paths that could access system modules
        unsafe_patterns = [
            "os",
            "sys",
            "subprocess",
            "importlib",
            "__builtin__",
            "builtins",
            "eval",
            "exec",
            "compile",
            "open",
            "file",
            "input",
            "raw_input",
            "__import__",
        ]

        # Check for path traversal attempts
        if ".." in module_path or module_path.startswith("/"):
            return False

        # Check against unsafe patterns
        module_parts = module_path.split(".")
        for part in module_parts:
            if part.lower() in unsafe_patterns:
                return False

        # Ensure it starts with allowed prefixes
        allowed_prefixes = [
            "traigent_plugins",
            "traigent.plugins.contrib",
            "custom_plugins",
        ]

        return any(module_path.startswith(prefix) for prefix in allowed_prefixes)

    def _is_valid_identifier(self, name: str) -> bool:
        """Check if name is a valid Python identifier.

        Args:
            name: Name to validate

        Returns:
            True if valid identifier, False otherwise
        """
        import keyword

        return name.isidentifier() and not keyword.iskeyword(name)

    def _is_safe_directory(self, directory: Path) -> bool:
        """Check if directory is in allowed locations.

        Args:
            directory: Directory path to validate

        Returns:
            True if directory is safe, False otherwise
        """
        directory = directory.resolve()

        # Define allowed plugin directories
        allowed_dirs = [
            Path.home() / ".traigent" / "plugins",
            Path.cwd() / "traigent_plugins",
            Path(__file__).parent / "contrib",
            Path.cwd() / "plugins",  # Current working directory plugins
        ]

        # Check if directory is within allowed locations
        for allowed_dir in allowed_dirs:
            try:
                allowed_dir = allowed_dir.resolve()
                # Check if directory is subdirectory of allowed location
                directory.relative_to(allowed_dir)
                return True
            except (ValueError, RuntimeError):
                continue

        return False


# Global plugin registry instance
_global_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get global plugin registry.

    Returns:
        Global plugin registry instance
    """
    return _global_registry


def register_plugin(plugin: TraigentPlugin) -> None:
    """Register plugin with global registry.

    Args:
        plugin: Plugin to register
    """
    _global_registry.register_plugin(plugin)


def load_plugin(module_path: str, plugin_class_name: str) -> None:
    """Load plugin from module into global registry.

    Args:
        module_path: Module path
        plugin_class_name: Plugin class name
    """
    _global_registry.load_plugin_from_module(module_path, plugin_class_name)


def discover_plugins() -> None:
    """Auto-discover plugins and load into global registry."""
    _global_registry.auto_discover_plugins()


def list_available_plugins() -> list[dict[str, Any]]:
    """List all available plugins.

    Returns:
        List of plugin information
    """
    return _global_registry.list_plugins()


def get_available_optimizers() -> list[str]:
    """Get names of available optimizer plugins.

    Returns:
        List of optimizer names
    """
    return list(_global_registry.get_optimizers().keys())


def get_available_evaluators() -> list[str]:
    """Get names of available evaluator plugins.

    Returns:
        List of evaluator names
    """
    return list(_global_registry.get_evaluators().keys())


def get_available_metrics() -> list[str]:
    """Get names of available metric plugins.

    Returns:
        List of metric names
    """
    return list(_global_registry.get_metrics().keys())


def get_available_integrations() -> list[str]:
    """Get names of available integration plugins.

    Returns:
        List of integration names
    """
    return list(_global_registry.get_integrations().keys())


def has_feature(feature: str) -> bool:
    """Check if a feature is available via any installed plugin.

    Convenience function that wraps get_plugin_registry().has_feature().

    Args:
        feature: Feature flag name (use FEATURE_* constants)

    Returns:
        True if feature is available, False otherwise

    Example:
        >>> from traigent.plugins import has_feature, FEATURE_PARALLEL
        >>> if has_feature(FEATURE_PARALLEL):
        ...     print("Parallel execution available!")
    """
    return _global_registry.has_feature(feature)
