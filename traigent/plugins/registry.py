"""Plugin registry system for Traigent SDK."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import importlib
import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TraigentPlugin(ABC):
    """Abstract base class for Traigent plugins."""

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


class PluginRegistry:
    """Registry for managing Traigent plugins."""

    def __init__(self) -> None:
        """Initialize plugin registry."""
        self._plugins: dict[str, TraigentPlugin] = {}
        self._optimizers: dict[str, Callable[..., Any]] = {}
        self._evaluators: dict[str, Callable[..., Any]] = {}
        self._metrics: dict[str, Callable[..., Any]] = {}
        self._integrations: dict[str, dict[str, Callable[..., Any]]] = {}

    def register_plugin(self, plugin: TraigentPlugin) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin to register
        """
        plugin_name = plugin.name

        if plugin_name in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' already registered, replacing")

        try:
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

            logger.info(
                f"Successfully registered plugin: {plugin_name} v{plugin.version}"
            )

        except Exception as e:
            logger.error(f"Failed to register plugin '{plugin_name}': {e}")
            raise

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin.

        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' not found")
            return

        plugin = self._plugins[plugin_name]

        try:
            # Cleanup plugin
            plugin.cleanup()

            # Remove from registries
            if isinstance(plugin, OptimizerPlugin):
                self._optimizers.pop(plugin.optimizer_name, None)

            if isinstance(plugin, EvaluatorPlugin):
                self._evaluators.pop(plugin.evaluator_name, None)

            if isinstance(plugin, MetricPlugin):
                self._metrics.pop(plugin.metric_name, None)

            if isinstance(plugin, IntegrationPlugin):
                self._integrations.pop(plugin.integration_name, None)

            # Remove plugin
            del self._plugins[plugin_name]

            logger.info(f"Unregistered plugin: {plugin_name}")

        except Exception as e:
            logger.error(f"Failed to unregister plugin '{plugin_name}': {e}")

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
        return self._optimizers.copy()

    def get_evaluators(self) -> dict[str, Callable[..., Any]]:
        """Get registered evaluators.

        Returns:
            Dict mapping evaluator names to factory functions
        """
        return self._evaluators.copy()

    def get_metrics(self) -> dict[str, Callable[..., Any]]:
        """Get registered metrics.

        Returns:
            Dict mapping metric names to calculation functions
        """
        return self._metrics.copy()

    def get_integrations(self) -> dict[str, dict[str, Callable[..., Any]]]:
        """Get registered integrations.

        Returns:
            Dict mapping integration names to their functions
        """
        return self._integrations.copy()

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
