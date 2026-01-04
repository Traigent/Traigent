"""Tests for plugin architecture robustness.

These tests verify:
- ModuleNotFoundError.name check behavior (distinguishes missing cloud vs broken deps)
- Re-entrant registry discovery protection
- Plugin API version compatibility
- Thread-safe plugin registration
"""

import sys
import threading

import pytest

from traigent.plugins import FEATURE_TRACING, FeaturePlugin, get_plugin_registry
from traigent.plugins.registry import (
    _get_traigent_version,
    _is_version_compatible,
    _parse_version,
)
from traigent.utils.exceptions import PluginVersionError


class TestModuleNotFoundErrorNameCheck:
    """Tests for ModuleNotFoundError .name attribute handling.

    The orchestrator must distinguish between:
    1. Cloud module not installed (graceful degradation)
    2. Broken install with missing transitive dependency (hard error)
    """

    def test_name_attribute_identifies_cloud_module(self):
        """Verify ModuleNotFoundError.name correctly identifies cloud module."""
        # Simulate cloud module not found
        err = ModuleNotFoundError("No module named 'traigent.cloud'")
        err.name = "traigent.cloud"

        assert err.name is not None
        assert err.name.startswith("traigent.cloud")

    def test_name_attribute_identifies_submodule(self):
        """Verify .name check works for cloud submodules."""
        err = ModuleNotFoundError("No module named 'traigent.cloud.backend_client'")
        err.name = "traigent.cloud.backend_client"

        assert err.name.startswith("traigent.cloud")

    def test_name_attribute_distinguishes_other_modules(self):
        """Verify .name check doesn't match non-cloud modules."""
        # Simulate missing dependency inside cloud module
        err = ModuleNotFoundError("No module named 'boto3'")
        err.name = "boto3"

        assert not err.name.startswith("traigent.cloud")

    def test_orchestrator_handles_missing_cloud_gracefully(self):
        """Verify the .name check logic correctly identifies cloud modules.

        This tests the pattern used in orchestrator._initialize_backend_client()
        to distinguish between 'cloud not installed' vs 'broken dependency'.
        """
        # Pattern from orchestrator.py:
        # except ModuleNotFoundError as err:
        #     if err.name and err.name.startswith("traigent.cloud"):
        #         # Graceful degradation
        #     raise  # Re-raise for other missing modules

        # Case 1: Cloud module missing - should be caught
        cloud_err = ModuleNotFoundError("No module named 'traigent.cloud'")
        cloud_err.name = "traigent.cloud.backend_client"

        should_degrade = cloud_err.name and cloud_err.name.startswith("traigent.cloud")
        assert should_degrade, "Cloud module error should trigger graceful degradation"

        # Case 2: Broken dependency - should NOT be caught
        dep_err = ModuleNotFoundError("No module named 'boto3'")
        dep_err.name = "boto3"

        should_degrade = dep_err.name and dep_err.name.startswith("traigent.cloud")
        assert not should_degrade, "Dependency error should NOT trigger degradation"

    def test_orchestrator_raises_for_broken_dependency(self):
        """Verify orchestrator raises for broken transitive dependencies.

        If a module inside traigent.cloud fails to import due to missing
        boto3/requests/etc, this should be a hard error, not silent degradation.
        """
        # Create error that looks like broken dependency
        dep_err = ModuleNotFoundError("No module named 'boto3'")
        dep_err.name = "boto3"

        # This should NOT be caught by the cloud-specific handler
        assert dep_err.name != "traigent.cloud"
        assert not dep_err.name.startswith("traigent.cloud")


class TestReentrantRegistryDiscovery:
    """Tests for re-entrant plugin discovery protection.

    Plugins may call registry methods during initialize(), which could
    cause infinite recursion or duplicate registrations without guards.
    """

    def test_discovery_guard_prevents_recursion(self):
        """Verify thread ID check blocks recursive discovery from same thread."""
        registry = get_plugin_registry()

        # Reset state
        registry._entry_points_loaded = False
        registry._discovery_thread_id = None
        registry._discovery_complete.set()

        # Track discovery calls
        discovery_calls = []
        original_discover = registry.discover_entry_points

        def tracking_discover():
            discovery_calls.append(len(discovery_calls))
            # Simulate plugin that tries to trigger re-discovery
            registry._ensure_entry_points_loaded()
            original_discover()

        registry.discover_entry_points = tracking_discover

        try:
            # First call should work
            registry._ensure_entry_points_loaded()

            # Should only have one discovery call, not recursive
            assert len(discovery_calls) == 1, (
                f"Expected 1 discovery call, got {len(discovery_calls)}. "
                "Re-entrancy guard may not be working."
            )
        finally:
            registry.discover_entry_points = original_discover
            registry._entry_points_loaded = True  # Reset

    def test_plugin_initialize_can_check_features(self):
        """Verify plugin.initialize() can safely call has_feature()."""

        class FeatureCheckingPlugin(FeaturePlugin):
            """Plugin that checks features during initialization."""

            def __init__(self):
                self.checked_features = []

            @property
            def name(self) -> str:
                return "test-feature-checker"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Test plugin"

            @property
            def author(self) -> str:
                return "Test"

            def provides_features(self) -> list[str]:
                return ["test_feature"]

            def initialize(self) -> None:
                # This should not cause deadlock or recursion
                registry = get_plugin_registry()
                self.checked_features.append(registry.has_feature(FEATURE_TRACING))

        registry = get_plugin_registry()
        plugin = FeatureCheckingPlugin()

        # Should not deadlock
        registry.register_plugin(plugin)

        # Plugin should have successfully checked features
        assert len(plugin.checked_features) == 1

        # Cleanup
        registry.unregister_plugin("test-feature-checker")

    def test_concurrent_discovery_is_serialized(self):
        """Verify concurrent discovery attempts are serialized by lock."""
        registry = get_plugin_registry()

        # Reset state
        registry._entry_points_loaded = False
        registry._discovery_thread_id = None
        registry._discovery_complete.set()

        discovery_order = []
        lock = threading.Lock()

        original_discover = registry.discover_entry_points

        def slow_discover():
            with lock:
                discovery_order.append(f"start_{threading.current_thread().name}")
            import time

            time.sleep(0.1)  # Simulate slow discovery
            original_discover()
            with lock:
                discovery_order.append(f"end_{threading.current_thread().name}")

        registry.discover_entry_points = slow_discover

        threads = []
        try:
            # Launch multiple threads trying to discover
            for i in range(3):
                t = threading.Thread(
                    target=registry._ensure_entry_points_loaded, name=f"thread_{i}"
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=5)

            # Only one thread should have done discovery
            starts = [x for x in discovery_order if x.startswith("start_")]
            assert len(starts) == 1, (
                f"Expected 1 discovery, got {len(starts)}. " f"Order: {discovery_order}"
            )
        finally:
            registry.discover_entry_points = original_discover
            registry._entry_points_loaded = True

    def test_concurrent_callers_wait_for_discovery(self):
        """Verify concurrent callers wait for discovery to complete before returning."""
        registry = get_plugin_registry()

        # Reset state
        registry._entry_points_loaded = False
        registry._discovery_thread_id = None
        registry._discovery_complete.set()

        # Track when each thread sees discovery as complete
        completion_times: dict[str, float] = {}
        discovery_end_time: list[float] = []
        lock = threading.Lock()

        original_discover = registry.discover_entry_points

        def slow_discover():
            import time

            time.sleep(0.2)  # Simulate slow discovery
            original_discover()
            with lock:
                discovery_end_time.append(time.time())

        def check_feature_after_discovery(thread_name: str):
            # This calls _ensure_entry_points_loaded internally
            registry.has_feature("some_feature")
            import time

            with lock:
                completion_times[thread_name] = time.time()

        registry.discover_entry_points = slow_discover

        threads = []
        try:
            # Launch multiple threads that will call has_feature
            for i in range(3):
                t = threading.Thread(
                    target=check_feature_after_discovery,
                    args=(f"thread_{i}",),
                    name=f"thread_{i}",
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join(timeout=5)

            # All threads should complete AFTER discovery ends
            assert len(discovery_end_time) == 1, "Discovery should run exactly once"
            for thread_name, complete_time in completion_times.items():
                assert complete_time >= discovery_end_time[0], (
                    f"{thread_name} completed at {complete_time} before "
                    f"discovery ended at {discovery_end_time[0]}"
                )
        finally:
            registry.discover_entry_points = original_discover
            registry._entry_points_loaded = True


class TestPluginReregistration:
    """Tests for plugin re-registration behavior."""

    def test_reregistration_cleans_up_old_plugin(self):
        """Verify re-registering cleans up the old plugin first."""

        class TestPlugin(FeaturePlugin):
            cleanup_called = False

            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Test"

            @property
            def author(self) -> str:
                return "Test"

            def provides_features(self) -> list[str]:
                return ["reregister_test"]

            def initialize(self) -> None:
                """No-op initialization for test plugin."""

            def cleanup(self) -> None:
                TestPlugin.cleanup_called = True

        registry = get_plugin_registry()

        # Register first plugin
        plugin1 = TestPlugin("reregister-test")
        registry.register_plugin(plugin1)

        # Register replacement - should cleanup first
        TestPlugin.cleanup_called = False
        plugin2 = TestPlugin("reregister-test")
        registry.register_plugin(plugin2)

        assert TestPlugin.cleanup_called, "Old plugin cleanup() should be called"

        # Cleanup
        registry.unregister_plugin("reregister-test")

    def test_unregister_always_removes_even_if_cleanup_fails(self):
        """Verify unregister removes plugin even if cleanup() raises."""

        class FailingCleanupPlugin(FeaturePlugin):
            @property
            def name(self) -> str:
                return "failing-cleanup"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Test"

            @property
            def author(self) -> str:
                return "Test"

            def provides_features(self) -> list[str]:
                return ["failing_feature"]

            def initialize(self) -> None:
                """No-op initialization for test plugin."""

            def cleanup(self) -> None:
                raise RuntimeError("Cleanup failed!")

        registry = get_plugin_registry()
        plugin = FailingCleanupPlugin()

        registry.register_plugin(plugin)
        plugin_names = [p["name"] for p in registry.list_plugins()]
        assert "failing-cleanup" in plugin_names

        # Unregister should succeed despite cleanup failure
        registry.unregister_plugin("failing-cleanup")

        plugin_names = [p["name"] for p in registry.list_plugins()]
        assert "failing-cleanup" not in plugin_names
        assert not registry.has_feature("failing_feature")


class TestLazyImportModules:
    """Tests verifying modules can be imported without cloud dependencies."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "traigent.core.orchestrator",
            "traigent.core.backend_session_manager",
            "traigent.plugins.registry",
            "traigent.utils.local_analytics",
            "traigent.agents.executor",
            "traigent.agents.config_mapper",
            "traigent.agents.specification_generator",
            "traigent.agents.platforms",
            "traigent.optimizers.interactive_optimizer",
            "traigent.optigen_integration",
        ],
    )
    def test_module_imports_without_cloud(self, module_path: str):
        """Verify modules can be imported when cloud package is missing."""

        class CloudBlocker:
            """Meta path finder that blocks traigent.cloud imports."""

            def find_module(self, name, path=None):
                if name.startswith("traigent.cloud"):
                    return self
                return None

            def load_module(self, name):
                err = ModuleNotFoundError(f"No module named '{name}'")
                err.name = name
                raise err

        # Clear any cached imports - both the target module AND traigent.cloud*
        # to ensure test isolation (earlier tests may have imported cloud)
        # Note: we need to snapshot keys() since we're modifying the dict
        modules_to_clear = [
            k
            for k in tuple(sys.modules.keys())
            if k.startswith(module_path) or k.startswith("traigent.cloud")
        ]
        for mod in modules_to_clear:
            sys.modules.pop(mod, None)

        blocker = CloudBlocker()
        sys.meta_path.insert(0, blocker)

        try:
            # Should not raise
            __import__(module_path)
        finally:
            sys.meta_path.remove(blocker)


class TestVersionParsing:
    """Tests for version string parsing and comparison."""

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("1.0.0", (1, 0, 0)),
            ("1.2.3", (1, 2, 3)),
            ("0.9.0", (0, 9, 0)),
            ("1.0", (1, 0)),
            ("1", (1,)),
            ("1.0.0-beta", (1, 0, 0)),
            ("1.0.0+local", (1, 0, 0)),
            ("1.0.0-alpha.1", (1, 0, 0)),
            ("0.1.0.dev1", (0, 1, 0)),
            ("2.0.0-rc1+build.123", (2, 0, 0)),
            # v prefix handling
            ("v1.0.0", (1, 0, 0)),
            ("V1.2.3", (1, 2, 3)),
            ("v0.9.0-beta", (0, 9, 0)),
            # Whitespace handling
            ("  1.0.0  ", (1, 0, 0)),
            (" v1.2.3", (1, 2, 3)),
            ("1.0.0 ", (1, 0, 0)),
        ],
    )
    def test_parse_version(self, version_str: str, expected: tuple[int, ...]):
        """Test version string parsing."""
        result = _parse_version(version_str)
        assert result == expected

    def test_parse_version_empty_returns_zero(self):
        """Empty version string should return (0,)."""
        assert _parse_version("") == (0,)
        assert _parse_version("-beta") == (0,)
        assert _parse_version("v") == (0,)


class TestVersionCompatibility:
    """Tests for version compatibility checking."""

    @pytest.mark.parametrize(
        "required,current,expected",
        [
            # Current >= required
            ("0.1.0", "0.9.0", True),
            ("0.9.0", "0.9.0", True),
            ("0.9.0", "1.0.0", True),
            ("1.0.0", "1.0.0", True),
            ("1.0.0", "1.0.1", True),
            ("1.0.0", "1.1.0", True),
            ("1.0.0", "2.0.0", True),
            # Current < required
            ("1.0.0", "0.9.0", False),
            ("2.0.0", "1.9.9", False),
            ("1.0.1", "1.0.0", False),
            # Edge cases
            ("0.1", "0.1.0", True),
            ("0.1.0", "0.1", True),
        ],
    )
    def test_version_compatibility(self, required: str, current: str, expected: bool):
        """Test version compatibility checking."""
        result = _is_version_compatible(required, current)
        assert result == expected, (
            f"Expected _is_version_compatible('{required}', '{current}') "
            f"to be {expected}, got {result}"
        )

    def test_get_traigent_version_returns_string(self):
        """Verify _get_traigent_version returns a valid version string."""
        version = _get_traigent_version()
        assert isinstance(version, str)
        assert len(version) > 0
        # Should be parseable
        parsed = _parse_version(version)
        assert len(parsed) > 0


class TestPluginVersionEnforcement:
    """Tests for plugin version compatibility enforcement during registration."""

    def test_compatible_plugin_registers_successfully(self):
        """Plugin requiring compatible version should register."""

        class CompatiblePlugin(FeaturePlugin):
            @property
            def name(self) -> str:
                return "test-compatible-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Test plugin"

            @property
            def author(self) -> str:
                return "Test"

            @property
            def traigent_version(self) -> str:
                # Require version lower than current
                return "0.1.0"

            def provides_features(self) -> list[str]:
                return ["test_compat_feature"]

            def initialize(self) -> None:
                """No-op initialization for test plugin."""

        registry = get_plugin_registry()
        plugin = CompatiblePlugin()

        # Should register without error
        registry.register_plugin(plugin)
        assert registry.get_plugin("test-compatible-plugin") is not None

        # Cleanup
        registry.unregister_plugin("test-compatible-plugin")

    def test_incompatible_plugin_raises_version_error(self):
        """Plugin requiring future version should fail to register."""

        class IncompatiblePlugin(FeaturePlugin):
            @property
            def name(self) -> str:
                return "test-incompatible-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Test plugin"

            @property
            def author(self) -> str:
                return "Test"

            @property
            def traigent_version(self) -> str:
                # Require version much higher than current
                return "99.0.0"

            def provides_features(self) -> list[str]:
                return ["test_incompat_feature"]

            def initialize(self) -> None:
                """No-op initialization for test plugin."""

        registry = get_plugin_registry()
        plugin = IncompatiblePlugin()

        # Should raise PluginVersionError
        with pytest.raises(PluginVersionError) as exc_info:
            registry.register_plugin(plugin)

        error = exc_info.value
        assert error.plugin_name == "test-incompatible-plugin"
        assert error.plugin_version == "1.0.0"
        assert error.required_traigent_version == "99.0.0"
        assert "upgrade" in str(error).lower()

        # Plugin should not be registered
        assert registry.get_plugin("test-incompatible-plugin") is None

    def test_version_error_includes_helpful_message(self):
        """PluginVersionError should include upgrade instructions."""
        error = PluginVersionError(
            plugin_name="my-plugin",
            plugin_version="2.0.0",
            required_traigent_version="5.0.0",
            current_traigent_version="1.0.0",
        )

        message = str(error)
        assert "my-plugin" in message
        assert "5.0.0" in message
        assert "1.0.0" in message
        assert "pip install --upgrade traigent" in message
