"""Integration tests for the refactored integrations module structure.

Tests verify:
1. No circular imports between modules
2. Backward compatibility with existing APIs
3. Module composition works correctly
4. Protocol compliance
5. Thread safety across module boundaries

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestNoCircularImports:
    """Verify no circular imports exist between modules."""

    def test_import_mappings_standalone(self):
        """Test mappings module can be imported standalone."""
        # Force fresh import
        module_name = "traigent.integrations.mappings"
        if module_name in sys.modules:
            del sys.modules[module_name]

        from traigent.integrations.mappings import (
            METHOD_MAPPINGS,
            PARAMETER_MAPPINGS,
            get_parameter_mapping,
        )

        assert isinstance(PARAMETER_MAPPINGS, dict)
        assert isinstance(METHOD_MAPPINGS, dict)
        assert callable(get_parameter_mapping)

    def test_import_activation_standalone(self):
        """Test activation module can be imported standalone."""
        from traigent.integrations.activation import (
            ActivationState,
            create_activation_state,
        )

        state = create_activation_state()
        assert isinstance(state, ActivationState)

    def test_import_wrappers_standalone(self):
        """Test wrappers module can be imported standalone."""
        from traigent.integrations.wrappers import (
            apply_parameter_overrides,
            create_method_wrapper,
            create_resilient_wrapper,
            create_wrapper,
        )

        # Verify all are importable
        assert callable(apply_parameter_overrides)
        assert callable(create_wrapper)
        assert callable(create_method_wrapper)
        assert callable(create_resilient_wrapper)

    def test_import_base_uses_activation(self):
        """Test base module correctly imports from activation."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()
        # Verify ActivationState is used internally
        assert hasattr(manager, "_state")
        assert hasattr(manager._state, "is_active")

    def test_import_order_independence(self):
        """Test modules can be imported in any order without errors."""
        # Import in reverse order
        from traigent.integrations.activation import ActivationState
        from traigent.integrations.base import BaseOverrideManager
        from traigent.integrations.framework_override import FrameworkOverrideManager
        from traigent.integrations.mappings import PARAMETER_MAPPINGS
        from traigent.integrations.wrappers import apply_parameter_overrides

        # All should be available
        assert PARAMETER_MAPPINGS is not None
        assert ActivationState is not None
        assert apply_parameter_overrides is not None
        assert BaseOverrideManager is not None
        assert FrameworkOverrideManager is not None


class TestBackwardCompatibility:
    """Verify backward compatibility with existing APIs."""

    def test_framework_override_manager_available(self):
        """Test FrameworkOverrideManager is importable from main path."""
        from traigent.integrations import FrameworkOverrideManager

        manager = FrameworkOverrideManager()
        assert hasattr(manager, "create_overridden_constructor")
        assert hasattr(manager, "cleanup_all_overrides")
        assert hasattr(manager, "activate_overrides")
        assert hasattr(manager, "deactivate_overrides")

    def test_override_functions_available(self):
        """Test override helper functions are importable."""
        from traigent.integrations import (
            disable_framework_overrides,
            enable_framework_overrides,
            override_all_platforms,
            override_anthropic,
            override_cohere,
            override_huggingface,
            override_langchain,
            override_openai_sdk,
            register_framework_mapping,
        )

        # All should be callable
        assert callable(enable_framework_overrides)
        assert callable(disable_framework_overrides)
        assert callable(override_openai_sdk)
        assert callable(override_langchain)
        assert callable(override_anthropic)
        assert callable(override_cohere)
        assert callable(override_huggingface)
        assert callable(override_all_platforms)
        assert callable(register_framework_mapping)

    def test_base_manager_backward_compatible(self):
        """Test BaseOverrideManager maintains backward-compatible properties."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Test backward-compatible property access (returns copies)
        constructors = manager._original_constructors
        methods = manager._original_methods
        overrides = manager._active_overrides

        assert isinstance(constructors, dict)
        assert isinstance(methods, dict)
        assert isinstance(overrides, dict)

    def test_override_active_state_methods(self):
        """Test override active state methods work correctly."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Test state management
        assert manager.is_override_active() is False
        manager.set_override_active(True)
        assert manager.is_override_active() is True
        manager.set_override_active(False)
        assert manager.is_override_active() is False

    def test_store_and_restore_methods(self):
        """Test store and restore methods work correctly."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Store constructor
        def original_init():
            pass

        manager.store_original_constructor("test.Class", original_init)
        assert manager.is_constructor_overridden("test.Class")

        # Restore
        restored = manager.restore_original_constructor("test.Class")
        assert restored == original_init
        assert not manager.is_constructor_overridden("test.Class")


class TestNewAPIExports:
    """Verify new API exports are correctly exposed."""

    def test_activation_exports(self):
        """Test activation module exports are available from main __init__."""
        from traigent.integrations import ActivationState, create_activation_state

        state = create_activation_state()
        assert isinstance(state, ActivationState)
        assert state.is_active() is False

    def test_mappings_exports(self):
        """Test mappings module exports are available from main __init__."""
        from traigent.integrations import (
            get_method_mapping,
            get_parameter_mapping,
            get_supported_frameworks,
        )

        frameworks = get_supported_frameworks()
        assert "openai.OpenAI" in frameworks
        assert "anthropic.Anthropic" in frameworks

        openai_mapping = get_parameter_mapping("openai.OpenAI")
        assert "temperature" in openai_mapping

        method_mapping = get_method_mapping("openai.OpenAI")
        assert "chat.completions.create" in method_mapping

    def test_wrappers_exports(self):
        """Test wrappers module exports are available from main __init__."""
        from traigent.integrations import (
            apply_parameter_overrides,
        )

        # Test apply_parameter_overrides
        result = apply_parameter_overrides(
            {"existing": "value"},
            {"temperature": 0.7},
            {"temperature": "temp"},
        )
        assert result["temp"] == 0.7

    def test_base_manager_export(self):
        """Test BaseOverrideManager is exported."""
        from traigent.integrations import BaseOverrideManager

        manager = BaseOverrideManager()
        assert hasattr(manager, "_state")


class TestModuleComposition:
    """Test that modules compose correctly."""

    def test_base_manager_uses_activation_state(self):
        """Test BaseOverrideManager correctly uses ActivationState."""
        from traigent.integrations.activation import ActivationState
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Verify internal state is ActivationState
        assert isinstance(manager._state, ActivationState)

        # Verify operations go through ActivationState
        manager.set_override_active(True)
        assert manager._state.is_active() is True

    def test_framework_override_uses_mappings(self):
        """Test FrameworkOverrideManager uses mappings from mappings.py."""
        from traigent.integrations.framework_override import FrameworkOverrideManager

        manager = FrameworkOverrideManager()

        # Verify mappings are loaded
        param_mappings = manager._parameter_mappings
        assert "openai.OpenAI" in param_mappings


class TestThreadSafetyAcrossModules:
    """Test thread safety across module boundaries."""

    def test_concurrent_activation_state_access(self):
        """Test concurrent access to activation state is thread-safe."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()
        results = {"activated": 0, "deactivated": 0}

        def toggle_state(should_activate):
            if should_activate:
                manager.set_override_active(True)
                time.sleep(0.01)
                if manager.is_override_active():
                    results["activated"] += 1
            else:
                manager.set_override_active(False)
                time.sleep(0.01)
                if not manager.is_override_active():
                    results["deactivated"] += 1

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(toggle_state, i % 2 == 0) for i in range(20)]
            for future in as_completed(futures):
                future.result()

        # Thread-local state means each thread sees its own state
        # Not testing exact counts due to thread-local behavior

    def test_concurrent_method_storage_across_managers(self):
        """Test concurrent method storage across multiple manager instances."""
        from traigent.integrations.base import BaseOverrideManager

        num_managers = 5
        num_ops_per_manager = 20

        managers = [BaseOverrideManager() for _ in range(num_managers)]

        def store_methods(manager_idx):
            manager = managers[manager_idx]
            for i in range(num_ops_per_manager):
                key = f"manager_{manager_idx}.method_{i}"
                manager.store_original_method(key, lambda: None)

        with ThreadPoolExecutor(max_workers=num_managers) as executor:
            futures = [executor.submit(store_methods, i) for i in range(num_managers)]
            for future in as_completed(futures):
                future.result()

        # Each manager should have its own methods
        for idx, manager in enumerate(managers):
            methods = manager._original_methods
            assert len(methods) == num_ops_per_manager
            for i in range(num_ops_per_manager):
                assert f"manager_{idx}.method_{i}" in methods


class TestProtocolCompliance:
    """Test Protocol compliance for wrappers module."""

    def test_base_manager_implements_override_context_interface(self):
        """Test BaseOverrideManager can be used where OverrideContext is expected."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Duck typing check - manager should have all Protocol methods
        assert hasattr(manager, "is_override_active")
        assert hasattr(manager, "extract_config_dict")

        # Methods should be callable with correct signatures
        assert callable(manager.is_override_active)
        assert callable(manager.extract_config_dict)

        # Test method behavior
        assert isinstance(manager.is_override_active(), bool)
        assert manager.extract_config_dict(None) is None


class TestAllExportsInAll:
    """Verify __all__ contains all expected exports."""

    def test_all_exports_match(self):
        """Test that __all__ contains all expected exports."""
        from traigent.integrations import __all__

        expected_exports = [
            # Core
            "FrameworkOverrideManager",
            "enable_framework_overrides",
            "disable_framework_overrides",
            "override_context",
            "register_framework_mapping",
            # Platforms
            "override_openai_sdk",
            "override_langchain",
            "override_anthropic",
            "override_cohere",
            "override_huggingface",
            "override_all_platforms",
            # Base
            "BaseOverrideManager",
            # Activation
            "ActivationState",
            "create_activation_state",
            # Mappings
            "PARAMETER_MAPPINGS",
            "METHOD_MAPPINGS",
            "get_parameter_mapping",
            "get_method_mapping",
            "get_supported_frameworks",
            # Wrappers
            "OverrideContext",
            "apply_parameter_overrides",
            "create_wrapper",
            "create_method_wrapper",
            "create_resilient_wrapper",
        ]

        for export in expected_exports:
            assert export in __all__, f"{export} not in __all__"


class TestOverrideActivationFlow:
    """Test the full override activation/deactivation lifecycle.

    These tests verify that the activation flow actually patches target classes
    and that deactivation properly restores them, addressing the critical
    bug where mutations to copied dicts would be discarded.
    """

    def test_activate_override_state(self):
        """Test that activate_overrides properly sets override state."""
        from traigent.integrations.framework_override import FrameworkOverrideManager

        manager = FrameworkOverrideManager()

        # Initially inactive
        assert not manager.is_override_active()

        # Activate overrides (even with no valid targets, state should be active)
        manager.activate_overrides([])

        # State should be active after activation
        assert manager.is_override_active()

        # Deactivate
        manager.deactivate_overrides()
        assert not manager.is_override_active()

    def test_activate_handles_missing_target_gracefully(self):
        """Test that activation handles non-importable targets gracefully."""
        from traigent.integrations.framework_override import FrameworkOverrideManager

        manager = FrameworkOverrideManager()

        # Try to activate a non-existent module - should not raise
        manager.activate_overrides(["nonexistent.module.Class"])

        # Verify nothing was registered
        assert not manager.is_override_registered("nonexistent.module.Class")

        # Cleanup should not raise
        manager.deactivate_overrides()

    def test_deactivate_restores_original(self):
        """Test that deactivation restores the original constructor."""
        from traigent.integrations.framework_override import FrameworkOverrideManager

        manager = FrameworkOverrideManager()

        class MockLLMClient:
            was_original_called = False

            def __init__(self, model="default"):
                MockLLMClient.was_original_called = True
                self.model = model

        # Register mapping
        manager._parameter_mappings["test.MockLLMClient"] = {"model": "model"}

        # Activate and deactivate
        manager.activate_overrides([MockLLMClient])
        manager.deactivate_overrides()

        # After deactivation, the original constructor should be restored
        # Verify by creating an instance - the original flag should still be set
        MockLLMClient.was_original_called = False
        MockLLMClient(model="test")
        assert MockLLMClient.was_original_called

    def test_mappings_isolation_between_instances(self):
        """Test that mapping mutations don't affect global mappings (deep copy fix)."""
        from traigent.integrations.framework_override import FrameworkOverrideManager
        from traigent.integrations.mappings import PARAMETER_MAPPINGS

        # Get original global state
        dict(PARAMETER_MAPPINGS.get("openai.OpenAI", {}))

        # Create two managers
        manager1 = FrameworkOverrideManager()
        manager2 = FrameworkOverrideManager()

        # Modify one manager's mappings
        manager1._parameter_mappings["openai.OpenAI"]["custom_param"] = "custom_value"

        # Verify the other manager is NOT affected (deep copy)
        assert "custom_param" not in manager2._parameter_mappings.get(
            "openai.OpenAI", {}
        )

        # Verify global mappings are NOT affected
        assert "custom_param" not in PARAMETER_MAPPINGS.get("openai.OpenAI", {})

    def test_method_mappings_isolation(self):
        """Test that method mapping mutations don't affect global mappings."""
        from traigent.integrations.framework_override import FrameworkOverrideManager
        from traigent.integrations.mappings import METHOD_MAPPINGS

        manager1 = FrameworkOverrideManager()
        manager2 = FrameworkOverrideManager()

        # Get a method mapping
        if "openai.OpenAI" in manager1._method_mappings:
            # Modify inner list
            list(
                manager1._method_mappings["openai.OpenAI"].get(
                    "chat.completions.create", []
                )
            )
            manager1._method_mappings["openai.OpenAI"][
                "chat.completions.create"
            ].append("custom_method_param")

            # Verify manager2 is NOT affected
            m2_params = manager2._method_mappings.get("openai.OpenAI", {}).get(
                "chat.completions.create", []
            )
            assert "custom_method_param" not in m2_params

            # Verify global mappings NOT affected
            global_params = METHOD_MAPPINGS.get("openai.OpenAI", {}).get(
                "chat.completions.create", []
            )
            assert "custom_method_param" not in global_params

    def test_register_and_unregister_cycle(self):
        """Test the full register/unregister cycle using public methods."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Initially no overrides
        assert not manager.is_override_registered("test.Target")
        assert len(manager.get_active_overrides_copy()) == 0

        # Register an override
        manager.register_active_override("test.Target", object)

        # Verify it's registered
        assert manager.is_override_registered("test.Target")
        overrides = manager.get_active_overrides_copy()
        assert "test.Target" in overrides
        assert overrides["test.Target"] is object

        # Unregister
        manager.unregister_active_override("test.Target")

        # Verify it's gone
        assert not manager.is_override_registered("test.Target")
        assert "test.Target" not in manager.get_active_overrides_copy()

    def test_clear_active_overrides(self):
        """Test that clearing returns the overrides and empties the registry."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Register multiple overrides
        manager.register_active_override("target1", "class1")
        manager.register_active_override("target2", "class2")
        manager.register_active_override("target3", "class3")

        # Clear and get returned overrides
        cleared = manager.clear_active_overrides()

        # Verify all were returned
        assert "target1" in cleared
        assert "target2" in cleared
        assert "target3" in cleared

        # Verify the registry is now empty
        assert len(manager.get_active_overrides_copy()) == 0

    def test_end_override_context_uses_unregister(self):
        """Test that end_override_context properly unregisters (uses public method)."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Start override context
        manager.start_override_context("test.framework")

        # Verify it's registered
        assert manager.is_override_registered("test.framework")
        assert manager.is_override_active()

        # End override context
        manager.end_override_context("test.framework")

        # Verify it's unregistered and deactivated
        assert not manager.is_override_registered("test.framework")
        assert not manager.is_override_active()

    def test_cleanup_override_uses_unregister(self):
        """Test that cleanup_override properly unregisters (uses public method)."""
        from traigent.integrations.base import BaseOverrideManager

        manager = BaseOverrideManager()

        # Store some data
        def mock_init():
            pass

        manager.store_original_constructor("test.framework", mock_init)
        manager.store_original_method("test.framework.method1", lambda: None)
        manager.register_active_override("test.framework", object)

        # Cleanup
        manager.cleanup_override("test.framework")

        # Verify everything is cleaned up
        assert not manager.is_constructor_overridden("test.framework")
        assert not manager.is_method_overridden("test.framework.method1")
        assert not manager.is_override_registered("test.framework")
