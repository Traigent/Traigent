"""Tests for PydanticAI integration import availability."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


class TestPydanticAIAvailability:
    """Test graceful degradation when pydantic-ai is not installed."""

    def test_pydanticai_available_flag_exists(self) -> None:
        """The PYDANTICAI_AVAILABLE flag should be importable regardless."""
        from traigent.integrations.pydantic_ai import PYDANTICAI_AVAILABLE

        assert isinstance(PYDANTICAI_AVAILABLE, bool)

    def test_init_exports_all_names(self) -> None:
        """__init__ should export all expected names regardless of availability."""
        import traigent.integrations.pydantic_ai as pkg

        assert hasattr(pkg, "PYDANTICAI_AVAILABLE")
        assert hasattr(pkg, "PydanticAIHandler")
        assert hasattr(pkg, "PydanticAIPlugin")
        assert hasattr(pkg, "create_pydantic_ai_handler")

    def test_init_when_pydantic_ai_missing(self) -> None:
        """When pydantic_ai cannot be imported, stubs raise ImportError."""
        import traigent.integrations.pydantic_ai as pkg

        # Simulate pydantic_ai not being installed
        with patch.dict(sys.modules, {"pydantic_ai": None}):
            importlib.reload(pkg)
            assert pkg.PYDANTICAI_AVAILABLE is False
            with pytest.raises(ImportError, match="pydantic-ai"):
                pkg.PydanticAIHandler()  # type: ignore[operator]
            with pytest.raises(ImportError, match="pydantic-ai"):
                pkg.create_pydantic_ai_handler()  # type: ignore[operator]

        # Restore
        importlib.reload(pkg)

    def test_handler_raises_when_unavailable(self) -> None:
        """PydanticAIHandler should raise ImportError with install instructions."""
        import traigent.integrations.pydantic_ai.handler as handler_mod

        original = handler_mod.PYDANTICAI_AVAILABLE
        try:
            handler_mod.PYDANTICAI_AVAILABLE = False
            from traigent.integrations.pydantic_ai.handler import PydanticAIHandler

            with pytest.raises(ImportError, match="pydantic-ai"):
                PydanticAIHandler(agent=object())
        finally:
            handler_mod.PYDANTICAI_AVAILABLE = original

    def test_plugin_importable_without_pydantic_ai(self) -> None:
        """PydanticAIPlugin should be importable — it doesn't import pydantic_ai."""
        from traigent.integrations.pydantic_ai.plugin import PydanticAIPlugin

        plugin = PydanticAIPlugin()
        assert plugin.FRAMEWORK.value == "pydantic_ai"

    def test_types_importable(self) -> None:
        """Data types should be importable without pydantic-ai."""
        from traigent.integrations.pydantic_ai._types import (
            AgentRunMetrics,
            PydanticAIHandlerMetrics,
        )

        m = AgentRunMetrics()
        assert m.input_tokens == 0

        pm = PydanticAIHandlerMetrics()
        assert pm.run_count == 0


class TestPluginRegistryIntegration:
    """Test that the plugin is discoverable via the registry."""

    def test_pydantic_ai_in_builtin_plugins(self) -> None:
        """PydanticAIPlugin should be in the builtin plugins list."""
        from traigent.integrations.plugin_registry import PluginRegistry

        # Reset singleton for clean test
        PluginRegistry._instance = None
        try:
            registry = PluginRegistry()
            plugin_names = registry.list_plugins()
            assert "pydantic_ai" in plugin_names
        finally:
            PluginRegistry._instance = None


class TestFrameworkEnum:
    """Test that PYDANTIC_AI is in the Framework enum."""

    def test_pydantic_ai_in_framework_enum(self) -> None:
        from traigent.integrations.utils.parameter_normalizer import Framework

        assert hasattr(Framework, "PYDANTIC_AI")
        assert Framework.PYDANTIC_AI.value == "pydantic_ai"

    def test_framework_from_string(self) -> None:
        from traigent.integrations.utils.parameter_normalizer import (
            ParameterNormalizer,
        )

        fw = ParameterNormalizer.get_framework_from_string("pydantic_ai")
        assert fw is not None
        assert fw.value == "pydantic_ai"
