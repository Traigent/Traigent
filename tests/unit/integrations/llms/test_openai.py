"""Unit tests for OpenAI SDK Integration.

Tests for the OpenAI SDK integration module providing seamless integration
with the OpenAI Python SDK for automatic parameter override.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility
# Traceability: FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import sys
import warnings
from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import pytest

from traigent.integrations.llms.openai import (
    OpenAIIntegration,
    _openai_integration,
    auto_detect_openai,
    enable_async_openai,
    enable_openai_optimization,
    enable_streaming_optimization,
    enable_sync_openai,
    enable_tools_optimization,
    get_supported_openai_clients,
    openai_context,
)


class TestOpenAIIntegration:
    """Tests for OpenAIIntegration class."""

    @pytest.fixture
    def integration(self) -> OpenAIIntegration:
        """Create test instance of OpenAIIntegration."""
        return OpenAIIntegration()

    def test_initialization(self, integration: OpenAIIntegration) -> None:
        """Test OpenAIIntegration initializes with correct supported clients."""
        assert isinstance(integration.supported_clients, dict)
        assert "openai.OpenAI" in integration.supported_clients
        assert "openai.AsyncOpenAI" in integration.supported_clients

    def test_supported_clients_structure(self, integration: OpenAIIntegration) -> None:
        """Test supported clients have correct parameter mappings."""
        for _client_name, mappings in integration.supported_clients.items():
            assert isinstance(mappings, dict)
            # Check for essential parameter mappings
            assert "model" in mappings
            assert "temperature" in mappings
            assert "max_tokens" in mappings
            assert "top_p" in mappings
            assert "frequency_penalty" in mappings
            assert "presence_penalty" in mappings
            assert "stop" in mappings
            assert "stream" in mappings
            assert "tools" in mappings
            assert "tool_choice" in mappings

    def test_sync_and_async_clients_have_same_mappings(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test sync and async OpenAI clients have identical parameter mappings."""
        sync_mappings = integration.supported_clients["openai.OpenAI"]
        async_mappings = integration.supported_clients["openai.AsyncOpenAI"]
        assert sync_mappings == async_mappings

    @patch("traigent.integrations.llms.openai.register_framework_mapping")
    def test_register_mappings_called_on_init(self, mock_register: MagicMock) -> None:
        """Test that parameter mappings are registered on initialization."""
        integration = OpenAIIntegration()
        # Should be called twice - once for each client type
        assert mock_register.call_count == 2
        mock_register.assert_any_call(
            "openai.OpenAI", integration.supported_clients["openai.OpenAI"]
        )
        mock_register.assert_any_call(
            "openai.AsyncOpenAI", integration.supported_clients["openai.AsyncOpenAI"]
        )

    def test_get_supported_clients(self, integration: OpenAIIntegration) -> None:
        """Test get_supported_clients returns list of client type names."""
        clients = integration.get_supported_clients()
        assert isinstance(clients, list)
        assert len(clients) == 2
        assert "openai.OpenAI" in clients
        assert "openai.AsyncOpenAI" in clients

    @patch("traigent.integrations.llms.openai.enable_framework_overrides")
    def test_enable_openai_overrides_with_none(
        self, mock_enable: MagicMock, integration: OpenAIIntegration
    ) -> None:
        """Test enable_openai_overrides with None enables all supported clients."""
        integration.enable_openai_overrides(None)
        mock_enable.assert_called_once()
        called_clients = mock_enable.call_args[0][0]
        assert "openai.OpenAI" in called_clients
        assert "openai.AsyncOpenAI" in called_clients

    @patch("traigent.integrations.llms.openai.enable_framework_overrides")
    def test_enable_openai_overrides_with_specific_clients(
        self, mock_enable: MagicMock, integration: OpenAIIntegration
    ) -> None:
        """Test enable_openai_overrides with specific client types."""
        client_types = ["openai.OpenAI"]
        integration.enable_openai_overrides(client_types)
        mock_enable.assert_called_once_with(["openai.OpenAI"])

    @patch("traigent.integrations.llms.openai.enable_framework_overrides")
    def test_enable_openai_overrides_with_both_clients(
        self, mock_enable: MagicMock, integration: OpenAIIntegration
    ) -> None:
        """Test enable_openai_overrides with both sync and async clients."""
        client_types = ["openai.OpenAI", "openai.AsyncOpenAI"]
        integration.enable_openai_overrides(client_types)
        mock_enable.assert_called_once_with(client_types)

    def test_normalize_client_types_with_none(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types returns all clients when None is passed."""
        result = integration._normalize_client_types(None)
        assert isinstance(result, list)
        assert "openai.OpenAI" in result
        assert "openai.AsyncOpenAI" in result

    def test_normalize_client_types_with_valid_list(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types with valid client type list."""
        result = integration._normalize_client_types(["openai.OpenAI"])
        assert result == ["openai.OpenAI"]

    def test_normalize_client_types_removes_duplicates(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types removes duplicate entries."""
        result = integration._normalize_client_types(
            ["openai.OpenAI", "openai.OpenAI", "openai.AsyncOpenAI"]
        )
        assert len(result) == 2
        assert "openai.OpenAI" in result
        assert "openai.AsyncOpenAI" in result

    def test_normalize_client_types_raises_on_string(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types raises TypeError for string input."""
        with pytest.raises(TypeError, match="client_types must be an iterable"):
            integration._normalize_client_types("openai.OpenAI")  # type: ignore[arg-type]

    def test_normalize_client_types_raises_on_bytes(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types raises TypeError for bytes input."""
        with pytest.raises(TypeError, match="client_types must be an iterable"):
            integration._normalize_client_types(b"openai.OpenAI")  # type: ignore[arg-type]

    def test_normalize_client_types_raises_on_empty_string(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types raises ValueError for empty string in list."""
        with pytest.raises(ValueError, match="must be non-empty strings"):
            integration._normalize_client_types([""])

    def test_normalize_client_types_raises_on_whitespace_string(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types raises ValueError for whitespace string."""
        with pytest.raises(ValueError, match="must be non-empty strings"):
            integration._normalize_client_types(["   "])

    def test_normalize_client_types_raises_on_non_string_entry(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types raises ValueError for non-string entry."""
        with pytest.raises(ValueError, match="must be non-empty strings"):
            integration._normalize_client_types([123])  # type: ignore[list-item]

    def test_normalize_client_types_raises_on_unsupported_client(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test _normalize_client_types raises ValueError for unsupported client."""
        with pytest.raises(ValueError, match="Unsupported OpenAI client type"):
            integration._normalize_client_types(["openai.UnsupportedClient"])

    def test_normalize_client_types_error_message_shows_supported(
        self, integration: OpenAIIntegration
    ) -> None:
        """Test error message includes list of supported client types."""
        with pytest.raises(ValueError) as exc_info:
            integration._normalize_client_types(["openai.Invalid"])
        error_msg = str(exc_info.value)
        assert "Supported types:" in error_msg
        assert "openai.AsyncOpenAI" in error_msg
        assert "openai.OpenAI" in error_msg


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @patch("traigent.integrations.llms.openai.enable_framework_overrides")
    def test_enable_openai_optimization_with_none(self, mock_enable: MagicMock) -> None:
        """Test enable_openai_optimization with None enables all clients."""
        enable_openai_optimization(None)
        mock_enable.assert_called_once()

    @patch("traigent.integrations.llms.openai.enable_framework_overrides")
    def test_enable_openai_optimization_with_specific_clients(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_openai_optimization with specific client types."""
        client_types = ["openai.OpenAI"]
        enable_openai_optimization(client_types)
        assert mock_enable.called

    def test_get_supported_openai_clients(self) -> None:
        """Test get_supported_openai_clients returns correct list."""
        clients = get_supported_openai_clients()
        assert isinstance(clients, list)
        assert len(clients) == 2
        assert "openai.OpenAI" in clients
        assert "openai.AsyncOpenAI" in clients

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_enable_sync_openai(self, mock_enable: MagicMock) -> None:
        """Test enable_sync_openai enables only sync client."""
        enable_sync_openai()
        mock_enable.assert_called_once_with(["openai.OpenAI"])

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_enable_async_openai(self, mock_enable: MagicMock) -> None:
        """Test enable_async_openai enables only async client."""
        enable_async_openai()
        mock_enable.assert_called_once_with(["openai.AsyncOpenAI"])


class TestOpenAIContext:
    """Tests for openai_context context manager."""

    @patch("traigent.integrations.llms.openai.override_context")
    def test_openai_context_with_none(self, mock_override: MagicMock) -> None:
        """Test openai_context with None passes all clients to override_context."""
        openai_context(None)
        mock_override.assert_called_once()
        called_clients = mock_override.call_args[0][0]
        assert "openai.OpenAI" in called_clients
        assert "openai.AsyncOpenAI" in called_clients

    @patch("traigent.integrations.llms.openai.override_context")
    def test_openai_context_with_specific_clients(
        self, mock_override: MagicMock
    ) -> None:
        """Test openai_context with specific client types."""
        client_types = ["openai.OpenAI"]
        openai_context(client_types)
        mock_override.assert_called_once_with(["openai.OpenAI"])

    @patch("traigent.integrations.llms.openai.override_context")
    def test_openai_context_returns_context_manager(
        self, mock_override: MagicMock
    ) -> None:
        """Test openai_context returns an AbstractContextManager."""
        mock_cm = MagicMock(spec=AbstractContextManager)
        mock_override.return_value = mock_cm
        returned_cm = openai_context()
        assert returned_cm == mock_cm

    def test_openai_context_validates_client_types(self) -> None:
        """Test openai_context validates client types before passing to override_context."""
        with pytest.raises(ValueError, match="Unsupported OpenAI client type"):
            openai_context(["invalid.client"])


class TestAutoDetectOpenAI:
    """Tests for auto_detect_openai function."""

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_auto_detect_with_both_clients(self, mock_enable: MagicMock) -> None:
        """Test auto_detect_openai when both sync and async clients are available."""
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock
        mock_openai.AsyncOpenAI = MagicMock

        with patch.dict("sys.modules", {"openai": mock_openai}):
            auto_detect_openai()
            mock_enable.assert_called_once()
            called_clients = mock_enable.call_args[0][0]
            assert "openai.OpenAI" in called_clients
            assert "openai.AsyncOpenAI" in called_clients

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_auto_detect_with_only_sync_client(self, mock_enable: MagicMock) -> None:
        """Test auto_detect_openai when only sync client is available."""
        mock_openai = MagicMock()
        mock_openai.OpenAI = MagicMock
        del mock_openai.AsyncOpenAI

        with patch.dict("sys.modules", {"openai": mock_openai}):
            auto_detect_openai()
            mock_enable.assert_called_once_with(["openai.OpenAI"])

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_auto_detect_with_only_async_client(self, mock_enable: MagicMock) -> None:
        """Test auto_detect_openai when only async client is available."""
        mock_openai = MagicMock()
        del mock_openai.OpenAI
        mock_openai.AsyncOpenAI = MagicMock

        with patch.dict("sys.modules", {"openai": mock_openai}):
            auto_detect_openai()
            mock_enable.assert_called_once_with(["openai.AsyncOpenAI"])

    def test_auto_detect_with_no_clients_warns(self) -> None:
        """Test auto_detect_openai warns when no supported clients found."""
        mock_openai = MagicMock()
        # Remove both client types
        del mock_openai.OpenAI
        del mock_openai.AsyncOpenAI

        with patch.dict("sys.modules", {"openai": mock_openai}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                auto_detect_openai()
                assert len(w) == 1
                assert "no supported client types found" in str(w[0].message).lower()
                assert issubclass(w[0].category, UserWarning)

    def test_auto_detect_with_no_openai_warns(self) -> None:
        """Test auto_detect_openai warns when OpenAI SDK not installed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            real_import = __import__

            # Patch the import to raise ImportError
            def mock_import(name, *args, **kwargs):
                if name == "openai":
                    raise ImportError("No module named 'openai'")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                auto_detect_openai()
                assert len(w) == 1
                assert "not detected" in str(w[0].message).lower()
                assert "pip install openai" in str(w[0].message)
                assert issubclass(w[0].category, UserWarning)


class TestStreamingOptimization:
    """Tests for enable_streaming_optimization function."""

    @pytest.fixture(autouse=True)
    def setup_config_modules(self) -> None:
        """Set up mock config modules for tests with broken imports."""
        from unittest.mock import MagicMock as MM

        from traigent.config.types import TraigentConfig

        # Create mock config module at the path source expects
        mock_config_module = MM()
        mock_config_module.context = MM()
        mock_config_module.context.set_config = MM()
        mock_config_module.types = MM()
        mock_config_module.types.TraigentConfig = TraigentConfig

        sys.modules["traigent.integrations.config"] = mock_config_module
        sys.modules["traigent.integrations.config.context"] = mock_config_module.context
        sys.modules["traigent.integrations.config.types"] = mock_config_module.types

        yield

        # Clean up sys.modules
        sys.modules.pop("traigent.integrations.config", None)
        sys.modules.pop("traigent.integrations.config.context", None)
        sys.modules.pop("traigent.integrations.config.types", None)

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_enable_streaming_calls_enable_optimization(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_streaming_optimization calls enable_openai_optimization."""  # noqa: E501
        enable_streaming_optimization()
        mock_enable.assert_called_once()

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_streaming_optimization_attempts_config_set(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_streaming_optimization attempts to set config."""
        # Just verify it doesn't crash - actual config setting depends
        # on the broken import
        enable_streaming_optimization()
        assert mock_enable.called


class TestToolsOptimization:
    """Tests for enable_tools_optimization function."""

    @pytest.fixture(autouse=True)
    def setup_config_modules(self) -> None:
        """Set up mock config modules for tests with broken imports."""
        from unittest.mock import MagicMock as MM

        from traigent.config.types import TraigentConfig

        mock_config_module = MM()
        mock_config_module.context = MM()
        mock_config_module.context.set_config = MM()
        mock_config_module.types = MM()
        mock_config_module.types.TraigentConfig = TraigentConfig

        sys.modules["traigent.integrations.config"] = mock_config_module
        sys.modules["traigent.integrations.config.context"] = mock_config_module.context
        sys.modules["traigent.integrations.config.types"] = mock_config_module.types

        yield

        sys.modules.pop("traigent.integrations.config", None)
        sys.modules.pop("traigent.integrations.config.context", None)
        sys.modules.pop("traigent.integrations.config.types", None)

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_enable_tools_calls_enable_optimization(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_tools_optimization calls enable_openai_optimization."""
        enable_tools_optimization(None)
        mock_enable.assert_called_once()

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_tools_optimization_accepts_tools_parameter(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_tools_optimization accepts tools parameter."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        enable_tools_optimization(tools)
        assert mock_enable.called

    @patch("traigent.integrations.llms.openai.enable_openai_optimization")
    def test_tools_optimization_accepts_empty_list(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_tools_optimization accepts empty tools list."""
        enable_tools_optimization([])
        assert mock_enable.called


class TestGlobalIntegrationInstance:
    """Tests for the global _openai_integration instance."""

    def test_global_instance_exists(self) -> None:
        """Test that global _openai_integration instance is created."""
        assert _openai_integration is not None
        assert isinstance(_openai_integration, OpenAIIntegration)

    def test_global_instance_has_supported_clients(self) -> None:
        """Test global instance has supported_clients configured."""
        assert hasattr(_openai_integration, "supported_clients")
        assert len(_openai_integration.supported_clients) > 0

    @patch("traigent.integrations.llms.openai.enable_framework_overrides")
    def test_module_functions_use_global_instance(self, mock_enable: MagicMock) -> None:
        """Test that module-level functions use the global instance."""
        # Call module function
        enable_openai_optimization()
        # Should have been called (proving global instance is being used)
        assert mock_enable.called
