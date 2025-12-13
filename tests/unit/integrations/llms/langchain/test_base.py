"""Unit tests for LangChain base integration module.

Tests for LangChain integration functionality that enables zero-code-change
optimization of LangChain applications through automatic parameter override.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Patch imports before loading the module
sys.modules["traigent.integrations.llms.utils"] = Mock()
sys.modules["traigent.integrations.llms.utils.logging"] = Mock()
sys.modules["traigent.integrations.llms.langchain.framework_override"] = Mock()


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies for the module."""
    with (
        patch(
            "traigent.integrations.llms.langchain.base.get_logger"
        ) as mock_get_logger,
        patch(
            "traigent.integrations.llms.langchain.base.enable_framework_overrides"
        ) as mock_enable,
        patch(
            "traigent.integrations.llms.langchain.base.register_framework_mapping"
        ) as mock_register,
    ):
        mock_get_logger.return_value = MagicMock()
        yield {
            "get_logger": mock_get_logger,
            "enable_framework_overrides": mock_enable,
            "register_framework_mapping": mock_register,
        }


from traigent.integrations.llms.langchain.base import (  # noqa: E402
    LangChainIntegration,
    add_langchain_llm_mapping,
    auto_detect_langchain_llms,
    enable_anthropic_langchain,
    enable_chatgpt_optimization,
    enable_claude_optimization,
    enable_langchain_optimization,
    enable_openai_langchain,
    get_supported_langchain_llms,
)


class TestLangChainIntegration:
    """Tests for LangChainIntegration class."""

    @pytest.fixture
    def integration(self) -> LangChainIntegration:
        """Create LangChainIntegration instance for testing."""
        with patch(
            "traigent.integrations.llms.langchain.base.register_framework_mapping"
        ):
            return LangChainIntegration()

    # Initialization tests
    def test_init_creates_supported_llms_dict(
        self, integration: LangChainIntegration
    ) -> None:
        """Test initialization creates supported_llms dictionary."""
        assert hasattr(integration, "supported_llms")
        assert isinstance(integration.supported_llms, dict)
        assert len(integration.supported_llms) > 0

    def test_init_includes_openai_llms(self, integration: LangChainIntegration) -> None:
        """Test initialization includes OpenAI LangChain LLMs."""
        assert "langchain_openai.ChatOpenAI" in integration.supported_llms
        assert "langchain_openai.OpenAI" in integration.supported_llms
        assert "langchain.llms.OpenAI" in integration.supported_llms

    def test_init_includes_anthropic_llms(
        self, integration: LangChainIntegration
    ) -> None:
        """Test initialization includes Anthropic LangChain LLMs."""
        assert "langchain_anthropic.ChatAnthropic" in integration.supported_llms
        assert "langchain.llms.Anthropic" in integration.supported_llms

    def test_init_mappings_have_required_parameters(
        self, integration: LangChainIntegration
    ) -> None:
        """Test that LLM mappings include required parameters."""
        for llm_class, mappings in integration.supported_llms.items():
            assert isinstance(mappings, dict), f"{llm_class} mappings should be a dict"
            # All LLMs should have model and temperature mappings
            assert "model" in mappings, f"{llm_class} missing model mapping"
            assert "temperature" in mappings, f"{llm_class} missing temperature mapping"

    def test_init_calls_register_mappings(self) -> None:
        """Test initialization calls _register_mappings method."""
        with patch.object(LangChainIntegration, "_register_mappings") as mock_register:
            with patch(
                "traigent.integrations.llms.langchain.base.register_framework_mapping"
            ):
                _ = LangChainIntegration()
                mock_register.assert_called_once()

    # _register_mappings tests
    @patch("traigent.integrations.llms.langchain.base.register_framework_mapping")
    def test_register_mappings_registers_all_llms(
        self, mock_register: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test _register_mappings registers all supported LLMs."""
        # Call the method
        integration._register_mappings()

        # Verify register_framework_mapping was called for each LLM
        expected_calls = len(integration.supported_llms)
        assert mock_register.call_count >= expected_calls

    @patch("traigent.integrations.llms.langchain.base.register_framework_mapping")
    def test_register_mappings_calls_with_correct_arguments(
        self, mock_register: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test _register_mappings calls register_framework_mapping with correct args."""
        integration._register_mappings()

        # Check that each call has the correct structure
        for call_args in mock_register.call_args_list:
            llm_class, mappings = call_args[0]
            assert isinstance(llm_class, str)
            assert isinstance(mappings, dict)
            assert llm_class in integration.supported_llms

    # enable_langchain_overrides tests
    @patch("traigent.integrations.llms.langchain.base.enable_framework_overrides")
    def test_enable_langchain_overrides_with_none_enables_all(
        self, mock_enable: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test enable_langchain_overrides with None enables all LLMs."""
        integration.enable_langchain_overrides(None)

        # Should call with all supported LLMs
        mock_enable.assert_called_once()
        called_llms = mock_enable.call_args[0][0]
        assert set(called_llms) == set(integration.supported_llms.keys())

    @patch("traigent.integrations.llms.langchain.base.enable_framework_overrides")
    def test_enable_langchain_overrides_with_specific_llms(
        self, mock_enable: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test enable_langchain_overrides with specific LLM types."""
        llm_types = ["langchain_openai.ChatOpenAI", "langchain_anthropic.ChatAnthropic"]
        integration.enable_langchain_overrides(llm_types)

        mock_enable.assert_called_once_with(llm_types)

    @patch("traigent.integrations.llms.langchain.base.enable_framework_overrides")
    @patch("traigent.integrations.llms.langchain.base.logger")
    def test_enable_langchain_overrides_logs_info(
        self,
        mock_logger: MagicMock,
        mock_enable: MagicMock,
        integration: LangChainIntegration,
    ) -> None:
        """Test enable_langchain_overrides logs information message."""
        llm_types = ["langchain_openai.ChatOpenAI"]
        integration.enable_langchain_overrides(llm_types)

        # Verify logger.info was called
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "LangChain overrides enabled" in log_message
        assert "langchain_openai.ChatOpenAI" in log_message

    # get_supported_llms tests
    def test_get_supported_llms_returns_list(
        self, integration: LangChainIntegration
    ) -> None:
        """Test get_supported_llms returns a list."""
        result = integration.get_supported_llms()
        assert isinstance(result, list)

    def test_get_supported_llms_contains_all_llms(
        self, integration: LangChainIntegration
    ) -> None:
        """Test get_supported_llms returns all supported LLM types."""
        result = integration.get_supported_llms()
        assert set(result) == set(integration.supported_llms.keys())

    def test_get_supported_llms_includes_expected_types(
        self, integration: LangChainIntegration
    ) -> None:
        """Test get_supported_llms includes expected LLM types."""
        result = integration.get_supported_llms()
        assert "langchain_openai.ChatOpenAI" in result
        assert "langchain_anthropic.ChatAnthropic" in result

    # add_custom_llm_mapping tests
    @patch("traigent.integrations.llms.langchain.base.register_framework_mapping")
    def test_add_custom_llm_mapping_adds_to_supported(
        self, mock_register: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test add_custom_llm_mapping adds new LLM to supported_llms."""
        custom_class = "custom_package.CustomLLM"
        custom_mapping = {
            "model": "model_name",
            "temperature": "temp",
            "max_tokens": "max_length",
        }

        integration.add_custom_llm_mapping(custom_class, custom_mapping)

        assert custom_class in integration.supported_llms
        assert integration.supported_llms[custom_class] == custom_mapping

    @patch("traigent.integrations.llms.langchain.base.register_framework_mapping")
    def test_add_custom_llm_mapping_calls_register(
        self, mock_register: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test add_custom_llm_mapping calls register_framework_mapping."""
        custom_class = "custom_package.CustomLLM"
        custom_mapping = {"model": "model_name"}

        integration.add_custom_llm_mapping(custom_class, custom_mapping)

        mock_register.assert_called_with(custom_class, custom_mapping)

    @patch("traigent.integrations.llms.langchain.base.register_framework_mapping")
    @patch("traigent.integrations.llms.langchain.base.logger")
    def test_add_custom_llm_mapping_logs_info(
        self,
        mock_logger: MagicMock,
        mock_register: MagicMock,
        integration: LangChainIntegration,
    ) -> None:
        """Test add_custom_llm_mapping logs information message."""
        custom_class = "custom_package.CustomLLM"
        custom_mapping = {"model": "model_name"}

        integration.add_custom_llm_mapping(custom_class, custom_mapping)

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Added custom LangChain LLM mapping" in log_message
        assert custom_class in log_message


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    # enable_langchain_optimization tests
    @patch("traigent.integrations.llms.langchain.base._langchain_integration")
    def test_enable_langchain_optimization_with_none(
        self, mock_integration: MagicMock
    ) -> None:
        """Test enable_langchain_optimization with None parameter."""
        enable_langchain_optimization(None)
        mock_integration.enable_langchain_overrides.assert_called_once_with(None)

    @patch("traigent.integrations.llms.langchain.base._langchain_integration")
    def test_enable_langchain_optimization_with_specific_types(
        self, mock_integration: MagicMock
    ) -> None:
        """Test enable_langchain_optimization with specific LLM types."""
        llm_types = ["langchain_openai.ChatOpenAI"]
        enable_langchain_optimization(llm_types)
        mock_integration.enable_langchain_overrides.assert_called_once_with(llm_types)

    # get_supported_langchain_llms tests
    @patch("traigent.integrations.llms.langchain.base._langchain_integration")
    def test_get_supported_langchain_llms_delegates_to_integration(
        self, mock_integration: MagicMock
    ) -> None:
        """Test get_supported_langchain_llms delegates to integration instance."""
        mock_integration.get_supported_llms.return_value = ["llm1", "llm2"]

        result = get_supported_langchain_llms()

        mock_integration.get_supported_llms.assert_called_once()
        assert result == ["llm1", "llm2"]

    # add_langchain_llm_mapping tests
    @patch("traigent.integrations.llms.langchain.base._langchain_integration")
    def test_add_langchain_llm_mapping_delegates_to_integration(
        self, mock_integration: MagicMock
    ) -> None:
        """Test add_langchain_llm_mapping delegates to integration instance."""
        llm_class = "custom.LLM"
        mapping = {"model": "model_name"}

        add_langchain_llm_mapping(llm_class, mapping)

        mock_integration.add_custom_llm_mapping.assert_called_once_with(
            llm_class, mapping
        )

    # enable_chatgpt_optimization tests
    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_enable_chatgpt_optimization_enables_openai_llms(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_chatgpt_optimization enables OpenAI LLM types."""
        enable_chatgpt_optimization()

        mock_enable.assert_called_once()
        llm_types = mock_enable.call_args[0][0]
        assert "langchain_openai.ChatOpenAI" in llm_types
        assert "langchain_openai.OpenAI" in llm_types
        assert "langchain.llms.OpenAI" in llm_types

    # enable_claude_optimization tests
    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_enable_claude_optimization_enables_anthropic_llms(
        self, mock_enable: MagicMock
    ) -> None:
        """Test enable_claude_optimization enables Anthropic LLM types."""
        enable_claude_optimization()

        mock_enable.assert_called_once()
        llm_types = mock_enable.call_args[0][0]
        assert "langchain_anthropic.ChatAnthropic" in llm_types
        assert "langchain.llms.Anthropic" in llm_types

    # Alias function tests
    @patch("traigent.integrations.llms.langchain.base.enable_chatgpt_optimization")
    def test_enable_openai_langchain_is_alias(self, mock_chatgpt: MagicMock) -> None:
        """Test enable_openai_langchain is an alias for enable_chatgpt_optimization."""
        enable_openai_langchain()
        mock_chatgpt.assert_called_once()

    @patch("traigent.integrations.llms.langchain.base.enable_claude_optimization")
    def test_enable_anthropic_langchain_is_alias(self, mock_claude: MagicMock) -> None:
        """Test enable_anthropic_langchain is an alias for enable_claude_optimization."""
        enable_anthropic_langchain()
        mock_claude.assert_called_once()


class TestAutoDetection:
    """Tests for auto-detection functionality."""

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_auto_detect_with_langchain_openai(self, mock_enable: MagicMock) -> None:
        """Test auto_detect_langchain_llms detects langchain_openai."""
        # Mock successful import of langchain_openai
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_openai": mock_module}):
            auto_detect_langchain_llms()

            # Should enable optimization for detected LLMs
            mock_enable.assert_called_once()
            enabled_llms = mock_enable.call_args[0][0]
            assert "langchain_openai.ChatOpenAI" in enabled_llms
            assert "langchain_openai.OpenAI" in enabled_llms

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_auto_detect_with_langchain_anthropic(self, mock_enable: MagicMock) -> None:
        """Test auto_detect_langchain_llms detects langchain_anthropic."""
        # Mock successful import of langchain_anthropic
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"langchain_anthropic": mock_module}):
            auto_detect_langchain_llms()

            mock_enable.assert_called_once()
            enabled_llms = mock_enable.call_args[0][0]
            assert "langchain_anthropic.ChatAnthropic" in enabled_llms

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_auto_detect_with_legacy_langchain_llms(
        self, mock_enable: MagicMock
    ) -> None:
        """Test auto_detect_langchain_llms detects legacy langchain.llms."""
        # Mock successful import of langchain.llms
        mock_llms_module = MagicMock()
        with patch.dict("sys.modules", {"langchain.llms": mock_llms_module}):
            auto_detect_langchain_llms()

            mock_enable.assert_called_once()
            enabled_llms = mock_enable.call_args[0][0]
            # Should include legacy OpenAI and/or Anthropic
            assert any("langchain.llms" in llm for llm in enabled_llms)

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_auto_detect_with_multiple_providers(self, mock_enable: MagicMock) -> None:
        """Test auto_detect_langchain_llms with multiple providers available."""
        # Mock multiple LangChain packages
        with patch.dict(
            "sys.modules",
            {
                "langchain_openai": MagicMock(),
                "langchain_anthropic": MagicMock(),
                "langchain.llms": MagicMock(),
            },
        ):
            auto_detect_langchain_llms()

            mock_enable.assert_called_once()
            enabled_llms = mock_enable.call_args[0][0]
            # Should detect all available providers
            assert len(enabled_llms) > 0
            assert any("openai" in llm.lower() for llm in enabled_llms)
            assert any("anthropic" in llm.lower() for llm in enabled_llms)

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    @patch("traigent.integrations.llms.langchain.base.warnings.warn")
    def test_auto_detect_with_no_langchain_installed(
        self, mock_warn: MagicMock, mock_enable: MagicMock
    ) -> None:
        """Test auto_detect_langchain_llms warns when no LangChain packages found."""

        # Mock all imports to fail
        def mock_import(name: str, *args, **kwargs):
            if "langchain" in name:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()

        with patch("builtins.__import__", side_effect=mock_import):
            auto_detect_langchain_llms()

            # Should issue a warning
            mock_warn.assert_called_once()
            warning_message = mock_warn.call_args[0][0]
            assert "No LangChain LLMs detected" in warning_message

            # Should not call enable_langchain_optimization
            mock_enable.assert_not_called()

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    @patch("traigent.integrations.llms.langchain.base.logger")
    def test_auto_detect_logs_detected_llms(
        self, mock_logger: MagicMock, mock_enable: MagicMock
    ) -> None:
        """Test auto_detect_langchain_llms logs number of detected LLMs."""
        # Mock successful detection
        with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
            auto_detect_langchain_llms()

            # Should log info about detected LLMs
            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args[0][0]
            assert "Auto-detected" in log_message
            assert "LangChain LLMs" in log_message


class TestParameterMappings:
    """Tests for parameter mapping configurations."""

    @pytest.fixture
    def integration(self) -> LangChainIntegration:
        """Create integration instance for mapping tests."""
        with patch(
            "traigent.integrations.llms.langchain.base.register_framework_mapping"
        ):
            return LangChainIntegration()

    def test_chatopenai_mapping_includes_streaming(
        self, integration: LangChainIntegration
    ) -> None:
        """Test ChatOpenAI mapping includes streaming parameters."""
        mapping = integration.supported_llms["langchain_openai.ChatOpenAI"]
        assert "streaming" in mapping
        # Should support both 'streaming' and 'stream' aliases
        assert mapping["streaming"] == "streaming"

    def test_chatopenai_mapping_includes_penalties(
        self, integration: LangChainIntegration
    ) -> None:
        """Test ChatOpenAI mapping includes frequency and presence penalties."""
        mapping = integration.supported_llms["langchain_openai.ChatOpenAI"]
        assert "frequency_penalty" in mapping
        assert "presence_penalty" in mapping

    def test_chatanthropic_mapping_includes_top_k(
        self, integration: LangChainIntegration
    ) -> None:
        """Test ChatAnthropic mapping includes top_k parameter."""
        mapping = integration.supported_llms["langchain_anthropic.ChatAnthropic"]
        assert "top_k" in mapping

    def test_chatanthropic_mapping_includes_stop_sequences(
        self, integration: LangChainIntegration
    ) -> None:
        """Test ChatAnthropic mapping includes stop_sequences parameter."""
        mapping = integration.supported_llms["langchain_anthropic.ChatAnthropic"]
        assert "stop_sequences" in mapping
        # Should map to 'stop' in LangChain
        assert mapping["stop_sequences"] == "stop"

    def test_legacy_openai_uses_model_name(
        self, integration: LangChainIntegration
    ) -> None:
        """Test legacy OpenAI LLM uses model_name instead of model."""
        mapping = integration.supported_llms["langchain.llms.OpenAI"]
        # Legacy LangChain used model_name
        assert "model" in mapping
        assert mapping["model"] == "model_name"

    def test_legacy_anthropic_uses_max_tokens_to_sample(
        self, integration: LangChainIntegration
    ) -> None:
        """Test legacy Anthropic LLM uses max_tokens_to_sample."""
        mapping = integration.supported_llms["langchain.llms.Anthropic"]
        assert "max_tokens" in mapping
        assert mapping["max_tokens"] == "max_tokens_to_sample"

    def test_all_mappings_have_consistent_structure(
        self, integration: LangChainIntegration
    ) -> None:
        """Test all mappings have consistent structure."""
        for llm_class, mapping in integration.supported_llms.items():
            # All should be dicts
            assert isinstance(mapping, dict), f"{llm_class} mapping is not a dict"
            # All values should be strings (parameter names)
            for key, value in mapping.items():
                assert isinstance(key, str), f"{llm_class} key {key} is not string"
                assert isinstance(
                    value, str
                ), f"{llm_class} value {value} is not string"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def integration(self) -> LangChainIntegration:
        """Create integration instance for edge case tests."""
        with patch(
            "traigent.integrations.llms.langchain.base.register_framework_mapping"
        ):
            return LangChainIntegration()

    def test_enable_with_empty_list(self, integration: LangChainIntegration) -> None:
        """Test enable_langchain_overrides with empty list."""
        with patch(
            "traigent.integrations.llms.langchain.base.enable_framework_overrides"
        ) as mock:
            integration.enable_langchain_overrides([])
            mock.assert_called_once_with([])

    def test_add_custom_mapping_overwrites_existing(
        self, integration: LangChainIntegration
    ) -> None:
        """Test adding custom mapping for existing LLM overwrites it."""
        original_mapping = integration.supported_llms[
            "langchain_openai.ChatOpenAI"
        ].copy()
        new_mapping = {"model": "custom_model_param"}

        with patch(
            "traigent.integrations.llms.langchain.base.register_framework_mapping"
        ):
            integration.add_custom_llm_mapping(
                "langchain_openai.ChatOpenAI", new_mapping
            )

        # Should overwrite
        assert integration.supported_llms["langchain_openai.ChatOpenAI"] == new_mapping
        assert (
            integration.supported_llms["langchain_openai.ChatOpenAI"]
            != original_mapping
        )

    def test_get_supported_llms_returns_copy(
        self, integration: LangChainIntegration
    ) -> None:
        """Test get_supported_llms returns list that can be modified safely."""
        result1 = integration.get_supported_llms()
        result2 = integration.get_supported_llms()

        # Modifying one shouldn't affect the other
        result1.append("fake_llm")
        assert "fake_llm" not in result2

    @patch("traigent.integrations.llms.langchain.base.enable_langchain_optimization")
    def test_auto_detect_handles_partial_import_failures(
        self, mock_enable: MagicMock
    ) -> None:
        """Test auto_detect handles when some imports fail but others succeed."""
        # Mock scenario where only langchain_openai is available
        mock_openai = MagicMock()

        def mock_import(name: str, *args, **kwargs):
            if name == "langchain_openai":
                return mock_openai
            elif "langchain" in name:
                raise ImportError(f"No module named '{name}'")
            # Let other imports work normally
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            auto_detect_langchain_llms()

            # Should still enable for available LLMs
            mock_enable.assert_called_once()
            enabled = mock_enable.call_args[0][0]
            assert any("openai" in llm.lower() for llm in enabled)

    def test_integration_instance_is_reusable(self) -> None:
        """Test that LangChainIntegration instance can be used multiple times."""
        with patch(
            "traigent.integrations.llms.langchain.base.register_framework_mapping"
        ):
            integration = LangChainIntegration()

            # Call methods multiple times
            with patch(
                "traigent.integrations.llms.langchain.base.enable_framework_overrides"
            ):
                integration.enable_langchain_overrides(None)
                integration.enable_langchain_overrides(["langchain_openai.ChatOpenAI"])

            result1 = integration.get_supported_llms()
            result2 = integration.get_supported_llms()

            # Should return consistent results
            assert result1 == result2

    @patch("traigent.integrations.llms.langchain.base.register_framework_mapping")
    def test_add_custom_mapping_with_special_characters(
        self, mock_register: MagicMock, integration: LangChainIntegration
    ) -> None:
        """Test adding custom mapping with class names containing special characters."""
        # Some package names might have underscores, numbers, etc.
        custom_class = "my_package_v2.CustomLLM_2024"
        custom_mapping = {"model": "model_param"}

        integration.add_custom_llm_mapping(custom_class, custom_mapping)

        assert custom_class in integration.supported_llms
        mock_register.assert_called_with(custom_class, custom_mapping)
