"""Unit tests for LangChain discovery module.

Tests for auto-discovery utilities that find LangChain components
that can be optimized by Traigent.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility
# Traceability: FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traigent.integrations.llms.langchain.discovery import (
    LangChainDiscovery,
)


class TestLangChainDiscovery:
    """Tests for LangChainDiscovery class."""

    @pytest.fixture
    def discovery(self) -> LangChainDiscovery:
        """Create LangChainDiscovery instance."""
        return LangChainDiscovery()

    @pytest.fixture
    def mock_base_llm_class(self) -> type:
        """Create mock BaseLLM class."""

        class MockBaseLLM:
            """Mock base LLM class."""

            pass

        return MockBaseLLM

    @pytest.fixture
    def mock_base_chat_model_class(self) -> type:
        """Create mock BaseChatModel class."""

        class MockBaseChatModel:
            """Mock base chat model class."""

            pass

        return MockBaseChatModel

    @pytest.fixture
    def mock_chain_class(self) -> type:
        """Create mock Chain class."""

        class MockChain:
            """Mock chain class."""

            pass

        return MockChain

    @pytest.fixture
    def mock_concrete_llm(self, mock_base_llm_class: type) -> type:
        """Create mock concrete LLM class."""

        class MockConcreteLLM(mock_base_llm_class):
            """Mock concrete LLM implementation."""

            def __init__(self, model: str = "test-model") -> None:
                """Initialize mock LLM."""
                self.model = model

        return MockConcreteLLM

    @pytest.fixture
    def mock_concrete_chat_model(self, mock_base_chat_model_class: type) -> type:
        """Create mock concrete chat model class."""

        class MockConcreteChatModel(mock_base_chat_model_class):
            """Mock concrete chat model implementation."""

            def __init__(self, model_name: str = "test-chat") -> None:
                """Initialize mock chat model."""
                self.model_name = model_name

        return MockConcreteChatModel

    @pytest.fixture
    def mock_module_with_llm(self, mock_base_llm_class: type) -> ModuleType:
        """Create mock module containing an LLM class."""

        class TestLLM(mock_base_llm_class):
            """Test LLM class."""

            pass

        module = ModuleType("test_module")
        module.TestLLM = TestLLM
        module._PrivateClass = type("_PrivateClass", (), {})
        return module

    @pytest.fixture
    def mock_module_with_chat_model(self, mock_concrete_chat_model: type) -> ModuleType:
        """Create mock module containing a chat model class."""
        module = ModuleType("test_module")
        module.TestChatModel = mock_concrete_chat_model
        module._PrivateClass = type("_PrivateClass", (), {})
        return module

    # Test initialization
    def test_init_creates_empty_cache(self, discovery: LangChainDiscovery) -> None:
        """Test that initialization creates an empty discovery cache."""
        assert discovery._discovered_cache == {}

    def test_langchain_packages_list_is_defined(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test that LANGCHAIN_PACKAGES constant is properly defined."""
        assert isinstance(discovery.LANGCHAIN_PACKAGES, list)
        assert len(discovery.LANGCHAIN_PACKAGES) > 0
        assert "langchain" in discovery.LANGCHAIN_PACKAGES
        assert "langchain_openai" in discovery.LANGCHAIN_PACKAGES

    # Test discover_all_llms
    @patch.object(LangChainDiscovery, "_get_base_llm_class")
    @patch.object(LangChainDiscovery, "_scan_package_for_llms")
    def test_discover_all_llms_success(
        self,
        mock_scan: MagicMock,
        mock_get_base: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_llm_class: type,
        mock_concrete_llm: type,
    ) -> None:
        """Test successful discovery of all LLM classes."""
        mock_get_base.return_value = mock_base_llm_class
        mock_scan.return_value = [mock_concrete_llm]

        result = discovery.discover_all_llms()

        assert len(result) == len(discovery.LANGCHAIN_PACKAGES)
        assert mock_concrete_llm in result
        mock_get_base.assert_called_once()
        assert mock_scan.call_count == len(discovery.LANGCHAIN_PACKAGES)

    @patch.object(LangChainDiscovery, "_get_base_llm_class")
    def test_discover_all_llms_returns_cached_result(
        self,
        mock_get_base: MagicMock,
        discovery: LangChainDiscovery,
        mock_concrete_llm: type,
    ) -> None:
        """Test that discover_all_llms returns cached result on second call."""
        discovery._discovered_cache["llms"] = [mock_concrete_llm]

        result = discovery.discover_all_llms()

        assert result == [mock_concrete_llm]
        mock_get_base.assert_not_called()

    @patch.object(LangChainDiscovery, "_get_base_llm_class")
    def test_discover_all_llms_when_base_class_not_found(
        self, mock_get_base: MagicMock, discovery: LangChainDiscovery
    ) -> None:
        """Test discover_all_llms returns empty list when BaseLLM not found."""
        mock_get_base.return_value = None

        result = discovery.discover_all_llms()

        assert result == []
        mock_get_base.assert_called_once()

    # Test discover_chat_models
    @patch.object(LangChainDiscovery, "_get_base_chat_model_class")
    @patch.object(LangChainDiscovery, "_scan_package_for_chat_models")
    def test_discover_chat_models_success(
        self,
        mock_scan: MagicMock,
        mock_get_base: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_chat_model_class: type,
        mock_concrete_chat_model: type,
    ) -> None:
        """Test successful discovery of all chat model classes."""
        mock_get_base.return_value = mock_base_chat_model_class
        mock_scan.return_value = [mock_concrete_chat_model]

        result = discovery.discover_chat_models()

        assert len(result) == len(discovery.LANGCHAIN_PACKAGES)
        assert mock_concrete_chat_model in result
        mock_get_base.assert_called_once()
        assert mock_scan.call_count == len(discovery.LANGCHAIN_PACKAGES)

    @patch.object(LangChainDiscovery, "_get_base_chat_model_class")
    def test_discover_chat_models_returns_cached_result(
        self,
        mock_get_base: MagicMock,
        discovery: LangChainDiscovery,
        mock_concrete_chat_model: type,
    ) -> None:
        """Test that discover_chat_models returns cached result on second call."""
        discovery._discovered_cache["chat_models"] = [mock_concrete_chat_model]

        result = discovery.discover_chat_models()

        assert result == [mock_concrete_chat_model]
        mock_get_base.assert_not_called()

    @patch.object(LangChainDiscovery, "_get_base_chat_model_class")
    def test_discover_chat_models_when_base_class_not_found(
        self, mock_get_base: MagicMock, discovery: LangChainDiscovery
    ) -> None:
        """Test discover_chat_models returns empty list when BaseChatModel not found."""
        mock_get_base.return_value = None

        result = discovery.discover_chat_models()

        assert result == []
        mock_get_base.assert_called_once()

    # Test discover_chains
    @patch.object(LangChainDiscovery, "_scan_module_for_subclasses")
    def test_discover_chains_success(
        self,
        mock_scan: MagicMock,
        discovery: LangChainDiscovery,
        mock_chain_class: type,
    ) -> None:
        """Test successful discovery of chain classes."""
        mock_chain_module = ModuleType("langchain.chains.base")
        mock_chain_module.Chain = mock_chain_class

        import sys

        sys.modules["langchain.chains.base"] = mock_chain_module
        mock_scan.return_value = [mock_chain_class]

        try:
            with patch(
                "importlib.import_module",
                side_effect=lambda x: (
                    mock_chain_module if "base" in x else ModuleType(x)
                ),
            ):
                result = discovery.discover_chains()
                assert len(result) > 0
        finally:
            sys.modules.pop("langchain.chains.base", None)

    def test_discover_chains_returns_cached_result(
        self, discovery: LangChainDiscovery, mock_chain_class: type
    ) -> None:
        """Test that discover_chains returns cached result on second call."""
        discovery._discovered_cache["chains"] = [mock_chain_class]

        result = discovery.discover_chains()

        assert result == [mock_chain_class]

    def test_discover_chains_when_chain_import_fails(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test discover_chains returns empty list when Chain class import fails."""
        with patch(
            "traigent.integrations.llms.langchain.discovery.importlib.import_module",
            side_effect=ImportError("Chain not found"),
        ):
            # Make sure the import statement in discover_chains fails
            import sys

            sys.modules.pop("langchain.chains.base", None)
            result = discovery.discover_chains()
            assert result == []

    # Test _get_base_llm_class
    def test_get_base_llm_class_from_langchain_llms_base(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test _get_base_llm_class returns BaseLLM from langchain.llms.base."""
        mock_base_llm = type("BaseLLM", (), {})
        mock_module = ModuleType("langchain.llms.base")
        mock_module.BaseLLM = mock_base_llm

        import sys

        sys.modules["langchain.llms.base"] = mock_module
        try:
            result = discovery._get_base_llm_class()
            assert result == mock_base_llm
        finally:
            sys.modules.pop("langchain.llms.base", None)

    def test_get_base_llm_class_from_langchain_core(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test _get_base_llm_class returns BaseLLM from langchain_core."""
        mock_base_llm = type("BaseLLM", (), {})
        mock_module = ModuleType("langchain_core.language_models.llms")
        mock_module.BaseLLM = mock_base_llm

        import sys

        # Make first import fail, second succeed
        sys.modules["langchain.llms.base"] = None  # type: ignore
        sys.modules["langchain_core.language_models.llms"] = mock_module

        with patch(
            "traigent.integrations.llms.langchain.discovery.importlib.import_module",
            side_effect=[
                ImportError("Not found"),
                mock_module,
            ],
        ):
            try:
                result = discovery._get_base_llm_class()
                assert result == mock_base_llm
            finally:
                sys.modules.pop("langchain.llms.base", None)
                sys.modules.pop("langchain_core.language_models.llms", None)

    def test_get_base_llm_class_returns_none_when_not_found(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test _get_base_llm_class returns None when BaseLLM not available."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> None:
            raise ImportError(f"No module named '{name}'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = discovery._get_base_llm_class()
            assert result is None

    # Test _get_base_chat_model_class
    def test_get_base_chat_model_class_from_langchain_chat_models_base(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test _get_base_chat_model_class returns BaseChatModel from langchain.chat_models.base."""
        mock_base_chat = type("BaseChatModel", (), {})
        mock_module = ModuleType("langchain.chat_models.base")
        mock_module.BaseChatModel = mock_base_chat

        import sys

        sys.modules["langchain.chat_models.base"] = mock_module
        try:
            result = discovery._get_base_chat_model_class()
            assert result == mock_base_chat
        finally:
            sys.modules.pop("langchain.chat_models.base", None)

    def test_get_base_chat_model_class_from_langchain_core(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test _get_base_chat_model_class returns BaseChatModel from langchain_core."""
        mock_base_chat = type("BaseChatModel", (), {})
        mock_module = ModuleType("langchain_core.language_models.chat_models")
        mock_module.BaseChatModel = mock_base_chat

        import sys

        # Make first import fail, second succeed
        sys.modules["langchain.chat_models.base"] = None  # type: ignore
        sys.modules["langchain_core.language_models.chat_models"] = mock_module

        with patch(
            "traigent.integrations.llms.langchain.discovery.importlib.import_module",
            side_effect=[
                ImportError("Not found"),
                mock_module,
            ],
        ):
            try:
                result = discovery._get_base_chat_model_class()
                assert result == mock_base_chat
            finally:
                sys.modules.pop("langchain.chat_models.base", None)
                sys.modules.pop("langchain_core.language_models.chat_models", None)

    def test_get_base_chat_model_class_returns_none_when_not_found(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test _get_base_chat_model_class returns None when not available."""

        def mock_import(name: str, *args: Any, **kwargs: Any) -> None:
            raise ImportError(f"No module named '{name}'")

        with patch("builtins.__import__", side_effect=mock_import):
            result = discovery._get_base_chat_model_class()
            assert result is None

    # Test _scan_package_for_llms
    @patch("importlib.import_module")
    @patch.object(LangChainDiscovery, "_scan_module_for_subclasses")
    def test_scan_package_for_llms_success(
        self,
        mock_scan_module: MagicMock,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_llm_class: type,
        mock_concrete_llm: type,
    ) -> None:
        """Test _scan_package_for_llms successfully finds LLM classes."""
        mock_module = ModuleType("test_package.llms")
        # First import succeeds, second fails
        mock_import.side_effect = [mock_module, ImportError()]
        mock_scan_module.return_value = [mock_concrete_llm]

        result = discovery._scan_package_for_llms("test_package", mock_base_llm_class)

        assert mock_concrete_llm in result
        assert len(result) == 1
        mock_import.assert_called()

    @patch("importlib.import_module")
    @patch.object(LangChainDiscovery, "_scan_module_for_subclasses")
    def test_scan_package_for_llms_tries_multiple_paths(
        self,
        mock_scan_module: MagicMock,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_llm_class: type,
        mock_concrete_llm: type,
    ) -> None:
        """Test _scan_package_for_llms tries both package.llms and package paths."""
        # First path fails, second succeeds
        mock_import.side_effect = [
            ImportError("Not found"),
            ModuleType("test_package"),
        ]
        mock_scan_module.return_value = [mock_concrete_llm]

        result = discovery._scan_package_for_llms("test_package", mock_base_llm_class)

        assert mock_concrete_llm in result
        assert mock_import.call_count == 2

    @patch("importlib.import_module")
    def test_scan_package_for_llms_handles_import_error(
        self,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_llm_class: type,
    ) -> None:
        """Test _scan_package_for_llms returns empty list on import error."""
        mock_import.side_effect = ImportError("Package not found")

        result = discovery._scan_package_for_llms("nonexistent", mock_base_llm_class)

        assert result == []

    @patch("importlib.import_module")
    def test_scan_package_for_llms_handles_general_exception(
        self,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_llm_class: type,
    ) -> None:
        """Test _scan_package_for_llms handles general exceptions gracefully."""
        mock_import.side_effect = RuntimeError("Unexpected error")

        result = discovery._scan_package_for_llms("error_package", mock_base_llm_class)

        assert result == []

    # Test _scan_package_for_chat_models
    @patch("importlib.import_module")
    @patch.object(LangChainDiscovery, "_scan_module_for_subclasses")
    def test_scan_package_for_chat_models_success(
        self,
        mock_scan_module: MagicMock,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_chat_model_class: type,
        mock_concrete_chat_model: type,
    ) -> None:
        """Test _scan_package_for_chat_models finds chat models."""
        mock_module = ModuleType("test_package.chat_models")
        # First import succeeds, second fails
        mock_import.side_effect = [mock_module, ImportError()]
        mock_scan_module.return_value = [mock_concrete_chat_model]

        result = discovery._scan_package_for_chat_models(
            "test_package", mock_base_chat_model_class
        )

        assert mock_concrete_chat_model in result
        assert len(result) == 1
        mock_import.assert_called()

    @patch("importlib.import_module")
    @patch.object(LangChainDiscovery, "_scan_module_for_subclasses")
    def test_scan_package_for_chat_models_tries_multiple_paths(
        self,
        mock_scan_module: MagicMock,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_chat_model_class: type,
        mock_concrete_chat_model: type,
    ) -> None:
        """Test _scan_package_for_chat_models tries both package.chat_models and package paths."""
        # First path fails, second succeeds
        mock_import.side_effect = [
            ImportError("Not found"),
            ModuleType("test_package"),
        ]
        mock_scan_module.return_value = [mock_concrete_chat_model]

        result = discovery._scan_package_for_chat_models(
            "test_package", mock_base_chat_model_class
        )

        assert mock_concrete_chat_model in result
        assert mock_import.call_count == 2

    @patch("importlib.import_module")
    def test_scan_package_for_chat_models_handles_import_error(
        self,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_chat_model_class: type,
    ) -> None:
        """Test _scan_package_for_chat_models returns empty list on import error."""
        mock_import.side_effect = ImportError("Package not found")

        result = discovery._scan_package_for_chat_models(
            "nonexistent", mock_base_chat_model_class
        )

        assert result == []

    @patch("importlib.import_module")
    def test_scan_package_for_chat_models_handles_general_exception(
        self,
        mock_import: MagicMock,
        discovery: LangChainDiscovery,
        mock_base_chat_model_class: type,
    ) -> None:
        """Test _scan_package_for_chat_models handles general exceptions gracefully."""
        mock_import.side_effect = RuntimeError("Unexpected error")

        result = discovery._scan_package_for_chat_models(
            "error_package", mock_base_chat_model_class
        )

        assert result == []

    # Test _scan_module_for_subclasses
    def test_scan_module_for_subclasses_finds_valid_subclass(
        self,
        discovery: LangChainDiscovery,
        mock_module_with_llm: ModuleType,
        mock_base_llm_class: type,
    ) -> None:
        """Test _scan_module_for_subclasses finds valid subclasses."""
        result = discovery._scan_module_for_subclasses(
            mock_module_with_llm, mock_base_llm_class
        )

        assert len(result) == 1
        assert result[0].__name__ == "TestLLM"

    def test_scan_module_for_subclasses_skips_private_classes(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses skips classes starting with underscore."""
        module = ModuleType("test_module")

        class _PrivateClass(mock_base_llm_class):
            pass

        module._PrivateClass = _PrivateClass

        result = discovery._scan_module_for_subclasses(module, mock_base_llm_class)

        assert len(result) == 0

    def test_scan_module_for_subclasses_skips_base_class(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses skips the base class itself."""
        module = ModuleType("test_module")
        module.BaseLLM = mock_base_llm_class

        result = discovery._scan_module_for_subclasses(module, mock_base_llm_class)

        assert len(result) == 0

    def test_scan_module_for_subclasses_skips_abstract_classes(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses skips abstract classes."""
        from abc import ABC, abstractmethod

        module = ModuleType("test_module")

        class AbstractLLM(mock_base_llm_class, ABC):
            @abstractmethod
            def method(self) -> None:
                pass

        module.AbstractLLM = AbstractLLM

        result = discovery._scan_module_for_subclasses(module, mock_base_llm_class)

        assert len(result) == 0

    def test_scan_module_for_subclasses_handles_non_class_attributes(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses handles non-class module attributes."""
        module = ModuleType("test_module")
        module.some_function = lambda: None
        module.some_variable = "test"
        module.some_dict = {}

        result = discovery._scan_module_for_subclasses(module, mock_base_llm_class)

        assert result == []

    def test_scan_module_for_subclasses_handles_getattr_exception(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses handles exceptions during getattr."""
        module = ModuleType("test_module")

        # Create a property that raises an exception
        class BadClass:
            @property
            def bad_property(self) -> None:
                raise RuntimeError("Cannot access")

        # Add an attribute name that will cause getattr to fail
        original_getattr = getattr

        def mock_getattr(obj: Any, name: str) -> Any:
            if name == "BadAttribute":
                raise RuntimeError("Getattr failed")
            return original_getattr(obj, name)

        with patch("builtins.getattr", side_effect=mock_getattr):
            module.BadAttribute = "placeholder"
            result = discovery._scan_module_for_subclasses(module, mock_base_llm_class)
            assert isinstance(result, list)

    # Test create_universal_langchain_mapping
    def test_create_universal_langchain_mapping_returns_dict(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test create_universal_langchain_mapping returns a dictionary."""
        result = discovery.create_universal_langchain_mapping()

        assert isinstance(result, dict)

    def test_create_universal_langchain_mapping_contains_expected_keys(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test create_universal_langchain_mapping contains expected parameter keys."""
        result = discovery.create_universal_langchain_mapping()

        expected_keys = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stop",
            "streaming",
            "timeout",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "n",
        ]

        for key in expected_keys:
            assert key in result

    def test_create_universal_langchain_mapping_values_are_lists(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test create_universal_langchain_mapping values are lists of strings."""
        result = discovery.create_universal_langchain_mapping()

        for key, value in result.items():
            assert isinstance(value, list), f"{key} value is not a list"
            assert all(
                isinstance(item, str) for item in value
            ), f"{key} contains non-string items"
            assert len(value) > 0, f"{key} has empty list"

    def test_create_universal_langchain_mapping_model_aliases(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test model parameter has common aliases."""
        result = discovery.create_universal_langchain_mapping()

        assert "model" in result["model"]
        assert "model_name" in result["model"]
        assert "model_id" in result["model"]

    def test_create_universal_langchain_mapping_max_tokens_aliases(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test max_tokens parameter has various provider-specific aliases."""
        result = discovery.create_universal_langchain_mapping()

        assert "max_tokens" in result["max_tokens"]
        assert "max_length" in result["max_tokens"]
        assert "max_new_tokens" in result["max_tokens"]
        assert "max_tokens_to_sample" in result["max_tokens"]

    # Test get_class_info
    def test_get_class_info_returns_dict(
        self, discovery: LangChainDiscovery, mock_concrete_llm: type
    ) -> None:
        """Test get_class_info returns a dictionary."""
        result = discovery.get_class_info(mock_concrete_llm)

        assert isinstance(result, dict)

    def test_get_class_info_contains_required_keys(
        self, discovery: LangChainDiscovery, mock_concrete_llm: type
    ) -> None:
        """Test get_class_info returns dictionary with required keys."""
        result = discovery.get_class_info(mock_concrete_llm)

        assert "name" in result
        assert "module" in result
        assert "full_name" in result
        assert "init_params" in result
        assert "docstring" in result

    def test_get_class_info_extracts_class_name(
        self, discovery: LangChainDiscovery, mock_concrete_llm: type
    ) -> None:
        """Test get_class_info correctly extracts class name."""
        result = discovery.get_class_info(mock_concrete_llm)

        assert result["name"] == mock_concrete_llm.__name__

    def test_get_class_info_extracts_module_name(
        self, discovery: LangChainDiscovery, mock_concrete_llm: type
    ) -> None:
        """Test get_class_info correctly extracts module name."""
        result = discovery.get_class_info(mock_concrete_llm)

        assert result["module"] == mock_concrete_llm.__module__

    def test_get_class_info_constructs_full_name(
        self, discovery: LangChainDiscovery, mock_concrete_llm: type
    ) -> None:
        """Test get_class_info constructs correct full class name."""
        result = discovery.get_class_info(mock_concrete_llm)

        expected = f"{mock_concrete_llm.__module__}.{mock_concrete_llm.__name__}"
        assert result["full_name"] == expected

    def test_get_class_info_extracts_init_parameters(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info extracts __init__ parameters."""

        class TestClass:
            """Test class with init params."""

            def __init__(self, param1: str, param2: int = 42) -> None:
                """Initialize test class."""
                self.param1 = param1
                self.param2 = param2

        result = discovery.get_class_info(TestClass)

        assert "param1" in result["init_params"]
        assert "param2" in result["init_params"]
        assert result["init_params"]["param2"]["default"] == 42

    def test_get_class_info_skips_self_args_kwargs(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info skips self, args, and kwargs parameters."""

        class TestClass:
            """Test class."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                """Initialize test class."""
                pass

        result = discovery.get_class_info(TestClass)

        assert "self" not in result["init_params"]
        assert "args" not in result["init_params"]
        assert "kwargs" not in result["init_params"]

    def test_get_class_info_handles_parameters_without_annotations(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info handles parameters without type annotations."""

        class TestClass:
            """Test class."""

            def __init__(self, untyped_param):  # type: ignore[no-untyped-def]
                """Initialize test class."""
                self.untyped_param = untyped_param

        result = discovery.get_class_info(TestClass)

        assert "untyped_param" in result["init_params"]
        assert result["init_params"]["untyped_param"]["annotation"] is None

    def test_get_class_info_handles_parameters_without_defaults(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info handles parameters without default values."""

        class TestClass:
            """Test class."""

            def __init__(self, required_param: str) -> None:
                """Initialize test class."""
                self.required_param = required_param

        result = discovery.get_class_info(TestClass)

        assert "required_param" in result["init_params"]
        assert result["init_params"]["required_param"]["default"] is None

    def test_get_class_info_extracts_docstring(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info extracts class docstring."""

        class TestClass:
            """This is a test docstring."""

            pass

        result = discovery.get_class_info(TestClass)

        assert result["docstring"] == "This is a test docstring."

    def test_get_class_info_handles_class_without_docstring(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info handles class without docstring."""

        class TestClass:
            pass

        result = discovery.get_class_info(TestClass)

        assert result["docstring"] is None

    def test_get_class_info_handles_signature_inspection_error(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test get_class_info handles errors during signature inspection."""

        class ProblematicClass:
            """Class with problematic init."""

            def __init__(self) -> None:
                """Initialize."""
                pass

        # Mock inspect.signature to raise an exception
        with patch("inspect.signature", side_effect=ValueError("Cannot get signature")):
            result = discovery.get_class_info(ProblematicClass)

            assert "init_params" in result
            assert result["init_params"] == {}

    # Edge cases and error handling
    def test_discover_all_llms_with_empty_scan_results(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test discover_all_llms when all package scans return empty lists."""
        with patch.object(
            discovery, "_get_base_llm_class", return_value=type("BaseLLM", (), {})
        ):
            with patch.object(discovery, "_scan_package_for_llms", return_value=[]):
                result = discovery.discover_all_llms()

                assert result == []

    def test_discover_chat_models_with_empty_scan_results(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test discover_chat_models when all package scans return empty lists."""
        with patch.object(
            discovery,
            "_get_base_chat_model_class",
            return_value=type("BaseChatModel", (), {}),
        ):
            with patch.object(
                discovery, "_scan_package_for_chat_models", return_value=[]
            ):
                result = discovery.discover_chat_models()

                assert result == []

    def test_scan_module_for_subclasses_with_empty_module(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses with empty module."""
        empty_module = ModuleType("empty_module")

        result = discovery._scan_module_for_subclasses(
            empty_module, mock_base_llm_class
        )

        assert result == []

    def test_scan_module_for_subclasses_with_multiple_valid_classes(
        self, discovery: LangChainDiscovery, mock_base_llm_class: type
    ) -> None:
        """Test _scan_module_for_subclasses finds multiple valid subclasses."""
        module = ModuleType("test_module")

        class LLM1(mock_base_llm_class):
            pass

        class LLM2(mock_base_llm_class):
            pass

        class LLM3(mock_base_llm_class):
            pass

        module.LLM1 = LLM1
        module.LLM2 = LLM2
        module.LLM3 = LLM3

        result = discovery._scan_module_for_subclasses(module, mock_base_llm_class)

        assert len(result) == 3
        assert LLM1 in result
        assert LLM2 in result
        assert LLM3 in result

    def test_cache_persistence_across_different_discovery_methods(
        self, discovery: LangChainDiscovery
    ) -> None:
        """Test that cache persists and is separate for different discovery methods."""
        mock_llm = type("MockLLM", (), {})
        mock_chat = type("MockChat", (), {})
        mock_chain = type("MockChain", (), {})

        discovery._discovered_cache["llms"] = [mock_llm]
        discovery._discovered_cache["chat_models"] = [mock_chat]
        discovery._discovered_cache["chains"] = [mock_chain]

        assert discovery.discover_all_llms() == [mock_llm]
        assert discovery.discover_chat_models() == [mock_chat]
        assert discovery.discover_chains() == [mock_chain]
