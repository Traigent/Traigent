"""Auto-discovery utilities for LangChain components.

This module provides utilities to automatically discover all LangChain components
that can be optimized by Traigent.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import importlib
import inspect
from typing import Any, cast

from ....utils.logging import get_logger

logger = get_logger(__name__)


class LangChainDiscovery:
    """Auto-discover all LangChain components."""

    # Known LangChain packages to scan
    LANGCHAIN_PACKAGES = [
        "langchain",
        "langchain_community",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_google_genai",
        "langchain_cohere",
        "langchain_huggingface",
        "langchain_aws",
        "langchain_azure",
        "langchain_mistralai",
        "langchain_ollama",
        "langchain_together",
        "langchain_groq",
        "langchain_nvidia",
    ]

    def __init__(self) -> None:
        """Initialize the discovery system."""
        self._discovered_cache: dict[str, Any] = {}

    def discover_all_llms(self) -> list[type]:
        """Discover all LLM classes in installed LangChain packages.

        Returns:
            List of discovered LLM classes
        """
        if "llms" in self._discovered_cache:
            return cast(list[type], self._discovered_cache["llms"])

        llms = []

        # First, try to get base class
        base_llm = self._get_base_llm_class()
        if not base_llm:
            logger.warning("Could not find LangChain BaseLLM class")
            return []

        # Scan all packages
        for package_name in self.LANGCHAIN_PACKAGES:
            llms.extend(self._scan_package_for_llms(package_name, base_llm))

        self._discovered_cache["llms"] = llms
        return llms

    def discover_chat_models(self) -> list[type]:
        """Discover all chat model classes.

        Returns:
            List of discovered chat model classes
        """
        if "chat_models" in self._discovered_cache:
            return cast(list[type], self._discovered_cache["chat_models"])

        chat_models = []

        # Try to get base chat model class
        base_chat = self._get_base_chat_model_class()
        if not base_chat:
            logger.warning("Could not find LangChain BaseChatModel class")
            return []

        # Scan all packages
        for package_name in self.LANGCHAIN_PACKAGES:
            chat_models.extend(
                self._scan_package_for_chat_models(package_name, base_chat)
            )

        self._discovered_cache["chat_models"] = chat_models
        return chat_models

    def discover_chains(self) -> list[type]:
        """Discover all chain classes that accept LLM parameters.

        Returns:
            List of discovered chain classes
        """
        if "chains" in self._discovered_cache:
            return cast(list[type], self._discovered_cache["chains"])

        chains = []

        # Try to get base chain class
        try:
            from langchain.chains.base import Chain

            base_chain = Chain
        except ImportError:
            logger.warning("Could not import LangChain Chain class")
            return []

        # Scan for chains
        for package_name in ["langchain", "langchain_community"]:
            try:
                chains_module = importlib.import_module(f"{package_name}.chains")
                chains.extend(
                    self._scan_module_for_subclasses(chains_module, base_chain)
                )
            except ImportError:
                continue

        self._discovered_cache["chains"] = chains
        return chains

    def _get_base_llm_class(self) -> type | None:
        """Get the BaseLLM class from LangChain."""
        try:
            from langchain.llms.base import BaseLLM

            return cast(type, BaseLLM)
        except ImportError:
            try:
                from langchain_core.language_models.llms import BaseLLM

                return cast(type, BaseLLM)
            except ImportError:
                return None

    def _get_base_chat_model_class(self) -> type | None:
        """Get the BaseChatModel class from LangChain."""
        try:
            from langchain.chat_models.base import BaseChatModel

            return cast(type, BaseChatModel)
        except ImportError:
            try:
                from langchain_core.language_models.chat_models import BaseChatModel

                return cast(type, BaseChatModel)
            except ImportError:
                return None

    def _scan_package_for_llms(self, package_name: str, base_class: type) -> list[type]:
        """Scan a package for LLM classes.

        Args:
            package_name: Name of the package to scan
            base_class: Base LLM class to check inheritance

        Returns:
            List of LLM classes found
        """
        llms = []

        try:
            # Try different module structures
            for module_path in [f"{package_name}.llms", package_name]:
                try:
                    module = importlib.import_module(module_path)
                    llms.extend(self._scan_module_for_subclasses(module, base_class))
                except ImportError:
                    continue

        except Exception as e:
            logger.debug(f"Error scanning {package_name} for LLMs: {e}")

        return llms

    def _scan_package_for_chat_models(
        self, package_name: str, base_class: type
    ) -> list[type]:
        """Scan a package for chat model classes.

        Args:
            package_name: Name of the package to scan
            base_class: Base chat model class to check inheritance

        Returns:
            List of chat model classes found
        """
        chat_models = []

        try:
            # Try different module structures
            for module_path in [f"{package_name}.chat_models", package_name]:
                try:
                    module = importlib.import_module(module_path)
                    chat_models.extend(
                        self._scan_module_for_subclasses(module, base_class)
                    )
                except ImportError:
                    continue

        except Exception as e:
            logger.debug(f"Error scanning {package_name} for chat models: {e}")

        return chat_models

    def _scan_module_for_subclasses(self, module: Any, base_class: type) -> list[type]:
        """Scan a module for subclasses of a base class.

        Args:
            module: Module to scan
            base_class: Base class to check inheritance

        Returns:
            List of subclasses found
        """
        subclasses = []

        for name in dir(module):
            if name.startswith("_"):
                continue

            try:
                obj = getattr(module, name)
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and obj != base_class
                    and not inspect.isabstract(obj)
                ):
                    subclasses.append(obj)
            except Exception as e:
                logger.debug(f"Could not inspect {name} in {module.__name__}: {e}")
                continue

        return subclasses

    def create_universal_langchain_mapping(self) -> dict[str, list[str]]:
        """Create mapping that works for most LangChain components.

        Returns:
            Universal parameter mapping for LangChain
        """
        return {
            "model": ["model", "model_name", "model_id", "engine", "model_key"],
            "temperature": ["temperature", "temp"],
            "max_tokens": [
                "max_tokens",
                "max_length",
                "max_new_tokens",
                "max_tokens_to_sample",
                "max_output_tokens",
            ],
            "top_p": ["top_p", "top_p_sampling", "nucleus_sampling"],
            "top_k": ["top_k"],
            "stop": ["stop", "stop_sequences", "stop_words", "stop_tokens"],
            "streaming": ["streaming", "stream"],
            "timeout": ["timeout", "request_timeout", "max_retries"],
            "seed": ["seed", "random_seed"],
            "frequency_penalty": [
                "frequency_penalty",
                "freq_penalty",
                "repetition_penalty",
            ],
            "presence_penalty": ["presence_penalty", "pres_penalty"],
            "n": ["n", "num_completions", "best_of"],
        }

    def get_class_info(self, cls: type) -> dict[str, Any]:
        """Get detailed information about a class.

        Args:
            cls: Class to analyze

        Returns:
            Dictionary with class information
        """
        info: dict[str, Any] = {
            "name": cls.__name__,
            "module": cls.__module__,
            "full_name": f"{cls.__module__}.{cls.__name__}",
            "init_params": {},
            "docstring": inspect.getdoc(cls),
        }

        # Get __init__ parameters
        try:
            sig = inspect.signature(cls.__init__)  # type: ignore[misc]
            for name, param in sig.parameters.items():
                if name not in ["self", "args", "kwargs"]:
                    info["init_params"][name] = {
                        "annotation": (
                            str(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else None
                        ),
                        "default": (
                            param.default
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                    }
        except Exception as e:
            logger.debug(f"Could not analyze component {cls}: {e}")

        return info
