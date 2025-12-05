"""Dynamic parameter discovery for framework integrations.

This module provides utilities to dynamically discover parameters from any framework,
enabling automatic parameter mapping without hardcoding.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any, cast

from traigent.security.config import get_security_flags

from ...utils.logging import get_logger

logger = get_logger(__name__)


class ParameterDiscovery:
    """Dynamically discover and map parameters for any framework."""

    @staticmethod
    def discover_init_parameters(cls: type) -> dict[str, inspect.Parameter]:
        """Extract all parameters from a class's __init__ method.

        Args:
            cls: The class to inspect

        Returns:
            Dictionary mapping parameter names to Parameter objects
        """
        if cls is None:
            return {}

        try:
            sig = inspect.signature(cls)
            return {
                name: param for name, param in sig.parameters.items() if name != "self"
            }
        except Exception as e:
            logger.debug(f"Could not discover parameters for {cls}: {e}")
            return {}

    @staticmethod
    def discover_method_parameters(
        obj: Any, method_path: str
    ) -> dict[str, inspect.Parameter]:
        """Extract parameters from a specific method.

        Args:
            obj: The object containing the method
            method_path: Dot-separated path to the method (e.g., "messages.create")

        Returns:
            Dictionary mapping parameter names to Parameter objects
        """
        if obj is None:
            return {}

        try:
            current_obj = obj
            for part in method_path.split("."):
                current_obj = getattr(current_obj, part)

            sig = inspect.signature(current_obj)
            return {
                name: param
                for name, param in sig.parameters.items()
                if name not in ["self", "cls"]
            }
        except Exception as e:
            logger.debug(f"Could not discover method parameters for {method_path}: {e}")
            return {}

    @staticmethod
    def find_similar_parameters(
        source_param: str, target_params: list[str], threshold: float = 0.8
    ) -> str | None:
        """Use fuzzy matching to find similar parameter names.

        Args:
            source_param: The parameter name to match
            target_params: List of possible target parameter names
            threshold: Similarity threshold (0-1)

        Returns:
            Best matching parameter name or None
        """
        # Simple implementation - can be enhanced with Levenshtein distance
        source_lower = source_param.lower()

        # Exact match (case-insensitive)
        for target in target_params:
            if source_lower == target.lower():
                return target

        # Common variations
        variations = {
            "model": ["model_name", "model_id", "engine"],
            "temperature": ["temp"],
            "max_tokens": ["max_length", "max_new_tokens", "max_tokens_to_sample"],
            "top_p": ["top_p_sampling"],
            "stop": ["stop_sequences", "stop_words"],
            "stream": ["streaming"],
        }

        if source_param in variations:
            for variant in variations[source_param]:
                if variant in target_params:
                    return variant

        # Substring matching
        for target in target_params:
            if source_lower in target.lower() or target.lower() in source_lower:
                return target

        return None

    @staticmethod
    def infer_parameter_type(param: inspect.Parameter) -> type | None:
        """Infer parameter type from annotations or defaults.

        Args:
            param: The parameter to analyze

        Returns:
            Inferred type or None
        """
        # Check annotation first
        if param.annotation != inspect.Parameter.empty:
            return cast(type | None, param.annotation)

        # Infer from default value
        if param.default != inspect.Parameter.empty:
            return type(param.default)

        return None

    @staticmethod
    def get_common_llm_parameters() -> set[str]:
        """Get a set of common LLM parameters across frameworks."""
        return {
            "model",
            "model_name",
            "model_id",
            "engine",
            "temperature",
            "temp",
            "max_tokens",
            "max_length",
            "max_new_tokens",
            "max_tokens_to_sample",
            "top_p",
            "top_p_sampling",
            "top_k",
            "frequency_penalty",
            "freq_penalty",
            "presence_penalty",
            "pres_penalty",
            "stop",
            "stop_sequences",
            "stop_words",
            "stream",
            "streaming",
            "tools",
            "functions",
            "system",
            "system_prompt",
            "system_message",
            "messages",
            "prompt",
            "seed",
            "random_seed",
            "logit_bias",
            "logit_biases",
            "n",
            "num_completions",
            "timeout",
            "request_timeout",
        }

    @classmethod
    def create_universal_mapping(cls) -> dict[str, list[str]]:
        """Create a universal parameter mapping for common LLM parameters."""
        return {
            "model": ["model", "model_name", "model_id", "engine"],
            "temperature": ["temperature", "temp"],
            "max_tokens": [
                "max_tokens",
                "max_length",
                "max_new_tokens",
                "max_tokens_to_sample",
            ],
            "top_p": ["top_p", "top_p_sampling"],
            "top_k": ["top_k"],
            "frequency_penalty": ["frequency_penalty", "freq_penalty"],
            "presence_penalty": ["presence_penalty", "pres_penalty"],
            "stop": ["stop", "stop_sequences", "stop_words"],
            "stream": ["stream", "streaming"],
            "tools": ["tools", "functions"],
            "system_prompt": ["system", "system_prompt", "system_message"],
            "seed": ["seed", "random_seed"],
            "messages": ["messages", "prompt"],
            "n": ["n", "num_completions"],
            "timeout": ["timeout", "request_timeout"],
            "logit_bias": ["logit_bias", "logit_biases"],
        }

    @staticmethod
    def discover_package_classes(
        package_name: str, base_class: type | None = None
    ) -> list[type]:
        """Discover all classes in a package that optionally inherit from a base class.

        Args:
            package_name: Name of the package to scan
            base_class: Optional base class to filter by

        Returns:
            List of discovered classes
        """
        classes = []

        try:
            package = importlib.import_module(package_name)

            for name in dir(package):
                obj = getattr(package, name)
                if inspect.isclass(obj):
                    if base_class:
                        if issubclass(obj, base_class) and obj != base_class:
                            classes.append(obj)
                    else:
                        classes.append(obj)

        except ImportError as e:
            logger.debug(f"Could not import package {package_name}: {e}")

        return classes

    @staticmethod
    def introspect_framework_structure(cls: type) -> dict[str, Any]:
        """Introspect complete framework structure.

        Args:
            cls: The class to introspect

        Returns:
            Dictionary with init_parameters, methods, and attributes
        """
        try:
            structure: dict[str, Any] = {
                "init_parameters": ParameterDiscovery.discover_init_parameters(cls),
                "methods": {},
                "attributes": {},
            }

            # Discover methods
            for name in dir(cls):
                if not name.startswith("_"):
                    try:
                        attr = getattr(cls, name)
                        if inspect.ismethod(attr) or inspect.isfunction(attr):
                            try:
                                # Try to get method signature without instantiating
                                sig = inspect.signature(attr)
                                params = {
                                    param_name: param
                                    for param_name, param in sig.parameters.items()
                                    if param_name not in ["self", "cls"]
                                }
                                structure["methods"][name] = {"parameters": params}
                            except Exception as sig_err:
                                logger.debug(
                                    f"Could not get signature for method {name}: {sig_err}"
                                )
                                structure["methods"][name] = {"parameters": {}}
                    except Exception as method_err:
                        logger.debug(f"Could not inspect method {name}: {method_err}")
                        continue

            return structure
        except Exception as e:
            logger.debug(f"Could not introspect framework structure for {cls}: {e}")
            return {"init_parameters": {}, "methods": {}, "attributes": {}}

    @staticmethod
    def find_completion_methods(cls: type) -> list[dict[str, Any]]:
        """Find completion-like methods in framework.

        Args:
            cls: The class to search

        Returns:
            List of method information dictionaries
        """
        completion_methods = []
        completion_keywords = ["create", "complete", "generate", "chat", "completion"]

        def _search_nested_methods(obj, path_prefix="", max_depth=3) -> None:
            """Recursively search for completion methods in nested objects."""
            if max_depth <= 0:
                return

            try:
                for name in dir(obj):
                    if name.startswith("_"):
                        continue

                    try:
                        attr = getattr(obj, name)
                        current_path = f"{path_prefix}.{name}" if path_prefix else name

                        # Check if this is a completion method
                        if any(
                            keyword in name.lower() for keyword in completion_keywords
                        ):
                            if inspect.ismethod(attr) or inspect.isfunction(attr):
                                completion_methods.append(
                                    {
                                        "name": name,
                                        "path": current_path,
                                        "parameters": {},
                                    }
                                )

                        # Recursively search nested objects (but not functions/methods)
                        if not (
                            inspect.ismethod(attr)
                            or inspect.isfunction(attr)
                            or inspect.isbuiltin(attr)
                            or inspect.isclass(attr)
                        ):
                            _search_nested_methods(attr, current_path, max_depth - 1)

                    except Exception as attr_err:
                        logger.debug(
                            f"Could not inspect attribute at {current_path}: {attr_err}"
                        )
                        continue
            except Exception as search_err:
                logger.debug(f"Error during nested method search: {search_err}")

        try:
            # Search class methods first
            _search_nested_methods(cls)

            # Also try to search instance methods by creating a dummy instance
            if inspect.isclass(cls):
                flags = get_security_flags()
                explicit_setting = getattr(
                    cls, "__traigent_allow_instance_discovery__", None
                )
                allow_instance_discovery = (
                    explicit_setting
                    if explicit_setting is not None
                    else flags.auto_discovery
                )

                if explicit_setting is None and os.getenv("PYTEST_CURRENT_TEST"):
                    allow_instance_discovery = False

                if allow_instance_discovery is False:
                    logger.debug(
                        "Skipping instantiation of %s during discovery (explicit opt-out)",
                        cls,
                    )
                elif allow_instance_discovery:
                    if (
                        flags.emit_security_telemetry
                        and explicit_setting is None
                        and not flags.auto_discovery
                    ):
                        logger.info(
                            "Auto discovery instantiating %s under profile %s",
                            cls.__name__,
                            flags.profile.value,
                        )
                    try:
                        # Try to create instance with minimal parameters
                        init_params = ParameterDiscovery.discover_init_parameters(cls)
                        if init_params:
                            # Try to create instance with dummy values for required parameters
                            dummy_args: dict[str, Any] = {}
                            for param_name, param in init_params.items():
                                if param.default == inspect.Parameter.empty:
                                    # Required parameter - provide dummy value based on type
                                    param_type = (
                                        ParameterDiscovery.infer_parameter_type(param)
                                    )
                                    if param_type is str:
                                        dummy_args[param_name] = "dummy"
                                    elif param_type is int:
                                        dummy_args[param_name] = 0
                                    elif param_type is float:
                                        dummy_args[param_name] = 0.0
                                    elif param_type is bool:
                                        dummy_args[param_name] = False
                                    else:
                                        dummy_args[param_name] = None
                            instance = cls(**dummy_args)
                        else:
                            instance = cls()

                        _search_nested_methods(instance)
                    except Exception as instance_err:
                        logger.debug(
                            f"Could not create instance for discovery: {instance_err}"
                        )

        except Exception as e:
            logger.debug(f"Could not find completion methods for {cls}: {e}")

        return completion_methods

    @staticmethod
    def analyze_parameter_types(cls: type) -> dict[str, Any]:
        """Analyze parameter types across framework.

        Args:
            cls: The class to analyze

        Returns:
            Dictionary with type analysis results
        """
        try:
            params = ParameterDiscovery.discover_init_parameters(cls)
            by_type: dict[str, list[str]] = {}
            by_category: dict[str, list[str]] = {}

            for name, param in params.items():
                param_type = ParameterDiscovery.infer_parameter_type(param)
                if param_type:
                    type_name = getattr(param_type, "__name__", str(param_type))
                    if type_name not in by_type:
                        by_type[type_name] = []
                    by_type[type_name].append(name)

                    # Categorize types
                    if type_name in ["str", "string"]:
                        category = "string"
                    elif type_name in ["int", "float", "number"]:
                        category = "number"
                    elif type_name in ["bool", "boolean"]:
                        category = "boolean"
                    elif type_name in ["list", "dict", "tuple"]:
                        category = "collection"
                    else:
                        category = "other"

                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(name)

            return {"by_type": by_type, "by_category": by_category}
        except Exception as e:
            logger.debug(f"Could not analyze parameter types for {cls}: {e}")
            return {"by_type": {}, "by_category": {}}

    @staticmethod
    def get_parameter_defaults(cls: type) -> dict[str, Any]:
        """Extract parameter defaults.

        Args:
            cls: The class to extract defaults from

        Returns:
            Dictionary mapping parameter names to default values
        """
        try:
            params = ParameterDiscovery.discover_init_parameters(cls)
            defaults = {}

            for name, param in params.items():
                if param.default != inspect.Parameter.empty:
                    defaults[name] = param.default

            return defaults
        except Exception as e:
            logger.debug(f"Could not get parameter defaults for {cls}: {e}")
            return {}

    @staticmethod
    def validate_parameter_compatibility(
        cls: type, traigent_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate parameter compatibility.

        Args:
            cls: The framework class
            traigent_params: TraiGent parameters to validate

        Returns:
            Dictionary with compatible, incompatible, and missing parameters
        """
        try:
            framework_params = ParameterDiscovery.discover_init_parameters(cls)
            framework_names = set(framework_params.keys())
            traigent_names = set(traigent_params.keys())

            compatible = []
            incompatible = []

            for param_name in traigent_names:
                if param_name in framework_names:
                    compatible.append(param_name)
                else:
                    # Check for similar names
                    similar = ParameterDiscovery.find_similar_parameters(
                        param_name, list(framework_names)
                    )
                    if similar:
                        compatible.append(param_name)
                    else:
                        incompatible.append(param_name)

            missing = list(framework_names - traigent_names)

            return {
                "compatible": compatible,
                "incompatible": incompatible,
                "missing": missing,
            }
        except Exception as e:
            logger.debug(f"Could not validate parameter compatibility for {cls}: {e}")
            return {"compatible": [], "incompatible": [], "missing": []}

    @staticmethod
    def create_automatic_mapping(
        cls: type, traigent_params: list[str]
    ) -> dict[str, str]:
        """Create automatic parameter mapping.

        Args:
            cls: The framework class
            traigent_params: List of TraiGent parameter names

        Returns:
            Dictionary mapping TraiGent params to framework params
        """
        try:
            framework_params = ParameterDiscovery.discover_init_parameters(cls)
            framework_names = list(framework_params.keys())
            mapping = {}

            for traigent_param in traigent_params:
                # Try exact match first
                if traigent_param in framework_names:
                    mapping[traigent_param] = traigent_param
                else:
                    # Try similarity matching
                    similar = ParameterDiscovery.find_similar_parameters(
                        traigent_param, framework_names
                    )
                    if similar:
                        mapping[traigent_param] = similar

            return mapping
        except Exception as e:
            logger.debug(f"Could not create automatic mapping for {cls}: {e}")
            return {}

    @staticmethod
    def suggest_parameter_mappings(
        cls: type, source_params: list[str]
    ) -> dict[str, str]:
        """Suggest parameter mappings based on similarity.

        Args:
            cls: The framework class
            source_params: List of source parameter names

        Returns:
            Dictionary with suggested mappings
        """
        return ParameterDiscovery.create_automatic_mapping(cls, source_params)

    @staticmethod
    def fuzzy_parameter_matching(cls: type, source_params: list[str]) -> dict[str, str]:
        """Perform fuzzy matching of parameter names.

        Args:
            cls: The framework class
            source_params: List of source parameter names

        Returns:
            Dictionary with fuzzy matches
        """
        return ParameterDiscovery.create_automatic_mapping(cls, source_params)

    @staticmethod
    def generate_mapping_confidence(
        cls: type, mappings: dict[str, str]
    ) -> dict[str, float]:
        """Generate confidence scores for mappings.

        Args:
            cls: The framework class
            mappings: Parameter mappings to score

        Returns:
            Dictionary with confidence scores (0.0-1.0)
        """
        try:
            framework_params = ParameterDiscovery.discover_init_parameters(cls)
            framework_names = set(framework_params.keys())
            confidence = {}

            for source_param, target_param in mappings.items():
                if target_param in framework_names:
                    if source_param == target_param:
                        # Exact match - high confidence
                        confidence[source_param] = 1.0
                    elif source_param.lower() == target_param.lower():
                        # Case-insensitive match - high confidence
                        confidence[source_param] = 0.95
                    elif source_param in target_param or target_param in source_param:
                        # Substring match - medium confidence
                        confidence[source_param] = 0.7
                    else:
                        # Poor match - low confidence
                        confidence[source_param] = 0.3
                else:
                    # Target doesn't exist - no confidence
                    confidence[source_param] = 0.0

            return confidence
        except Exception as e:
            logger.debug(f"Could not generate mapping confidence for {cls}: {e}")
            return {}

    @staticmethod
    def detect_framework_version(cls: type) -> str | None:
        """Detect framework version.

        Args:
            cls: The framework class

        Returns:
            Version string or None
        """
        try:
            # Try different version attributes
            version_attrs = ["__version__", "_version", "VERSION", "version"]

            for attr in version_attrs:
                if hasattr(cls, attr):
                    version = getattr(cls, attr)
                    if isinstance(version, str):
                        return version

            # Try module version
            module = inspect.getmodule(cls)
            if module:
                for attr in version_attrs:
                    if hasattr(module, attr):
                        version = getattr(module, attr)
                        if isinstance(version, str):
                            return version

            return None
        except Exception as e:
            logger.debug(f"Could not detect framework version for {cls}: {e}")
            return None

    @staticmethod
    def check_parameter_version_compatibility(
        v1_params: list[str], v2_params: list[str]
    ) -> dict[str, list[str]]:
        """Check parameter compatibility across versions.

        Args:
            v1_params: Parameters from version 1
            v2_params: Parameters from version 2

        Returns:
            Dictionary with added, removed, and common parameters
        """
        v1_set = set(v1_params)
        v2_set = set(v2_params)

        return {
            "added": list(v2_set - v1_set),
            "removed": list(v1_set - v2_set),
            "common": list(v1_set & v2_set),
        }

    @staticmethod
    def adapt_parameters_for_version(
        params: dict[str, Any], version: str, exclude_params: list[str] | None = None
    ) -> dict[str, Any]:
        """Adapt parameters for specific version.

        Args:
            params: Original parameters
            version: Target version
            exclude_params: Parameters to exclude

        Returns:
            Adapted parameters dictionary
        """
        adapted = params.copy()

        if exclude_params:
            for param in exclude_params:
                adapted.pop(param, None)

        return adapted
