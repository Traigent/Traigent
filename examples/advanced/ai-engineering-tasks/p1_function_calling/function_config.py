"""
Configuration for function calling optimization.

This module defines the search space for optimizing function calling reliability,
including tool description formats, parameter validation strategies, and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DescriptionFormat(Enum):
    """Tool description formats."""

    OPENAI_FUNCTIONS = "openai_functions"
    ANTHROPIC_TOOLS = "anthropic_tools"
    JSON_SCHEMA = "json_schema"
    TYPESCRIPT = "typescript"
    DOCSTRING = "docstring"
    XML_STRUCTURED = "xml_structured"


class SelectionStrategy(Enum):
    """Tool selection strategies."""

    ALL_TOOLS_ALWAYS = "all_tools_always"
    FILTERED_RELEVANT = "filtered_relevant"
    PROGRESSIVE_DISCLOSURE = "progressive_disclosure"
    LLM_SUGGESTED = "llm_suggested"


class ErrorStrategy(Enum):
    """Error handling strategies."""

    NONE = "none"
    RETRY_WITH_ERROR = "retry_with_error"
    REPHRASE_DESCRIPTION = "rephrase_description"
    PROVIDE_EXAMPLE = "provide_example"
    BREAK_DOWN_STEPS = "break_down_steps"


class ValidationApproach(Enum):
    """Parameter validation approaches."""

    NONE = "none"
    SCHEMA_VALIDATION = "schema_validation"
    MOCK_EXECUTION = "mock_execution"
    LLM_VERIFICATION = "llm_verification"


@dataclass
class FunctionConfig:
    """Configuration for function calling optimization."""

    # Tool description format
    description_format: str = "openai_functions"

    # Parameter schema representation
    parameter_schema: str = "json_schema_strict"

    # Tool selection strategy
    selection_strategy: str = "filtered_relevant"

    # Error handling approach
    error_strategy: str = "retry_with_error"

    # Parameter validation
    validation: str = "schema_validation"

    # Retry configuration
    max_retries: int = 1
    temperature: float = 0.0

    # Advanced options
    include_examples: bool = True
    example_format: str = "json"
    context_aware_selection: bool = False
    progressive_hints: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "description_format": self.description_format,
            "parameter_schema": self.parameter_schema,
            "selection_strategy": self.selection_strategy,
            "error_strategy": self.error_strategy,
            "validation": self.validation,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "include_examples": self.include_examples,
            "example_format": self.example_format,
            "context_aware_selection": self.context_aware_selection,
            "progressive_hints": self.progressive_hints,
        }


# TraiGent search space for function calling optimization
FUNCTION_SEARCH_SPACE = {
    # Tool description formats
    "description_format": [
        "openai_functions",  # OpenAI function format
        "anthropic_tools",  # Anthropic tool format
        "json_schema",  # Pure JSON Schema
        "typescript",  # TypeScript definitions
        "docstring",  # Python docstring style
        "xml_structured",  # XML-tagged format
    ],
    # Parameter schema strategies
    "parameter_schema": [
        "json_schema_strict",  # Strict JSON Schema with validation
        "pydantic_model",  # Pydantic model definition
        "typescript_interface",  # TypeScript interface format
        "examples_based",  # Learn from examples
        "minimal_description",  # Brief parameter descriptions
    ],
    # Tool selection strategies
    "selection_strategy": [
        "all_tools_always",  # Show all available tools
        "filtered_relevant",  # Pre-filter by category/relevance
        "progressive_disclosure",  # Start simple, expand as needed
        "llm_suggested",  # Model suggests needed tools
    ],
    # Error handling approaches
    "error_strategy": [
        "none",  # No special error handling
        "retry_with_error",  # Show error message and retry
        "rephrase_description",  # Rephrase tool description
        "provide_example",  # Give concrete usage example
        "break_down_steps",  # Break complex calls into steps
    ],
    # Validation approaches
    "validation": [
        "none",  # Trust model output
        "schema_validation",  # Validate against JSON schema
        "mock_execution",  # Dry-run with mock tools
        "llm_verification",  # Ask model to verify parameters
    ],
    # Model parameters
    "max_retries": [0, 1, 2, 3],
    "temperature": [0.0, 0.1, 0.3],
    # Advanced features
    "include_examples": [True, False],
    "example_format": ["json", "yaml", "python"],
    "context_aware_selection": [True, False],
    "progressive_hints": [True, False],
}


def create_function_config(**config_params) -> FunctionConfig:
    """Create a FunctionConfig from parameter dictionary."""
    return FunctionConfig(**config_params)


def get_tool_description(tool_info: dict[str, Any], format_type: str) -> str:
    """Generate tool description in specified format."""

    name = tool_info.get("name", "unknown_tool")
    description = tool_info.get("description", "No description available")
    parameters = tool_info.get("parameters", {})

    if format_type == "openai_functions":
        return f"""{{
    "name": "{name}",
    "description": "{description}",
    "parameters": {parameters}
}}"""

    elif format_type == "anthropic_tools":
        return f"""<tool_description>
<tool_name>{name}</tool_name>
<description>{description}</description>
<parameters>{parameters}</parameters>
</tool_description>"""

    elif format_type == "json_schema":
        return f"""{{
    "type": "function",
    "function": {{
        "name": "{name}",
        "description": "{description}",
        "parameters": {parameters}
    }}
}}"""

    elif format_type == "typescript":
        param_types = []
        for param, info in parameters.get("properties", {}).items():
            param_type = info.get("type", "any")
            param_types.append(f"  {param}: {param_type}")

        return f"""interface {name.title()}Params {{
{chr(10).join(param_types)}
}}

function {name}(params: {name.title()}Params): any {{
    // {description}
}}"""

    elif format_type == "docstring":
        param_docs = []
        for param, info in parameters.get("properties", {}).items():
            param_desc = info.get("description", f"{param} parameter")
            param_type = info.get("type", "any")
            param_docs.append(f"        {param} ({param_type}): {param_desc}")

        return f"""def {name}(**kwargs):
    \"\"\"
    {description}

    Parameters:
{chr(10).join(param_docs)}
    \"\"\"
    pass"""

    elif format_type == "xml_structured":
        return f"""<tool>
    <name>{name}</name>
    <description>{description}</description>
    <parameters>
        <schema>{parameters}</schema>
    </parameters>
</tool>"""

    else:
        return f"{name}: {description}\nParameters: {parameters}"


def validate_function_call(
    call_data: dict[str, Any], tool_schema: dict[str, Any]
) -> dict[str, Any]:
    """Validate a function call against tool schema."""

    errors = []
    warnings = []

    # Check if function name matches
    expected_name = tool_schema.get("name", "")
    actual_name = call_data.get("name", "")

    if expected_name != actual_name:
        errors.append(
            f"Function name mismatch: expected '{expected_name}', got '{actual_name}'"
        )

    # Check required parameters
    required_params = tool_schema.get("parameters", {}).get("required", [])
    provided_params = call_data.get("arguments", {})

    for param in required_params:
        if param not in provided_params:
            errors.append(f"Missing required parameter: {param}")

    # Check parameter types (simplified)
    param_schema = tool_schema.get("parameters", {}).get("properties", {})
    for param, value in provided_params.items():
        if param in param_schema:
            expected_type = param_schema[param].get("type", "any")
            actual_type = type(value).__name__

            # Simple type checking
            type_mapping = {
                "string": "str",
                "number": ["int", "float"],
                "integer": "int",
                "boolean": "bool",
                "array": "list",
                "object": "dict",
            }

            expected_python_types = type_mapping.get(expected_type, expected_type)
            if isinstance(expected_python_types, list):
                if actual_type not in expected_python_types:
                    warnings.append(
                        f"Type mismatch for {param}: expected {expected_python_types}, got {actual_type}"
                    )
            elif actual_type != expected_python_types:
                warnings.append(
                    f"Type mismatch for {param}: expected {expected_python_types}, got {actual_type}"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "score": 1.0 if len(errors) == 0 else max(0.0, 1.0 - len(errors) * 0.2),
    }


def get_baseline_configs() -> list[dict[str, Any]]:
    """Get baseline configurations for comparison."""

    return [
        {
            "name": "simple_baseline",
            "description_format": "openai_functions",
            "parameter_schema": "json_schema_strict",
            "selection_strategy": "all_tools_always",
            "error_strategy": "none",
            "validation": "none",
            "max_retries": 0,
            "temperature": 0.0,
            "include_examples": False,
            "context_aware_selection": False,
        },
        {
            "name": "standard_practice",
            "description_format": "openai_functions",
            "parameter_schema": "json_schema_strict",
            "selection_strategy": "filtered_relevant",
            "error_strategy": "retry_with_error",
            "validation": "schema_validation",
            "max_retries": 1,
            "temperature": 0.0,
            "include_examples": True,
            "context_aware_selection": False,
        },
        {
            "name": "robust_approach",
            "description_format": "anthropic_tools",
            "parameter_schema": "pydantic_model",
            "selection_strategy": "progressive_disclosure",
            "error_strategy": "provide_example",
            "validation": "mock_execution",
            "max_retries": 2,
            "temperature": 0.1,
            "include_examples": True,
            "context_aware_selection": True,
        },
    ]
