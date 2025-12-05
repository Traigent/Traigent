"""
Extraction Configuration for Structured Output Engineering
==========================================================

Configuration dataclass defining the search space for structured output optimization.
Based on the Modern Structured Output Engineering use case specification.
"""

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration for structured data extraction optimization.

    This dataclass defines the complete search space for optimizing
    structured output extraction, covering modern 2024/2025 approaches.
    """

    # Output format strategies (2024/2025 options)
    output_format: str  # "json_mode", "function_calling", "xml_tags", "markdown_code_block", "yaml_format"

    # Schema communication strategies
    schema_strategy: str  # "pydantic_in_prompt", "json_schema", "typescript_types", "examples_only", "minimal_description"

    # Validation and correction approaches
    validation_approach: str  # "none", "constrained_decoding", "guided_generation", "retry_with_error_feedback", "self_correction"

    # Prompt engineering aspects
    prompt_structure: str  # "xml_sections", "markdown_headers", "flat"

    # Example strategies
    n_examples: int  # 0, 1, 3
    example_selection: str  # "random", "diverse", "similar"

    # Model parameters
    temperature: float  # 0.0, 0.1, 0.2
    max_retries: int  # 0, 1, 2


# Search space definition for TraiGent optimization
EXTRACTION_SEARCH_SPACE = {
    "output_format": [
        "json_mode",  # Native JSON mode (GPT-4, etc.)
        "function_calling",  # Tool/function call syntax
        "xml_tags",  # XML-wrapped structure (Claude preference)
        "markdown_code_block",  # ```json blocks
        "yaml_format",  # YAML structure
    ],
    "schema_strategy": [
        "pydantic_in_prompt",  # Full Pydantic model
        "json_schema",  # JSON Schema format
        "typescript_types",  # TypeScript interface
        "examples_only",  # Learn from examples
        "minimal_description",  # Brief field descriptions
    ],
    "validation_approach": [
        "none",  # Trust the model
        "constrained_decoding",  # Guarantee valid JSON
        "guided_generation",  # Token-level guidance
        "retry_with_error_feedback",  # Show parsing errors
        "self_correction",  # Ask model to verify
    ],
    "prompt_structure": [
        "xml_sections",  # <instructions>, <schema>, <input>
        "markdown_headers",  # ## Instructions, ## Schema
        "flat",  # Simple concatenation
    ],
    "n_examples": [0, 1, 3],
    "example_selection": ["random", "diverse", "similar"],
    "temperature": [0.0, 0.1, 0.2],
    "max_retries": [0, 1, 2],
}


def create_extraction_config(**kwargs) -> ExtractionConfig:
    """Create an ExtractionConfig with validation."""
    return ExtractionConfig(
        output_format=kwargs.get("output_format", "json_mode"),
        schema_strategy=kwargs.get("schema_strategy", "pydantic_in_prompt"),
        validation_approach=kwargs.get("validation_approach", "none"),
        prompt_structure=kwargs.get("prompt_structure", "xml_sections"),
        n_examples=kwargs.get("n_examples", 0),
        example_selection=kwargs.get("example_selection", "random"),
        temperature=kwargs.get("temperature", 0.0),
        max_retries=kwargs.get("max_retries", 0),
    )
