"""
Evaluator for Structured Output Engineering
===========================================

Evaluation functions for scoring structured data extraction performance.
Implements the metrics specified in the use case: parsing success, schema validation,
field-level F1 scores, precision, and recall.
"""

import json
import time
from typing import Any, Dict, List, Optional, Type

from extraction_config import ExtractionConfig
from pydantic import BaseModel, ValidationError


def extract_structured(
    text: str,
    schema: Type[BaseModel],
    config: "ExtractionConfig",
    examples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Extract structured data from text according to schema.

    This function simulates the extraction process using the specified configuration.
    In a real implementation, this would call the actual LLM with the optimized parameters.

    Args:
        text: Input text to extract from
        schema: Pydantic schema defining expected structure
        config: ExtractionConfig with optimization parameters
        examples: Optional few-shot examples

    Returns:
        dict: Extracted dictionary (may be invalid if parsing fails)
    """
    start_time = time.time()

    try:
        # Build prompt based on configuration
        _build_extraction_prompt(text, schema, config, examples)

        # Simulate LLM call with configuration parameters
        # In real implementation, this would use the actual LLM API
        extracted_data = _simulate_llm_extraction(text, schema, config)

        # Apply validation based on config
        if config.validation_approach != "none":
            extracted_data = _apply_validation(extracted_data, schema, config)

        return extracted_data

    except Exception as e:
        # Return error info for analysis
        return {
            "_extraction_error": str(e),
            "_extraction_failed": True,
            "_extraction_time": time.time() - start_time,
        }


def score_extraction(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any], schema: Type[BaseModel]
) -> Dict[str, float]:
    """
    Score extraction attempt against ground truth.

    Returns comprehensive metrics as specified in the use case:
    - parsing_success: 0.0 or 1.0
    - schema_valid: 0.0 or 1.0
    - field_micro_f1: float
    - field_precision: float
    - field_recall: float

    Args:
        predicted: Extracted dictionary
        ground_truth: Expected dictionary
        schema: Pydantic schema for validation

    Returns:
        Dict with evaluation metrics
    """
    metrics = {}

    # Check if extraction failed completely
    if predicted.get("_extraction_failed", False):
        return {
            "parsing_success": 0.0,
            "schema_valid": 0.0,
            "field_micro_f1": 0.0,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "latency_ms": predicted.get("_extraction_time", 0) * 1000,
        }

    # 1. Parsing Success Rate
    parsing_success = 1.0
    try:
        json.dumps(predicted)  # Test if it's JSON serializable
    except (TypeError, ValueError):
        parsing_success = 0.0

    metrics["parsing_success"] = parsing_success

    # 2. Schema Validation
    schema_valid = 0.0
    try:
        schema(**predicted)
        schema_valid = 1.0
    except (ValidationError, TypeError, ValueError):
        pass

    metrics["schema_valid"] = schema_valid

    # 3. Field-level metrics (micro-averaged F1, precision, recall)
    field_metrics = _calculate_field_metrics(predicted, ground_truth)
    metrics.update(field_metrics)

    # 4. Latency metrics
    metrics["latency_ms"] = predicted.get("_extraction_time", 0) * 1000

    return metrics


def _build_extraction_prompt(
    text: str,
    schema: Type[BaseModel],
    config: "ExtractionConfig",
    examples: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build extraction prompt based on configuration."""

    # Schema representation based on strategy
    if config.schema_strategy == "pydantic_in_prompt":
        schema_str = str(schema.model_json_schema())
    elif config.schema_strategy == "json_schema":
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
    elif config.schema_strategy == "typescript_types":
        schema_str = _convert_to_typescript(schema)
    elif config.schema_strategy == "examples_only":
        schema_str = _format_examples(examples) if examples else ""
    else:  # minimal_description
        schema_str = _minimal_schema_description(schema)

    # Prompt structure based on configuration
    if config.prompt_structure == "xml_sections":
        prompt = f"""<instructions>
Extract structured data from the following text according to the schema.
</instructions>

<schema>
{schema_str}
</schema>

<input>
{text}
</input>"""
    elif config.prompt_structure == "markdown_headers":
        prompt = f"""## Instructions
Extract structured data from the following text according to the schema.

## Schema
{schema_str}

## Input Text
{text}"""
    else:  # flat
        prompt = f"Extract data from: {text}\n\nSchema: {schema_str}"

    # Add examples if configured
    if config.n_examples > 0 and examples:
        example_text = _format_examples(examples[: config.n_examples])
        if config.prompt_structure == "xml_sections":
            prompt = prompt.replace(
                "</instructions>",
                f"\n\n<examples>\n{example_text}\n</examples>\n</instructions>",
            )
        else:
            prompt = f"Examples:\n{example_text}\n\n{prompt}"

    return prompt


def _simulate_llm_extraction(
    text: str, schema: Type[BaseModel], config: "ExtractionConfig"
) -> Dict[str, Any]:
    """Simulate LLM extraction with realistic success/failure patterns."""

    # Simulate different success rates based on configuration
    import random

    # Base success rate depends on output format and validation approach
    base_success_rate = 0.85

    if config.output_format == "json_mode":
        base_success_rate += 0.1
    elif config.output_format == "xml_tags":
        base_success_rate += 0.05

    if config.validation_approach == "retry_with_error_feedback":
        base_success_rate += 0.08
    elif config.validation_approach == "self_correction":
        base_success_rate += 0.05

    # Temperature affects consistency
    base_success_rate -= config.temperature * 0.1

    # Simulate extraction success
    if random.random() < base_success_rate:
        # Generate realistic extracted data based on schema
        return _generate_realistic_extraction(schema, text)
    else:
        # Simulate extraction failure
        return {"_extraction_failed": True, "error": "Parsing failed"}


def _generate_realistic_extraction(
    schema: Type[BaseModel], text: str
) -> Dict[str, Any]:
    """Generate realistic extraction data for testing."""
    # This would normally be the actual LLM output
    # For demo purposes, create a basic structure

    # Get schema fields
    fields = schema.model_fields

    result = {}
    for field_name, field_info in fields.items():
        # Generate realistic values based on field type and text content
        if field_info.annotation is str:
            # Extract relevant text snippets
            words = text.split()[:5]
            result[field_name] = " ".join(words)
        elif field_info.annotation is int:
            result[field_name] = len(text.split())
        elif field_info.annotation is float:
            result[field_name] = len(text) / 100.0
        elif field_info.annotation is bool:
            result[field_name] = len(text) > 100
        else:
            result[field_name] = str(field_info.annotation)

    return result


def _apply_validation(
    data: Dict[str, Any], schema: Type[BaseModel], config: "ExtractionConfig"
) -> Dict[str, Any]:
    """Apply validation based on configuration."""

    if config.validation_approach == "retry_with_error_feedback":
        # Simulate retry logic
        try:
            schema(**data)
            return data
        except ValidationError as e:
            # In real implementation, would retry with error feedback
            return {"_validation_error": str(e), "_extraction_failed": True}

    elif config.validation_approach == "self_correction":
        # Simulate self-correction
        try:
            validated = schema(**data)
            return validated.model_dump()
        except ValidationError:
            # Simulate correction attempt
            return _attempt_correction(data, schema)

    return data


def _calculate_field_metrics(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, float]:
    """Calculate field-level precision, recall, and F1 scores."""

    # Remove internal fields
    predicted_clean = {k: v for k, v in predicted.items() if not k.startswith("_")}

    if not predicted_clean and not ground_truth:
        return {"field_micro_f1": 1.0, "field_precision": 1.0, "field_recall": 1.0}

    if not predicted_clean:
        return {"field_micro_f1": 0.0, "field_precision": 0.0, "field_recall": 0.0}

    if not ground_truth:
        return {"field_micro_f1": 0.0, "field_precision": 0.0, "field_recall": 1.0}

    # Calculate matches
    matches = 0
    total_predicted = len(predicted_clean)
    total_ground_truth = len(ground_truth)

    for key, value in predicted_clean.items():
        if key in ground_truth and str(value).strip() == str(ground_truth[key]).strip():
            matches += 1

    precision = matches / total_predicted if total_predicted > 0 else 0.0
    recall = matches / total_ground_truth if total_ground_truth > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {"field_micro_f1": f1, "field_precision": precision, "field_recall": recall}


# Helper functions
def _convert_to_typescript(schema: Type[BaseModel]) -> str:
    """Convert Pydantic schema to TypeScript interface."""
    return f"// TypeScript interface for {schema.__name__}\n// Simplified for demo"


def _format_examples(examples: Optional[List[Dict[str, Any]]]) -> str:
    """Format examples for prompt inclusion."""
    if not examples:
        return ""
    return "\n".join(
        [f"Example {i+1}: {json.dumps(ex, indent=2)}" for i, ex in enumerate(examples)]
    )


def _minimal_schema_description(schema: Type[BaseModel]) -> str:
    """Generate minimal schema description."""
    fields = schema.model_fields
    descriptions = []
    for name, field_info in fields.items():
        descriptions.append(f"{name}: {field_info.annotation.__name__}")
    return "Fields: " + ", ".join(descriptions)


def _attempt_correction(
    data: Dict[str, Any], schema: Type[BaseModel]
) -> Dict[str, Any]:
    """Simulate correction attempt."""
    # Simple correction logic for demo
    corrected = data.copy()

    # Try to fix common issues
    for key, value in corrected.items():
        if isinstance(value, str) and value.isdigit():
            try:
                corrected[key] = int(value)
            except ValueError:
                pass

    return corrected
