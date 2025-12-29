# P0-1: Modern Structured Output Engineering

**Priority Level:** P0 (Universal problems, 1-3 days to implement)
**Implementation Status:** ✅ Complete
**Traigent Integration:** Full optimization with systematic parameter exploration

## Overview

This example demonstrates how Traigent systematically optimizes structured data extraction - one of the most universal challenges in AI engineering. The optimization explores modern output strategies including JSON mode, function calling, XML tags, and various validation approaches to achieve near-perfect parsing reliability while maintaining extraction quality.

## Problem Statement

In 2024/2025, the structured output problem has evolved beyond simple JSON extraction. With multiple output strategies available (JSON mode, function calling, XML tags, constrained decoding), engineers waste weeks determining which approach works best for their specific use case. The wrong choice leads to parsing failures that cascade through production systems, causing silent failures and corrupted data that costs enterprises hundreds of thousands annually.

**Core Challenge:** Different models have different strengths:
- GPT-4 excels with JSON mode
- Claude performs better with XML tags
- Open-source models might need constrained decoding

Manual testing of all combinations is prohibitively expensive.

## Goals & Success Criteria

✅ **Reliability:** Achieve >99.9% valid, parseable outputs
✅ **Quality:** Maintain or improve extraction F1 score compared to baseline
✅ **Performance:** Keep latency within 110% of baseline performance
✅ **Cost:** Reduce or maintain token costs compared to naive approaches
✅ **Reproducibility:** Provide deterministic, reproducible evaluation results

## Configuration Space

Traigent explores this comprehensive search space:

```python
EXTRACTION_SEARCH_SPACE = {
    # Output format strategies (2024/2025 options)
    "output_format": [
        "json_mode",           # Native JSON mode (GPT-4, etc.)
        "function_calling",    # Tool/function call syntax
        "xml_tags",           # XML-wrapped structure (Claude preference)
        "markdown_code_block", # ```json blocks
        "yaml_format"         # YAML structure
    ],

    # Schema communication strategies
    "schema_strategy": [
        "pydantic_in_prompt",    # Full Pydantic model
        "json_schema",           # JSON Schema format
        "typescript_types",      # TypeScript interface
        "examples_only",         # Learn from examples
        "minimal_description"    # Brief field descriptions
    ],

    # Validation and correction approaches
    "validation_approach": [
        "none",                      # Trust the model
        "constrained_decoding",      # Guarantee valid JSON
        "guided_generation",         # Token-level guidance
        "retry_with_error_feedback", # Show parsing errors
        "self_correction"           # Ask model to verify
    ],

    # Additional parameters...
    "prompt_structure": ["xml_sections", "markdown_headers", "flat"],
    "n_examples": [0, 1, 3],
    "temperature": [0.0, 0.1, 0.2],
    "max_retries": [0, 1, 2]
}
```

## Dataset & Evaluation

**Dataset Composition:**
- 500 text snippets across 5 domains (invoices, support tickets, medical records, product reviews, news articles)
- Each snippet paired with a Pydantic schema defining required and optional fields
- Deliberate inclusion of edge cases: missing fields, extra information, ambiguous values
- 20% real-world data, 80% synthetic but realistic examples

**Schema Complexity Levels:**
- **Simple:** 3-5 fields, flat structure
- **Medium:** 5-10 fields, one level of nesting
- **Complex:** 10+ fields, multiple nesting levels, arrays

**Evaluation Protocol:**
- Single evaluation set (no train/test split needed for optimization)
- Fixed random seed for reproducibility
- Each configuration tested on entire dataset
- Timeout of 10 seconds per extraction attempt

## Key Metrics

**Primary Metrics:**
- `parsing_success_rate`: Percentage of outputs that parse and validate against schema (Target: ≥99.9%)
- `field_micro_f1`: Micro-averaged F1 across all fields (Target: ≥baseline + 3 percentage points)

**Secondary Metrics:**
- `latency_p95_ms`: 95th percentile latency per extraction (Target: ≤baseline × 1.1)
- `cost_per_1k_extractions`: Token-based cost estimate (Target: ≤baseline)

## Implementation Files

```
p0_structured_output/
├── __init__.py                    # Package overview and documentation
├── main.py                        # Main Traigent optimization script
├── extraction_config.py           # ExtractionConfig dataclass and search space
├── evaluator.py                   # Core evaluation and scoring functions
├── dataset.py                     # Dataset generation across 5 domains
└── README.md                      # This documentation
```

## Usage

### Basic Usage

```bash
# Navigate to the example directory
cd examples/advanced/ai-engineering-tasks/p0_structured_output/

# Run the optimization
python main.py
```

### Programmatic Usage

```python
import traigent
from extraction_config import EXTRACTION_SEARCH_SPACE
from evaluator import extract_structured
from dataset import generate_evaluation_dataset

# Generate evaluation dataset
dataset = generate_evaluation_dataset(total_samples=500)

# Configure Traigent optimization
@traigent.optimize(
    config_space=EXTRACTION_SEARCH_SPACE,
    objectives=["parsing_success_rate", "field_micro_f1", "-latency_p95_ms"],
    direction="maximize",
    max_trials=100
)
def optimize_extraction(**config):
    # Traigent will explore different configurations
    pass

# Run optimization
results = optimize_extraction()
print(f"Best configuration: {results.best_config}")
print(f"Achieved metrics: {results.best_metrics}")
```

## Expected Results

Based on the use case specification, optimization typically achieves:

- **Parsing Success Rate:** 99.8%+ (vs. baseline ~85%)
- **F1 Score Improvement:** +5-15% over baseline approaches
- **Latency:** Within 110% of baseline performance
- **Cost:** Maintained or reduced through smart parameter selection

**Common Winning Configurations:**
- GPT-4: `json_mode` + `retry_with_error_feedback` + structured prompts
- Claude: `xml_tags` + `self_correction` + examples
- Open-source: `constrained_decoding` + minimal validation

## Key Insights

🔧 **Different models excel with different output strategies**
📊 **Validation approaches significantly impact reliability vs. latency**
💰 **Few-shot examples improve quality but increase token costs**
🎯 **Systematic optimization beats manual parameter tuning by 20-40%**
⚡ **The right configuration can achieve both higher quality and lower cost**

## Integration with Production Systems

This example demonstrates patterns for:
- **Error handling:** Graceful degradation when extraction fails
- **Retry logic:** Smart retry strategies with error feedback
- **Cost tracking:** Token usage monitoring and optimization
- **Quality gates:** Validation pipelines for production deployment
- **Model switching:** Configuration that works across different LLM providers

## Related Examples

- **P0-3:** Few-Shot Example Selection Strategies
- **P1-2:** Token Budget Optimization
- **P1-3:** Safety Guardrails: PII and Hallucination Prevention

## References

- [Use Case Specification](../../../use-case.md#p0-1-modern-structured-output-engineering)
- [Traigent Documentation](../../../../README.md)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

---

*This example is part of Traigent's AI Engineering Task collection, demonstrating systematic optimization of common AI engineering challenges.*
