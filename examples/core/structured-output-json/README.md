# Structured Output JSON

Optimize JSON extraction from unstructured text with custom validation.

## What You'll Learn

- How to create metrics that validate structured output (JSON parsing, field matching)
- How to test different prompt strategies (`strict_json` vs `relaxed_json`)
- How schema enforcement levels affect extraction accuracy

## The Task

We want to extract structured data (vendor name, amount) from invoice text. We are optimizing:

1. **Temperature**: `0.0` (deterministic) vs `0.2` (slight variation)
2. **Format Hint**: `strict_json` vs `relaxed_json` (prompt guidance)
3. **Schema Rigidity**: `strict` vs `lenient` (how strict the schema enforcement)

## Quick Start

```bash
# Install (from repo root, if not already done)
pip install -e ".[dev,integrations,analytics]"

# Run in mock mode (no API key needed)
export TRAIGENT_MOCK_LLM=true
python examples/core/structured-output-json/run.py
```

## Expected Output

```text
============================================================
Structured JSON Extraction Example
============================================================

Objective: json_score (maximize)
Configuration space:
  - temperature: 0.0, 0.2
  - format_hint: strict_json, relaxed_json
  - schema_rigidity: strict, lenient
Total configurations: 8 (2 x 2 x 2)
Mode: MOCK (no LLM API calls)
------------------------------------------------------------

============================================================
OPTIMIZATION COMPLETE
============================================================
Best config: {'format_hint': 'strict_json', 'schema_rigidity': 'strict', 'temperature': 0.0}
Best score: 1.00
Total trials: 8
Runtime: 0.05s
```

## How the Custom Metric Works

The `json_score` metric validates:

1. Is the output valid JSON?
2. Does it have the required fields?
3. Do the values match the expected output?

```python
def json_score_metric(output, expected, **_):
    try:
        obj = json.loads(output)
    except Exception:
        return 0.0  # Invalid JSON = score 0

    # Count how many fields match
    matched = sum(1 for k, v in expected.items()
                  if k in obj and str(obj[k]).lower() == str(v).lower())
    return matched / len(expected)
```

## Running with Real APIs

```bash
export ANTHROPIC_API_KEY="your-key"
export TRAIGENT_MOCK_LLM=false
python examples/core/structured-output-json/run.py
```

## Next Steps

- [safety-guardrails](../safety-guardrails/) - Add safety validation
- [tool-use-calculator](../tool-use-calculator/) - Structured tool calling
