# Structured Output JSON

Optimize JSON extraction from unstructured text with custom validation.

## Quick Start

```bash
export TRAIGENT_MOCK_LLM=true
python examples/core/structured-output-json/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `temperature` | 0.0, 0.2 | Response variance |
| `format_hint` | strict_json, relaxed_json | JSON formatting guidance |
| `schema_rigidity` | strict, lenient | Schema enforcement level |

## What It Optimizes

- JSON validity and parsing success
- Field extraction accuracy
- Format hint effectiveness

## Expected Output

```
Best config: {'temperature': 0.0, 'format_hint': 'strict_json', 'schema_rigidity': 'strict'}
Best score: 0.95
```

## Custom Metric

The `json_score` metric validates:
1. Valid JSON parsing
2. Required field presence
3. Value correctness against expected output

```python
def json_score_metric(output, expected, _llm_metrics):
    obj = json.loads(output)
    # Score based on field matches
```

## Key Concepts

- **Custom metrics**: Define your own evaluation function
- **Seamless injection**: Parameters applied automatically
- **Schema validation**: Test different rigidity levels

## Next Steps

- [safety-guardrails](../safety-guardrails/) - Add safety validation
- [tool-use-calculator](../tool-use-calculator/) - Structured tool calling
