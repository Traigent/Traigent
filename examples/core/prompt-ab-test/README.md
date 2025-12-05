# Prompt A/B Test

Compare prompt variants to find the best performing version.

## Quick Start

```bash
export TRAIGENT_MOCK_MODE=true
python examples/core/prompt-ab-test/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `prompt_variant` | a, b | Prompt version to test |
| `model` | claude-3-haiku, claude-3-5-sonnet | Model selection |
| `temperature` | 0.0, 0.2 | Response variance |

## What It Optimizes

- Compares two prompt versions (A vs B)
- Tests each variant across models and temperatures
- Finds statistically significant winner

## Expected Output

```
Best config: {'prompt_variant': 'b', 'model': 'claude-3-5-sonnet-20241022', 'temperature': 0.0}
Best score: 0.88
```

## Key Concepts

- **Grid search**: Exhaustive combination testing
- **Parallel trials**: Configurable concurrency profiles
- **Prompt variants**: External prompt files (prompt_a.txt, prompt_b.txt)

## Files

- `prompt_a.txt` - First prompt variant
- `prompt_b.txt` - Second prompt variant

## Next Steps

- [prompt-style-optimization](../prompt-style-optimization/) - Optimize tone and style
- [few-shot-classification](../few-shot-classification/) - Add examples to prompts
