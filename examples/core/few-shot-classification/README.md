# Few-Shot Classification

Optimize few-shot prompting parameters for sentiment classification.

## Quick Start

```bash
export TRAIGENT_MOCK_LLM=true
python examples/core/few-shot-classification/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `model` | claude-3-haiku, claude-3-5-sonnet | Model selection |
| `temperature` | 0.0, 0.3 | Response variance |
| `k` | 0, 2, 4 | Number of examples |
| `selection_strategy` | top_k, diverse | Example selection method |

## What It Optimizes

- Number of few-shot examples needed
- Example selection strategy (diverse vs sequential)
- Model and temperature for classification tasks

## Expected Output

```
Best config: {'model': 'claude-3-5-sonnet-20241022', 'k': 4, 'selection_strategy': 'diverse'}
Best score: 0.92
```

## Key Concepts

- **Parameter injection mode**: Explicit config parameter
- **Grid search algorithm**: Tests all combinations
- **Example selection**: Round-robin by label for diversity

## Next Steps

- [prompt-ab-test](../prompt-ab-test/) - Test different prompt versions
- [structured-output-json](../structured-output-json/) - Extract structured data
