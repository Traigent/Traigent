# Hello World (RAG Optimization)

> **Note:** If you are new to Traigent, start with the [Simple Prompt Optimization](../simple-prompt/) example first.

Optimize a RAG-enabled Q&A function to find the best model, temperature, and RAG settings.

## Quick Start

```bash
export TRAIGENT_MOCK_MODE=true  # Skip API calls
python examples/core/hello-world/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `model` | claude-3-5-sonnet, claude-3-haiku | Model selection |
| `temperature` | 0.0 | Response consistency |
| `use_rag` | true, false | Toggle RAG retrieval |
| `top_k` | 1, 2, 3 | Documents to retrieve |

## What It Optimizes

- Tests whether RAG improves answer quality
- Compares model performance on Q&A tasks
- Finds optimal `top_k` for your knowledge base

## Expected Output

```
Best config: {'model': 'claude-3-haiku-20240307', 'use_rag': True, 'top_k': 2}
Best score: 0.85
```

## Key Concepts

- **Seamless injection**: Parameters auto-applied to LangChain
- **Custom cost metric**: Realistic mock telemetry
- **Edge analytics**: Local execution with no cloud dependency

## Next Steps

- [few-shot-classification](../few-shot-classification/) - Add few-shot examples
- [multi-objective-tradeoff](../multi-objective-tradeoff/) - Balance cost vs accuracy
