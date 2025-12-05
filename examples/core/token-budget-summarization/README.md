# Token Budget Summarization

Optimize summarization under strict token constraints.

## Quick Start

```bash
export TRAIGENT_MOCK_MODE=true
python examples/core/token-budget-summarization/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `max_tokens` | 64, 96, 128 | Token budget |
| `temperature` | 0.0, 0.2 | Response variance |
| `style` | bulleted, paragraph | Output format |

## What It Optimizes

- Maximum tokens for cost control
- Output style (bullets vs prose)
- Information density within constraints

## Expected Output

```
Best config: {'max_tokens': 96, 'temperature': 0.0, 'style': 'bulleted'}
Best score: 0.85
```

## Key Concepts

- **Token budgeting**: Hard limits on output length
- **Style selection**: Structured vs narrative output
- **Random search**: Efficient parameter exploration

## Use Cases

- Meeting summary extraction
- Document key point identification
- Cost-constrained content generation

## Next Steps

- [multi-objective-tradeoff](../multi-objective-tradeoff/) - Balance cost vs accuracy
- [chunking-long-context](../chunking-long-context/) - Handle long documents
