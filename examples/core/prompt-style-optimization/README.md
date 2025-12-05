# Prompt Style Optimization

Optimize email drafting style and tone parameters.

## Quick Start

```bash
export TRAIGENT_MOCK_MODE=true
python examples/core/prompt-style-optimization/run.py
```

## Configuration Space

| Parameter | Values | Purpose |
|-----------|--------|---------|
| `style` | bulleted, paragraph | Output structure |
| `tone` | formal, friendly | Communication tone |
| `temperature` | 0.0, 0.2 | Response variance |

## What It Optimizes

- Output formatting (bullets vs prose)
- Tone matching (formal vs casual)
- Style consistency across outputs

## Expected Output

```
Best config: {'style': 'bulleted', 'tone': 'formal', 'temperature': 0.0}
Best score: 0.88
```

## Custom Metric

The `style_accuracy` metric validates:
- Bullet point presence (for bulleted style)
- Greeting format (Dear vs Hi based on tone)

```python
def style_accuracy_metric(output, expected, _llm_metrics):
    exp_style, exp_tone = expected.split(",")
    ok_style = ("- " in output) if exp_style == "bulleted" else True
    ok_tone = ("Dear" in output) if exp_tone == "formal" else ("Hi" in output)
    return 1.0 if (ok_style and ok_tone) else 0.0
```

## Key Concepts

- **Bayesian optimization**: Efficient parameter search
- **Seamless injection**: Auto-applied via `traigent.get_current_config()`
- **Style validation**: Rule-based output checking

## Next Steps

- [prompt-ab-test](../prompt-ab-test/) - Compare prompt versions
- [token-budget-summarization](../token-budget-summarization/) - Add length constraints
