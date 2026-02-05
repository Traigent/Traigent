# Prompt Style Optimization

Optimize email drafting style and tone parameters to find the best configuration for your use case.

## What You'll Learn

- How to define custom metrics for subjective qualities (style, tone)
- How configuration parameters are automatically injected via `traigent.get_config()`
- How grid search exhaustively tests all parameter combinations

## The Task

We want to draft emails in different styles. We are optimizing:

1. **Style**: `bulleted` (bullet points) vs `paragraph` (prose)
2. **Tone**: `formal` (Dear...) vs `friendly` (Hi...)
3. **Temperature**: `0.0` (consistent) vs `0.2` (slight variation)

## Quick Start

```bash
# Install (from repo root, if not already done)
pip install -e ".[dev,integrations,analytics]"

# Run in mock mode (no API key needed)
export TRAIGENT_MOCK_LLM=true
python examples/core/prompt-style-optimization/run.py
```

## Expected Output

```text
============================================================
Prompt Style Optimization Example
============================================================

Objective: style_accuracy (maximize)
Configuration space:
  - style: bulleted, paragraph
  - tone: formal, friendly
  - temperature: 0.0, 0.2
Total configurations: 8 (2 x 2 x 2)
Mode: MOCK (no LLM API calls)
------------------------------------------------------------

============================================================
OPTIMIZATION COMPLETE
============================================================
Best config: {'style': 'bulleted', 'temperature': 0.0, 'tone': 'formal'}
Best score: 0.80
Total trials: 8
Runtime: 0.05s
```

**Note**: Mock mode achieves 80% accuracy because the dataset intentionally includes some examples expecting different styles to simulate realistic optimization scenarios.

## How the Custom Metric Works

The `style_accuracy` metric checks if the output matches what the dataset expects:

```python
def style_accuracy_metric(output, expected, **_):
    exp_style, exp_tone = expected.split(",")  # e.g., "bulleted,formal"
    ok_style = ("- " in output) if exp_style == "bulleted" else ("- " not in output)
    ok_tone = ("Dear" in output) if exp_tone == "formal" else ("Hi" in output)
    return 1.0 if (ok_style and ok_tone) else 0.0
```

## Running with Real APIs

```bash
export ANTHROPIC_API_KEY="your-key"
export TRAIGENT_MOCK_LLM=false
python examples/core/prompt-style-optimization/run.py
```

## Next Steps

- [prompt-ab-test](../prompt-ab-test/) - Compare prompt versions head-to-head
- [token-budget-summarization](../token-budget-summarization/) - Add length constraints
