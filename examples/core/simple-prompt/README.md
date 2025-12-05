# Simple Prompt Optimization

This is the "Hello World" of TraiGent. It demonstrates the most basic usage: optimizing a prompt and model parameters for a simple summarization task.

## What You'll Learn

- How to use the `@traigent.optimize` decorator.
- How to define a `configuration_space` (parameters to tune).
- How to access injected parameters using `traigent.get_current_config()`.
- How to run an optimization loop.

## The Task

We want to summarize short text snippets. We are optimizing:

1. **Model**: `claude-3-haiku` vs `claude-3-5-sonnet`.
2. **Temperature**: `0.0` (deterministic) vs `0.7` (creative).
3. **Prompt Style**: `concise` vs `detailed`.

## Quick Start

```bash
# Run in mock mode (no API key needed)
export TRAIGENT_MOCK_MODE=true
python examples/core/simple-prompt/run.py
```

## Expected Output

You should see TraiGent trying different combinations of parameters and reporting the results.

```text
Starting Simple Prompt Optimization...
Running with: model=claude-3-haiku-20240307, temp=0.0, style=concise
...
Optimization Complete!
Best Score: ...
Best Configuration: ...
```

## Next Steps

Once you understand this example, check out:

- [hello-world](../hello-world/) - Adds RAG (Retrieval Augmented Generation) to the mix.
- [few-shot-classification](../few-shot-classification/) - Optimizes few-shot examples.
