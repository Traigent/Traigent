# Simple Prompt Optimization

This is the "Hello World" of Traigent. It demonstrates the most basic usage: optimizing a prompt and model parameters for a simple summarization task.

## What You'll Learn

- How to use the `@traigent.optimize` decorator.
- How to define a `configuration_space` (parameters to tune).
- How to access injected parameters using `traigent.get_config()`.
- How to run an optimization loop.

## The Task

We want to summarize short text snippets. We are optimizing:

1. **Model**: `claude-haiku-4-5-20251001` vs `claude-sonnet-4-6`.
2. **Temperature**: `0.0` (deterministic) vs `0.7` (creative).
3. **Prompt Style**: `concise` vs `detailed`.

## Quick Start

```bash
# Runs in local mock mode automatically when ANTHROPIC_API_KEY is unset.
python examples/core/simple-prompt/run.py
```

## Expected Output

You should see Traigent trying different combinations of parameters and reporting the results.

```text
Starting Simple Prompt Optimization...
Running with: model=claude-haiku-4-5-20251001, temp=0.0, style=concise
...
Optimization Complete!
Best Score: ...
Best Configuration: ...
```

## Next Steps

Once you understand this example, check out:

- [rag-optimization](../rag-optimization/) - Adds RAG (Retrieval Augmented Generation) to the mix.
- [few-shot-classification](../few-shot-classification/) - Optimizes few-shot examples.
