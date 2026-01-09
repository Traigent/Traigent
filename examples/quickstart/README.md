# Traigent Quickstart Examples

This directory contains the exact examples from the main README.md, ready to run.

## Prerequisites

```bash
# Install Traigent (from repo root)
pip install -e ".[dev,integrations,analytics]"
```

## Examples

### 1. Simple Q&A Agent (`01_simple_qa.py`)

The basic example from the README showing Tuned Variables optimization.

```bash
# Run in mock mode (no API keys needed)
export TRAIGENT_MOCK_LLM=true
python examples/quickstart/01_simple_qa.py
```

### 2. Customer Support with RAG (`02_customer_support_rag.py`)

Full example with RAG integration for customer support optimization.

```bash
export TRAIGENT_MOCK_LLM=true
python examples/quickstart/02_customer_support_rag.py
```

### 3. Custom Objectives (`03_custom_objectives.py`)

Shows how to use custom weights and objective definitions.

```bash
export TRAIGENT_MOCK_LLM=true
python examples/quickstart/03_custom_objectives.py
```

## Running with Real APIs

Once you're ready to use real LLM APIs:

```bash
# Set your API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Disable mock mode
export TRAIGENT_MOCK_LLM=false

# Run any example
python examples/quickstart/01_simple_qa.py
```

## Dataset

These examples use `data/qa_samples.jsonl` located at:

- `data/qa_samples.jsonl`

## DSPy Integration

Traigent integrates with DSPy for automatic prompt optimization. Combine DSPy's prompt engineering with Traigent's hyperparameter optimization.

```bash
# Install DSPy support
pip install traigent[dspy]
```

### Quick Example

```python
from traigent.integrations.dspy_adapter import DSPyPromptOptimizer
import traigent

@traigent.optimize(
    # DSPy generates prompt variants that Traigent explores
    prompt=DSPyPromptOptimizer.create_prompt_choices([
        "You are a helpful assistant. Answer concisely.",
        "Think step by step before answering.",
        "Answer only with the most relevant information.",
    ]),
    model=["gpt-3.5-turbo", "gpt-4o-mini"],
    temperature=[0.1, 0.3, 0.5],
    objectives=["accuracy", "cost", "latency"],
    eval_dataset="validation.jsonl",
)
def qa_agent(question: str) -> str:
    config = traigent.get_config()
    # ... implementation
```

### Important: Dataset Separation

Use **different datasets** for DSPy and Traigent to avoid overfitting:

| Dataset        | Purpose                                          |
| -------------- | ------------------------------------------------ |
| `trainset`     | DSPy prompt optimization (small, ~10-50 examples)|
| `eval_dataset` | Traigent config optimization (validation set)    |
| `test_dataset` | Final evaluation (held-out, never seen)          |

See the full [DSPy Integration Guide](../docs/DSPY_INTEGRATION.md) for detailed patterns.

## Interactive Notebook

For a visual walkthrough, run the Jupyter notebook:

```bash
jupyter notebook examples/quickstart/demo_walkthrough.ipynb
```

## Next Steps

After trying these quickstart examples:

1. Explore `examples/core/` for more comprehensive examples
2. Check `examples/advanced/` for specialized patterns
3. Read the [DSPy Integration Guide](../docs/DSPY_INTEGRATION.md) for prompt optimization
4. Read the [Evaluation Guide](../../docs/guides/evaluation.md)
5. Try the CLI: `traigent --help`
