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
export TRAIGENT_MOCK_MODE=true
python examples/quickstart/01_simple_qa.py
```

### 2. Customer Support with RAG (`02_customer_support_rag.py`)

Full example with RAG integration for customer support optimization.

```bash
export TRAIGENT_MOCK_MODE=true
python examples/quickstart/02_customer_support_rag.py
```

### 3. Custom Objectives (`03_custom_objectives.py`)

Shows how to use custom weights and objective definitions.

```bash
export TRAIGENT_MOCK_MODE=true
python examples/quickstart/03_custom_objectives.py
```

## Running with Real APIs

Once you're ready to use real LLM APIs:

```bash
# Set your API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Disable mock mode
export TRAIGENT_MOCK_MODE=false

# Run any example
python examples/quickstart/01_simple_qa.py
```

## Dataset

These examples use `data/qa_samples.jsonl` located at:

- `data/qa_samples.jsonl`

## Next Steps

After trying these quickstart examples:

1. Explore `examples/core/` for more comprehensive examples
2. Check `examples/advanced/` for specialized patterns
3. Read the [Evaluation Guide](../../docs/guides/evaluation.md)
4. Try the CLI: `traigent --help`
