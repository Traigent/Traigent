# Traigent Examples

Simple, clean examples showing how to use Traigent for LLM optimization.

## Quick Start

```bash
# Mock examples - no API keys needed
python walkthrough/examples/mock/01_simple.py

# Real examples - requires OpenAI API key
export OPENAI_API_KEY="your-key"
python walkthrough/examples/real/01_simple.py
```

## Structure

```text
examples/
  mock/          # No API keys needed, instant results
  real/          # Requires API keys, real LLM calls
  datasets/      # Pre-built evaluation datasets
```

## Examples

| #  | Example          | Description                                |
|----|------------------|--------------------------------------------|
| 01 | Simple           | Basic model and temperature tuning         |
| 02 | Zero Code Change | Seamless mode intercepts hardcoded values  |
| 03 | Parameter Mode   | Explicit configuration control             |
| 04 | Multi-Objective  | Balance accuracy, cost, and latency        |
| 05 | RAG              | Optimize retrieval and generation together |
| 06 | Custom Evaluator | Define your own success metrics            |
| 07 | Privacy Modes    | Local, Cloud, and Hybrid execution         |

## Mock vs Real

| Mock (`mock/`)     | Real (`real/`)          |
|--------------------|-------------------------|
| No API keys        | Requires OPENAI_API_KEY |
| Instant results    | Real LLM calls          |
| Great for learning | Production-ready        |
| ~50 lines each     | ~60 lines each          |
