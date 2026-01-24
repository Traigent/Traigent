# Traigent Examples Guide

What lives in `examples/`, how to run it, and where to start.

## Layout (trimmed to what matters)
```text
examples/
|- core/                    # Start here
|  |- simple-prompt/       # Minimal prompt tuning
|  |- hello-world/         # RAG toggle
|  |- few-shot-classification/
|  |- multi-objective-tradeoff/
|  |- prompt-ab-test/
|  |- structured-output-json/
|  |- token-budget-summarization/
|  |- safety-guardrails/
|  |- tool-use-calculator/
|  |- prompt-style-optimization/
|  `- chunking-long-context/
|
|- advanced/                # Patterns and deep dives
|  |- execution-modes/
|  |- results-analysis/
|  |- ai-engineering-tasks/
|  |- ragas/
|  `- metric-registry/
|
|- integrations/           # CI/CD and Bedrock samples
|- datasets/               # Shared evaluation data
|- templates/              # Example boilerplates
`- docs/                   # Guides like this one
```

## Core examples at a glance
| Example | Optimizes | Run (mock mode) |
| --- | --- | --- |
| simple-prompt | Model, temperature, prompt style | `TRAIGENT_MOCK_LLM=true python examples/core/simple-prompt/run.py` |
| hello-world | RAG toggle, model, top_k | `TRAIGENT_MOCK_LLM=true python examples/core/hello-world/run.py` |
| few-shot-classification | Example count/strategy | `TRAIGENT_MOCK_LLM=true python examples/core/few-shot-classification/run.py` |
| multi-objective-tradeoff | Accuracy, cost, latency | `TRAIGENT_MOCK_LLM=true python examples/core/multi-objective-tradeoff/run_anthropic.py` |
| prompt-ab-test | Prompt variants | `TRAIGENT_MOCK_LLM=true python examples/core/prompt-ab-test/run.py` |
| structured-output-json | Schema validation | `TRAIGENT_MOCK_LLM=true python examples/core/structured-output-json/run.py` |
| token-budget-summarization | Budgeted summaries | `TRAIGENT_MOCK_LLM=true python examples/core/token-budget-summarization/run.py` |
| safety-guardrails | Moderation tuning | `TRAIGENT_MOCK_LLM=true python examples/core/safety-guardrails/run.py` |
| tool-use-calculator | Tool calling | `TRAIGENT_MOCK_LLM=true python examples/core/tool-use-calculator/run.py` |
| prompt-style-optimization | Tone/voice tuning | `TRAIGENT_MOCK_LLM=true python examples/core/prompt-style-optimization/run.py` |
| chunking-long-context | RAG chunking | `TRAIGENT_MOCK_LLM=true python examples/core/chunking-long-context/run.py` |

## How to run
```bash
# Install from repo root (includes example deps)
pip install -e ".[examples]"

# Mock mode (recommended)
export TRAIGENT_MOCK_LLM=true
python examples/core/simple-prompt/run.py

# Real APIs (set keys you need)
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
python examples/core/multi-objective-tradeoff/run_openai.py
```

## Pick by goal
- Lower cost/latency: `multi-objective-tradeoff`, `token-budget-summarization`, `execution-modes` (advanced).
- Improve accuracy: `few-shot-classification`, `prompt-style-optimization`, `ai-engineering-tasks/p0_few_shot_selection`.
- Safer outputs: `safety-guardrails`, `ai-engineering-tasks/p1_safety_guardrails`.
- Structured JSON: `structured-output-json`, `ai-engineering-tasks/p0_structured_output`.
- Tool calling: `tool-use-calculator`.
- Long docs: `chunking-long-context`, `ai-engineering-tasks/p0_context_engineering`.

## Quick pattern
```python
import traigent

@traigent.optimize(
    eval_dataset="examples/datasets/simple-prompt/evaluation_set.jsonl",
    configuration_space={"model": ["claude-3-haiku-20240307"], "temperature": [0.0, 0.7]},
    objectives=["accuracy"],
    injection_mode="seamless",
)
def summarize(text: str) -> str:
    config = traigent.get_config()
    return f"Summary for: {text} | model={config['model']}"
```

Need a deeper map? Check `START_HERE.md` for sequencing or `LEARNING_ROADMAP.md` for a multi-week path.
