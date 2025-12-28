# Getting Started with Traigent SDK

The fastest path to optimize an LLM workflow with **zero code changes**.

## 🚀 Quick Start

1) Install from source (recommended for examples):

```bash
pip install -e ".[integrations]"        # Core + LangChain/OpenAI/Anthropic
export TRAIGENT_MOCK_MODE=true          # Run examples without API keys
python examples/quickstart/01_simple_qa.py
```

2) Wrap your existing function:

```python
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    eval_dataset="examples/datasets/hello-world/evaluation_set.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.0, 0.7]},
)
def answer_question(question: str) -> str:
    cfg = traigent.get_config()  # Active trial/applied config
    llm = ChatOpenAI(model=cfg.get("model"), temperature=cfg.get("temperature"))
    return llm.invoke(question).content

# Async-safe in Traigent - use asyncio.run in sync contexts
if __name__ == "__main__":
    import asyncio
    results = asyncio.run(answer_question.optimize(max_trials=5, algorithm="grid"))
    print({"best_config": results.best_config, "best_score": results.best_score})
```

## 📋 Config Access Lifecycle

| When | Use | Notes |
| --- | --- | --- |
| Inside the optimized function | `traigent.get_config()` | Works during trials and after `apply_best_config()` |
| During optimization only | `traigent.get_trial_config()` | Raises if no active trial (strict) |
| After the run | `results.best_config` | Returned from `func.optimize()` |
| Future calls | `answer_question.current_config` | Automatically set to the last applied best config |

## 🧪 Datasets

- Format: JSONL with `input` and optional `output`/`expected_output`
- Minimal example:

```jsonl
{"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"}
{"input": {"question": "Capital of France?"}, "output": "Paris"}
```

## 🎛️ Multiple Objectives

```python
from traigent.api.decorators import EvaluationOptions

@traigent.optimize(
    objectives=["accuracy", "cost", "latency"],
    evaluation=EvaluationOptions(eval_dataset="data.jsonl"),
    configuration_space={"temperature": (0.0, 1.0), "model": ["gpt-4o-mini", "gpt-4o"]},
)
def classify(text: str) -> str:
    ...
```

## 🧪 Mock Mode & Examples

- `TRAIGENT_MOCK_MODE=true python examples/core/hello-world/run.py` (no API keys)
- Examples Navigator: `python -m http.server -d examples 8000` → http://localhost:8000

## 🛠️ CLI Snippets

```bash
traigent info                                   # Version/features
traigent algorithms                             # Available strategies
traigent optimize examples/core/hello-world/run.py -a grid -n 5
traigent validate examples/datasets/hello-world/evaluation_set.jsonl
traigent plot my_run -p progress
```

## 🔀 Multi-Provider Model Testing

To test models from different providers (OpenAI, Anthropic, Google, etc.) with a unified interface, we recommend [LiteLLM](https://github.com/BerriAI/litellm):

```bash
pip install litellm
```

```python
from litellm import completion

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini/gemini-pro"],
    },
    ...
)
def my_agent(query: str) -> str:
    config = traigent.get_config()
    response = completion(model=config.get("model"), messages=[{"role": "user", "content": query}])
    return response.choices[0].message.content
```

## 🔒 Execution Model

Open-source builds support `execution_mode="edge_analytics"` (local). Cloud/hybrid orchestration is a roadmap item—keep runs in local mode unless you have a managed backend wired up.

---

Ready for more? Dive into the [examples](../examples/) and the [API reference](../api-reference/complete-function-specification.md).
