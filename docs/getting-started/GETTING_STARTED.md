# Getting Started with Traigent SDK

The fastest path to optimize an LLM workflow with **zero code changes**.

## 🚀 Quick Start

1) Install and run - no API keys needed:

```bash
pip install "traigent[integrations]"
python -m traigent.examples.quickstart
```

Or from a source checkout:

Pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[recommended]"
python hello_world.py
```

Uv:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[recommended]"
python hello_world.py
```

The quickstart runs in mock mode by default - it simulates LLM calls so you can see the full optimization flow instantly.

2) Here's what it does - one decorator, automatic optimization:

```python
import asyncio
from langchain_openai import ChatOpenAI
import traigent

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.7, 1.0],
    },
    objectives=["accuracy"],
    eval_dataset="qa_samples.jsonl",
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    return llm.invoke(question).content

# Run optimization
result = asyncio.run(answer.optimize(max_trials=6, algorithm="grid"))
print(f"Best config: {result.best_config}")
print(f"Best score:  {result.best_score:.2%}")
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

- `TRAIGENT_MOCK_LLM=true python examples/core/rag-optimization/run.py` (no API keys)
- Examples Navigator: `python -m http.server 8000` → http://localhost:8000/examples/

## 🛠️ CLI Snippets

```bash
traigent info                                   # Version/features
traigent algorithms                             # Available strategies
traigent optimize examples/core/rag-optimization/run.py -a grid -n 5
traigent validate examples/datasets/rag-optimization/evaluation_set.jsonl
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
    objectives=["accuracy"],
    eval_dataset="data/qa_samples.jsonl",
)
def my_agent(query: str) -> str:
    config = traigent.get_config()
    response = completion(model=config.get("model"), messages=[{"role": "user", "content": query}])
    return str(response.choices[0].message.content)
```

## 🔒 Execution Model

Traigent executes your code locally. The default is `execution_mode="edge_analytics"` (local).

`execution_mode="hybrid"` runs trials locally while sending session and trial metrics to the backend so results appear in the Traigent portal.

`execution_mode="cloud"` is reserved for future remote execution. It is not available yet and fails with: “Cloud remote execution is not available yet; use hybrid for portal-tracked optimization.”

If you want website-visible results today, use `hybrid`; do not switch to `cloud`.

To run fully local (no Traigent backend communication), set `TRAIGENT_OFFLINE_MODE=true`.

---

Ready for more? Dive into the [examples](../../examples/) and the [API reference](../api-reference/complete-function-specification.md).
