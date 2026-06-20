# Getting Started with Traigent SDK

The fastest path to optimize an LLM workflow with **zero code changes**.

## 🚀 Quick Start

1) Install and run - no API keys needed:

```bash
pip install "traigent[recommended]"
traigent quickstart
```

For guided setup after the quickstart:

```bash
traigent onboard
traigent auth device-login
traigent first-prompt --agent codex
```

From a source checkout for development:

Pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "traigent[recommended]"
python hello_world.py
```

Uv:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install "traigent[recommended]"
python hello_world.py
```

The quickstart runs in mock mode by default and pre-approves the local cost gate for the demo - it simulates LiteLLM/LangChain calls so you can see the full optimization flow instantly. For your own dry runs, use mock mode plus `cost_approved=True` or `TRAIGENT_COST_APPROVED=true`; raw `openai.chat.completions.create(...)` and `anthropic.messages.create(...)` calls are not mocked.

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
traigent quickstart                             # Packaged mock-mode demo
traigent onboard                                # Guided project setup
traigent auth device-login                      # Browser device login
traigent first-prompt --agent codex             # Coding-agent prompt
traigent recommend rag                          # Configuration recommendations
traigent mcp serve                              # Local stdio MCP server
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

Traigent executes your code locally. Configure routing with `algorithm` and
`offline`; there is no user-facing execution-mode selector for new code.

The default `algorithm="auto"` path is cloud-first: the backend suggests trial
configs and your process runs the trials. If cloud connectivity is unavailable,
auto falls back locally unless `TRAIGENT_REQUIRE_CLOUD=1` is set.

Explicit `algorithm="grid"` and `"random"` run locally with no cloud optimizer
round trip. Explicit smart algorithms require cloud and error if offline or
unavailable.

To run fully local with no Traigent backend communication, set `offline=True` or
`TRAIGENT_OFFLINE=1`.

---

Ready for more? Dive into the [examples](../examples/) and the [API reference](../api-reference/complete-function-specification.md).
