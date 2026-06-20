# Run-Plan: Text-to-SQL via OpenRouter

Pre-filled example for running the text-to-SQL optimization using
[OpenRouter](https://openrouter.ai/) as the LLM provider. OpenRouter gives
access to dozens of models through a single OpenAI-compatible API endpoint.

## Prerequisites

```bash
pip install "traigent[recommended]"
```

Set your OpenRouter API key (get one at https://openrouter.ai/keys):

```bash
export OPENROUTER_API_KEY="sk-or-..."  # pragma: allowlist secret
```

## Quickest path (mock mode — no API key spend)

```bash
TRAIGENT_MOCK_LLM=true python examples/core/text-to-sql/run.py
```

## Real run with OpenRouter

Replace the provider-specific import and LLM instantiation in `run.py` with:

```python
import os
from langchain_openai import ChatOpenAI
import traigent

@traigent.optimize(
    eval_dataset="examples/datasets/text-to-sql/evaluation_set.jsonl",
    objectives=["sql_accuracy"],
    configuration_space={
        # Use paid models — free-tier slots (:free suffix on OpenRouter) hit
        # 429 rate limits under the example's trial concurrency and will score
        # 0% accuracy, making the optimization silently useless.
        "model": [
            "openai/gpt-4o-mini",        # cheap, reliable, recommended default
            "deepseek/deepseek-chat",     # low-cost alternative
        ],
        "temperature": [0.0, 0.2],
        "include_schema": ["true", "false"],
    },
    metric_functions={"sql_accuracy": sql_accuracy},
    injection_mode="seamless",
    offline=True,
)
def generate_sql(question: str) -> str:
    cfg = traigent.get_config()
    llm = ChatOpenAI(
        model=cfg["model"],
        temperature=float(cfg["temperature"]),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    ...
```

> **Why not `qwen/qwen3-coder:free`?**
> Free-tier OpenRouter slots share a very limited rate-limit quota. Under the
> concurrency the optimizer uses (multiple parallel trials), `:free` models
> immediately return `429 Too Many Requests`. Every trial for that model scores
> 0% accuracy and completes in ~0.25 s, which silently eliminates it from
> contention — you end up optimizing over one model instead of two and never
> notice. Use any paid model to avoid this. `openai/gpt-4o-mini` via OpenRouter
> costs ~$0.15 / 1 M input tokens and is reliable at the concurrency levels
> this example uses.

## Running with SPIDER-30

Download Spider 1.0, convert to JSONL (see `README.md` for the conversion
script), then pass the dataset path:

```bash
OPENROUTER_API_KEY=sk-or-... python examples/core/text-to-sql/run.py \
    --eval-dataset path/to/spider_eval.jsonl
```

> SPIDER is released under a non-commercial research license. Do not
> redistribute modified copies of the dataset.

## Expected output

```
Text-to-SQL Optimization Example
...
OPTIMIZATION COMPLETE
Best config : {'model': 'openai/gpt-4o-mini', 'temperature': 0.0, 'include_schema': 'true'}
Best score  : 0.XXX
Total trials: 8
```
