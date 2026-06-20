# Traigent API Patterns

This page shows current Traigent SDK patterns for decorating a function, reading
tuned values, running optimization, and inspecting results.

## Canonical Pattern

```python
import asyncio
import traigent


def my_score(output: str, expected: str) -> float:
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def call_model(prompt: str, *, temperature: float) -> str:
    # Replace with your provider call.
    return prompt


@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3, 0.7]},
    eval_dataset="data/eval.jsonl",
    scoring_function=my_score,
    objectives=["accuracy", "cost"],
)
def fn(prompt: str) -> str:
    cfg = traigent.get_config()
    return call_model(prompt, temperature=cfg["temperature"])


results = asyncio.run(fn.optimize(max_trials=20))
df = results.to_dataframe()
```

Key points:

- Use `configuration_space`, not `config_space`.
- Put datasets and scoring on the decorator through `eval_dataset`,
  `scoring_function`, `metric_functions`, or the `evaluation={...}` bundle.
- Read tuned values with `traigent.get_config()` in default context mode.
- Run optimization with `await fn.optimize(...)` or `asyncio.run(fn.optimize(...))`.
- Use `fn.optimize_sync(...)` only from synchronous code that is not already
  inside an event loop.

## Custom Scoring

```python
def contains_expected(output: str, expected: str) -> float:
    return 1.0 if str(expected).lower() in str(output).lower() else 0.0


@traigent.optimize(
    configuration_space={"style": ["short", "detailed"]},
    eval_dataset="data/qa.jsonl",
    scoring_function=contains_expected,
    objectives=["accuracy"],
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    return render_answer(question, style=cfg["style"])
```

## Runtime Controls

Runtime controls belong on `.optimize()`, not on the decorator.

```python
results = await answer.optimize(
    algorithm="grid",
    max_trials=50,
    timeout=3600,
    cost_limit=5.00,
    cost_approved=True,
)
```

`cost_approved=True` must be a real Python boolean. String values such as
`"true"` are ignored.

## Configuration Spaces

```python
space = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.0, 0.3, 0.7],
    "max_tokens": [128, 256, 512],
}
```

You can also use structured tuned-variable objects:

```python
from traigent import Choices, Range

space = {
    "model": Choices(["gpt-4o-mini", "gpt-4o"]),
    "temperature": Range(0.0, 1.0),
}
```

## Parameter Injection

Context injection is the default and works with `traigent.get_config()`:

```python
@traigent.optimize(configuration_space={"temperature": [0.0, 0.5]})
def summarize(text: str) -> str:
    cfg = traigent.get_config()
    return call_model(text, temperature=cfg["temperature"])
```

If you prefer explicit config arguments, opt into parameter injection:

```python
@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.5]},
    injection={"injection_mode": "parameter", "config_param": "config"},
)
def summarize(text: str, config: dict[str, object]) -> str:
    return call_model(text, temperature=float(config["temperature"]))
```

Valid injection modes are `context`, `parameter`, and `seamless`.

## Async Functions

```python
@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset="data/eval.jsonl",
    scoring_function=my_score,
)
async def async_answer(question: str) -> str:
    cfg = traigent.get_config()
    return await async_call_model(question, temperature=cfg["temperature"])


results = await async_answer.optimize(max_trials=10)
```

## Results

```python
print(results.best_config)
print(results.best_score)
print(results.status)
print(results.source)

successful = results.successful_trials
df = results.to_dataframe()
```

`successful_trials` is a list of successful `TrialResult` objects. Use
`len(results.successful_trials)` when you need a count.

`results.source` tells you which execution path ran:
`cloud_brain`, `local_fallback`, `explicit_local`, or `offline`.

## Progressive Runs

```python
quick = await answer.optimize(algorithm="random", max_trials=5)

refined_space = {
    "model": [quick.best_config["model"]],
    "temperature": [0.0, 0.2, 0.4, 0.6],
}

refined = await answer.optimize(
    configuration_space=refined_space,
    algorithm="grid",
    max_trials=20,
)
```

## External Service Evaluator

Use `ExternalServiceEvaluator` when trials are delegated to an external service
that implements the Traigent Hybrid API contract. This replaces the removed
`execution_mode="hybrid_api"` plus flat `hybrid_api_*` arguments; the optimizer
still runs locally and each trial evaluation is dispatched through the external
service.

```python
from traigent.api.decorators import ExternalServiceEvaluator, HybridAPIOptions

@traigent.optimize(
    evaluator=ExternalServiceEvaluator(
        hybrid_api=HybridAPIOptions(
            endpoint="http://localhost:8080",
            transport_type="http",
        )
    ),
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset="data/eval.jsonl",
    scoring_function=my_score,
)
def external_agent(_query: str) -> str:
    return ""


results = await external_agent.optimize(max_trials=10)
```

Because the optimizer is local in this setup, successful runs report
`results.source == "explicit_local"`.

## Multi-Field Evaluation Inputs

Many tasks need more than a single input string. A text2SQL task, for example,
requires both the natural-language question (`input`) and the database name
(`db_id`) and ground-truth SQL (`gold_sql`) for evaluation. The SDK supports
this through `example.metadata`.

### How extra JSONL fields are routed

When the SDK loads a JSONL dataset it applies the following mapping for each row:

| JSONL key | Maps to |
|---|---|
| `input` (or `input_data`) | `example.input_data` |
| `expected_output` (or `expected`, `output`, `answer`, `target`, `label`) | `example.expected_output` |
| **every other key** | `example.metadata` (a `dict`) |

A row like:

```jsonl
{"input": "How many employees live in each city?", "expected_output": "3 rows", "db_id": "company", "gold_sql": "SELECT city, COUNT(*) FROM employees GROUP BY city"}
```

produces an `EvaluationExample` where:
- `example.input_data` → `"How many employees live in each city?"`
- `example.expected_output` → `"3 rows"`
- `example.metadata` → `{"db_id": "company", "gold_sql": "SELECT city, COUNT(*) FROM employees GROUP BY city"}`

### Accessing metadata in a custom evaluator

Use `custom_evaluator` to read extra fields alongside the function output.
The evaluator receives the raw `EvaluationExample` and must return an
`ExampleResult`:

```python
import traigent
from traigent.api.types import ExampleResult


def sql_evaluator(func, config, example):
    db_id = example.metadata.get("db_id", "default")
    gold_sql = example.metadata.get("gold_sql", "")

    predicted_sql = func(example.input_data)
    correct = predicted_sql.strip().lower() == gold_sql.strip().lower()
    return ExampleResult(
        actual_output=predicted_sql,
        score=1.0 if correct else 0.0,
        success=correct,
    )


@traigent.optimize(
    eval_dataset="data/text2sql.jsonl",
    custom_evaluator=sql_evaluator,
    objectives=["accuracy"],
    configuration_space={
        "model": ["claude-3-haiku-20240307"],
        "temperature": [0.0, 0.3],
    },
)
def text2sql_agent(question: str) -> str:
    cfg = traigent.get_config()
    return call_llm(question, model=cfg["model"], temperature=cfg["temperature"])
```

> **Note:** a plain `scoring_function(output, expected)` does not receive
> `example.metadata`. Use `custom_evaluator` whenever the task needs extra
> fields beyond the predicted and expected outputs.

### Multi-field input dict

When the optimized function itself needs multiple inputs, set `input` to a
JSON object:

```jsonl
{"input": {"question": "Find all cities with more than 100 employees", "schema": "employees(id, city, dept)"}, "expected_output": "SELECT city FROM employees GROUP BY city HAVING COUNT(*) > 100"}
```

The SDK passes the dict's keys as keyword arguments if the function signature
matches (either via named parameters or `**kwargs`):

```python
@traigent.optimize(
    eval_dataset="data/text2sql.jsonl",
    scoring_function=my_score,
    objectives=["accuracy"],
    configuration_space={"model": ["claude-3-haiku-20240307"], "temperature": [0.0]},
)
def text2sql_agent(question: str, schema: str) -> str:
    cfg = traigent.get_config()
    return call_llm(
        f"Schema: {schema}\nQuestion: {question}",
        model=cfg["model"],
        temperature=cfg["temperature"],
    )
```

Alternatively, accept `**kwargs` to keep the signature flexible:

```python
@traigent.optimize(...)
def text2sql_agent(**kwargs) -> str:
    cfg = traigent.get_config()
    question = kwargs["question"]
    schema = kwargs.get("schema", "")
    return call_llm(f"Schema: {schema}\nQ: {question}", model=cfg["model"])
```

## Mock Smoke Tests

For tutorials and CI smoke tests, call mock mode explicitly near the top of the
file:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

Mock mode avoids provider spend and skips model-pricing preflight, but it is not
a production billing bypass.
