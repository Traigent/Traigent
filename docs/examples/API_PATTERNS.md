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

successful = results.successful_trials
df = results.to_dataframe()
```

`successful_trials` is a list of successful `TrialResult` objects. Use
`len(results.successful_trials)` when you need a count.

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

## Hybrid API

Use `execution_mode="hybrid_api"` when trials are delegated to an external
service that implements the Traigent Hybrid API contract.

```python
@traigent.optimize(
    execution={
        "execution_mode": "hybrid_api",
        "hybrid_api_endpoint": "http://localhost:8080",
    },
    configuration_space={"temperature": [0.0, 0.3]},
    eval_dataset="data/eval.jsonl",
    scoring_function=my_score,
)
def external_agent(_query: str) -> str:
    return ""


results = await external_agent.optimize(max_trials=10)
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
