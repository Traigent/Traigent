# EvaluationOptions Reference

`EvaluationOptions` is a Pydantic model that groups all evaluation-related settings for the `@traigent.optimize()` decorator.

```python
from traigent.api.decorators import EvaluationOptions
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `eval_dataset` | `str \| list[str] \| Dataset \| None` | `None` | Path to a JSONL evaluation dataset, a list of dataset paths, or a `Dataset` object. Each row should contain `input` or `input_data` plus an expected output field. |
| `custom_evaluator` | `Callable \| None` | `None` | A callable with signature `(func, config, example: EvaluationExample) -> ExampleResult`. Gives full control over how each example is executed and scored. |
| `scoring_function` | `Callable \| None` | `None` | A lightweight callable with signature `(prediction, expected) -> float`. Returns a numeric score for each example. |
| `metric_functions` | `dict[str, Callable] \| None` | `None` | Dictionary mapping metric names to callables. Each callable has signature `(prediction, expected, input_data) -> float`. |

## Custom Evaluator

The `custom_evaluator` gives you full control over trial execution and measurement. It receives the function, the trial config, and each dataset example.

### Signature

```python
def custom_evaluator(
    func: Callable,
    config: dict[str, Any],
    example: EvaluationExample,
) -> ExampleResult:
    ...
```

### Parameters

- `func` - The original decorated function (not wrapped).
- `config` - The configuration being tested in this trial (e.g., `{"model": "gpt-4", "temperature": 0.5}`).
- `example` - An `EvaluationExample` object. Read function inputs from `example.input_data`, expected values from `example.expected_output`, and extra JSONL fields from `example.metadata`.

### Return Value

Must return an `ExampleResult` containing at minimum:
- `example_id` (str) - Unique identifier for this example.
- `input_data` (dict) - The inputs used for the function call.
- `expected_output` - The expected value from the dataset.
- `actual_output` - The function output.
- `metrics` (dict[str, float]) - Per-objective scores, such as `{"accuracy": 1.0}`.
- `execution_time` (float) - Elapsed time in seconds.
- `success` (bool) - Whether the example evaluation succeeded.

### Example

```python
import time
from collections.abc import Callable
from typing import Any

from traigent.api.types import ExampleResult
from traigent.evaluators.base import EvaluationExample

def my_evaluator(
    func: Callable[..., Any],
    config: dict[str, Any],
    example: EvaluationExample,
) -> ExampleResult:
    start = time.perf_counter()
    actual_output = func(example.input_data["question"])
    execution_time = time.perf_counter() - start

    expected = example.expected_output
    accuracy = (
        1.0
        if expected is not None
        and str(expected).lower() in str(actual_output).lower()
        else 0.0
    )

    metadata = dict(example.metadata or {})
    if "model" in config:
        metadata["model"] = config["model"]

    return ExampleResult(
        example_id=str(metadata.get("example_id", "example")),
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output=actual_output,
        metrics={"accuracy": accuracy},
        execution_time=execution_time,
        success=True,
        metadata=metadata,
    )
```

```python
import traigent
from traigent.api.decorators import EvaluationOptions

@traigent.optimize(
    evaluation=EvaluationOptions(custom_evaluator=my_evaluator),
    objectives=["accuracy"],
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=question)
```

### Validation

Traigent validates the `custom_evaluator` signature at decoration time. If your callable has parameters named `prediction`, `expected`, and `input_data`, Traigent will raise a `ValidationError` suggesting you use `metric_functions` instead. This catches a common mistake where a metric evaluator is passed as a custom evaluator.

## Scoring Function

A simpler alternative to `custom_evaluator`. Traigent handles function execution and passes the output and expected value to your scorer.

### Signature

```python
def scoring_function(prediction: str, expected: str) -> float:
    ...
```

### Example

```python
def fuzzy_match(prediction: str, expected: str) -> float:
    pred = prediction.strip().lower()
    exp = expected.strip().lower()
    if pred == exp:
        return 1.0
    if exp in pred:
        return 0.8
    return 0.0

@traigent.optimize(
    evaluation=EvaluationOptions(
        eval_dataset="test_data.jsonl",
        scoring_function=fuzzy_match,
    ),
    objectives=["accuracy"],
    configuration_space={"temperature": [0.0, 0.5, 1.0]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(temperature=cfg["temperature"], prompt=query)
```

## Metric Functions

Use `metric_functions` when you want to track multiple named metrics per example. Each metric function receives the prediction, expected output, and input data.

### Signature

```python
def metric_fn(prediction: Any, expected: Any, input_data: dict) -> float:
    ...
```

### Example

```python
def accuracy(prediction, expected, input_data) -> float:
    return 1.0 if prediction.strip() == expected.strip() else 0.0

def conciseness(prediction, expected, input_data) -> float:
    return max(0.0, 1.0 - len(prediction) / 2000)

def relevance(prediction, expected, input_data) -> float:
    keywords = input_data.get("keywords", [])
    if not keywords:
        return 1.0
    found = sum(1 for kw in keywords if kw.lower() in prediction.lower())
    return found / len(keywords)

@traigent.optimize(
    evaluation=EvaluationOptions(
        eval_dataset="eval_set.jsonl",
        metric_functions={
            "accuracy": accuracy,
            "conciseness": conciseness,
            "relevance": relevance,
        },
    ),
    objectives=["accuracy"],
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def summarize(text: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=f"Summarize: {text}")
```

## Choosing Between Evaluation Approaches

| Scenario | Recommended Approach |
|---|---|
| Standard accuracy on a JSONL dataset | `eval_dataset` alone (uses built-in evaluation) |
| Simple right/wrong scoring | `scoring_function` |
| Multiple metrics (accuracy + latency + cost) | `metric_functions` |
| Custom execution logic (retries, pre/post processing) | `custom_evaluator` |
| LLM-as-judge evaluation | `custom_evaluator` (call judge LLM inside evaluator) |

## Dataset Format

The `eval_dataset` JSONL file should have one JSON object per line. At minimum, include an `input` object whose keys match your function parameters and an expected output field. Extra top-level fields become `example.metadata`:

```json
{"input": {"question": "What is Python?"}, "expected": "A programming language", "db_path": "eval.sqlite"}
{"input": {"question": "What is 2+2?"}, "expected": "4", "db_path": "eval.sqlite"}
```

Multiple datasets can be provided as a list:

```python
EvaluationOptions(eval_dataset=["train.jsonl", "validation.jsonl"])
```
