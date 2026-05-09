# EvaluationOptions Reference

`EvaluationOptions` is a Pydantic model that groups all evaluation-related settings for the `@traigent.optimize()` decorator.

```python
from traigent.api.decorators import EvaluationOptions
```

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `eval_dataset` | `str \| list[str] \| Dataset \| None` | `None` | Path to a JSONL evaluation dataset, a list of dataset paths, or a `Dataset` object. Each row should contain input fields and an expected output. |
| `custom_evaluator` | `Callable \| None` | `None` | A callable with signature `(func, config, example) -> ExampleResult`. Gives full control over how each example is executed and scored. |
| `scoring_function` | `Callable \| None` | `None` | A lightweight callable with signature `(prediction, expected) -> float`. Returns a numeric score for each example. |
| `metric_functions` | `dict[str, Callable] \| None` | `None` | Dictionary mapping metric names to callables. Each callable has signature `(prediction, expected, input_data) -> float`. |

## Custom Evaluator

The `custom_evaluator` gives you full control over trial execution and measurement. It receives the function, the trial config, and each dataset example.

### Signature

```python
def custom_evaluator(
    func: Callable,
    config: dict[str, Any],
    example: dict[str, Any],
) -> ExampleResult:
    ...
```

### Parameters

- `func` - The original decorated function (not wrapped).
- `config` - The configuration being tested in this trial (e.g., `{"model": "gpt-4", "temperature": 0.5}`).
- `example` - A single row from the eval dataset as a dictionary.

### Return Value

Must return an `ExampleResult` (or compatible dict) containing at minimum:
- `score` (float) - The primary score for this example.
- `prediction` (str) - The model output.

### Example

```python
from traigent.evaluators.base import ExampleResult

def my_evaluator(func, config, example):
    import time
    start = time.time()
    prediction = func(example["question"])
    latency = time.time() - start

    score = 1.0 if example["expected"] in prediction else 0.0

    return ExampleResult(
        score=score,
        prediction=prediction,
        measures={"latency_ms": latency * 1000},
    )

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

The `eval_dataset` JSONL file should have one JSON object per line. At minimum, include input fields that match your function parameters and an `expected` field:

```json
{"question": "What is Python?", "expected": "A programming language"}
{"question": "What is 2+2?", "expected": "4"}
```

Multiple datasets can be provided as a list:

```python
EvaluationOptions(eval_dataset=["train.jsonl", "validation.jsonl"])
```
