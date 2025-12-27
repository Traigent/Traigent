# Stop Condition Test Scenario Templates

This file provides copyable templates for optimizer validation tests. All placeholders
use the same variable names as `TestScenario` and the fixtures in
`tests/optimizer_validation/conftest.py` so you can paste and adjust quickly.

## Base Scenario Template
```python
from tests.optimizer_validation.specs import (
    EvaluatorSpec,
    ExpectedOutcome,
    ExpectedResult,
    TestScenario,
)
from tests.optimizer_validation.specs.trace_expectations import TraceExpectations

name = "..."
description = "..."
config_space = {"model": ["gpt-3.5-turbo", "gpt-4"]}
max_trials = 5
timeout = 30.0
parallel_config = {"mode": "sequential"}
dataset_size = 3
evaluator = EvaluatorSpec(type="default")

expected = ExpectedResult(
    outcome=ExpectedOutcome.SUCCESS,
    min_trials=1,
    max_trials=max_trials,
    expected_stop_reason=None,
)

trace_expectations = TraceExpectations(
    min_trial_spans=1,
)

scenario = TestScenario(
    name=name,
    description=description,
    config_space=config_space,
    max_trials=max_trials,
    timeout=timeout,
    parallel_config=parallel_config,
    dataset_size=dataset_size,
    evaluator=evaluator,
    expected=expected,
    trace_expectations=trace_expectations,
)
```

## Stop Reason: Max Trials (Deterministic)
```python
max_trials = 3
scenario = TestScenario(
    name="max_trials_reason",
    description="Stop reason when max_trials reached",
    config_space={"model": ["gpt-3.5-turbo"]},
    max_trials=max_trials,
)

func, result = await scenario_runner(scenario)
assert not isinstance(result, Exception)
assert len(result.trials) == max_trials
if result.stop_reason is not None:
    assert result.stop_reason == "max_trials_reached"
```

## Stop Reason: Config Space Exhaustion
Use a tiny config space and assert unique configs do not exceed the space size.
```python
config_space = {"model": ["gpt-3.5-turbo"]}
max_trials = 5
scenario = TestScenario(
    name="config_space_exhaustion",
    description="Config space exhaustion behavior",
    config_space=config_space,
    max_trials=max_trials,
)

func, result = await scenario_runner(scenario)
assert not isinstance(result, Exception)
unique_configs = {tuple(sorted(t.config.items())) for t in result.trials}
assert len(unique_configs) <= len(config_space["model"])
if result.stop_reason is not None:
    assert result.stop_reason in ("optimizer", "max_trials_reached")
```

## Sample Budget + LLM/Judge Invocation Visibility
Track calls explicitly and verify trace spans even if the final result does not expose
per-example counts.
```python
from traigent.api.types import ExampleResult
from tests.optimizer_validation.specs import EvaluatorSpec
from tests.optimizer_validation.specs.trace_expectations import TraceExpectations

max_trials = 4
dataset_size = 20
llm_calls = {"count": 0}

def counting_evaluator(func, config, example) -> ExampleResult:
    llm_calls["count"] += 1
    actual_output = func(example.input_data.get("text", ""))
    return ExampleResult(
        example_id=str(id(example)),
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output=actual_output,
        metrics={"accuracy": 1.0},
        execution_time=0.01,
        success=True,
    )

def check_example_budget(trace):
    for span in trace.evaluation_spans:
        examples_count = span.attributes.get("evaluation.examples_count", 0)
        if examples_count > dataset_size:
            return f"evaluation.examples_count {examples_count} > {dataset_size}"
    return None

trace_expectations = TraceExpectations(
    custom_validators=[check_example_budget],
)

scenario = TestScenario(
    name="sample_budget_visibility",
    description="Ensure per-example work stays within budget",
    max_trials=max_trials,
    dataset_size=dataset_size,
    evaluator=EvaluatorSpec(type="custom", evaluator_fn=counting_evaluator),
    trace_expectations=trace_expectations,
)

func, result, trace = await traced_scenario_runner(scenario)
assert not isinstance(result, Exception)
assert llm_calls["count"] <= max_trials * dataset_size
trace_validation = trace_validator(scenario, trace, result)
assert trace_validation.passed, trace_validation.errors
```

## Concurrency: Require Overlap of Trial Spans
Make each trial slow enough to overlap so the trace can verify actual concurrency.
```python
import time
from traigent.api.types import ExampleResult

trial_concurrency = 4
max_trials = 8

def slow_evaluator(func, config, example) -> ExampleResult:
    time.sleep(0.2)
    actual_output = func(example.input_data.get("text", ""))
    return ExampleResult(
        example_id=str(id(example)),
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output=actual_output,
        metrics={"accuracy": 1.0},
        execution_time=0.2,
        success=True,
    )

def check_overlap(trace):
    events = []
    for span in trace.trial_spans:
        events.append((span.start_time_ns, 1))
        events.append((span.end_time_ns, -1))
    events.sort()
    current = 0
    max_overlap = 0
    for _, delta in events:
        current += delta
        max_overlap = max(max_overlap, current)
    if max_overlap < trial_concurrency:
        return f"max overlap {max_overlap} < trial_concurrency {trial_concurrency}"
    return None

scenario = TestScenario(
    name="trial_overlap",
    description="Parallel trials overlap in time",
    max_trials=max_trials,
    parallel_config={"mode": "parallel", "trial_concurrency": trial_concurrency},
    evaluator=EvaluatorSpec(type="custom", evaluator_fn=slow_evaluator),
    trace_expectations=TraceExpectations(custom_validators=[check_overlap]),
)

func, result, trace = await traced_scenario_runner(scenario)
assert not isinstance(result, Exception)
trace_validation = trace_validator(scenario, trace, result)
assert trace_validation.passed, trace_validation.errors
```

## Async Parallel Failure Handling (Function/Eval/Dataset)
Ensure the orchestrator terminates all trials and does not deadlock under errors.
```python
from tests.optimizer_validation.specs import ExpectedOutcome, ExpectedResult
from tests.optimizer_validation.specs.trace_expectations import failure_expectations

parallel_config = {"mode": "parallel", "trial_concurrency": 4}

scenario = TestScenario(
    name="parallel_failure_function",
    description="Function raises mid-run, all trials terminate",
    parallel_config=parallel_config,
    function_should_raise=RuntimeError,
    function_raise_on_call=2,
    expected=ExpectedResult(outcome=ExpectedOutcome.FAILURE),
    trace_expectations=failure_expectations(allow_zero_trials=True),
)

func, result, trace = await traced_scenario_runner(scenario)
if isinstance(result, Exception):
    trace_validation = trace_validator(scenario, trace, result=None)
else:
    trace_validation = trace_validator(scenario, trace, result)
assert trace_validation.passed, trace_validation.errors
```

To cover evaluator and dataset failures, swap in:
- `EvaluatorSpec(type="custom", evaluator_fn=failing_evaluator)`
- `dataset_path` pointing to a dataset built with `include_malformed=True`

## Max Samples Stop Condition (Requires Harness Support)
If you extend `scenario_runner` to pass `max_samples` into `decorated.optimize`,
use this template to verify `max_samples_reached`:
```python
max_samples = 20
scenario = TestScenario(
    name="max_samples_reached",
    description="Stop when max_samples reached",
    dataset_size=100,
    max_trials=100,
)

func, result = await scenario_runner(scenario, max_samples=max_samples)
assert not isinstance(result, Exception)
if result.stop_reason is not None:
    assert result.stop_reason == "max_samples_reached"
```
