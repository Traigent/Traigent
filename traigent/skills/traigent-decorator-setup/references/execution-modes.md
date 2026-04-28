# Execution Modes Reference

`ExecutionOptions` is a Pydantic model that controls where and how optimization runs execute.

```python
from traigent.api.decorators import ExecutionOptions
```

## ExecutionOptions Fields

### Core Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `execution_mode` | `str` | `"edge_analytics"` | Where to run: `"edge_analytics"` for local, `"hybrid"` for portal-tracked local execution, or `"cloud"` for the future remote execution path. |
| `local_storage_path` | `str \| None` | `None` | Directory path for local result storage. |
| `minimal_logging` | `bool` | `True` | Minimize logging output during optimization. |
| `parallel_config` | `ParallelConfig \| dict \| None` | `None` | Parallel execution settings. See ParallelConfig section. |
| `privacy_enabled` | `bool \| None` | `None` | Enable privacy-preserving mode (no raw data sent to cloud). |
| `max_total_examples` | `int \| None` | `None` | Cap total examples evaluated across all trials. |
| `samples_include_pruned` | `bool` | `True` | Whether pruned trials count toward sample limits. |
| `cloud_fallback_policy` | `str \| None` | `None` | Legacy setting for future cloud execution. `cloud` is not available yet and fails closed. |

### Repetition Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `reps_per_trial` | `int` | `1` | Number of times to repeat each configuration. Useful for noisy LLM evaluations. Set to 3-5 for statistical stability. |
| `reps_aggregation` | `str` | `"mean"` | How to aggregate metrics across repetitions: `"mean"`, `"median"`, `"min"`, or `"max"`. |

### JS Bridge Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `runtime` | `str` | `"python"` | Runtime for trial execution: `"python"` or `"node"`. |
| `js_module` | `str \| None` | `None` | Path to the JS module containing the trial function. Required when `runtime="node"`. |
| `js_function` | `str` | `"runTrial"` | Name of the exported function to call in the JS module. |
| `js_timeout` | `float` | `300.0` | Timeout for JS trial execution in seconds. |
| `js_parallel_workers` | `int` | `1` | Number of parallel JS worker processes. |

### Hybrid API Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `hybrid_api_endpoint` | `str \| None` | `None` | URL of the hybrid API server. |
| `tunable_id` | `str \| None` | `None` | Identifier for this tunable in the hybrid API. |
| `hybrid_api_transport` | `Any \| None` | `None` | Custom transport instance for the hybrid API. |
| `hybrid_api_transport_type` | `str` | `"auto"` | Transport type: `"auto"`, `"http"`, `"grpc"`, etc. |
| `hybrid_api_batch_size` | `int` | `1` | Batch size for hybrid API requests. |
| `hybrid_api_batch_parallelism` | `int` | `1` | Number of parallel batches. |
| `hybrid_api_keep_alive` | `bool` | `True` | Keep the hybrid API connection alive between trials. |
| `hybrid_api_heartbeat_interval` | `float` | `30.0` | Heartbeat interval in seconds. |
| `hybrid_api_timeout` | `float \| None` | `None` | Request timeout for hybrid API calls. |
| `hybrid_api_auth_header` | `str \| None` | `None` | Authorization header value for hybrid API. |
| `hybrid_api_auto_discover_tvars` | `bool` | `False` | Automatically discover tunable variables from the hybrid API. |

## Execution Modes Explained

### Edge Analytics (Default)

Optimization runs on your local machine. Set `TRAIGENT_OFFLINE_MODE=true` when you want no Traigent backend communication. Raw data, prompts, and model outputs never leave your environment in local mode.

```python
@traigent.optimize(
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        local_storage_path="./results",
        privacy_enabled=True,
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)
```

**When to use**: Most local/private use cases.

### Hybrid

Supported portal-tracked execution. Trials run locally, while the backend stores sessions and trial metrics for website visibility.

```python
@traigent.optimize(
    execution=ExecutionOptions(execution_mode="hybrid"),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)
```

**When to use**: When you want results in the Traigent portal while keeping trial execution local.

### Cloud

Reserved for future remote execution. It is not implemented yet and fails with: “Cloud remote execution is not available yet; use hybrid for portal-tracked optimization.”

Do not configure new runs with `execution_mode="cloud"` yet. It raises a clear
unavailable error instead of starting a synthetic remote optimization.

**When to use**: Do not use yet. Choose `hybrid` for portal-tracked optimization.

### Cloud Fallback Policy

Cloud fallback policy is retained for compatibility with the future cloud path. It does not make `execution_mode="cloud"` available today:

| Policy | Behavior |
|---|---|
| `"auto"` | Reserved for future cloud behavior; currently inert because cloud mode fails validation before fallback. |
| `"warn"` | Reserved for future cloud behavior; currently inert because cloud mode fails validation before fallback. |
| `"never"` | Reserved for future cloud behavior; currently inert because cloud mode fails validation before fallback. |

## ParallelConfig Integration

Pass a `ParallelConfig` to run trials and/or examples in parallel:

```python
from traigent.config.parallel import ParallelConfig

@traigent.optimize(
    execution=ExecutionOptions(
        parallel_config=ParallelConfig(
            mode="parallel",
            trial_concurrency=2,
            example_concurrency=4,
        ),
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)
```

You can also pass it as a dictionary:

```python
@traigent.optimize(
    execution=ExecutionOptions(
        parallel_config={
            "mode": "parallel",
            "trial_concurrency": 2,
            "example_concurrency": 4,
        },
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg["model"], prompt=query)
```

## JS Bridge (Node.js Execution)

Run trials in a Node.js subprocess. Useful when your LLM application is written in JavaScript/TypeScript.

```python
@traigent.optimize(
    execution=ExecutionOptions(
        runtime="node",
        js_module="./src/trial.js",
        js_function="runTrial",
        js_timeout=120.0,
        js_parallel_workers=2,
    ),
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
def my_func(query: str) -> str:
    # This function body is not executed when runtime="node"
    # Instead, the JS module's runTrial function is called
    pass
```

The JS module should export a function matching `js_function`:

```javascript
// src/trial.js
async function runTrial(config, example) {
    const response = await callLLM({
        model: config.model,
        prompt: example.question,
    });
    return {
        prediction: response.text,
        score: response.text.includes(example.expected) ? 1.0 : 0.0,
    };
}

module.exports = { runTrial };
```

## Repetitions for Statistical Stability

LLM outputs are non-deterministic. Use `reps_per_trial` to repeat each configuration multiple times and aggregate the results:

```python
@traigent.optimize(
    execution=ExecutionOptions(
        reps_per_trial=5,          # Run each config 5 times
        reps_aggregation="median", # Use median score (robust to outliers)
    ),
    configuration_space={"temperature": [0.0, 0.3, 0.7, 1.0]},
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(temperature=cfg["temperature"], prompt=query)
```

Aggregation options:

| Method | Use When |
|---|---|
| `"mean"` | Default. Good for normally distributed scores. |
| `"median"` | Robust to outliers. Good for noisy evaluations. |
| `"min"` | Worst-case analysis. Conservative. |
| `"max"` | Best-case analysis. Optimistic. |
