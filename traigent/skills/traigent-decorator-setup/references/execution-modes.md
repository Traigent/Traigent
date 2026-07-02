# Execution Modes Reference

`ExecutionOptions` is a Pydantic model that controls where and how optimization runs execute.

```python
from traigent.api.decorators import ExecutionOptions
```

## ExecutionOptions Fields

### Core Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `execution_mode` | `str` | auto-selected | Auto-selected based on algorithm and transport. `"local"` for grid/random; `"hybrid"` for smart algorithms; `"hybrid_api"` for REST external services. Portal tracking is controlled by `TRAIGENT_API_KEY`, not this field. `"cloud"` is reserved for future use. |
| `local_storage_path` | `str \| None` | `None` | Directory path for local result storage. |
| `minimal_logging` | `bool` | `True` | Minimize logging output during optimization. |
| `parallel_config` | `ParallelConfig \| dict \| None` | `None` | Parallel execution settings. See ParallelConfig section. |
| `privacy_enabled` | `bool \| None` | `None` | Enable privacy-preserving mode (no raw data sent to cloud). |
| `max_total_examples` | `int \| None` | `None` | Cap total examples evaluated across all trials. |
| `samples_include_pruned` | `bool` | `True` | Whether pruned trials count toward sample limits. |
| `cloud_fallback_policy` | `str \| None` | `None` | Legacy setting for future cloud execution. `cloud` is not available yet and fails closed. |

### Repetition Fields (enterprise-only)

| Field | Type | Default | Description |
|---|---|---|---|
| `reps_per_trial` | `int` | `1` | **Enterprise-only.** Number of times to repeat each configuration. Only `1` is accepted in the OSS SDK; any other value raises `ValidationError` at `ExecutionOptions` construction. Per-configuration repetition requires Traigent Enterprise (contact `sales@traigent.ai`). |
| `reps_aggregation` | `str` | `"mean"` | **Enterprise-only.** How to aggregate metrics across repetitions. Only `"mean"` is accepted in the OSS SDK; any other value raises `ValidationError`. Per-configuration repetition aggregation requires Traigent Enterprise. |

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
        execution_mode="local",
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

Auto-selected when smart algorithms (Bayesian, TPE, CMA-ES, NSGA-II) are used. The cloud assists with suggestion generation; trials still run locally.

```python
@traigent.optimize(
    # execution_mode is auto-selected — no need to set it manually
    # Set TRAIGENT_API_KEY for portal tracking
    configuration_space={“model”: [“gpt-3.5-turbo”, “gpt-4”]},
    objectives=[“accuracy”, “cost”],
)
def my_func(query: str) -> str:
    cfg = traigent.get_config()
    return call_llm(model=cfg[“model”], prompt=query)
```

**When to use**: Automatically used for smart algorithm runs. For portal tracking with grid/random search, just set `TRAIGENT_API_KEY` — mode stays `local`.

### Cloud

Reserved for future remote execution. It is not implemented yet and fails closed.

Do not configure new runs with `execution_mode=”cloud”` yet.

**When to use**: Do not use yet. For portal-tracked runs, set `TRAIGENT_API_KEY` — mode is auto-selected.

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

## Repetitions for Statistical Stability (enterprise-only)

LLM outputs are non-deterministic. Per-configuration repetition aggregation is
an **enterprise-only** feature in the current OSS SDK release. Constructing
`ExecutionOptions` with any non-default value for `reps_per_trial` or
`reps_aggregation` raises a `pydantic.ValidationError` at construction time
because the gate is enforced at the contract boundary instead of
late in the optimization run.

```python
# Enterprise-only - raises ValidationError in the OSS SDK:
# ExecutionOptions(reps_per_trial=5, reps_aggregation="median")
```

Aggregation options shipped with Traigent Enterprise:

| Method | Use When |
|---|---|
| `"mean"` | Default. Good for normally distributed scores. Only value accepted by the OSS SDK. |
| `"median"` | Enterprise-only. Robust to outliers; good for noisy evaluations. |
| `"min"` | Enterprise-only. Worst-case analysis; conservative. |
| `"max"` | Enterprise-only. Best-case analysis; optimistic. |

Contact `sales@traigent.ai` to unlock per-configuration repetitions.
