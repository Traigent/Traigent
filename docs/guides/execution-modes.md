# Traigent Execution Model

Traigent no longer exposes an execution-mode selector for normal optimization
workflows. Configure optimization with these public knobs:

- `algorithm` (default: `"auto"`): `"auto"`, `"grid"`, `"random"`, or a smart
  optimizer name such as Bayesian/TPE/CMA-ES when enabled by your account.
- `offline` (default: `False`): force local-only operation with zero Traigent
  backend egress.

Trials always run in your process. The cloud path supplies optimizer decisions;
it does not execute your dataset examples remotely.

## Behavior at a Glance

| Request | Backend behavior | Result `source` |
| --- | --- | --- |
| `algorithm="auto"` | Uses the Traigent cloud optimizer (`/next-trial`) when available; trials run locally | `cloud_brain` |
| `algorithm="auto"` and backend unavailable due to connectivity, missing credentials, or local backend unavailability | Warns and degrades to local grid/random | `local_fallback` |
| `algorithm="grid"` or `"random"` | Runs locally with no backend round trip | `explicit_local` |
| `offline=True`, `TRAIGENT_OFFLINE=1`, or `TRAIGENT_OFFLINE_MODE=1` | Runs locally with zero backend egress | `offline` |
| Smart algorithm name | Requires the cloud optimizer | `cloud_brain` |

Set `TRAIGENT_REQUIRE_CLOUD=1` to disable the automatic local fallback. With that
environment variable set, an unavailable backend is an error instead of a
`local_fallback` result.

Explicit smart algorithms require the cloud optimizer. They hard-error when
`offline=True` is set or when the backend is unavailable; they do not silently
downgrade to a local optimizer.

## Cloud-First Auto

The default `@optimize(...)` path uses `algorithm="auto"` and `offline=False`.
When cloud access is configured, the SDK creates a cloud-brain session and asks
the backend for each next configuration. Each suggested trial still executes
locally in your Python process.

```python
import traigent

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.5],
    },
    objectives=["accuracy"],
    evaluation={"eval_dataset": "data.jsonl"},
)
def answer(question: str) -> str:
    return run_agent(question)

result = answer.optimize(max_trials=8)
print(result.source)  # "cloud_brain" or "local_fallback"
```

Automatic fallback is limited to `algorithm="auto"`. It exists to keep the
default developer workflow usable when cloud optimization is unavailable,
including temporary backend/network failures, missing API keys, or an
unavailable local backend.

## Explicit Local

Use `grid` or `random` when you want a local optimizer and no cloud optimizer
round trip.

```python
result = answer.optimize(algorithm="grid", max_trials=8)
print(result.source)  # "explicit_local"
```

## Offline and No Egress

Use `offline=True`, `TRAIGENT_OFFLINE=1`, or the legacy
`TRAIGENT_OFFLINE_MODE=1` when no Traigent backend egress is allowed. This is
the air-gapped/no-network setting.

```python
import traigent

@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3, 0.7]},
    objectives=["accuracy"],
    evaluation={"eval_dataset": "data.jsonl"},
    algorithm="grid",
    offline=True,
)
def answer(question: str) -> str:
    return run_agent(question)

result = answer.optimize(max_trials=3)
print(result.source)  # "offline"
```

`offline=True`, `TRAIGENT_OFFLINE=1`, and `TRAIGENT_OFFLINE_MODE=1` prevent
Traigent backend calls. They do not prevent calls your own function makes to LLM
providers, databases, tools, or other services.

## Privacy Boundary

The default cloud-brain path is privacy-on by default, but privacy-on is not the
same as offline/no-network.

On the cloud-brain path, Traigent sends only optimizer metadata such as:

- configuration-space schema
- configuration and trial identifiers
- numeric metrics
- numeric counts and status values

It does not send dataset example content: no example inputs, expected/reference
outputs, prompts, responses, or example metadata. If your policy requires no
Traigent backend egress at all, use `offline=True`, `TRAIGENT_OFFLINE=1`, or
`TRAIGENT_OFFLINE_MODE=1`.

## Smart Algorithms

Smart algorithms are cloud optimizers. Requesting one explicitly means you want
the cloud optimizer to choose configurations.

```python
result = answer.optimize(algorithm="bayesian", max_trials=20)
print(result.source)  # "cloud_brain"
```

If the cloud is unavailable, or if `offline=True` is set, this run raises an
error. Use `algorithm="auto"` when you want cloud-first behavior with local
fallback.

## External Service Evaluation

The former `hybrid_api_*` flat options moved into the evaluator bundle.

```python
import traigent
from traigent.api.decorators import ExternalServiceEvaluator, HybridAPIOptions

@traigent.optimize(
    evaluator=ExternalServiceEvaluator(
        hybrid_api=HybridAPIOptions(
            endpoint="http://your-service:8080",
            tunable_id="my_agent",
            auth_header="Bearer <token>",
            batch_size=10,
            batch_parallelism=2,
        )
    ),
    configuration_space={
        "model": ["fast", "accurate", "balanced"],
        "temperature": [0.0, 0.5, 1.0],
    },
    objectives=["accuracy", "cost", "latency"],
    evaluation={"eval_dataset": "data.jsonl"},
)
def my_agent(_query: str) -> str:
    return ""  # Execution is delegated to the external service evaluator.
```

## Result Provenance

Optimization results expose one of these `source` values:

- `cloud_brain`: backend optimizer selected configurations; trials ran locally.
- `local_fallback`: `algorithm="auto"` fell back to local optimization because
  cloud connectivity was unavailable.
- `explicit_local`: caller requested `algorithm="grid"` or `"random"`.
- `offline`: caller set `offline=True`, `TRAIGENT_OFFLINE=1`, or
  `TRAIGENT_OFFLINE_MODE=1`.

## Legacy Mapping

Legacy fields are accepted for compatibility today and emit deprecation
warnings. They are targeted for removal in a future major release.

| Legacy setting | New behavior |
| --- | --- |
| `execution_mode="edge_analytics"` or `"local"` | `offline=True` / no Traigent backend egress |
| `execution_mode="hybrid"` or `"standard"` | cloud-first `algorithm="auto"` |
| `execution_mode="privacy"` | cloud-first `algorithm="auto"`; use `offline=True` for no egress |
| `execution_mode="cloud"` | cloud-first `algorithm="auto"` with a loud warning because historical semantics flipped |
| `execution_mode="hybrid_api"` | `evaluator=ExternalServiceEvaluator(hybrid_api=HybridAPIOptions(...))` |
| `privacy_enabled` | removed no-op with a deprecation warning |
| `cloud_fallback_policy` | removed no-op with a deprecation warning |
