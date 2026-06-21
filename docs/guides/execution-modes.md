# Optimization Routing

Most users should not set routing options. Set `TRAIGENT_API_KEY`, decorate your
function, and call `.optimize(...)`. The default uses Traigent's smart optimizer
and syncs results to the portal.

Traigent exposes two routing knobs for cases that need them:

- `algorithm`: omit it or use `"auto"` for the default smart optimizer; use
  `"grid"` or `"random"` for explicit local search; use smart algorithm names
  such as `"bayesian"` or `"optuna_tpe"` only when cloud optimization is
  available.
- `offline`: set `True` only for zero Traigent backend egress. Offline runs do
  not sync to the portal.

Trials run in your Python process. The cloud optimizer chooses configurations;
it does not execute your dataset examples remotely.

## Behavior

| Request | Optimization decision source | Portal sync |
| --- | --- | --- |
| Omit routing settings | Traigent smart optimizer when authenticated | Yes |
| `algorithm="auto"` | Same as the default | Yes |
| `algorithm="grid"` or `"random"` | Local search in your Python process | Yes, unless `offline=True` |
| Smart algorithm name | Traigent cloud optimizer; errors if unavailable | Yes |
| `offline=True` | Local only, zero Traigent backend egress | No |

Local search is not the same as offline. `grid` and `random` run locally but
still upload run metadata and results to the portal when credentials are
available. Add `offline=True` only when no Traigent backend traffic is allowed.

## Default Smart Optimization

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
```

## Explicit Local Search

Use local search when you want a simple sweep or reproducible baseline. Results
still sync to the portal unless you also set `offline=True`.

```python
result = answer.optimize(algorithm="grid", max_trials=8)
```

## No-Egress Local Runs

Use `offline=True` when policy requires zero Traigent backend egress. This only
controls Traigent traffic; your function may still call LLM providers, tools, or
databases.

```python
@traigent.optimize(
    configuration_space={"temperature": [0.0, 0.3, 0.7]},
    objectives=["accuracy"],
    evaluation={"eval_dataset": "data.jsonl"},
    offline=True,
)
def answer(question: str) -> str:
    return run_agent(question)

result = answer.optimize(algorithm="grid", max_trials=3)
```

## Smart Algorithms

Explicit smart algorithms are cloud-required. Do not request them for local-only
runs.

```python
result = answer.optimize(algorithm="bayesian", max_trials=20)
```

If cloud optimization is unavailable, this run raises an error. Use the default
when you want Traigent to pick the best available path.

## Migration Note

Older examples may contain `execution_mode=...`. That selector is deprecated.
Remove it for the default portal-tracked path. If the old intent was zero
Traigent backend egress, use `offline=True`. If the old intent was local search
that can still sync results, use `algorithm="grid"` or `"random"`.
