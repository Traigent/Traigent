# Execution Mode Migration

Traigent's execution model is now defined by `algorithm` plus `offline`. The
legacy `execution_mode`, `privacy_enabled`, and `cloud_fallback_policy` inputs
are deprecated compatibility shims, not the current public model.

## Canonical Model

- `algorithm="auto"`: cloud-first optimizer selection with local trial
  execution. If the backend is unavailable because of missing credentials,
  offline connectivity, or backend failure, Traigent warns once and degrades to
  local grid/random.
- `algorithm="grid"` or `"random"`: explicit local optimization with no
  backend round trip.
- Explicit smart algorithms such as `"bayesian"`, `"tpe"`, `"optuna_tpe"`,
  `"nsga2"`, or `"cmaes"`: cloud-only. If the cloud is unavailable, the run
  errors. It does not silently downgrade.
- `offline=True`: force zero Traigent backend egress. Environment equivalents:
  `TRAIGENT_OFFLINE=1` and the legacy `TRAIGENT_OFFLINE_MODE=1`.
- `TRAIGENT_REQUIRE_CLOUD=1`: disable the automatic local fallback for
  `algorithm="auto"` and hard-error instead.

## Deprecated Mapping Table

| Deprecated input | Current meaning | What to write now |
| --- | --- | --- |
| `execution_mode="edge_analytics"` | local-only / no Traigent backend egress | `offline=True` |
| `execution_mode="local"` | local-only / no Traigent backend egress | `offline=True` |
| `execution_mode="hybrid"` | cloud-first automatic optimization | omit it, or set `algorithm="auto"` |
| `execution_mode="standard"` | cloud-first automatic optimization | omit it, or set `algorithm="auto"` |
| `execution_mode="privacy"` | cloud-first automatic optimization; not no-egress | omit it, or set `algorithm="auto"`; use `offline=True` for no egress |
| `execution_mode="cloud"` | cloud-first automatic optimization, despite the historical local meaning | omit it, or set `algorithm="auto"` |
| `execution_mode="hybrid_api"` | external-service evaluation moved off the mode axis | `evaluator=ExternalServiceEvaluator(hybrid_api=HybridAPIOptions(...))` |
| `privacy_enabled` | deprecated no-op | remove it; use `offline=True` only when no egress is required |
| `cloud_fallback_policy` | deprecated no-op | remove it; use `TRAIGENT_REQUIRE_CLOUD=1` when fallback must be disabled |

## Copy-Paste Replacements

```python
import traigent

# Before
@traigent.optimize(execution_mode="hybrid")
def answer(question: str) -> str:
    return run_agent(question)

# After
@traigent.optimize()
def answer(question: str) -> str:
    return run_agent(question)
```

```python
import traigent

# Before
@traigent.optimize(execution_mode="edge_analytics")
def answer(question: str) -> str:
    return run_agent(question)

# After
@traigent.optimize(offline=True)
def answer(question: str) -> str:
    return run_agent(question)
```

```python
import traigent

# Before
@traigent.optimize(
    execution_mode="hybrid_api",
    hybrid_api_endpoint="https://svc",
)
def my_agent(_query: str) -> str:
    return ""

# After
from traigent.api.decorators import ExternalServiceEvaluator, HybridAPIOptions

@traigent.optimize(
    evaluator=ExternalServiceEvaluator(
        hybrid_api=HybridAPIOptions(endpoint="https://svc")
    )
)
def my_agent(_query: str) -> str:
    return ""
```

```python
import traigent

# Before
@traigent.optimize(privacy_enabled=True)
def answer(question: str) -> str:
    return run_agent(question)

# After
@traigent.optimize()
def answer(question: str) -> str:
    return run_agent(question)
```

If the old intent behind `privacy_enabled=True` was "no Traigent network
egress", the correct replacement is `offline=True`.

## Privacy and No-Egress

The cloud-first path is content-free by design: it sends configuration IDs and
numeric metrics, not dataset example content. That default privacy boundary is
not the same as offline operation.

- Use the default cloud-first path when backend egress is allowed.
- Use `offline=True` when policy requires zero Traigent backend egress.

## Result Provenance

Inspect `result.metadata["source"]` to see how a run actually executed:

- `cloud_brain`: the cloud optimizer selected configurations.
- `local_fallback`: `algorithm="auto"` degraded to local optimization after a
  connectivity failure.
- `explicit_local`: you requested `algorithm="grid"` or `"random"`.
- `offline`: you forced offline mode.
