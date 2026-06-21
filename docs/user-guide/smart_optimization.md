# Smart Optimization

Traigent SDKs expose a small local optimizer surface by default:

- `grid`
- `random`

Smart optimization is backend-routed. Use a TraigentBackend typed session or
hybrid SDK mode when you need guided search beyond the local defaults.

## Public Smart Strategies

| Strategy         | Runs In         | Use Case                                                 |
| ---------------- | --------------- | -------------------------------------------------------- |
| `bayesian`       | TraigentBackend | Public alias for guided TPE-style search                 |
| `tpe`            | TraigentBackend | Guided search over categorical and continuous spaces     |
| `hyperband`      | TraigentBackend | Budget-aware early-stopping/pruning for expensive trials |
| `frontier_scout` | TraigentBackend | Evidence-aware frontier tracking for multi-metric runs   |

Direct optimizer-engine names are not part of the SDK contract. SDKs should
send the public strategy name and let TraigentBackend resolve the implementation.

## Python Hybrid Session Example

```python
from traigent.cloud.sessions import TraigentSessionClient

client = TraigentSessionClient()

session = await client.create_optimization_session(
    function_name="answer_quality",
    configuration_space={
        "model": {"type": "categorical", "choices": ["fast", "accurate"]},
        "temperature": {"type": "float", "low": 0.0, "high": 1.0},
    },
    objectives=["accuracy", "cost"],
    dataset_metadata={"size": 20},
    max_trials=12,
    optimization_strategy={"algorithm": "tpe"},
)
```

Use `hyperband` when trial results can be reported progressively and
`frontier_scout` when the trial metadata includes per-example evidence rows for
frontier analysis.

## Local Defaults

For offline/local SDK runs, keep using `grid` or `random`. These require no
backend service and remain deterministic/reproducible under the existing local
SDK execution path.
