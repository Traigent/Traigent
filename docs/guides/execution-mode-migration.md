# Algorithm and Offline Migration

The public routing model is now `algorithm` plus `offline`.

Most code should remove old routing settings entirely:

```python
@traigent.optimize()
def answer(question: str) -> str:
    return run_agent(question)
```

With `TRAIGENT_API_KEY` set, the default uses Traigent's smart optimizer and
syncs results to the portal. `grid` and `random` run local search, but they also
sync results when authenticated. `offline=True` is the only no-egress local path.

## Replacements

| Old intent | Write now |
| --- | --- |
| Default portal-tracked optimization | Omit routing settings |
| Explicit local search with portal results | `answer.optimize(algorithm="grid")` or `answer.optimize(algorithm="random")` |
| Zero Traigent backend egress | `@traigent.optimize(offline=True)` and use `grid` or `random` |
| Explicit smart optimizer | `answer.optimize(algorithm="bayesian")` or another cloud-required smart algorithm |

Legacy `execution_mode=...` inputs are deprecated compatibility shims. Remove
them in new code. If you are unsure what an old value meant, start by omitting
it; add `offline=True` only when the requirement is no Traigent backend egress.

## External API Contracts

External-service API contract docs may still use `/hybrid` endpoint names
because those are backend route names. Do not use those names as optimization
routing selectors in new SDK code.
