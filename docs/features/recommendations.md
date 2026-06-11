# Configuration Recommendations

Traigent 0.12.0 exposes catalog-backed recommendations for configuration spaces.
Use them to bootstrap candidate knobs for an agent or task type before validating
the choices on your own evaluation dataset.

## Public API

```python
from traigent.api.functions import (
    list_recommendation_agent_types,
    recommend_configuration_space,
)

print(list_recommendation_agent_types())

recommendations = recommend_configuration_space(
    "rag",
    min_impact="medium",
    min_confidence="medium",
)
```

`list_recommendation_agent_types()` returns the valid catalog task types.

`recommend_configuration_space(agent_type, *, min_impact=None,
min_confidence=None)` returns a JSON-serializable response with version metadata,
a caveat, a `configuration_space`, and recommendation rows.

The optional filters accept:

- `low`
- `medium`
- `high`

## CLI

```bash
traigent recommend --list-types
traigent recommend rag
traigent recommend code_gen --min-impact medium
traigent recommend rag --min-confidence high --json
```

The CLI uses the same public API. Use `--json` when another tool needs to parse
the response.

## What The Response Contains

Recommendation rows are public catalog metadata. They include knob names,
suggested ranges, impact estimates, evidence labels, effectuation status, and
apply guidance.

## Copy-Paste Example

```python
import traigent as tg
from traigent.api.functions import recommend_configuration_space

space = recommend_configuration_space("rag", min_impact="medium")

@tg.optimize(configuration_space=space["configuration_space"])
def answer_question(query: str) -> str:
    cfg = tg.get_config()
    return query
```

Honesty note: recommendations are catalog-backed starter guidance. They are not
proof that a knob will improve your task; validate them on your own evaluation
dataset before treating them as winners.
