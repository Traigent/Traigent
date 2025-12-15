# TVL Constraint Expressions (SDK Semantics)

TraiGent compiles TVL `constraints` expressions into a safe subset of Python.
This applies to both:

- Legacy constraints (`constraints: [{id, type, rule/when/then, ...}]`)
- TVL 0.9 structural constraints (`constraints: { structural: [{expr|when/then}, ...], ... }`)

## Namespaces

- `params`: the candidate configuration (the current trial’s parameter values)
- `metrics`: evaluation metrics (may be empty/`None` for purely structural checks)

Examples:

- `params.model == "gpt-4o"`
- `metrics.token_cost_usd <= 0.05`

## Operators and literals

- Equality: use `==` (single `=` is not supported and will raise a validation error)
- Comparisons: `<`, `<=`, `>`, `>=`, `!=`
- Membership: `in`, `not in`
- Booleans: `and`, `or`, `not`
- CEL-style aliases are accepted and translated before parsing:
  - `&&` → `and`
  - `||` → `or`
  - `!x` → `not x` (but `!=` remains `!=`)
  - `true/false/null` → `True/False/None`

## Functions

The evaluator exposes a small set of safe helpers:

- `len`, `min`, `max`, `sum`, `abs`, `any`, `all`
- `math.<fn>` via Python’s `math` module (for example `math.log`, `math.sqrt`)

## Accessing keys (including dotted names)

`params.<name>` is attribute access over a dictionary. This works when `<name>` is a valid identifier key.

If your tuned-variable name contains dots or other non-identifier characters (for example `retriever.k`), use
subscript access:

- `params["retriever.k"] >= 3`
- `metrics["p95_latency_ms"] <= 1200`

Nested access only works when the configuration is actually nested:

- `params.retriever.k` works only if `params["retriever"]` is a dict with key `k`.

## Guarding against missing values

Missing `params.<name>` / `metrics.<name>` values resolve to `None`. Guard explicitly when needed:

- `params.temperature is None or params.temperature <= 0.8`

## `when/then` constraints

Conditional constraints use implication semantics:

- Spec form: `when: A`, `then: B`
- Equivalent expression: `not (A) or (B)`

