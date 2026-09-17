# Seamless Injection (Runtime Fallback)

Seamless injection lets Traigent override simple variable assignments without
modifying your function signature. It uses a safe AST transformation and falls
back to a runtime shim when no assignments can be rewritten.

## How It Works

1. **AST rewrite first**: Traigent replaces literal assignments that match
   configuration keys (for example `model = "gpt-4"`).
2. **Runtime shim fallback**: If nothing is rewritten and the configuration
   overlaps the function signature, Traigent injects values at call time.

The runtime shim respects caller-provided arguments and only fills missing
parameters or defaults.

## Supported Patterns

- Variable assignments inside the function body
- Signature defaults (`def fn(model="gpt-4")`)
- Keyword-only defaults (`def fn(*, model="gpt-4")`)
- Required parameters when callers omit them (shim fills from config)

## Known Limitations

- **Direct call literals** are not rewritten:
  - Example: `ChatAnthropic(model="claude")` stays literal.
- **Dynamic lookups** (for example `getattr`) are not modified.
- A `**kwargs` catch-all parameter never counts as an injectable target, even
  though a config key would bind into it at call time.

If you need those patterns, use `injection_mode="parameter"` or read configs via
`traigent.get_config()`.

### Fails closed on zero injectable targets

If a non-empty `configuration_space` has **no** injectable target for the
target function -- no local assignment and no parameter named after any
config key -- Traigent raises `SeamlessNoInjectableTargetsError` (a
`ConfigurationError`) before the first optimization trial, instead of running
every trial with the same unvaried code and returning a phantom
`best_config`. Set `TRAIGENT_ALLOW_SEAMLESS_NO_TARGETS=true` to opt back into
the previous warning-only behavior for the rare intentional case (for
example, a config key that only gates behavior via `traigent.get_config()`
inside the function body). Partial coverage -- some config keys injectable,
others not -- logs a `WARNING` naming the uncovered keys and does **not**
fail closed.

## Example

```python
@traigent.optimize(
    injection_mode="seamless",
    configuration_space={
        "model": ["gpt-4o", "gpt-4o-mini"],
        "temperature": [0.1, 0.5, 0.9],
    },
)
def answer(question: str, *, model: str = "gpt-4o", temperature: float = 0.7) -> str:
    # The runtime shim updates defaults when AST rewrite has nothing to change.
    return call_llm(question, model=model, temperature=temperature)
```

## Safety Notes

- No `exec()` is used.
- AST rewriting only injects safe literal values.
- The shim binds arguments using `inspect.signature` to avoid mutating caller data.

## Observability

Seamless injection logs when it falls back to the runtime shim in debug mode. If
you need deterministic, low-cost runs while testing injection behavior, use
`TRAIGENT_MOCK_LLM=true`.
