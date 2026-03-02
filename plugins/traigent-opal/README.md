# Traigent OPAL Plugin

Native Python API for OPAL-style optimization declarations in Traigent.

## Why this plugin

The plugin supports three authoring modes:

1. **Scoped Symbol-in DSL (recommended)**
   `with opal_program(...) as v: v.model in tv(...)` reads close to OPAL while
   remaining valid Python.
2. **ProgramSpec object API (supported)**
   `program(..., tvars=[...], objectives=[...])`.
3. **Raw OPAL source string/file mode (compatibility mode)**.

## Installation

```bash
pip install traigent-opal
```

## Recommended: Scoped Symbol-in DSL

```python
from traigent_opal import (
    opal_program,
    tv,
    choices,
    frange,
)
from langchain_openai import ChatOpenAI

with opal_program(module="corp.support.rag_bot", evaluation_set="eval_dataset") as v:
    v.model_name in tv(choices("gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"))
    v.retriever_kind in tv(choices("bm25", "hybrid", "dense"))
    v.top_k in tv(choices(3, 5, 8, 12))
    v.temperature in tv(frange(0.0, 0.7, 0.1))
    v.maximize("quality", weight=0.7)
    v.minimize("latency_p95", weight=0.3)
    v.when(v.retriever_kind, "dense", "top_k <= 8")
    v.constraint("cost_per_query <= 0.03")

@v.optimize(max_trials=40)
def answer(query: str) -> str:
    docs = search_docs(query, kind=v.retriever_kind, top_k=v.top_k)
    llm = ChatOpenAI(model=v.model_name, temperature=v.temperature)
    return llm.invoke(format_rag_prompt(query, docs)).content
```

`@v.optimize` is thin sugar over `traigent.optimize` bound to this specific
builder/spec. If you prefer explicit Traigent wiring, this is equivalent:

```python
import traigent
from traigent_opal import compile_opal_spec

artifact = compile_opal_spec(v.build())

@traigent.optimize(**artifact.to_optimize_kwargs())
def answer_explicit(query: str) -> str:
    ...
```

Builder calls map to OPAL keywords:
- `v.maximize(...)` -> `objective maximize ...`
- `v.minimize(...)` -> `objective minimize ...`
- `v.constraint(...)` / `v.when(...)` -> `constraint ...`

### Binding behavior (`v.<name>`)

- Inside `with opal_program(...) as v:`, `v.<name>` is a declaration symbol.
- Outside the block (runtime), `v.<name>` resolves to the currently bound value.
- If no binding exists, access raises an explicit `AttributeError` explaining how to bind.

For runtime safety:
- Prefer `v.using(config)` for scoped, thread/async-safe temporary bindings.
- Use `v.freeze(config)` for process-level static defaults (deployment-style use).

### Run without optimization (fixed/best config)

```python
best = {
    "model_name": "gpt-4o-mini",
    "retriever_kind": "bm25",
    "top_k": 5,
    "temperature": 0.2,
}

@v.deploy(config=best)
def answer_deploy(query: str) -> str:
    docs = search_docs(query, kind=v.retriever_kind, top_k=v.top_k)
    llm = ChatOpenAI(model=v.model_name, temperature=v.temperature)
    return llm.invoke(format_rag_prompt(query, docs)).content
```

## Why not bare `model_name in tvar_model_name`?

Python requires names to exist at runtime. A scoped symbol carrier (`v`) avoids
undefined-name behavior and keeps declarations explicit and deterministic.

Canonical pattern:

```python
with opal_program(...) as v:
    v.model_name in tv(choices("gpt-4o-mini", "gpt-4o"))
```

## ProgramSpec Object API (supported)

```python
from traigent_opal import opal_optimize, program, tvar, choices, frange, maximize

spec = program(
    tvars=[
        tvar("model_name", choices("gpt-4o-mini", "gpt-4o")),
        tvar("temperature", frange(0.0, 0.7, 0.1)),
    ],
    objectives=[maximize("quality")],
)

@opal_optimize(spec)
def answer(query: str, **config) -> str:
    ...
```

## Public API

```python
from traigent_opal import (
    ProgramSpec,
    ProgramBuilder,
    SymbolRef,
    TunedVariable,
    TVarSpec,
    DomainChoices,
    DomainRange,
    CallableTemplate,
    ObjectiveSpec,
    ConstraintSpec,
    ChanceConstraintSpec,
    opal_program,
    tv,
    program,
    tvar,
    choices,
    frange,
    callable_template,
    maximize,
    minimize,
    constraint,
    when,
    chance_constraint,
    opal_optimize,
    compile_opal_spec,
)
```

`opal_optimize` is a compatibility helper exported from `traigent_opal.__init__`.

## Compatibility mode (supported)

Raw OPAL source text and `.opal` files are still supported for existing users.

```python
from traigent_opal import opal_optimize

source = """
module examples.model_selection
model in {"gpt-4o-mini", "gpt-4o"}
temperature in [0.0, 1.0] step 0.1
# opal: objective maximize quality on eval_dataset
# opal: objective minimize cost_per_query
"""

@opal_optimize(source)
def agent(query: str, **config) -> str:
    ...
```

## Migration guide (0.2.x -> 0.3.0)

Before (string-first):

```python
source = """
model in {"gpt-4o-mini", "gpt-4o"}
# opal: objective maximize quality
"""

@opal_optimize(source)
def run(query: str, **config):
    ...
```

After (scoped Symbol-in DSL):

```python
with opal_program() as v:
    v.model in tv(choices("gpt-4o-mini", "gpt-4o"))
    v.maximize("quality")

@v.optimize
def run(query: str):
    ...
```

## Notes

- Constraints and chance constraints remain expression strings in v1 of the
  APIs (intentional scope limit).
- Objective direction and weights are preserved through `ObjectiveSchema` when
  using ProgramSpec/scoped DSL.
- `v.<name>` is dual-mode by design:
  - Inside `with opal_program(...) as v:` it is a declaration symbol.
  - Outside the `with` block it resolves to runtime/frozen bound values.
- Behavior access recommendations:
  - Preferred: direct `v.<tvar_name>` in function bodies.
  - Compatibility: `@opal_optimize(spec)` + `traigent.get_config()` / `**config`.
- Concurrency guidance:
  - Prefer `v.using(config)` for scoped, thread/async-safe bindings.
  - Use `v.freeze(config)` for process-level static deployment defaults.
- Behavior-plane function bodies remain regular Python and are intentionally
  outside plugin static analysis scope.

## Low-level compile APIs

```python
from traigent_opal import compile_opal_source, compile_opal_file, compile_opal_spec

artifact_a = compile_opal_source("model in {'a', 'b'}")
artifact_b = compile_opal_file("example.opal")
artifact_c = compile_opal_spec(program(tvars=[tvar("model", choices("a", "b"))]))

kwargs = artifact_c.to_optimize_kwargs()
metadata = artifact_c.to_directive_metadata()
```
