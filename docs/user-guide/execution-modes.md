# Execution Modes

Traigent has exactly three execution modes. Each has one public name. Use
these names in code, docs, and conversation; older names are listed at the end
with their replacements.

<!--
Source of truth: traigent/api/decorators.py (`optimize` decorator arguments
`algorithm`, `offline`, `evaluator`; `ExecutionOptions`; `ExternalServiceEvaluator`),
traigent/config/types.py (execution policy resolution, removed/fail-closed
legacy selectors), traigent/hybrid/ (external-service transports).
Re-verify against those files before editing this page. Any statement about
what leaves the machine must be backed by a wire-level witness, not a reading
of the code.
-->

| Mode | Where your agent runs | Who picks configurations | What leaves your machine | How to select it |
| --- | --- | --- | --- | --- |
| **local** | Your machine | The Traigent SDK, locally (`grid` / `random`) | Nothing is sent to Traigent | `offline=True`, optionally with `algorithm="grid"` or `algorithm="random"` |
| **cloud** | Your machine | The Traigent backend ("cloud brain") | Trial configuration values (including any prompt text in your configuration space), metrics, and run metadata such as the experiment and function names | `algorithm="auto"` (the default), or another backend algorithm |
| **hybrid_api** | An HTTP or MCP service you operate | Set by `algorithm` / `offline`, as for the other two modes | The requests the SDK sends to your service, plus, unless `offline=True`, what cloud mode sends | `evaluator=ExternalServiceEvaluator(kind="hybrid_api", ...)` |

`cloud` mode was formerly called a "hybrid session" in some older docs; call
it a **cloud session**. `hybrid_api` is the public product name "Hybrid Mode
API": the Traigent SDK calls your service to run each configuration (see the
[Hybrid Mode API Contract](../hybrid-mode-api-contract.md) and the
[Hybrid Mode Client Guide](../hybrid-mode-client-guide.md); the reference
contract lives in the `traigent-api` repo).

`cloud` mode is not a privacy mode. If nothing may leave the machine, use
`local` (`offline=True`).

## Code examples

The mode is chosen with arguments to the `@traigent.optimize` decorator; the
run itself is started with `await <function>.optimize()`. Each decorator also
takes your usual arguments (configuration space, evaluation dataset,
objectives), left out here.

```python
import traigent
from traigent.api.decorators import ExternalServiceEvaluator

# local: nothing is sent to Traigent
@traigent.optimize(offline=True, algorithm="grid")  # + your usual arguments
def my_agent(question: str) -> str: ...

# cloud: your agent runs locally, the Traigent backend picks configurations
@traigent.optimize(algorithm="auto")  # the default; + your usual arguments
def my_cloud_agent(question: str) -> str: ...

# hybrid_api: your agent is an external service the Traigent SDK calls
@traigent.optimize(  # + your usual arguments
    evaluator=ExternalServiceEvaluator(
        kind="hybrid_api",
        hybrid_api={"endpoint": "https://my-agent.example.com"},
    ),
)
def my_service_agent(question: str) -> str: ...

result = await my_agent.optimize()
```

See [Execution Modes / Optimization Routing](../guides/execution-modes.md) for
the full `algorithm` / `offline` routing reference within `local` and `cloud`.

## When `algorithm="auto"` falls back to a local search

`algorithm="auto"` asks the Traigent backend to pick configurations. When the
backend session cannot be created, the run falls back to a **local random
search** on your machine, but only for these failures: no API key; an
unreachable or unavailable backend (for example timeouts and 408, 500, 502, 503
or 504 responses); or an `auto` run where cloud is not required and both the
typed and the legacy session-creation requests were answered with HTTP 400.
Definitive authentication rejections, quota
refusals, closed backend refusals and other client errors (for example 401,
402, 403, 404, 409, 422 or 429) raise instead. Set `TRAIGENT_REQUIRE_CLOUD=1`
to fail instead of falling back.

The session-creation request may already have been sent to Traigent before the
failure, so a fallback run is not guaranteed to have sent nothing. After the
fallback, the search and trials run locally.

Traigent logs a warning when this happens, naming the reason. You can also
check the result:

- `result.metadata["source"] == "local_fallback"` (a managed run reports
  `"cloud_brain"`), with the reason in `result.metadata["fallback_reason"]`.
- `result.cloud_url` is not set, because no backend experiment exists.

## Removed and deprecated names

| Old name | Status | Replacement |
| --- | --- | --- |
| `edge_analytics` (as `execution_mode` or `ExecutionMode.EDGE_ANALYTICS`) | Removed; raises an error | `offline=True`, optionally with `algorithm="grid"` or `"random"` |
| `execution_mode="privacy"` | Removed; fails closed | `offline=True` |
| `execution_mode="cloud"` | Removed; fails closed | Omit `execution_mode`; use `algorithm="auto"` |
| `execution_mode="standard"` | Deprecated compatibility selector (warns) | Omit `execution_mode`; use `algorithm="auto"` |
| `execution_mode="hybrid"` | Deprecated compatibility selector (warns) | Omit `execution_mode`; use `algorithm="auto"` (this is `cloud` mode) |
| "hybrid session" (docs wording) | Deprecated wording, not a code selector | "cloud session" |
| `hybrid` as a standalone mode name | Deprecated wording | `cloud` |
| `execution_mode="hybrid_api"` | Deprecated as an execution selector (warns) | `evaluator=ExternalServiceEvaluator(kind="hybrid_api", ...)` |
| Flat `hybrid_api_*` keyword options (`hybrid_api_endpoint`, `hybrid_api_transport`, and so on) | Deprecated spellings, still accepted | The nested `hybrid_api={...}` options on `ExternalServiceEvaluator` |
| `ExecutionMode` enum (`traigent/config/types.py`) | Deprecated compatibility enum | `algorithm` and `offline` |
