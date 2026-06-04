# Guided Generation

Guided generation lets Traigent create **new tuning material** — improved prompt
candidates and new evaluation examples — guided by the backend's proprietary
tuning signals, while the generation itself runs on **your own LLM** so prompt
text and example content never leave your environment.

Two capabilities, one contract:

| Capability | What it produces | How it's used |
|------------|------------------|---------------|
| **Prompt rewrite** | New prompt variants aimed at measured weak spots | Folded into the config space as new `Choices` the optimizer searches |
| **Benchmark guide generation** | New evaluation examples near the informative/difficult frontier | Added to your dataset (`metadata.synthetic`) |

## The IP boundary

The differentiating asset is Traigent's **tuning signals** (informativeness,
difficulty, discriminative power, uniqueness, …) and the **selection policy** that
acts on them. These never cross the client boundary. Instead, the backend returns
an opaque **`GuidancePlan`**:

```
GuidancePlan
├── plan_id, policy_version, plan_token (opaque, signed), expires_at
├── plan_budget.total_generations          # client-requested, server-capped
└── items: [ { seed_ref, action, coarse_priority } ]
                 │         │            └── high | medium | low (ordinal only)
                 │         └── generate_similar | generate_harder |
                 │            diversify_around | rewrite_prompt
                 └── a reference the CLIENT owns and resolves locally
```

A plan carries **selection only** — no signal names, no values, no formulas, and
no per-seed count. It is a deliberately lossy, many-to-one projection of the
signals (≈12 signals → one action verb + one of three buckets per seed). This is
**oracle-hardening**, not information-theoretic secrecy: deterministic
per-(run, policy_version) plans, per-seed-independent buckets, minimum cohort
sizes, rate limits, and policy-version rotation raise the cost of treating the
endpoint as a black-box oracle. The defensible asset is the *recipe*, not the
*decision*.

## What privacy mode protects (and what it does not)

- **Protects:** customer content — prompt text, example inputs/outputs, and any
  synthesized content — never leaves the client; generation runs on the user's
  own LLM credentials.
- **Does not change:** the per-example *numeric* metrics (score, latency, tokens)
  keyed by stable `ex_{hash}_{index}` IDs still flow to the backend (that is the
  substrate the signals are computed from). The IP we protect is *our* signal
  definitions and policy — not the raw metrics the customer already owns.

## Using it (Python SDK)

The reachable entry point is `OptimizedFunction.optimize_with_guidance`:

```python
from traigent.generation import BackendGuidanceProvider

def my_llm(prompt: str) -> str:                     # your own model + creds
    return client.responses.create(model="gpt-4o", input=prompt).output_text

provider = BackendGuidanceProvider.from_async_post(session_id, my_authed_post)

best = my_agent.optimize_with_guidance(
    provider,
    plan_kind="prompt_rewrite",                     # or "benchmark_guide"
    rewrite_llm=my_llm,
    prompt_param="prompt_template",
    prompt_rewrite={"rounds": 2, "candidates_per_round": 3},
)
```

Or configure at decoration time:

```python
@traigent.optimize(eval_dataset=..., prompt_rewrite={"rounds": 2})
def my_agent(...): ...
```

The building blocks are also usable directly: `PromptRewriter`,
`ExampleSynthesizer`, `merge_prompt_candidates`, `GuidanceLoop`, and
`GuidancePlan`/`GuidancePlanItem`/`GuidanceAction` (all exported from
`traigent.generation`). Each engine accepts a `RewriteLLM`, a constructed client
with `.complete()`, or a bare `fn(prompt) -> str`. A missing provider fails
closed (raises) — it never fabricates candidates or builds a client from ambient
env keys.

## Using it (TypeScript SDK)

`@traigent/sdk` exposes the parity surface — `GuidancePlan*` Zod DTOs, the
`PromptRewriter` / `ExampleSynthesizer` / `GuidanceLoop` / `BackendGuidanceProvider`
engines, and `promptRewrite?` / `growDataset?` fields on `OptimizationSpec`.

## Required-by-design

There is **no offline fallback** that reconstructs guidance locally. If the
backend plan can't be fetched, generation is refused rather than fabricated — the
compass is essential, and that keeps the signals protected.

## See also

- Runnable offline example: [`examples/core/guided-generation/`](../../examples/core/guided-generation/README.md)
- Contract: `TraigentSchema/traigent_schema/schemas/guidance/`
