# Guided Generation

**Privacy-preserving prompt rewrite + benchmark growth, guided by backend tuning
signals — without ever revealing those signals.**

Traigent can generate *new* tuning material for you:

- **Prompt rewrite** — LLM-generated prompt candidates aimed at your prompt's
  measured weak spots, folded into the configuration space as new `Choices` the
  optimizer searches (no optimizer changes).
- **Benchmark guide generation** — new evaluation examples synthesized near the
  informative/difficult frontier, growing your dataset.

Both run **on your own LLM** in privacy mode, so prompt text and example content
never leave your environment. The backend returns only an opaque
**`GuidancePlan`** — which seeds to act on, an action verb
(`generate_similar` / `generate_harder` / `diversify_around` / `rewrite_prompt`),
and a coarse `high|medium|low` priority. It never reveals the proprietary tuning
signals, their values, or the selection policy behind the plan.

## Run it

```bash
python examples/core/guided-generation/run.py
```

Fully offline and deterministic — no API key, no backend. It substitutes a fake
"user LLM" (a plain callback) and a fake `GuidancePlanProvider` so you can see
the mechanics end-to-end.

## What it shows

1. **`PromptRewriter`** turns failing cases + your prompts into improved
   candidates via your LLM, and `merge_prompt_candidates(...)` folds them into a
   `Choices` (purely — your original config space is untouched).
2. **`ExampleSynthesizer`** turns seed examples + an action verb into new
   `EvaluationExample`s tagged `metadata.synthetic`.
3. **`GuidanceLoop`** drives the full loop: optimize → fetch the opaque plan
   (required; no offline fallback) → generate locally → re-optimize, tracking the
   best result across rounds. It asserts the provider only ever receives a
   **content-free** `GuidancePlanRequest`.

## Using it for real

In production you wire two things to real implementations:

```python
from traigent.generation import BackendGuidanceProvider

# 1. Your own LLM — any callable fn(prompt) -> str, or a client with .complete()
def my_llm(prompt: str) -> str:
    return openai_client.responses.create(model="gpt-4o", input=prompt).output_text

# 2. A provider bound to your Traigent session (async client bridged to sync)
provider = BackendGuidanceProvider.from_async_post(session_id, my_authed_post)

# Then, on any @traigent.optimize-decorated function:
best = my_agent.optimize_with_guidance(
    provider,
    plan_kind="prompt_rewrite",        # or "benchmark_guide"
    rewrite_llm=my_llm,
    prompt_param="prompt_template",
    prompt_rewrite={"rounds": 2, "candidates_per_round": 3},
)
```

Or configure it at decoration time and call with just the provider + LLM:

```python
@traigent.optimize(eval_dataset=..., prompt_rewrite={"rounds": 2})
def my_agent(...): ...

my_agent.optimize_with_guidance(provider, rewrite_llm=my_llm, prompt_param="prompt")
```

## Privacy & IP boundary

- **Your content stays local** — generation runs on your LLM; only the opaque
  plan request (plan kind + budget + scope) and your existing numeric metrics
  cross the wire.
- **Our signals stay server-side** — the plan is a lossy, opaque projection of
  the tuning signals; it carries no signal names, values, formulas, or per-seed
  counts.
- **The backend plan is required** — there is no offline fallback that
  reconstructs guidance locally; if the plan can't be fetched, generation is
  refused rather than fabricated.
