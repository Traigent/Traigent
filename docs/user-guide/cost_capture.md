# Token and Cost Capture

Traigent charges each trial for the LLM calls its examples make. It reads the
provider's reported usage (prompt and completion tokens), prices it, and
records `input_tokens`, `output_tokens`, `total_tokens`, `input_cost`,
`output_cost` and `total_cost` on every example and trial.

This page lists which calls are captured, how to price a model Traigent does
not know, and what a trial reports when nothing was captured.

## What is captured

| Client | Captured when | Notes |
| --- | --- | --- |
| LangChain `ChatOpenAI.invoke`, `ChatAnthropic.invoke`, Bedrock chat models | always (patched by the evaluator) | synchronous `invoke` only: `ainvoke`/`abatch` are not captured yet ([#2445](https://github.com/Traigent/Traigent/issues/2445)); `stream`/`astream` capture the last chunk, which carries usage only with `stream_usage=True` |
| `litellm.completion` / `litellm.acompletion` | always (patched by the evaluator) | streaming calls are not captured |
| Traigent's `BedrockChatClient` | always | |
| Raw OpenAI SDK: `openai.OpenAI` / `openai.AsyncOpenAI` `chat.completions.create` and `completions.create` | the OpenAI override is active (see below) | non-streaming calls that return `usage` |

### Raw OpenAI SDK calls (including OpenAI-compatible gateways)

An agent that calls the OpenAI SDK directly is captured when the OpenAI
framework override is active. Either turn it on for the whole process:

```python
from traigent.integrations.llms.openai import enable_openai_optimization

enable_openai_optimization()  # patches openai.OpenAI and openai.AsyncOpenAI
```

or name the classes on the decorator:

```python
client = openai.OpenAI(base_url=os.environ["LLM_BASE_URL"])  # build it once


@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
    objectives=["accuracy", "cost"],
    eval_dataset="data.jsonl",
    auto_override_frameworks=True,
    framework_targets=["openai.OpenAI", "openai.AsyncOpenAI"],
)
def answer(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # replaced by the trial's model
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content
```

Every call made while an example runs is charged, so an agent that calls the
model several times per example is billed for all of them.

Coverage limits:

- **The override must be active.** Without `enable_openai_optimization()` or
  `framework_targets`, raw OpenAI calls are not captured (issue
  [#2441](https://github.com/Traigent/Traigent/issues/2441) tracks
  capture that needs no opt-in).
- **Streaming calls (`stream=True`) are not captured.** A stream's usage
  arrives only after your code reads it, and only with
  `stream_options={"include_usage": True}`. The example is reported as
  unmeasured, not as `$0`.
- **Responses without `usage`** (some gateways omit it) are not captured.
  They are reported as unmeasured.
- `client.chat.completions.with_raw_response.create(...)` and the Responses
  API (`client.responses.create`) are not captured.
- When LangChain's `ChatOpenAI.invoke` or `litellm.completion` makes the
  OpenAI call for you, that wrapper records the usage and the underlying
  OpenAI call is not counted a second time.

## Pricing a model Traigent does not know

Traigent prices tokens from LiteLLM's model table. A gateway alias such as
`acme-gateway/house-model` is not in that table, so its calls are
**unpriced**: tokens are recorded, cost is not known. With
`TRAIGENT_STRICT_COST_ACCOUNTING=true` an unpriced model raises; otherwise its
calls are recorded at `$0` and the run's result carries a warning naming the
unpriced model.

Give the alias a price with `TRAIGENT_CUSTOM_MODEL_PRICING_JSON` (inline) or
`TRAIGENT_CUSTOM_MODEL_PRICING_FILE` (path to the same JSON). Prices are USD
per token:

```bash
export TRAIGENT_CUSTOM_MODEL_PRICING_JSON='{
  "acme-gateway/house-model": {
    "input_cost_per_token": 0.000002,
    "output_cost_per_token": 0.000005
  }
}'
```

The key must be the model name Traigent prices. In the custom-evaluator lane
that is the trial's `model` setting when the configuration has one, otherwise
the model the response reports.

## When nothing was captured

A missing measurement is not a free call. When a custom-evaluator example has
no captured usage, its result has `metadata["llm_usage_measured"] = False`
and no token or cost keys. When no example in a trial was measured, the
trial's token and cost totals are left out of its metrics (or set to `None`
when `TRAIGENT_STRICT_METRICS_NULLS=true`) instead of being reported as `0`.
When only some examples were measured, the totals cover those examples and a
warning names the coverage.

The same holds for a `cost` objective: a trial with no measured example has
no `cost` (or `None` under strict nulls), never `0.0`. The results table shows
it as `n/a`. Weighted `best_config` selection counts a missing cost as the
worst cost, so there an unmeasured trial does not win on cost. Other surfaces
do not handle it yet: the Pareto front keeps unmeasured trials, the batch
composite score ignores the missing cost, a constraint written as
`metrics.get("cost", 0) <= limit` accepts them, and workflow spans upload
`cost_usd: 0.0` ([#2446](https://github.com/Traigent/Traigent/issues/2446)).
A `metric_limit` on a cost metric leaves unmeasured trials out of its running
total rather than failing the run. Two run-level warnings say what happened:

- `COST_OBJECTIVE_NO_USAGE_CAPTURED`: no trial captured usage, so the cost
  column is unmeasured. Under strict cost accounting the run fails instead.
- `COST_OBJECTIVE_PARTIAL_USAGE_CAPTURED`: some successful trials captured
  usage and others did not; the cost comparison covers only the measured ones.

## When a run stops because cost was not measured

The cost limit (`cost_limit=` on `@traigent.optimize` or `.optimize()`, or
`TRAIGENT_RUN_COST_LIMIT`, default `$2.00`) can only bound spend it can see.
When any trial reports no cost, Traigent is conservative: the whole run
switches to a trial limit and stops after `max_unmeasured_trials` trials
(default `10`) with `stop_reason == "cost_limit"`. One unmeasured trial is
enough, even if every other trial was measured. `max_trials` alone does not
lift this limit. The result carries the `COST_UNMEASURED_TRIAL_LIMIT_REACHED` warning code
and a message, which is also logged, that says how many trials were unmeasured
and how many were measured.

To continue, either:

1. **Get cost measured for every configuration.** Call the model through a
   captured client (the table above; for LangChain, the synchronous `invoke`).
   In a mixed run, find why the unmeasured configurations report no usage: a
   gateway that omits `usage`, a streaming call, or an uncaptured client. A
   fully measured run is bounded by its cost budget instead of the trial limit,
   so raise `cost_limit` if the budget is what stops you.
2. **Accept untracked spend.** Raise the unmeasured-cost trial limit with the
   `max_unmeasured_trials` parameter, on the decorator or on the run:

   ```python
   @traigent.optimize(..., max_unmeasured_trials=50)
   def answer(question: str) -> str: ...

   answer.optimize_sync(max_trials=50, max_unmeasured_trials=50)
   ```

   It must be an int of at least 1. Without it, the
   `TRAIGENT_FALLBACK_TRIAL_LIMIT` environment variable applies, then the
   default of `10`; the parameter wins over the environment variable. Traigent
   cannot tell you what those trials cost.

Build the OpenAI client once, outside the optimized function, as in the
examples above. The override injects the trial's `model` and sampling
parameters at `chat.completions.create`, not into the client constructor.
