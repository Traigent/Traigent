# Token and Cost Capture

Traigent can use captured LLM usage to compute trial token counts and cost.
It can use a provider-reported cost or price captured tokens. To retain example
results in a direct local evaluation, set `detailed=True` on `LocalEvaluator`.

## What is captured

| Client | Captured when | Notes |
| --- | --- | --- |
| LangChain `ChatOpenAI`, `ChatAnthropic`, and supported Bedrock chat models | The evaluator patches supported calls such as `invoke` and `ainvoke` | Stream capture uses usage from the final chunk. |
| `litellm.completion` / `litellm.acompletion` | The evaluator's capture patch is active and the call is non-streaming | Token reporting depends on usage data. A positive response cost or an available LiteLLM cost estimate can supply cost without token usage. The patch skips streams. |
| Traigent's `BedrockChatClient` | Capture is active and the response carries usage | |
| OpenAI SDK, sync and async | The OpenAI override is active and the supported, non-streaming call returns usage | See the methods below. |
| Anthropic SDK, sync and async | Call `traigent.integrations.override_anthropic()` for the process, or set both `auto_override_frameworks=True` and `framework_targets=["anthropic.Anthropic", "anthropic.AsyncAnthropic"]` in the decorator's `injection` options | Captures usage from `messages.create` for non-streaming calls and from the final message of `messages.stream` on context exit, if available. |
| Direct `google.genai` calls | Outside the SDK's client capture | Gemini calls through LiteLLM follow the LiteLLM rules above. |

### Raw OpenAI SDK calls (including OpenAI-compatible gateways)

Enable the OpenAI framework override for the process with this call.

```python
from traigent.integrations.llms.openai import enable_openai_optimization

enable_openai_optimization()
```

For decorator-scoped overrides, set `auto_override_frameworks=True` and
`framework_targets=["openai.OpenAI", "openai.AsyncOpenAI"]` in the
`injection` options on `@traigent.optimize`.

- Supported capture methods include `chat.completions.create`,
  `chat.completions.parse`, `beta.chat.completions.parse`, `completions.create`,
  `responses.create`, and `responses.parse`, including their
  `with_raw_response` forms.
- The Responses API's `input_tokens` and `output_tokens` count as prompt and
  completion tokens.
- Activate the override before first accessing a client's `with_raw_response`
  methods. The SDK parses those responses for usage and caches the parsed object
  for your `.parse()` call.
- For parameter injection, the override supports `chat.completions.create`
  and `completions.create`.

## Pricing a model Traigent does not know

Traigent can price captured tokens using LiteLLM's model table. If a gateway
alias such as `acme-gateway/house-model` has no matching rate, supply one to
price its tokens.

`TRAIGENT_STRICT_COST_ACCOUNTING` sets strict runtime cost accounting
explicitly. When unset, a `cost` objective enables strict runtime cost
accounting for that optimization. If a trial reaches cost accounting with
missing or unpriced cost, strict mode raises `CostTrackingRequiredError`.
If a trial fails solely on provider errors and every call raises before
returning a response, the run can continue with that trial's spend marked unknown.

For an unpriced model, supply rates as shown below. For an optimized function
path that makes no model calls, return `traigent.with_usage(value, total_cost=0)`
to report zero explicitly. To turn off strict accounting, set
`TRAIGENT_STRICT_COST_ACCOUNTING=false`. If you enabled
`TRAIGENT_REQUIRE_COST_TRACKING`, turn that off as well to allow missing or
unpriced trial costs. An explicit cost limit can still refuse an unmeasured
trial.

Give the alias a price in USD per token with
`TRAIGENT_CUSTOM_MODEL_PRICING_JSON` or point
`TRAIGENT_CUSTOM_MODEL_PRICING_FILE` to a file with the same JSON.

```bash
export TRAIGENT_CUSTOM_MODEL_PRICING_JSON='{
  "acme-gateway/house-model": {
    "input_cost_per_token": 0.000002,
    "output_cost_per_token": 0.000005
  }
}'
```

Use the model name Traigent prices as the key. For a custom evaluator, the SDK
reads the model from the captured response. If the response omits the model,
the SDK uses the trial's `model` setting.

## Mock provenance in result metadata

Optimization results can record mock provenance in `result.metadata`.

| Key | Meaning |
|---|---|
| `mock_llm_run` | When the SDK observes calls, `True` means it served a mock response and `False` means it observed only real calls. |
| `mock_llm_mixed` | When the SDK observes calls, `True` means it observed both simulated and real calls. |

When the SDK observes no calls, it falls back to the mock settings it saw
during trial dispatch or at run start.

Read `result.metadata.get("mock_llm_run")` and treat `None` as "not recorded".
The results table labels a recorded mock run. With mock mode active, it also
labels results that lack mock provenance.

## When nothing was captured

When usage capture is enabled and available for a custom evaluator, an
`ExampleResult` that the SDK enriches without captured usage gets
`metadata["llm_usage_measured"] = False`. If the evaluator raises and the SDK
creates a failed row, that row can omit the key. The evaluator logs a coverage
warning if it captures usage for only some examples.

A custom evaluator can report zero provider attempts with
`metrics["provider_attempts"] = 0` on a failed example. If every example failed
before calling a provider and its metrics report zero attempts, the trial
can report `total_cost == 0.0` without setting `cost_unmeasured`.

## Reading cost results

For local evaluation metrics, `cost` is the total for the trial and
`cost_per_example_mean` is the mean across examples with measured cost.
`tokens_estimated` is set when some token counts were estimated from text
length. Those counts provide no evidence of billed spend.

Check `cost_unmeasured` before interpreting a trial's numeric `cost` as a
total. For a flagged trial, the results table shows `≥` when
`cost_lower_bound` is positive and otherwise shows `unmeasured`. For a
constraint requiring known cost within a limit, require a numeric
`metrics.get("cost")` within the limit and reject trials flagged
`cost_unmeasured`.

When run summaries (`get_summary()` and `experiment_stats`) count unmeasured
trials, they show `total_cost` as `None` and report `cost_unmeasured_trials`
and `cost_lower_bound`. The lower bound is `None` when there is no positive
known spend.

## Per-run cost limits

The SDK uses the per-run cost limit to check the estimate before asking for
approval and to decide whether to start more trials during the run. A trial
already running can spend more than estimated and take the run over the limit.
Parallel trials already running can add to the overrun.

Set `cost_limit` on the decorator or optimization call. When omitted, the SDK
normally reads `TRAIGENT_RUN_COST_LIMIT`, with a default of $2.00. Passing the
Python argument `cost_approved=True` without a Python `cost_limit` skips the
approval prompt and uses $2.00 even if `TRAIGENT_RUN_COST_LIMIT` sets a
different amount. In that case, a nonempty `TRAIGENT_RUN_COST_LIMIT` makes
the cap count as explicit for the unmeasured-cost check.

With an explicit cost limit, a non-failed trial that completes with
unmeasured cost raises `CostTrackingRequiredError`, even when you set
`max_trials` or `max_total_examples`.

## When unmeasured cost stops a run

When a run uses the inherited default cost limit and continues after an
unmeasured trial, it can switch to a trial-count fallback. The fallback
defaults to 10 trials and can be set with `TRAIGENT_FALLBACK_TRIAL_LIMIT`.
A fallback stop can report `stop_reason == "cost_limit"` and
`COST_UNMEASURED_TRIAL_LIMIT_REACHED` in `result.warning_codes`.

After a fallback stop, capture priced usage or report a cost metric for
every configuration. You can also explicitly size a local optimization
with `max_trials` or `max_total_examples` to waive the trial-count fallback.
Passing the default trial count explicitly also qualifies. `max_total_examples`
caps examples across all trials and leaves the ordinary `max_trials` default
of 10 in place. Managed optimization can still apply the fallback with
explicit sizing.

The per-run cost limit still applies to known spend. When a local result
returns after running unmeasured trials under the waiver, check
`result.warning_codes` for `COST_UNMEASURED_TRIALS_RAN`.

For runs stopped by a trial, sample, or cost limit, check
`result.metadata.get("stop_detail")` for an explanation. The results table
prints this explanation when present.

A `max_total_examples` passed to `.optimize()` or `.optimize_sync()` is kept on
the function object, so later calls on the same object keep that sample budget
(and keep counting as explicitly sized) until you change it.
