# Token and Cost Capture

Traigent can use captured LLM usage to compute trial token counts and cost.
It can use a provider-reported cost or price captured tokens. To retain example
results in a direct local evaluation, set `detailed=True` on `LocalEvaluator`.

## What is captured

| Client | Captured when | Notes |
| --- | --- | --- |
| LangChain `ChatOpenAI`, `ChatAnthropic`, and supported Bedrock chat models | The evaluator patches supported calls such as `invoke` and `ainvoke` | Stream capture uses usage from the final chunk. |
| `litellm.completion` / `litellm.acompletion` | The evaluator's capture patch is active and the non-streaming response includes usage | The patch skips streaming responses. |
| Traigent's `BedrockChatClient` | Capture is active and the response carries usage | |
| OpenAI SDK, sync and async | The OpenAI override is active and the supported, non-streaming call returns usage | See the methods below. |
| Anthropic SDK, sync and async | The override is active via `framework_targets=["anthropic.Anthropic", "anthropic.AsyncAnthropic"]` or `traigent.integrations.override_anthropic()` | Captures usage from `messages.create` for non-streaming calls and from the final message of `messages.stream` on context exit, if available. |
| Direct `google.genai` calls | Outside the SDK's client capture | For Gemini through LiteLLM, the same non-streaming and usage requirements apply. |

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
accounting for that optimization. An explicit cost limit can still refuse
an unmeasured trial when strict cost accounting is off.

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

When a custom evaluator has usage capture enabled and available, an example
without captured usage gets `metadata["llm_usage_measured"] = False`. The
evaluator logs a coverage warning if it captures usage for only some examples.
A custom evaluator can also report `provider_attempts=0` on failed rows. If
every row failed before calling a provider and reports zero attempts, the
trial can report `total_cost == 0.0` without setting `cost_unmeasured`.

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

## When a run stops because cost was not measured

Set `cost_limit` on the decorator or optimization call, or set
`TRAIGENT_RUN_COST_LIMIT`. The default cost limit is $2.00. With an explicit
cost limit, the SDK raises when a non-failed trial completes with unmeasured
cost, even if you set `max_trials` or `max_total_examples`.

With the default cost limit, a single unmeasured trial can activate a trial
count fallback. That fallback defaults to 10 trials and can be set with
`TRAIGENT_FALLBACK_TRIAL_LIMIT`. When it stops the run, the result can report
`stop_reason == "cost_limit"` with
`COST_UNMEASURED_TRIAL_LIMIT_REACHED`.

Passing `max_trials` or `max_total_examples` explicitly waives the trial count
fallback, even when you pass the default number of trials. The cost limit still
applies to known spend. When a result returns after running unmeasured trials
under this waiver, check for `COST_UNMEASURED_TRIALS_RAN`.

For runs stopped by a trial, sample, or cost limit, check
`result.metadata.get("stop_detail")` for an explanation. The results table
prints this explanation when present.

A `max_total_examples` passed to `.optimize()` or `.optimize_sync()` is kept on
the function object, so later calls on the same object keep that sample budget
(and keep counting as explicitly sized) until you change it.
