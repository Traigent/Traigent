# Raw OpenAI SDK Reference

## Overview

Agents that call the OpenAI SDK directly (`openai.OpenAI`, `openai.AsyncOpenAI`)
or any OpenAI-compatible gateway through it get both parameter injection and
token/cost capture from Traigent's OpenAI framework override.

## Enable It

```python
from traigent.integrations.llms.openai import enable_openai_optimization

enable_openai_optimization()  # patches openai.OpenAI and openai.AsyncOpenAI
```

or on the decorator:

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

## Cost Capture

While the override is active, every non-streaming `chat.completions.create` /
`completions.create` response that carries `usage` is charged to the running
example, including every call of a multi-call agent.

Not captured (the example is reported as unmeasured, never as `$0`):

- the override is not active;
- `stream=True` calls;
- responses without `usage`;
- `with_raw_response.create(...)` and `client.responses.create(...)`.

## Pricing Gateway Aliases

A gateway alias is not in LiteLLM's price table. Price it per token:

```bash
export TRAIGENT_CUSTOM_MODEL_PRICING_JSON='{"acme-gateway/house-model": {"input_cost_per_token": 0.000002, "output_cost_per_token": 0.000005}}'
```

`TRAIGENT_CUSTOM_MODEL_PRICING_FILE` accepts the same JSON from a file.

See `docs/user-guide/cost_capture.md` in the SDK repository for the full
capture matrix.
