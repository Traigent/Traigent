# LiteLLM Multi-Provider Reference

## Overview

LiteLLM provides a unified `completion()` API that works across 100+ LLM providers. Combined with Traigent, you can optimize model selection across providers in a single optimization run.

## Installation

```bash
pip install traigent litellm
```

Set API keys for each provider you want to test:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
export COHERE_API_KEY="..."
export MISTRAL_API_KEY="..."
```

## Supported Providers

LiteLLM uses model name prefixes to route to the correct provider. Common providers:

| Provider | Model Prefix | Example |
|---|---|---|
| OpenAI | (none) | `gpt-4o-mini`, `gpt-4o` |
| Anthropic | `anthropic/` or none | `claude-3-haiku-20240307`, `claude-3-5-sonnet-20241022` |
| Google Gemini | `gemini/` | `gemini/gemini-1.5-flash`, `gemini/gemini-1.5-pro` |
| Mistral | `mistral/` | `mistral/mistral-small-latest` |
| Cohere | `command-r` | `command-r`, `command-r-plus` |
| AWS Bedrock | `bedrock/` | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` |
| Azure OpenAI | `azure/` | `azure/my-deployment` |
| Together AI | `together_ai/` | `together_ai/meta-llama/Llama-3-70b-chat-hf` |
| Groq | `groq/` | `groq/llama3-70b-8192` |

See LiteLLM documentation for the full provider list.

## Basic Multi-Provider Example

```python
import traigent
import litellm

@traigent.optimize(
    configuration_space={
        "model": [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "gemini/gemini-1.5-flash",
        ],
        "temperature": [0.0, 0.3, 0.7],
        "max_tokens": [256, 512],
    },
    objectives=["accuracy"],
    max_trials=15,
)
def answer(question):
    config = traigent.get_config()

    response = litellm.completion(
        model=config["model"],
        messages=[{"role": "user", "content": question}],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content

results = answer.optimize(dataset="qa_eval.jsonl")
```

## Cost Tracking

LiteLLM tracks cost per completion call. You can access this through Traigent's results:

```python
results = answer.optimize(dataset="qa_eval.jsonl")

# Traigent aggregates cost across all trials
if results.total_cost is not None:
    print(f"Total optimization cost: ${results.total_cost:.4f}")

# Per-trial cost analysis
for trial in results.successful_trials:
    model = trial.config["model"]
    accuracy = trial.get_metric("accuracy", 0.0)
    print(f"{model}: accuracy={accuracy:.2%}")

# Compare cost vs accuracy across providers
stats = results.experiment_stats
print(f"Cost per config: ${stats.cost_per_configuration:.4f}")
```

### Cost-Accuracy Tradeoff Analysis

```python
results = answer.optimize(dataset="qa_eval.jsonl")

# Group trials by provider
from collections import defaultdict
provider_results = defaultdict(list)

for trial in results.successful_trials:
    model = trial.config["model"]
    # Extract provider from model name
    if "/" in model:
        provider = model.split("/")[0]
    elif model.startswith("claude"):
        provider = "anthropic"
    elif model.startswith("gpt"):
        provider = "openai"
    else:
        provider = "other"
    provider_results[provider].append(trial)

for provider, trials in provider_results.items():
    avg_acc = sum(t.get_metric("accuracy", 0.0) for t in trials) / len(trials)
    print(f"{provider}: avg accuracy={avg_acc:.2%} ({len(trials)} trials)")
```

## Complete Multi-Provider Example with Fallback

```python
import traigent
import litellm

@traigent.optimize(
    configuration_space={
        "model": [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022",
            "gemini/gemini-1.5-flash",
            "gemini/gemini-1.5-pro",
        ],
        "temperature": [0.0, 0.3, 0.5, 0.7],
        "max_tokens": [256, 512, 1024],
    },
    objectives=["accuracy"],
    max_trials=20,
)
def generate_response(prompt):
    config = traigent.get_config()

    response = litellm.completion(
        model=config["model"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content


# Run optimization
results = generate_response.optimize(dataset="eval_prompts.jsonl")

# Analyze results
print(f"Best model: {results.best_config['model']}")
print(f"Best temperature: {results.best_config['temperature']}")
print(f"Best score: {results.best_score:.3f}")
print(f"Total cost: ${results.total_cost:.4f}" if results.total_cost else "Cost not tracked")
print(f"Trials: {len(results.successful_trials)}/{len(results.trials)} successful")

# Apply best config
if results.best_score is not None and results.best_score >= 0.8:
    generate_response.apply_best_config(results)
    # Now generate_response() uses the best model/temperature/max_tokens
```

## Tips

- LiteLLM handles retries and fallbacks internally; Traigent optimizes the model choice
- Set `LITELLM_LOG=DEBUG` for detailed provider-level logging
- Use `litellm.set_verbose = True` during development to see API calls
- Some providers require additional setup (Azure needs deployment names, Bedrock needs AWS credentials)
- LiteLLM's cost tracking works automatically for most providers
