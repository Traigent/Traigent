# Traigent Execution Modes Guide

> **Current status:** Traigent executes your code locally. The default is `execution_mode="edge_analytics"` (local). `execution_mode="cloud"` and `execution_mode="hybrid"` are reserved for Traigent Cloud and are not yet supported in this build; they will raise `NotYetSupported` when optimization runs.

## Overview

Use `edge_analytics` (default) to run locally. Use `cloud` or `hybrid` only when Traigent Cloud is available for your build/deployment.

To run fully local (no Traigent backend communication), set `TRAIGENT_OFFLINE_MODE=true`.

## Modes at a Glance

| Mode | OSS availability | Status | Notes |
| --- | --- | --- | --- |
| `edge_analytics` | ✅ Available | ✅ Supported | Local execution |
| `hybrid` | ✅ Available | 🚧 Not yet supported | Raises `NotYetSupported` when optimization runs |
| `cloud` | ✅ Available | 🚧 Not yet supported | Raises `NotYetSupported` when optimization runs |

## Local Mode (`edge_analytics`)

### Overview
Runs entirely on your infrastructure. Only your LLM provider sees requests made during trials.

### Configuration

```python
import traigent
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

@traigent.optimize(
    execution=ExecutionOptions(
        execution_mode="edge_analytics",
        local_storage_path="./my_optimizations",
        minimal_logging=True,
    ),
    evaluation=EvaluationOptions(eval_dataset="data.jsonl"),
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.1, 0.5, 0.9]},
    objectives=["accuracy", "cost"],
)
def my_agent(query: str) -> str:
    return process_query(query)
```

### Features

- ✅ **Complete data privacy**: code and I/O stay local
- ✅ **Local storage**: defaults to `~/.traigent/`; override with `local_storage_path`
- ✅ **Fast iteration**: no orchestration latency; parallel trials supported
- ✅ **No registration**: works offline/air-gapped (aside from your LLM API calls)

### Limitations

- ⚠️ Advanced Bayesian orchestration and team features are cloud-only roadmap items
- ⚠️ Result sharing is manual (local files)

### When to use

- Regulated or sensitive data (healthcare, finance, legal)
- Air-gapped or restricted environments
- CI smoke tests and demos (`TRAIGENT_MOCK_LLM=true`)
- Budget-conscious experiments

## Roadmap Notes

Some fully managed backend capabilities (for example, cloud-executed trials / agent optimization) require a provisioned backend and may not be available in OSS builds yet. In this build, only `execution_mode="edge_analytics"` is supported.

## Privacy-Safe Analytics

Edge Analytics can submit aggregated, privacy-safe usage stats when a Traigent API key is configured. No prompts, inputs, outputs, or code are transmitted in OSS local mode. Set `TRAIGENT_OFFLINE_MODE=true` to disable any backend communication.

To disable usage analytics in managed setups, pass `enable_usage_analytics=False` via `traigent.initialize(config=TraigentConfig(...))`. In OSS, leave the Traigent API key unset to skip analytics submission entirely.
