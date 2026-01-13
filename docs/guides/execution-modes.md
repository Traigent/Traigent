# Traigent Execution Modes Guide

> **Current status:** Traigent executes your code locally. By default it runs in `execution_mode="cloud"` to enable Traigent Cloud insights when available, and falls back to local-only execution (with a notice) when no `TRAIGENT_API_KEY` is configured or cloud is unavailable.

## Overview

Use `cloud` (default) if you want Traigent Cloud session tracking + insights when available. Use `edge_analytics` to force local-only runs (no cloud attempts). Use `hybrid` (or legacy `privacy`) when you want local execution with privacy toggles and cloud insights when available.

## Modes at a Glance

| Mode | OSS availability | Privacy | Notes |
| --- | --- | --- | --- |
| `cloud` | ✅ Available | ⚠️ Metadata shared when configured | Default; falls back to local-only when cloud is unavailable |
| `edge_analytics` | ✅ Available | ✅ Data stays local | Local-only (no cloud attempts) |
| `hybrid` | ✅ Available | ✅ I/O local, metadata shared when configured | Local execution with privacy toggles; cloud insights when available |

> **Legacy alias:** `privacy` maps to `hybrid` and sets `privacy_enabled=True`.

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

Some fully managed backend capabilities (for example, cloud-executed trials / agent optimization) require a provisioned backend and may not be available in OSS builds yet. When cloud services are unavailable, the SDK falls back to local-only execution automatically.

## Privacy-Safe Analytics

Edge Analytics can submit aggregated, privacy-safe usage stats when a Traigent API key is configured. No prompts, inputs, outputs, or code are transmitted in OSS local mode.

To disable usage analytics in managed setups, pass `enable_usage_analytics=False` via `traigent.initialize(config=TraigentConfig(...))`. In OSS, leave the Traigent API key unset to skip analytics submission entirely.
