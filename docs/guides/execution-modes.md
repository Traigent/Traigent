# Traigent Execution Modes Guide

> **Current status:** Open-source builds support `edge_analytics` (local) only. Cloud and hybrid are roadmap-ready but require a managed backend.

## Overview

Use `edge_analytics` for all OSS runs. This guide focuses on local mode and notes roadmap items for awareness.

## Modes at a Glance

| Mode | OSS availability | Privacy | Notes |
| --- | --- | --- | --- |
| `edge_analytics` | ✅ Available | ✅ Data stays local | Default and supported today |
| `cloud` | 🚧 Roadmap | ⚠️ Metadata shared | Managed backend only (not shipped) |
| `hybrid` | 🚧 Roadmap | ✅ I/O local, metadata shared | Managed backend only (not shipped) |

> **Legacy alias:** `privacy` maps to `hybrid` and sets `privacy_enabled=True`, but hybrid execution is still a managed-backend feature.

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

## Roadmap (Awareness Only)

- `cloud`: Managed orchestration with Bayesian search, team history, dashboards.
- `hybrid`: Local execution with cloud-guided trial selection and privacy toggles.
- Until a managed backend is provisioned, keep `execution_mode="edge_analytics"` in OSS builds.

## Privacy-Safe Analytics

Edge Analytics can submit aggregated, privacy-safe usage stats when a Traigent API key is configured. No prompts, inputs, outputs, or code are transmitted in OSS local mode.

To disable usage analytics in managed setups, pass `enable_usage_analytics=False` via `traigent.initialize(config=TraigentConfig(...))`. In OSS, leave the Traigent API key unset to skip analytics submission entirely.
