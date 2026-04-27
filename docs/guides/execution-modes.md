# Traigent Execution Modes Guide

> **Current status:** Traigent executes trials locally. The default is `execution_mode="edge_analytics"` (local-only). Use `execution_mode="hybrid"` when you want local execution plus backend/portal tracking. `execution_mode="cloud"` is reserved for future remote execution and fails with: “Cloud remote execution is not available yet; use hybrid for portal-tracked optimization.”

## Overview

Use `edge_analytics` (default) to run locally. Use `hybrid` for website-visible results while keeping trials on your machine. Do not use `cloud` yet; remote agent execution is not implemented.

To run fully local (no Traigent backend communication), set `TRAIGENT_OFFLINE_MODE=true`.

## Modes at a Glance

| Mode | OSS availability | Status | Notes |
| --- | --- | --- | --- |
| `edge_analytics` | Available | Supported | Local execution and local results |
| `hybrid` | Available | Supported | Local execution plus backend session/trial tracking |
| `cloud` | Reserved | Not implemented | Remote execution path fails closed |

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

## Hybrid Mode

Hybrid mode is the production path for portal-tracked SDK runs today. The SDK creates a backend session through `/api/v1/sessions`, runs each trial locally, and submits trial metrics through `/api/v1/sessions/{session_id}/results`.

Use `hybrid` when you want results to appear in the Traigent website. Use `edge_analytics` when you want fully local runs with no backend dependency.

## Cloud Roadmap

Fully managed remote execution, including cloud-executed trials and remote agent optimization, is reserved for `execution_mode="cloud"`. That path is not implemented yet and must not return synthetic session IDs, trial suggestions, agent outputs, or completed statuses.

## Privacy-Safe Analytics

Edge Analytics can submit aggregated, privacy-safe usage stats when a Traigent API key is configured. No prompts, inputs, outputs, or code are transmitted in OSS local mode. Set `TRAIGENT_OFFLINE_MODE=true` to disable any backend communication.

To disable usage analytics in managed setups, pass `enable_usage_analytics=False` via `traigent.initialize(config=TraigentConfig(...))`. In OSS, leave the Traigent API key unset to skip analytics submission entirely.
