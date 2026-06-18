# Traigent Execution Modes Guide

> **Current status (0.13+):** Traigent auto-selects the execution mode — you do not need to set `execution_mode` explicitly. Portal tracking is enabled automatically when `TRAIGENT_API_KEY` is set, regardless of mode. `execution_mode=”cloud”` is reserved for future remote execution and is not available yet.

## Overview

**You do not need to set `execution_mode`.** The SDK picks the right mode automatically:

- **Grid/random search (typical Python decorator use)** → `edge_analytics` (local execution; portal-tracked when `TRAIGENT_API_KEY` is present)
- **Smart algorithms (Bayesian, TPE, CMA-ES)** → `hybrid` (cloud-assisted orchestration; requires API key)
- **External REST agent service** → `hybrid_api` (your service receives suggestions via the Hybrid Mode REST API)

Portal tracking (results visible in the Traigent portal) is a function of having `TRAIGENT_API_KEY` set, not of the execution mode.

To run fully local (no Traigent backend communication), set `TRAIGENT_OFFLINE_MODE=true` or omit `TRAIGENT_API_KEY`.

## Modes at a Glance

| Mode | When auto-selected | Notes |
| --- | --- | --- |
| `edge_analytics` | Grid/random search | Local execution; portal-visible when API key is set |
| `hybrid` | Smart algorithms (Bayesian, TPE, …) | Cloud-assisted orchestration; API key required |
| `hybrid_api` | External REST agent service configured | Your agent receives suggestions via REST |
| `cloud` | (reserved) | Remote execution — not implemented yet; fails closed |

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

## Hybrid Mode (`hybrid`)

Auto-selected when a smart algorithm (Bayesian, TPE, CMA-ES, NSGA-II) is requested. The SDK uses cloud-assisted orchestration for suggestion generation while running each trial locally.

`hybrid` is **not** the mode to set for "I want portal visibility with grid/random search" — that is `edge_analytics` with `TRAIGENT_API_KEY` set. Use `hybrid` only if you explicitly need smart cloud-assisted optimization algorithms.

## Hybrid API Mode (`hybrid_api`)

Auto-selected when a Hybrid Mode REST endpoint or transport is configured (`hybrid_api_endpoint` / `hybrid_api_transport`). Your external agent service receives trial suggestions via the Traigent Hybrid Mode REST API.

## Cloud Roadmap

Fully managed remote execution, including cloud-executed trials and remote agent optimization, is reserved for `execution_mode="cloud"`. That path is not implemented yet and must not return synthetic session IDs, trial suggestions, agent outputs, or completed statuses.

## Privacy-Safe Analytics

Edge Analytics can submit aggregated, privacy-safe usage stats when a Traigent API key is configured. No prompts, inputs, outputs, or code are transmitted in OSS local mode. Set `TRAIGENT_OFFLINE_MODE=true` to disable any backend communication.

To disable usage analytics in managed setups, pass `enable_usage_analytics=False` via `traigent.initialize(config=TraigentConfig(...))`. In OSS, leave the Traigent API key unset to skip analytics submission entirely.
