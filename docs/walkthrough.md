# Walkthrough - 8 Runnable Examples

Step through Traigent's features with 8 progressive examples. All run in mock mode - no API keys, no cost.

## Setup

The bundled `walkthrough/mock/*` scripts self-select mock behavior for local demos. For your own tutorial code, prefer the in-code helper:

```python
from traigent.testing import enable_mock_mode_for_quickstart

enable_mock_mode_for_quickstart()
```

The legacy `TRAIGENT_MOCK_LLM=true` env var remains available outside production for shell fixtures and backwards compatibility, but direct user-set activation emits `DeprecationWarning`.

## Steps

| # | Run | What you'll learn |
|---|-----|-------------------|
| 1 | `python walkthrough/mock/01_tuning_qa.py` | Basic model + temperature optimization |
| 2 | `python walkthrough/mock/02_zero_code_change.py` | Seamless mode — zero code changes to existing code |
| 3 | `python walkthrough/mock/03_parameter_mode.py` | Explicit config access via `traigent.get_config()` |
| 4 | `python walkthrough/mock/04_multi_objective.py` | Balance accuracy, cost, and latency |
| 5 | `python walkthrough/mock/05_rag_parallel.py` | RAG optimization with parallel evaluation |
| 6 | `python walkthrough/mock/06_custom_evaluator.py` | Define your own success metrics |
| 7 | `python walkthrough/mock/07_multi_provider.py` | Compare OpenAI, Anthropic, Google in one run |
| 8 | `python walkthrough/mock/08_privacy_modes.py` | Local-only privacy-first execution |

## What to expect

Each script prints trial results to the console. In mock mode the SDK
intercepts the LLM call layer with canned/deterministic responses; the
walkthrough scripts themselves use a helper called `get_mock_accuracy`
to produce illustrative accuracy numbers (the helper is example-only,
not part of the SDK runtime).

When you switch to real LLMs (do not call the helper, unset the legacy `TRAIGENT_MOCK_LLM` env var if present, set your API keys, and run the `walkthrough/real/*` scripts), your custom evaluator
or the built-in `LocalEvaluator` accuracy calculator scores the real
LLM outputs end-to-end — the SDK does not fabricate evaluator scores
in either mode.

> Note: The legacy `MockModeOptions` knobs (`enabled`,
> `override_evaluator`, `base_accuracy`, `variance`) are retained on the
> schema for backwards compatibility but are **all inert** — mock mode
> is activated by `traigent.testing.enable_mock_mode_for_quickstart()`
> in local code, not via that object. The legacy `TRAIGENT_MOCK_LLM=true`
> env var remains for shell fixtures and backwards compatibility, but emits
> `DeprecationWarning` when users set it directly. See issue #874.

## Optional Extras

These are not part of the core 8-step path, but they complement it:

- `python walkthrough/mock/09_rag_multi_objective.py`
- `python walkthrough/demo/rag_agent.py` (requires `OPENAI_API_KEY`)
- `python walkthrough/demo/optimize_rag.py`

## Browse source

Browse source in the repository: `walkthrough/mock/`.
