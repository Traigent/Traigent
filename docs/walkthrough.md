# Walkthrough — 8 Runnable Examples

Step through Traigent's features with 8 progressive examples. All run in mock mode — no API keys, no cost.

## Setup

```bash
export TRAIGENT_MOCK_LLM=true
```

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

Each script prints trial results to the console. In mock mode, scores are random — the point is to see the full pipeline run without errors.

When you're ready for real optimization, remove `TRAIGENT_MOCK_LLM` and set your API keys.

## Browse source

[walkthrough/mock/](../walkthrough/mock/)
