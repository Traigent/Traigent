# Bazak Agent Grid Optimization

Runs a full grid search over the Bazak ZAP agent (`ai.bazak.ai`) using the
Traigent SDK. Evaluates **3 hardcoded example IDs** against every model in the
discovered config space (currently 4 models = 12 total evaluations).

## Examples

| Input ID | Description |
|---|---|
| `no-filter-single-search-trashcan-blue` | Unfiltered product search |
| `product-search-specific-model` | Specific model product search |
| `consultant-fridge` | Consultant fridge scenario |

## Prerequisites

1. Traigent backend + frontend running locally (`docker compose up`)
2. Python venv with Traigent SDK installed (`make install-dev`)
3. `.env` file in this folder with `BAZAK_AUTH_TOKEN` and `TRAIGENT_API_KEY`

## Usage

```bash
cd /path/to/Traigent
.venv/bin/python examples/bazak/run_optimization.py
```

## Output

- Console: per-trial results and final summary
- `examples/bazak/results.json`: structured results (gitignored)
- `examples/bazak/REPORT.md`: human-readable report (gitignored)
- Traigent backend: experiment + configuration runs visible in FE

## Config Space (from /config-space, TVL 0.9)

```json
{
  "model": ["gemini-3-flash-preview", "gemini-2.5-flash", "gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07"]
}
```

## Metrics

The evaluate endpoint returns per-example:
- `tool_accuracy` — did the agent select the correct tool?
- `param_accuracy` — did the agent pass correct parameters?
- `text_accuracy` — was the text response correct?

The primary optimization objective is `tool_accuracy`.
