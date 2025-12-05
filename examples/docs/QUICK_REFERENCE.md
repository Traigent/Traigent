# Traigent Quick Reference Guide

Everything you need to run and adapt examples quickly.

## Common commands
```bash
# Install from repo root
pip install -e .

# Mock mode (no keys)
export TRAIGENT_MOCK_MODE=true
python examples/core/simple-prompt/run.py

# With real LLMs
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
python examples/core/multi-objective-tradeoff/run_openai_optuna.py
```

## Core optimize pattern
```python
import traigent

@traigent.optimize(
    eval_dataset="examples/datasets/simple-prompt/evaluation_set.jsonl",
    configuration_space={"model": ["claude-3-haiku-20240307"], "temperature": [0.0, 0.7]},
    objectives=["accuracy"],
)
def summarize(text: str) -> str:
    cfg = traigent.get_current_config()
    return f"Summary | model={cfg['model']}"  # call your LLM here
```

## Execution modes
- `edge_analytics` (default) - local execution with analytics.
- `cloud` - run in Traigent cloud.
- `hybrid` - mix local execution with cloud analytics.
- Mock: `TRAIGENT_MOCK_MODE=true` works with any mode.

## Quick knobs
- Constrain cost: smaller `max_trials`, cheaper models, `constraints={"cost_per_call": "<0.01"}`.
- Faster runs: discrete search spaces, `early_stopping=True`, sensible `trial_concurrency`.
- Safer outputs: use `safety-guardrails` example as a template.

## Directory cheat sheet
```text
examples/
|- core/              # Start here (mockable)
|- advanced/          # Deep dives and patterns
|- integrations/      # CI/CD and partner workflows
|- datasets/          # Shared evaluation data
`- docs/              # Guides like this one
```

## Troubleshooting highlights
- `ModuleNotFoundError: traigent` -> `pip install -e .` from repo root.
- `API key not found` -> `TRAIGENT_MOCK_MODE=true` or export keys.
- Slow runs -> lower `max_trials` or narrow the configuration space.
- Empty results -> verify dataset paths in `eval_dataset`.

Need more? See `START_HERE.md`, `EXAMPLES_GUIDE.md`, or `TROUBLESHOOTING.md`.
