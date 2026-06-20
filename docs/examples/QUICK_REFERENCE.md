# Traigent Quick Reference Guide

Everything you need to run and adapt examples quickly.

## Common commands
```bash
# Install from repo root (includes example deps)
pip install "traigent[recommended]"

# Auto-mocks when ANTHROPIC_API_KEY is unset
python examples/core/simple-prompt/run.py

# With real LLMs
export ANTHROPIC_API_KEY="your-key" # pragma: allowlist secret
export OPENAI_API_KEY="your-key" # pragma: allowlist secret
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
    cfg = traigent.get_config()
    return f"Summary | model={cfg['model']}"  # call your LLM here
```

## Execution model
- `algorithm="auto"` (default) - cloud-first optimizer decisions, local trials, local fallback on connectivity failures.
- `algorithm="grid"` / `"random"` - explicit local optimizers with no cloud optimizer round trip.
- `offline=True` - zero Traigent backend egress.
- Mock: call `traigent.testing.enable_mock_mode_for_quickstart()` in local tutorial code.

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
`- gallery/           # Browser gallery and inline demo assets
```

## Troubleshooting highlights
- `ModuleNotFoundError: traigent` -> `pip install "traigent[recommended]"`.
- `API key not found` -> call `traigent.testing.enable_mock_mode_for_quickstart()` in local tutorial code or export keys.
- Slow runs -> lower `max_trials` or narrow the configuration space.
- Empty results -> verify dataset paths in `eval_dataset`.

Need more? See [Start Here](START_HERE.md), [Examples Guide](README.md), or
[Troubleshooting](TROUBLESHOOTING.md).
