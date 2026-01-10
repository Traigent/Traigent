# Config Persistence Example

This example demonstrates the complete workflow from local optimization to production deployment.

## Workflow Overview

```
1. optimize.py    - Run optimization during development
2. export         - Export best config for version control
3. production.py  - Load exported config in production
```

## Files

- `optimize.py` - Development script that runs optimization and exports results
- `production.py` - Production script that loads the optimized config
- `configs/` - Directory for exported configs (committed to git)

## Quick Start

```bash
# 1. Run optimization (development)
TRAIGENT_MOCK_LLM=true python examples/persistence/optimize.py

# 2. Export via CLI (alternative to programmatic export)
traigent export sentiment_agent -o examples/persistence/configs/prod.json

# 3. Run in production mode
TRAIGENT_CONFIG_PATH=examples/persistence/configs/prod.json python examples/persistence/production.py
```

## Key Concepts

### Development Phase

```python
@traigent.optimize(
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"], ...},
    objectives=["accuracy", "cost"],
    eval_dataset="data/eval.jsonl",
)
async def my_agent(query: str) -> str:
    ...

# Run optimization
results = await my_agent.optimize()

# Export for deployment
my_agent.export_config("configs/prod.json")
```

### Production Phase

```python
@traigent.optimize(
    load_from="configs/prod.json",  # Auto-load on decoration
    configuration_space={"model": ["gpt-3.5-turbo", "gpt-4"]},
)
async def my_agent(query: str) -> str:
    ...

# Config already applied - no optimization runs!
assert my_agent.best_config is not None
```

### Environment Variable Override

```bash
# Override config path at runtime (useful for staging/prod environments)
export TRAIGENT_CONFIG_PATH=configs/staging.json
python production.py
```

## Best Practices

1. **Git Structure**: Commit `configs/` but gitignore `.traigent/`
2. **Environment Separation**: Use different config files for staging/prod
3. **Version Control**: Export configs when optimization improves results
4. **CI/CD**: Use `TRAIGENT_CONFIG_PATH` env var for deployment flexibility
