# Production Deployment Example

Demonstrates the full production deployment workflow with Traigent:

1. **Optimize** — Run the optimization loop to find the best configuration
2. **Save** — Persist the best configuration to a JSON file
3. **Load** — Load the saved configuration in a production context
4. **Run** — Execute the function with the frozen configuration

## Run

```bash
# Mock mode (no API key needed)
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python run.py

# Real mode (requires OPENAI_API_KEY and langchain-openai)
# pip install langchain-openai
python run.py
```

## Key Patterns

- `save_best_config(result, path)` — Persist optimization results
- `load_config(path)` — Load a saved configuration
- `run_with_config(config, query)` — Execute with a frozen config
