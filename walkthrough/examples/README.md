# TraiGent Walkthroughs

This directory contains the step-by-step walkthrough scripts (`01_*.py` – `07_*.py`)
used throughout the documentation. Every example follows the same structure and
relies on the shared helpers defined in `_shared.py` for:

- **Path setup** – `add_repo_root_to_sys_path` ensures the repository root is on
  `sys.path`, so running files directly (`python walkthrough/examples/01_simple_optimization.py`)
  works without extra environment tweaks.
- **Dataset generation** – `dataset_path` + `ensure_dataset` create reproducible
  JSONL datasets with realistic prompts the first time each example runs.
- **Mock mode** – `init_mock_mode()` reads `TRAIGENT_MOCK_MODE` and initializes
  the SDK in mock/edge analytics mode, allowing every example to execute without
  API keys.

## Running the walkthroughs

```bash
# Recommended: run in mock mode (no API calls)
TRAIGENT_MOCK_MODE=true python walkthrough/examples/01_simple_optimization.py

# Real API calls (set the relevant keys in your environment first)
python walkthrough/examples/05_rag_example.py
```

To exercise every script at once, use the smoke test:

```bash
bash scripts/test/smoke_test_walkthroughs.sh
```

The smoke test sets `TRAIGENT_MOCK_MODE=true` automatically and runs each script
with a timeout, making it safe to use in CI.

## Adding new examples

1. Drop a new `XX_descriptive_name.py` file in this folder.
2. Import the helpers:
   ```python
   from _shared import add_repo_root_to_sys_path, dataset_path, ensure_dataset, init_mock_mode

   add_repo_root_to_sys_path(__file__)
   ```
3. Call `dataset_path`/`ensure_dataset` to bootstrap any datasets you need.
4. Guard external dependencies or network calls behind `init_mock_mode()` checks
   so the example runs both online and offline.

Keeping these conventions ensures every walkthrough stays consistent, easy to
maintain, and quick to test.
