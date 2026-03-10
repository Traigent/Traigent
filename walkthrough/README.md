# Traigent JS Walkthrough

This walkthrough mirrors the Python SDK structure with runnable native Node
examples for this branch.

## Layout

- `mock/` - offline walkthrough runs
- `real/` - real-provider walkthrough runs
- `datasets/` - shared JSONL inputs
- `utils/` - walkthrough helpers

## Run a Mock Walkthrough

```bash
npm run build:sdk
node walkthrough/mock/01_tuning_qa.mjs
```

## Run a Real Walkthrough

```bash
export OPENAI_API_KEY="..."
npm run build:sdk
node walkthrough/real/01_tuning_qa.mjs
```

This branch is native-only. Backend-guided hybrid walkthroughs live on the
`feature/hybrid-optuna-session-mode` branch.
