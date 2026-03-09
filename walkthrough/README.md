# Traigent JS Walkthrough

This mirrors the sibling Python walkthrough structure while staying honest to the JS SDK:

- offline / mock examples default to native mode
- real examples can use local native execution or backend-guided hybrid mode when `TRAIGENT_BACKEND_URL` and `TRAIGENT_API_KEY` are set

## Run a Mock Example

```bash
npm run build:sdk
node walkthrough/mock/01_tuning_qa.mjs
```

## Run a Real Example

```bash
export OPENAI_API_KEY="your-key"
npm run build:sdk
node walkthrough/real/01_tuning_qa.mjs
```

If you also set:

```bash
export TRAIGENT_BACKEND_URL="http://127.0.0.1:5000"
export TRAIGENT_API_KEY="..."
```

the real walkthrough examples will use backend-guided hybrid optimization by default.

## Structure

```text
walkthrough/
├── README.md
├── mock/
├── real/
├── datasets/
├── utils/
├── run_with_env.sh
└── test_all_examples.sh
```

## Example Topics

1. `01_tuning_qa.mjs`
2. `02_zero_code_change.mjs`
3. `03_parameter_mode.mjs`
4. `04_multi_objective.mjs`
5. `05_rag_parallel.mjs`
6. `06_custom_evaluator.mjs`
7. `07_multi_provider.mjs`
8. `08_privacy_modes.mjs`

## Smoke Run

```bash
npm run smoke:walkthrough
```
