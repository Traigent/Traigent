# Traigent JS Examples

This directory mirrors the Python SDK’s examples layout while staying faithful to the JS SDK surface.

## Structure

- `quickstart/` - the fastest runnable examples
- `core/` - focused examples for native, hybrid, TVL, and conditional/constraint flows
- `datasets/` - shared JSONL inputs copied from the sibling Python walkthrough datasets
- `utils/` - example helpers

## Quickstart

```bash
npm run build:sdk
node examples/quickstart/01_simple_qa.mjs
node examples/quickstart/02_customer_support_rag.mjs
node examples/quickstart/03_custom_objectives.mjs
```

## Smoke Run

```bash
npm run smoke:examples
```

## Notes

- Example scripts default to offline/mock behavior unless they explicitly live under `walkthrough/real/`.
- The aligned JS walkthrough is in [../walkthrough/README.md](../walkthrough/README.md).

