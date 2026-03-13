# Traigent JS Examples

This directory mirrors the Python SDK example layout with examples that run on
this branch's native Node optimizer.

## Structure

- `quickstart/` - small runnable examples
- `adoption/` - minimal-change examples for existing client code
- `core/` - focused examples for specific SDK behaviors
- `datasets/` - shared JSONL inputs
- `utils/` - helpers used by the examples

## Run

```bash
npm run build:sdk
node examples/quickstart/00_agent_injection_map.mjs
node examples/quickstart/01_simple_qa.mjs
node examples/quickstart/02_customer_support_rag.mjs
node examples/quickstart/03_custom_objectives.mjs
node examples/core/tvl-loading/run.mjs
node examples/adoption/minimal-change/runner.mjs
node examples/adoption/seamless/runner.mjs
```

Read [quickstart/README.md](./quickstart/README.md) first if you want the
control-flow explanation before jumping into the runnable files.

For seamless support in this checkout:

- framework parameters: use the wrapped OpenAI, LangChain, or Vercel AI clients
- hardcoded local tuned vars: use `traigent migrate seamless` or the Babel plugin export
- plain Node functions: seamless also has an experimental runtime fallback for self-contained functions
