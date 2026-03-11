# Docs Index

This docs folder mirrors the Python SDK structure where it makes sense, but the
content here is specific to the JS SDK project and this native-first checkout.

## Start Here

- [../README.md](../README.md) for the package overview
- [CLIENT_CODE_GUIDE.md](./CLIENT_CODE_GUIDE.md) for the current integration model
- [NATIVE_JS_PARITY_MATRIX.md](./NATIVE_JS_PARITY_MATRIX.md) for Python vs JS feature parity
- [Hybrid JS Parity Matrix](../../traigent-js-hybrid-optuna/docs/HYBRID_JS_PARITY_MATRIX.md) for the backend-guided session branch
- [PYTHON_SDK_MODULE_CATALOG_AND_GAP_ANALYSIS.md](./PYTHON_SDK_MODULE_CATALOG_AND_GAP_ANALYSIS.md) for the Python SDK module-family inventory and JS gap analysis
- [JS_PARITY_ROADMAP.md](./JS_PARITY_ROADMAP.md) for the prioritized native-vs-hybrid execution roadmap
- [CROSS_SDK_VALIDATION.md](./CROSS_SDK_VALIDATION.md) for cross-SDK test strategy
- [SCHEMA_ALIGNMENT_AUDIT.md](./SCHEMA_ALIGNMENT_AUDIT.md) for JS SDK vs TraigentSchema entity alignment

## Current Branch Reality

- Native high-level agent optimization is the primary supported path.
- Hybrid spec authoring is supported through `toHybridConfigSpace()`.
- Backend-guided `execution.mode = 'hybrid'` is not implemented in this checkout.
- `seamless` now covers:
  - framework interception for wrapped OpenAI, LangChain, and Vercel AI clients
  - codemod/build-time rewriting for hardcoded tuned variables
  - an experimental runtime rewrite fallback for self-contained plain Node functions

## Project-Wide Note

The overall JS SDK project currently spans two active lines of work:

- this native-first checkout, which is the source of truth for local/native optimization ergonomics
- a separate hybrid-enabled worktree, which carries backend-guided session execution work that is not merged into this checkout yet

When a parity document says "this checkout", it refers to the code currently in
this repository root. When a document says "overall JS project", it includes the
separate hybrid worktree status as well.

## Reference Material

- [REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md) is retained as a legacy bridge reference.
- Mermaid diagrams under [diagrams/](./diagrams) describe the older bridge/orchestrated flow, not the current native-first JS API.
