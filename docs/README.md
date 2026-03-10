# Docs Index

This docs folder mirrors the Python SDK structure where it makes sense, but the
content here is specific to this JS checkout.

## Start Here

- [../README.md](../README.md) for the package overview
- [CLIENT_CODE_GUIDE.md](./CLIENT_CODE_GUIDE.md) for the current integration model
- [NATIVE_JS_PARITY_MATRIX.md](./NATIVE_JS_PARITY_MATRIX.md) for Python vs JS feature parity
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

## Reference Material

- [REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md) is retained as a legacy bridge reference.
- Mermaid diagrams under [diagrams/](./diagrams) describe the older bridge/orchestrated flow, not the current native-first JS API.
