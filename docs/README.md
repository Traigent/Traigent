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
- [RELEASING.md](./RELEASING.md) for the Changesets-based release flow
- [BRANCH_PROTECTION.md](./BRANCH_PROTECTION.md) for required `main` branch protection checks
- [TYPED_SESSION_SMOKE.md](./TYPED_SESSION_SMOKE.md) for the shared JS/Python typed-session smoke environment
- [SONAR_TRIAGE.md](./SONAR_TRIAGE.md) for current static-analysis triage and follow-up

## Current Branch Reality

- Native high-level agent optimization is supported.
- Backend-guided local execution through the typed session surface is supported.
- Hybrid session helpers are supported for:
  - create
  - list
  - status
  - finalize
  - delete
- `seamless` now covers:
  - framework interception for wrapped OpenAI, LangChain, and Vercel AI clients
  - codemod/build-time rewriting for hardcoded tuned variables
  - an experimental runtime rewrite fallback for self-contained plain Node functions when explicitly opted in
- Python-style remote cloud execution remains out-of-scope for JS.

## Project-Wide Note

The parity docs still distinguish native-first behavior from backend-guided
hybrid behavior when the constraints or guarantees differ. In this repository
root, both surfaces are now present and documented as one merged SDK.

## Reference Material

- [REAL_MODE_SEQUENCE_FLOW.md](./REAL_MODE_SEQUENCE_FLOW.md) is retained as a legacy bridge reference.
- Mermaid diagrams under [diagrams/](./diagrams) describe the older bridge/orchestrated flow, not the current native-first JS API.
