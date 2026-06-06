# JS Bridge Removed

Python SDK <= 0.11.x supported Python-orchestrated JavaScript optimization through `execution={"runtime": "node"}` as a temporary bridge. Python SDK 0.12.0 removes that bridge.

The Python JS bridge was removed. The native JS SDK publish is pending; do not
document `npm install @traigent/sdk` as available yet. Use the JS SDK source
checkout instructions when available.

The current native TypeScript SDK support exists in the internal `traigent-js`
repo; npm publication for `@traigent/sdk` is pending.

Migration reference:

- https://github.com/Traigent/traigent-js/blob/main/docs/getting-started/minimal-integration.md
- https://github.com/Traigent/traigent-js/blob/main/docs/MIGRATION_FROM_PYTHON.md

Removed Python APIs include:

- `ExecutionOptions.runtime`
- `ExecutionOptions.js_module`
- `ExecutionOptions.js_function`
- `ExecutionOptions.js_timeout`
- `ExecutionOptions.js_parallel_workers`
- `ExecutionOptions.js_use_npx`
- `ExecutionOptions.js_runner_path`
- `ExecutionOptions.js_node_executable`
- `traigent.bridges.*`
- `traigent.evaluators.JSEvaluator`
