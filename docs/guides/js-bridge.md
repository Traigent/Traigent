# JS Bridge Removed

Python SDK <= 0.11.x supported Python-orchestrated JavaScript optimization through `execution={"runtime": "node"}` as a temporary bridge. Python SDK 0.12.0 removes that bridge.

The Python JS bridge was removed. The native JS SDK publish is pending; do not
document `npm install @traigent/sdk` as available yet. Use the JS SDK source
checkout instructions when available.

The native TypeScript SDK is not public yet; npm publication for
`@traigent/sdk` is pending.

Migration reference: the JS SDK's `docs/getting-started/minimal-integration.md`
and `docs/MIGRATION_FROM_PYTHON.md`, published with the JS SDK once it is public.

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
