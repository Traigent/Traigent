# SDK smart-pruning producer result

## Scope

- Worktree: `core_project/worktrees/Traigent/sdk-smart-pruning-producer`
- Branch: `feat/sdk-smart-pruning-producer`
- Implementation commit: `e57feba4f32f3ba2dc477749ffb739ffa3df6a43`
- WorkIntent: `st_a5adda9b66c2`
- Result: SDK-only producer change. No TraigentSchema or TraigentBackend change is needed.

## Contract confirmation

- `TraigentSchema/traigent_schema/schemas/optimization/smart_pruning_schema.json` already defines the session-create `smart_pruning` object: required `label` with `aggressive|balanced|conservative`, plus optional bounded numeric parameters.
- `TraigentSchema/traigent_schema/schemas/optimization/intermediate_report_schema.json` already defines the intermediate-report request/response contract:

```json
{
  "session_id": "session_123",
  "trial_id": "trial_abc",
  "running_score": 0.75,
  "examples_attempted": 3,
  "partial_cost_usd": 0.03,
  "objective_name": "accuracy"
}
```

The SDK sends only schema fields. `partial_cost_usd` and `objective_name` are optional and omitted when unknown. The backend response consumed by the SDK is `{ "prune": boolean, "prune_reason": string | null }`.

## Design

- Public SDK surface:
  - `traigent/api/decorators.py:201` adds `SmartPruningOptions`, matching the schema labels and bounds.
  - `traigent/api/decorators.py:270` adds `ExecutionOptions.smart_pruning`.
  - `traigent/api/decorators.py:2154` adds the direct `@optimize(..., smart_pruning=...)` option.
  - `traigent/api/decorators.py:956` normalizes dict/model input to the wire dict and rejects extra fields via Pydantic.
  - `traigent/api/decorators.py:2714` and `traigent/api/decorators.py:2952` pass the normalized config into `OptimizedFunction`.

- Session-create producer:
  - `traigent/core/optimized_function.py:717` stores `self.smart_pruning`; `traigent/core/optimized_function.py:1876` passes it to the orchestrator.
  - `traigent/core/orchestrator.py:213`, `traigent/core/orchestrator.py:286`, `traigent/core/orchestrator.py:2208`, and `traigent/core/orchestrator.py:2379` preserve the config and pass it into `BackendSessionManager.create_session`.
  - `traigent/core/backend_session_manager.py:230`, `traigent/core/backend_session_manager.py:667`, `traigent/core/backend_session_manager.py:705`, and `traigent/core/backend_session_manager.py:796` carry the effective run-scoped config into the backend client.
  - `traigent/cloud/models.py:231`, `traigent/cloud/session_operations.py:501`, `traigent/cloud/session_operations.py:633`, `traigent/cloud/backend_client.py:1366`, and `traigent/cloud/backend_client.py:1383` add the request field and delegate it through the cloud client.
  - `traigent/cloud/api_operations.py:664` and `traigent/cloud/api_operations.py:711` add top-level `smart_pruning` to typed and legacy session-create payloads; `traigent/cloud/client.py:1942` keeps the direct cloud-client serializer aligned.

- Intermediate-report producer:
  - `traigent/core/backend_session_manager.py:857` gates reports strictly on configured `smart_pruning`, backend tracking, a real session id, not offline/no-egress, and a cloud-required/cloud-brain execution policy.
  - `traigent/core/trial_lifecycle.py:681` reintroduces progress tracking only through that cloud-aware gate.
  - Cadence is after each evaluated example/step. This matches pruning algorithms that need early partial curves, keeps payloads content-free, and is zero-egress unless `smart_pruning` is explicitly enabled on a managed/cloud run.
  - `traigent/core/trial_lifecycle.py:720`, `traigent/core/trial_lifecycle.py:739`, `traigent/core/trial_lifecycle.py:751`, and `traigent/core/trial_lifecycle.py:758` build the schema-correct payload from aggregate numeric progress only.
  - `traigent/cloud/api_operations.py:806` POSTs to `/sessions/{session_id}/intermediate-report` and returns the backend prune decision.

- Prune flow:
  - `traigent/core/trial_lifecycle.py:764` raises `TrialPrunedError` when `prune=true`.
  - `traigent/evaluators/base.py:1660`, `traigent/evaluators/base.py:1764`, `traigent/evaluators/base.py:2010`, `traigent/evaluators/base.py:2459`, and `traigent/evaluators/base.py:3112` preserve partial example results when the callback prunes.
  - Existing lifecycle PRUNED handling is reused; `traigent/core/trial_lifecycle.py:1133` keeps the prune reason on the submitted trial result.

## Changed files

- Smart-pruning implementation: `traigent/api/decorators.py`, `traigent/core/optimized_function.py`, `traigent/core/orchestrator.py`, `traigent/core/backend_session_manager.py`, `traigent/core/trial_lifecycle.py`, `traigent/evaluators/base.py`, `traigent/cloud/models.py`, `traigent/cloud/session_operations.py`, `traigent/cloud/api_operations.py`, `traigent/cloud/backend_client.py`, `traigent/cloud/client.py`.
- Tests: `tests/unit/api/decorator_tests/test_decorator_basic.py`, `tests/unit/cloud/test_session_creation_warm_start.py`, `tests/unit/cloud/test_api_operations.py`, `tests/unit/core/test_trial_lifecycle.py`.
- Type-only hook fixes required by full-repo mypy/pre-commit: `traigent/config/types.py`, `traigent/core/metadata_helpers.py`, `traigent/security/session_manager.py`, `traigent/security/input_validation.py`.

## Tests and checks

- `python3 -m pytest -q -n0 tests/unit/core/test_trial_lifecycle.py tests/unit/cloud/test_session_creation_warm_start.py tests/unit/cloud/test_api_operations.py::TestIntermediateReport tests/unit/api/decorator_tests/test_decorator_basic.py`
  - Result: `88 passed, 2 warnings`.
- `python3 -m ruff check ...changed files...`
  - Result: passed.
- `python3 -m mypy traigent`
  - Result: passed (`466 source files`; note-only output from existing untyped analytics functions and unused pyproject mypy sections).
- Commit hook on implementation commit:
  - isort, black, ruff, detect-secrets, whitespace/end-of-file, merge-conflict/private-key checks, bandit, mypy, TVL specs, strict cost accounting, public test assertion contracts, dependency floor drift, ignored-file check all passed.

## Open risks

- The intermediate-report egress path is covered by unit tests with mocked aiohttp, not by a live backend integration test.
- The producer sends aggregate numeric progress only; if future backend pruning needs richer curve metadata, TraigentSchema should be extended first.
- A direct `ruff format --check` on `traigent/config/types.py` disagrees with the repo's Black hook formatting for one cast expression. `ruff check`, Black, and pre-commit all pass.

## Schema/BE changes

None required. The SDK now produces the already-defined `smart_pruning` session-create field and the already-defined `intermediate-report` payload, and consumes the already-defined `{prune, prune_reason}` response.
