# Python SDK Module Catalog and Gap Analysis

This document audits the canonical Python SDK package under:

- [Traigent/traigent](../../Traigent/traigent)

It has two purposes:

1. create a complete module-family catalog using a consistent format
2. compare those Python capabilities to the current JS SDK project state

Branch-aware reading guide:

- `matched`: implemented and covered by passing tests or a verified public example
- `partial`: implemented with bounded semantics and covered, but still behind Python
- `gap`: the backend/API contract is reachable today and the JS side could implement it now
- `deferred-backend`: blocked on missing or insufficiently specified backend/protocol support
- `out-of-scope`: not a current JS SDK target

## Scope and Counting Method

- Root audited: [Traigent/traigent](../../Traigent/traigent)
- Included: tracked package artifacts under that root (`.py`, `.md`, `.yaml`, `py.typed`, root package files)
- Excluded: `__pycache__`, `.mypy_cache`
- Total tracked package artifacts covered: `366`

## Documentation Format

Each catalog row uses the same fields:

- `Category`: top-level package family
- `Label`: concise functional label
- `Description`: what the family does
- `Key Types / APIs`: main public or architectural surfaces
- `Responsibility`: what this family owns in the Python SDK
- `File Count`: tracked files under that family
- `Native Status`: current state of this native-first checkout
- `Hybrid Status`: current state of the hybrid-enabled worktree
- `Overall JS Status`: project-wide status across both active JS lines
- `Evidence`: passing tests or verified public examples for `matched` / `partial` rows

## Branch-aware Status Summary

This summary is the planning surface. It distinguishes the native checkout, the
hybrid worktree, and the overall JS project so the catalog below does not
collapse two different implementation lines into one label.

| Category | Native Status | Hybrid Status | Overall JS Status | Evidence |
| --- | --- | --- | --- | --- |
| `api` | `partial` | `partial` | `partial` | Native: [`tests/unit/optimization/spec.test.ts`](../tests/unit/optimization/spec.test.ts), [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/spec.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/spec.test.ts), [`../../traigent-js-hybrid-optuna/tests/unit/optimization/agent.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/agent.test.ts) |
| `config` | `partial` | `partial` | `partial` | Native: [`tests/unit/core/context.test.ts`](../tests/unit/core/context.test.ts), [`tests/unit/seamless/transform.test.ts`](../tests/unit/seamless/transform.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/agent.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/agent.test.ts) |
| `core` | `partial` | `partial` | `partial` | Native: [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts) |
| `optimizers` | `partial` | `partial` | `partial` | Native: [`tests/unit/optimization/native.test.ts`](../tests/unit/optimization/native.test.ts), [`tests/unit/optimization/native-bayesian.test.ts`](../tests/unit/optimization/native-bayesian.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts) |
| `evaluators` | `partial` | `partial` | `partial` | Native: [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/agent.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/agent.test.ts) |
| `integrations` | `partial` | `partial` | `partial` | Native: [`tests/unit/integrations/framework-interception.test.ts`](../tests/unit/integrations/framework-interception.test.ts), [`tests/unit/integrations/auto-wrap.test.ts`](../tests/unit/integrations/auto-wrap.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/integrations/framework-interception.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/integrations/framework-interception.test.ts), [`../../traigent-js-hybrid-optuna/tests/unit/integrations/auto-wrap.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/integrations/auto-wrap.test.ts) |
| `hybrid` | `out-of-scope` | `partial` | `partial` | Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts) |
| `cloud` | `out-of-scope` | `partial` | `partial` | Hybrid session helpers: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/hybrid.test.ts) |
| `tvl` | `partial` | `partial` | `partial` | Native: [`tests/unit/optimization/tvl.test.ts`](../tests/unit/optimization/tvl.test.ts), [`tests/unit/optimization/native-promotion.test.ts`](../tests/unit/optimization/native-promotion.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/optimization/tvl.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/optimization/tvl.test.ts) |
| `tuned_variables` | `partial` | `gap` | `partial` | Native: [`tests/unit/tuned-variables/discovery.test.ts`](../tests/unit/tuned-variables/discovery.test.ts), [`tests/unit/cli/detect.test.ts`](../tests/unit/cli/detect.test.ts); Hybrid: no tuned-variable discovery implementation today. |
| `agents` | `gap` | `gap` | `gap` | High-level agent optimization exists, but Python’s broader agent-platform mapping surface still has no JS equivalent. |
| `config_generator` | `out-of-scope` | `out-of-scope` | `out-of-scope` | Separate product layer, not a current JS SDK target. |
| `metrics` | `partial` | `partial` | `partial` | Native: [`tests/unit/optimization/native-cost.test.ts`](../tests/unit/optimization/native-cost.test.ts), [`tests/unit/optimization/agent.test.ts`](../tests/unit/optimization/agent.test.ts); Hybrid: [`../../traigent-js-hybrid-optuna/tests/unit/integrations/shared.test.ts`](../../traigent-js-hybrid-optuna/tests/unit/integrations/shared.test.ts) |
| `utils` | `partial` | `partial` | `partial` | Distributed across optimization/runtime helper coverage in both branches. |
| `cli` | `partial` | `partial` | `partial` | Native codemod/build-time seamless CLI exists; bridge/hybrid CLI breadth is still limited. |
| `security` / `analytics` / `experimental` | `out-of-scope` | `out-of-scope` | `out-of-scope` | Platform-side families are not being mirrored in the JS SDK package. |

## Functional Catalog

| Category | Label | Description | Key Types / APIs | Responsibility | File Count | Overall JS Status |
| --- | --- | --- | --- | --- | ---: | --- |
| `(root)` | Package entrypoints and compatibility surface | Public package exports, versioning, top-level client shim, package metadata | `optimize`, top-level helpers, `traigent_client.py`, `__init__.py` | Defines the Python SDK’s public entry surface and legacy compatibility affordances | 6 | `partial` |
| `api` | Decorator and public optimization API | High-level decorator surface, config builders, constraints, parameter ranges, public types | `decorators.py`, `functions.py`, `types.py`, `config_space.py` | Owns the user-facing optimization contract | 13 | `partial` |
| `config` | Runtime config and injection plumbing | Injection modes, AST transform, runtime injector, backend/provider config, context, parallel config | `seamless_injection.py`, `runtime_injector.py`, `ast_transformer.py`, `types.py` | Owns runtime config resolution and injection behavior | 13 | `partial` |
| `core` | Optimization orchestration runtime | Trial lifecycle, orchestration pipeline, stop conditions, cost enforcement, sampling, metrics aggregation, state | `orchestrator.py`, `optimization_pipeline.py`, `stop_conditions.py`, `cost_enforcement.py` | Owns the execution engine | 44 | `partial` |
| `optimizers` | Search algorithms and optimizer adapters | Grid/random/Bayesian/Optuna/cloud/interactive coordinators, pruners, checkpoints | `grid.py`, `random.py`, `bayesian.py`, `optuna_optimizer.py`, `pruners.py` | Owns search strategy execution | 20 | `partial` |
| `evaluators` | Dataset evaluation and metrics runtime | Local, JS, hybrid API evaluators; dataset registry; metrics tracking | `local.py`, `js_evaluator.py`, `hybrid_api.py`, `metrics.py` | Owns example-level evaluation loops | 8 | `partial` |
| `integrations` | Framework, provider, observability, vector-store integrations | LLM plugins, framework overrides, LangChain, PydanticAI, Langfuse, observability, discovery | `framework_override.py`, `langchain/handler.py`, `llms/*`, `observability/*` | Owns ecosystem integration and seamless framework control | 65 | `partial` |
| `hybrid` | Hybrid transport protocol layer | Discovery, lifecycle, protocol, HTTP/MCP transports | `protocol.py`, `transport.py`, `http_transport.py`, `mcp_transport.py` | Owns hybrid execution transport contracts | 7 | `partial` |
| `cloud` | Backend/cloud client and session stack | Backend clients, session/trial operations, auth, billing, sync, DTOs, privacy, subsets | `client.py`, `models.py`, `sessions.py`, `trial_operations.py`, `billing.py` | Owns both typed-session control-plane integration and Python-specific remote cloud execution families | 35 | `partial` |
| `tvl` | TVL specification system | TVL spec loading, models, validation, objectives, promotion, statistics, CLI | `spec_loader.py`, `models.py`, `promotion_gate.py`, `statistics.py` | Owns the TVL configuration language | 10 | `partial` |
| `tuned_variables` | Tuned-variable discovery | Detection strategies, discovery, dataflow analysis, typed results | `detector.py`, `discovery.py`, `dataflow_strategy.py` | Owns automatic tuned-variable discovery | 7 | `partial` |
| `agents` | Agent platform mapping and execution helpers | Config mapping, platform abstractions, agent spec generation, executor | `platforms.py`, `config_mapper.py`, `executor.py` | Owns agent-specific optimization adaptation | 5 | `gap` |
| `config_generator` | Config/TVL generation pipeline | Agent classification, benchmark/objective/constraint/tvar recommendation, apply pipeline | `pipeline.py`, `apply.py`, `subsystems/*`, `presets/*` | Owns assisted generation of optimization specs | 18 | `out-of-scope` |
| `metrics` | Metric registry and advanced metric helpers | Agent metrics, content scoring, RAGAS metrics, registry | `registry.py`, `content_scoring.py`, `ragas_metrics.py` | Owns reusable metric families | 5 | `partial` |
| `utils` | Shared utilities | Cost calculator, persistence, retry, objectives, analytics helpers, diagnostics, logging | `cost_calculator.py`, `persistence.py`, `multi_objective.py`, `validation.py` | Owns reusable cross-cutting helpers | 31 | `partial` |
| `cli` | CLI workflows | Auth, local commands, tuned-variable detection, config generation, validation, entrypoint | `main.py`, `detect_tvars_command.py`, `generate_config_command.py` | Owns the Python CLI UX | 10 | `partial` |
| `security` | Enterprise security and auth stack | Headers, JWT, crypto, rate limiting, enterprise policies, auth factors, sessions, tenancy | `jwt_validator.py`, `enterprise.py`, `auth/*`, `session_manager.py` | Owns security controls and enterprise hardening | 22 | `out-of-scope` |
| `analytics` | Post-run analytics and intelligence | Scheduling, predictive analytics, anomaly detection, meta-learning, cost optimization | `predictive.py`, `anomaly.py`, `meta_learning.py` | Owns analytics beyond core optimization execution | 8 | `out-of-scope` |
| `bridges` | Cross-runtime bridge layer | JS bridge and process pool plumbing | `js_bridge.py`, `process_pool.py` | Owns Python-to-JS bridge execution | 3 | `out-of-scope` |
| `invokers` | Invocation abstractions | Base/local/batch/streaming invokers | `base.py`, `local.py`, `batch.py`, `streaming.py` | Owns standardized invocation surfaces | 5 | `out-of-scope` |
| `storage` | Local persistence storage | Local storage package | `local_storage.py` | Owns on-disk storage helpers | 2 | `partial` |
| `wrapper` | Wrapper service/server layer | Service/server/errors for wrapper mode | `server.py`, `service.py`, `errors.py` | Owns wrapper-hosted execution surface | 4 | `deferred-backend` |
| `providers` | Provider validation helpers | Provider validation utilities | `validation.py` | Owns provider-side validation helpers | 2 | `partial` |
| `plugins` | Plugin registry | Plugin registration and loading | `registry.py` | Owns extensibility registration | 2 | `out-of-scope` |
| `telemetry` | Telemetry helpers | Optuna metrics telemetry | `optuna_metrics.py` | Owns optimizer telemetry helpers | 2 | `out-of-scope` |
| `reporting` | Reporting helpers | Example map reporting | `example_map.py` | Owns reporting outputs | 2 | `partial` |
| `visualization` | Visualization | Plot generation and quick plots | `plots.py` | Owns visualization layer | 2 | `partial` |
| `hooks` | Hook tooling | Hook installer, config, validation | `installer.py`, `validator.py` | Owns hook-based workflow tooling | 4 | `out-of-scope` |
| `adapters` | Execution adapter | Adapter between execution environments | `execution_adapter.py` | Owns adapter abstraction | 1 | `out-of-scope` |
| `experimental` | Experimental features | Simple cloud simulator/platform experiments | `simple_cloud/*` | Holds unstable or exploratory features | 10 | `out-of-scope` |

## Gap Analysis

### Areas Where Current JS Is Strong

- Public optimize API and high-level agent optimization mental model
- Native runtime search algorithms: grid, random, local Bayesian
- Evaluation loop ownership in native mode
- Explicit injection modes: `context`, `parameter`, and practical `seamless`
- Cost and stop-condition handling for native-local optimization
- Basic framework interception for OpenAI / LangChain / Vercel AI, plus
  `autoWrapFrameworkTarget(...)` / `autoWrapFrameworkTargets({...})` helpers
  and bounded explicit-object discovery via `discoverFrameworkTargets(...)` /
  `prepareFrameworkTargets(...)`
- DTO/measure validation and native runtime coverage

### Areas Where Python Still Exceeds This JS Checkout

#### 1. Cloud and Hybrid Execution

Python owns a full remote stack across:
- [cloud](../../Traigent/traigent/cloud)
- [hybrid](../../Traigent/traigent/hybrid)
- parts of [core](../../Traigent/traigent/core)
- parts of [optimizers](../../Traigent/traigent/optimizers)

Current JS project:
- native-first in this checkout
- backend-guided hybrid execution exists in the separate hybrid-enabled worktree
- that hybrid worktree now includes:
  - typed session lifecycle
  - low-level typed session helpers for create / next-trial / submit-result
  - list / health / status / finalize / delete helpers with normalized DTOs
  - high-level plain-agent evaluation
  - backend-guided config suggestion
  - OpenAI / LangChain / Vercel AI seamless interception with runtime metric collection
- fully targets backend-guided local execution over the typed `/sessions` surface
- does **not** target Python's server-side remote execution model, where agents are reconstructed and invoked in the cloud
- therefore, the remaining Python cloud-execution families should be treated as `out-of-scope` for JS unless that product decision changes

#### 2. TVL

Python has a full TVL language/runtime:
- spec loading
- models
- validation
- promotion policy
- banded/statistical objectives
- TVL CLI

Current JS checkout:
- implements a focused native TVL subset:
  - typed `tvars`
  - banded objectives
  - structural and derived constraints compiled to callbacks through a parsed
    safe-expression subset
  - exploration strategy/budget mapping
  - promotion-policy parsing
  - artifact-specific `nativeCompatibility` reporting on loaded TVL artifacts,
    including `usedFeatures` and summarized `warnings`
- still does not implement the full Python TVL runtime or CLI, and it does not
  expose the full Python promotion-gate lifecycle/reporting model; native JS now applies
  `minEffect` and `tieBreakers` during best-trial selection, uses sample-based
  paired promotion / TOST band promotion when trials expose metric samples,
  `chanceConstraints` can reject trials when they provide explicit counts
  or binary metric samples, and native results/trials now expose bounded
  `promotionDecision` reports

#### 3. Cloud Clients, Sessions, and Control Plane

Python includes:
- session lifecycle APIs
- DTO/model contracts
- subset selection
- billing/privacy/auth operations
- sync/resilience layers

Current JS project:
- typed interactive session orchestration exists in the hybrid-enabled worktree
- that worktree now covers the practical decorator path better than before:
  - plain agent functions
  - local evaluation
  - seamless framework overrides
  - framework auto-wrap helpers, bounded explicit-object discovery, and seamless diagnostics
  - low-level typed session helpers for create / next-trial / submit-result
  - session list / status / finalize / delete helpers with normalized DTOs
  - executable hybrid session-control example
  - provider-derived cost/token/latency submission
- now covers the reachable typed `/sessions` control-plane surface well
- still lacks Python's broader cloud/session/control-plane surface, resilience
  layers, and DTO breadth where those depend on:
  - backend routes that are not exposed on the current typed `/sessions` surface, or
  - Python's remote cloud-execution model, which JS does not target

#### 4. Config Generation and Assisted Authoring

Python includes:
- agent classification
- benchmark catalogs
- objective recommendation
- safety-constraint recommendation
- tuned-variable range generation still remains behind Python

Current JS checkout:
- no equivalent authoring-generation pipeline

#### 5. Enterprise Security / Platform Concerns

Python package contains an enterprise/platform layer:
- auth
- JWT/OIDC/SAML/MFA
- encryption
- rate limiting
- tenancy
- session security

Current JS checkout:
- not a product target for the local SDK package
- divergence is justified and expected

### Divergences That Are Justified

- Python has platform/security/analytics families that should not be copied directly into the JS SDK package.
- Python cloud/hybrid clients belong with the backend-enabled JS branch/worktree, not this native-first checkout.
- Python's remote cloud-execution APIs are not a JS parity target; JS hybrid is explicitly backend-guided **local** execution.
- TVL parity in JS should be deliberate; it should not be copied blindly if the JS user-facing authoring API is clearer.

### Gaps That Are Not Yet Justified

- full hybrid/cloud session parity in the dedicated hybrid JS branch should be
  reconciled against the **reachable** typed-session portions of
  [cloud](../../Traigent/traigent/cloud) and
  [hybrid](../../Traigent/traigent/hybrid), not Python's remote cloud-execution-only surfaces
- TVL support remains a meaningful Python-to-JS feature gap if the long-term goal is cross-SDK authoring parity
- tuned-variable discovery is now partially covered in the native checkout through bounded heuristics and a CLI surface, but both discovery breadth and config-generation parity are still materially behind Python

## Recommended Next Work

1. Treat this doc as the canonical category map for Python-to-JS parity discussions.
2. Keep native JS work focused on `api`, `config`, `core`, `optimizers`, `evaluators`, `integrations`, `metrics`, and selected `utils`.
3. Use the hybrid JS branch to close the `cloud` and `hybrid` gaps, not this native-first checkout.
4. If cross-SDK authoring parity matters, bring over TVL in a deliberate phase instead of re-encoding it ad hoc.
5. Consider a second pass that maps every Python file to a concrete JS file or explicit `deferred/out-of-scope` decision.

## Complete File Inventory

This appendix covers every tracked artifact counted in the audit.

```text
[(root)]
__init__.py
_version.py
conftest.py
optigen_integration.py
py.typed
traigent_client.py
[adapters]
adapters/execution_adapter.py
[agents]
agents/__init__.py
agents/config_mapper.py
agents/executor.py
agents/platforms.py
agents/specification_generator.py
[analytics]
analytics/__init__.py
analytics/anomaly.py
analytics/cost_optimization.py
analytics/example_insights.py
analytics/intelligence.py
analytics/meta_learning.py
analytics/predictive.py
analytics/scheduling.py
[api]
api/__init__.py
api/agent_inference.py
api/config_builder.py
api/config_space.py
api/constraint_builders.py
api/constraints.py
api/decorators.py
api/functions.py
api/parameter_ranges.py
api/parameter_validator.py
api/safety.py
api/types.py
api/validation_protocol.py
[bridges]
bridges/__init__.py
bridges/js_bridge.py
bridges/process_pool.py
[cli]
cli/__init__.py
cli/auth_commands.py
cli/detect_tvars_command.py
cli/function_discovery.py
cli/generate_config_command.py
cli/hooks_commands.py
cli/local_commands.py
cli/main.py
cli/optimization_validator.py
cli/validation_types.py
[cloud]
cloud/README.md
cloud/__init__.py
cloud/_aiohttp_compat.py
cloud/agent_dtos.py
cloud/api_key_manager.py
cloud/api_operations.py
cloud/auth.py
cloud/backend_bridges.py
cloud/backend_client.py
cloud/backend_components.py
cloud/backend_synchronizer.py
cloud/billing.py
cloud/client.py
cloud/cloud_operations.py
cloud/credential_manager.py
cloud/credential_resolver.py
cloud/dataset_converter.py
cloud/dtos.py
cloud/event_manager.py
cloud/integration_manager.py
cloud/models.py
cloud/optimizer_client.py
cloud/password_auth_handler.py
cloud/privacy_operations.py
cloud/production_mcp_client.py
cloud/resilient_client.py
cloud/service.py
cloud/session_operations.py
cloud/sessions.py
cloud/subset_selection.py
cloud/sync_manager.py
cloud/token_manager.py
cloud/trial_operations.py
cloud/trial_tracker.py
cloud/validators.py
[config]
config/__init__.py
config/api_keys.py
config/ast_transformer.py
config/backend_config.py
config/context.py
config/feature_flags.py
config/models.yaml
config/parallel.py
config/providers.py
config/runtime_injector.py
config/seamless_injection.py
config/seamless_optuna_adapter.py
config/types.py
[config_generator]
config_generator/__init__.py
config_generator/agent_classifier.py
config_generator/apply.py
config_generator/llm_backend.py
config_generator/pipeline.py
config_generator/presets/__init__.py
config_generator/presets/agent_type_catalog.py
config_generator/presets/benchmark_catalog.py
config_generator/presets/constraint_templates.py
config_generator/presets/range_presets.py
config_generator/subsystems/__init__.py
config_generator/subsystems/benchmarks.py
config_generator/subsystems/objectives.py
config_generator/subsystems/safety_constraints.py
config_generator/subsystems/structural_constraints.py
config_generator/subsystems/tvar_ranges.py
config_generator/subsystems/tvar_recommendations.py
config_generator/types.py
[core]
core/__init__.py
core/backend_session_manager.py
core/cache_policy.py
core/ci_approval.py
core/config_builder.py
core/config_state_manager.py
core/constants.py
core/cost_enforcement.py
core/cost_estimator.py
core/evaluator_wrapper.py
core/license.py
core/llm_processor.py
core/logger_facade.py
core/mandatory_metrics.py
core/meta_types.py
core/metadata_helpers.py
core/metric_registry.py
core/metrics_aggregator.py
core/namespace.py
core/objectives.py
core/optimization_pipeline.py
core/optimized_function.py
core/orchestrator.py
core/orchestrator_helpers.py
core/parallel_execution_manager.py
core/progress_manager.py
core/pruning_progress_tracker.py
core/refactoring_utils.py
core/result_selection.py
core/sample_budget.py
core/samplers/__init__.py
core/samplers/random_sampler.py
core/session_context.py
core/stat_significance.py
core/stop_condition_manager.py
core/stop_conditions.py
core/tracing.py
core/trial_context.py
core/trial_lifecycle.py
core/trial_result_factory.py
core/types.py
core/types_ext.py
core/utils.py
core/workflow_trace_manager.py
[evaluators]
evaluators/__init__.py
evaluators/base.py
evaluators/dataset_registry.py
evaluators/hybrid_api.py
evaluators/js_evaluator.py
evaluators/local.py
evaluators/metrics.py
evaluators/metrics_tracker.py
[experimental]
experimental/README.md
experimental/__init__.py
experimental/simple_cloud/README.md
experimental/simple_cloud/platforms/__init__.py
experimental/simple_cloud/platforms/anthropic_executor.py
experimental/simple_cloud/platforms/base_platform.py
experimental/simple_cloud/platforms/cohere_executor.py
experimental/simple_cloud/platforms/huggingface_executor.py
experimental/simple_cloud/platforms/parameter_mapping.py
experimental/simple_cloud/simulator.py
[hooks]
hooks/__init__.py
hooks/config.py
hooks/installer.py
hooks/validator.py
[hybrid]
hybrid/__init__.py
hybrid/discovery.py
hybrid/http_transport.py
hybrid/lifecycle.py
hybrid/mcp_transport.py
hybrid/protocol.py
hybrid/transport.py
[integrations]
integrations/__init__.py
integrations/activation.py
integrations/base.py
integrations/base_plugin.py
integrations/batch_wrapper.py
integrations/bedrock_client.py
integrations/config.py
integrations/dspy_adapter.py
integrations/framework_override.py
integrations/langchain/__init__.py
integrations/langchain/handler.py
integrations/langfuse/__init__.py
integrations/langfuse/callback.py
integrations/langfuse/client.py
integrations/langfuse/tracker.py
integrations/llms/__init__.py
integrations/llms/anthropic_plugin.py
integrations/llms/azure_openai_plugin.py
integrations/llms/base_llm_plugin.py
integrations/llms/bedrock_plugin.py
integrations/llms/cohere_plugin.py
integrations/llms/gemini_plugin.py
integrations/llms/huggingface_plugin.py
integrations/llms/langchain/__init__.py
integrations/llms/langchain/base.py
integrations/llms/langchain/discovery.py
integrations/llms/langchain_plugin.py
integrations/llms/llamaindex_plugin.py
integrations/llms/mistral_plugin.py
integrations/llms/openai.py
integrations/llms/openai_plugin.py
integrations/mappings.py
integrations/model_discovery/__init__.py
integrations/model_discovery/anthropic_discovery.py
integrations/model_discovery/azure_discovery.py
integrations/model_discovery/base.py
integrations/model_discovery/cache.py
integrations/model_discovery/gemini_discovery.py
integrations/model_discovery/mistral_discovery.py
integrations/model_discovery/openai_discovery.py
integrations/model_discovery/registry.py
integrations/observability/__init__.py
integrations/observability/mlflow.py
integrations/observability/wandb.py
integrations/observability/workflow_traces.py
integrations/plugin_registry.py
integrations/providers.py
integrations/pydantic_ai/__init__.py
integrations/pydantic_ai/_types.py
integrations/pydantic_ai/handler.py
integrations/pydantic_ai/plugin.py
integrations/utils/__init__.py
integrations/utils/discovery.py
integrations/utils/message_coercion.py
integrations/utils/mock_adapter.py
integrations/utils/parameter_normalizer.py
integrations/utils/response_wrapper.py
integrations/utils/validation.py
integrations/utils/version_compat.py
integrations/vector_stores/__init__.py
integrations/vector_stores/base.py
integrations/vector_stores/chromadb_plugin.py
integrations/vector_stores/pinecone_plugin.py
integrations/vector_stores/weaviate_plugin.py
integrations/wrappers.py
[invokers]
invokers/__init__.py
invokers/base.py
invokers/batch.py
invokers/local.py
invokers/streaming.py
[metrics]
metrics/__init__.py
metrics/agent_metrics.py
metrics/content_scoring.py
metrics/ragas_metrics.py
metrics/registry.py
[optimizers]
optimizers/__init__.py
optimizers/base.py
optimizers/batch_optimizers.py
optimizers/bayesian.py
optimizers/benchmarking.py
optimizers/cloud_optimizer.py
optimizers/grid.py
optimizers/interactive_optimizer.py
optimizers/optuna_adapter.py
optimizers/optuna_checkpoint.py
optimizers/optuna_coordinator.py
optimizers/optuna_optimizer.py
optimizers/optuna_utils.py
optimizers/pruners.py
optimizers/random.py
optimizers/registry.py
optimizers/remote.py
optimizers/remote_services.py
optimizers/results.py
optimizers/service_registry.py
[plugins]
plugins/__init__.py
plugins/registry.py
[providers]
providers/__init__.py
providers/validation.py
[reporting]
reporting/__init__.py
reporting/example_map.py
[security]
security/__init__.py
security/audit.py
security/auth/__init__.py
security/auth/helpers.py
security/auth/mfa.py
security/auth/models.py
security/auth/oidc.py
security/auth/saml.py
security/auth/sms.py
security/auth/totp.py
security/config.py
security/credentials.py
security/crypto_utils.py
security/deployment.py
security/encryption.py
security/enterprise.py
security/headers.py
security/input_validation.py
security/jwt_validator.py
security/rate_limiter.py
security/session_manager.py
security/tenant.py
[storage]
storage/__init__.py
storage/local_storage.py
[telemetry]
telemetry/__init__.py
telemetry/optuna_metrics.py
[tuned_variables]
tuned_variables/__init__.py
tuned_variables/dataflow_strategy.py
tuned_variables/detection_strategies.py
tuned_variables/detection_types.py
tuned_variables/detector.py
tuned_variables/discovery.py
tuned_variables/py.typed
[tvl]
tvl/__init__.py
tvl/__main__.py
tvl/models.py
tvl/objectives.py
tvl/options.py
tvl/promotion_gate.py
tvl/registry.py
tvl/spec_loader.py
tvl/spec_validator.py
tvl/statistics.py
[utils]
utils/__init__.py
utils/batch_optimizer_utils.py
utils/batch_processing.py
utils/callbacks.py
utils/constraints.py
utils/cost_calculator.py
utils/diagnostics.py
utils/env_config.py
utils/error_handler.py
utils/example_id.py
utils/exceptions.py
utils/file_versioning.py
utils/function_identity.py
utils/hashing.py
utils/importance.py
utils/incentives.py
utils/insights.py
utils/langchain_interceptor.py
utils/local_analytics.py
utils/logging.py
utils/multi_objective.py
utils/numpy_compat.py
utils/objectives.py
utils/optimization_analyzer.py
utils/optimization_logger.py
utils/persistence.py
utils/reproducibility.py
utils/retry.py
utils/secure_path.py
utils/user_prompts.py
utils/validation.py
[visualization]
visualization/__init__.py
visualization/plots.py
[wrapper]
wrapper/__init__.py
wrapper/errors.py
wrapper/server.py
wrapper/service.py
```
