# Feature Matrix for Plugin Architecture

Comprehensive coverage of Traigent capabilities to guide splitting the package into a core plus plugins.

| Area | Capabilities / Options | Config / Entry Points | Dependencies / Extras | Plugin Scope Note |
| --- | --- | --- | --- | --- |
| Injection and config delivery | Context, parameter, attribute, seamless injection; default_config; auto framework overrides and explicit framework_targets; allow_parallel_attribute opt-in | optimize decorator via `injection.*`, `default_config`, `injection_mode`, `config_param` | Core | Keep in base |
| Execution modes and storage | Edge analytics, hybrid/privacy, standard, cloud; local_storage_path, minimal_logging, privacy_enabled; repetitions aggregation | `execution.*`, env `TRAIGENT_RESULTS_FOLDER`, `TRAIGENT_MOCK_LLM` | Core | Base |
| Optimization algorithms | Random, Grid, Bayesian (scikit-optimize), Optuna (TPE, Random, CMAES, Grid, NSGAII), Parallel/Adaptive/Interactive, Remote/Cloud | Runtime `algorithm`, `max_trials`, `timeout`, `cache_policy` | bayesian extra, core optuna | Base: optuna/random/grid; plugin candidates: bayesian, parallel batch, remote/cloud, interactive |
| Objectives and multi-objective | ObjectiveSchema with weights/orientation, banded objectives, aggregation modes (sum, harmonic, chebyshev); multi-objective via Optuna NSGA-II and MultiObjectiveBatchOptimizer | `objectives`, `ObjectiveSchema`, TVL bands | Optuna for NSGA-II | Multi-objective/banded to plugins |
| Constraints, budgets, stopping | Inline constraints (pre/post eval), ConfigSpace constraints, TVL validation; stop conditions (plateau window/epsilon); budget_limit/budget_metric/include_pruned; cost_limit/cost_approved; sample budgets max_total_examples/samples_include_pruned; max_trials | Decorator args or runtime overrides; env `TRAIGENT_RUN_COST_LIMIT`, `TRAIGENT_COST_APPROVED` | Core | Constraint validation and budgets can move to plugins |
| Parallelism and concurrency | ParallelConfig (auto/sequential/parallel, trial_concurrency, example_concurrency, thread_workers), ParallelExecutionManager, legacy batch size bridging | `execution.parallel_config`, legacy `parallel_trials` and `batch_size` | Core | Parallel modes to plugins |
| Invokers | LocalInvoker, BatchInvoker, StreamingInvoker (streaming, batch, retry, timeout) | Orchestrator selection; streaming paths use StreamingInvoker | Core | Batch/streaming could ship with parallel plugin |
| Evaluation and metrics | EvaluationOptions (dataset path/list/Dataset, custom_evaluator, scoring_function, metric_functions); evaluators: LocalEvaluator (parallel, RAGAS, custom metrics, progress) and SimpleScoring; reps_per_trial and reps_aggregation | `evaluation.*`, `execution.reps_per_trial`, `reps_aggregation` | ragas, pandas (examples/test extras) | Advanced eval (RAGAS/parallel) to plugins; keep simple scoring in base |
| Mock mode | Mocked responses/metrics, optional evaluator override | `mock.*`, env `TRAIGENT_MOCK_LLM` | Core | Base |
| Cache and baseline reuse | Cache policy allow_repeats and prefer_new (dedup local storage), default_config baseline trial | `cache_policy`, `local_storage_path` | Core | Base |
| Integrations: LLM providers | Plugins for OpenAI/Azure, Anthropic, Gemini, Bedrock, Cohere, Mistral, HuggingFace; parameter mapping/validation/version checks | Auto via `auto_override_frameworks` or `framework_targets`; plugin configs via YAML/JSON | integrations extra | Strong plugin candidates per provider |
| Integrations: frameworks and vector stores | LangChain and LlamaIndex overrides, LiteLLM pattern support, vector stores (Chroma, Pinecone, Weaviate), Bedrock client; observability plugins (MLflow, W&B) | Same injection config; observability via integration settings | integrations extra | Plugin bundles |
| Agents and adapters | Agent config mapper/executor/spec generator; execution_adapter | API usage | Core | Stay base or move with integrations |
| TVL specification layer | tvl_spec, tvl_environment, TVLOptions apply_* toggles for config/objectives/constraints/budget, banded objectives, promotion gate | Decorator args; CLI `traigent optimize --tvl-spec ... --tvl-environment ...` | Core | Plugin candidate |
| Analytics and intelligence | Cost optimization, anomaly detection, scheduling, predictive analytics, meta-learning, trend analysis; visualization plots | Feature-driven; CLI `traigent plot` | analytics and visualization extras | Plugin bundle |
| Observability and tracking | MLflow/W&B logging, local analytics submission | Integration configs | integrations extra | Plugin |
| CI and policy hooks | Hook installer/validator, cost/performance/model allow/block constraints from traigent.yml | CLI `traigent hooks ...`; config in traigent.yml | Core deps | Plugin candidate |
| Security | JWT validation, encryption, headers, rate limiter, tenant/session management | Used in cloud/enterprise paths | security extra | Plugin/enterprise |
| Cloud/backend | Backend client/session sync, remote optimizers, cloud execution modes | `execution.execution_mode`, backend env vars | Core deps | Plugin/enterprise |
| CLI and developer tools | traigent optimize/results/plot/auth/hooks; persistence manager saves runs | CLI flags; `get_available_strategies` | Core | Base (could split UI commands later) |
| UI and playground | Streamlit control center and demos | Run via `streamlit run playground/traigent_control_center.py` | playground extra | Plugin/UI bundle |
| Storage and persistence | Local results (.traigent_local), locking, resume, cost logs | `execution.local_storage_path`, CLI results | Core | Base |

Sources reviewed: README, docs/feature_matrices/*, traigent/api/decorators.py, traigent/config/types.py, traigent/core/orchestrator.py, pyproject.toml, integrations inventory, hooks, integrations modules.
