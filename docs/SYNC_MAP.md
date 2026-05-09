# Skill-to-SDK Sync Map

When SDK source files change, review the corresponding skills for accuracy.

| Skill | SDK Source Dependencies |
|-------|----------------------|
| `traigent-quickstart` | `docs/getting-started/*`, `pyproject.toml`, `examples/quickstart/` |
| `traigent-configuration-space` | `traigent/api/parameter_ranges.py`, `traigent/api/constraint_builders.py`, `docs/user-guide/tuned_variables.md` |
| `traigent-decorator-setup` | `traigent/api/decorators.py` (EvaluationOptions, InjectionOptions, ExecutionOptions), `docs/user-guide/injection_modes.md` |
| `traigent-run-optimization` | `traigent/core/optimized_function.py` (optimize/optimize_sync), `traigent/core/orchestrator.py`, `traigent/config/parallel.py` |
| `traigent-analyze-results` | `traigent/api/types.py` (OptimizationResult, TrialResult, StopReason), `traigent/core/optimized_function.py` (apply_best_config) |
| `traigent-integrations` | `traigent/integrations/*`, `docs/architecture/integrations-inventory.md` |
| `traigent-debugging` | `traigent/utils/exceptions.py`, `traigent/utils/env_config.py` |
