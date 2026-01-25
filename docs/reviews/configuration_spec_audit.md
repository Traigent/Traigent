# Traigent Configuration Specification Review

## Scope
- Code paths: `traigent/api/decorators.py`, `traigent/core/optimized_function.py`, `traigent/core/objectives.py`, `traigent/api/parameter_ranges.py`, `traigent/api/config_space.py`, `traigent/api/constraints.py`, `traigent/tvl/spec_loader.py`, `traigent/tvl/models.py`, `traigent/tvl/options.py`, `traigent/evaluators/base.py`, `traigent/evaluators/dataset_registry.py`, `traigent/tuned_variables/discovery.py`.
- Docs: `docs/api-reference/decorator-reference.md`, `docs/api-reference/complete-function-specification.md`, `docs/guides/evaluation.md`, `docs/tvl/CONSTRAINT_EXPRESSIONS.md`, `docs/tvl/tvl-website/client/public/examples/ch2_hello_tvl.tvl.yml`, `docs/planned_features/TVL_UNIFICATION_DESIGN_V2.md`.
- DVC/config artifacts: `params.yaml`, `scripts/auto_tune/*`.

## Current Configuration Surfaces (What Exists)

### Objectives
- Decorator: `@traigent.optimize(objectives=...)` and legacy/runtime overrides in `traigent/api/decorators.py:1026`.
- Runtime override: `.optimize(objectives=...)` in `traigent/core/optimized_function.py:919`.
- Global default: `traigent.configure(objectives=...)` in `traigent/api/functions.py:31`.
- TVL spec: `objectives` parsed into `ObjectiveSchema` in `traigent/tvl/spec_loader.py:596`.

### Tuned Variables / Configuration Space (TVARs)
- `configuration_space` dict with lists/tuples or `Range/Choices/...` via `traigent/api/parameter_ranges.py:935`.
- `ConfigSpace` class (tvars + constraints) via `traigent/api/config_space.py:57`.
- TVL 0.9 `tvars` (or legacy `configuration_space`) parsed by `traigent/tvl/spec_loader.py:588`.
- Tuned variable discovery helpers (callable discovery) in `traigent/tuned_variables/discovery.py:1` (not wired into optimize flow).

### Constraints
- Decorator accepts `Constraint`/`BoolExpr`/callables and normalizes them in `traigent/api/decorators.py:1451`.
- `ConfigSpace` constraints and DSL in `traigent/api/constraints.py:493`.
- TVL `constraints` compiled into callables in `traigent/tvl/spec_loader.py:608`.
- Legacy constraint system (ConstraintManager, ParameterRangeConstraint) in `traigent/utils/constraints.py:1` (exported but not used in optimize flow).

### Evaluation & Evaluation Sets
- `eval_dataset`, `custom_evaluator`, `scoring_function`, `metric_functions` bundled in `EvaluationOptions` in `traigent/api/decorators.py:84`.
- Dataset loading, registry, and root enforcement in `traigent/evaluators/base.py:109` and `traigent/evaluators/dataset_registry.py:1`.
- TVL `evaluation_set` parsed in `traigent/tvl/spec_loader.py:584` and surfaced via `traigent/tvl/models.py:470`.

### DVC / params.yaml (Planned + Scripts)
- TVL↔DVC architecture is documented but not implemented in the SDK: `docs/planned_features/TVL_UNIFICATION_DESIGN_V2.md:13`.
- `params.yaml` and `scripts/auto_tune/*` describe a DVC-driven pipeline that is separate from TVL + decorator paths.

## Findings (Issues / Gaps / Inconsistencies)

- High: TVL runtime options ignore all `apply_*` flags and never apply `evaluation_set` in `.optimize(...)`. `traigent/core/optimized_function.py:846` + `traigent/tvl/options.py:15` vs the decorator path in `traigent/api/decorators.py:923`.
- High: Inline ParameterRange auto-naming does not propagate to constraints; missing `var_names` means conditions silently pass. `traigent/api/parameter_ranges.py:935` + `traigent/api/constraints.py:265`.
- Medium: `evaluation_set.seed` is parsed but not used anywhere; only the dataset string is applied. `traigent/tvl/models.py:470` + `traigent/api/decorators.py:933`.
- Medium: TVL docs/examples use `evaluation.workloads` or `evaluation_set.uri`, but the loader only reads `evaluation_set.dataset`. `docs/tvl/tvl-website/client/public/examples/ch2_hello_tvl.tvl.yml:40` + `docs/api-reference/decorator-reference.md:323` vs `traigent/tvl/spec_loader.py:584`.
- Medium: Configuration validation/builders are duplicated and drifted (`traigent/api/parameter_validator.py`, `traigent/api/config_builder.py`, `traigent/core/config_builder.py`), and injection mode validation disagrees (`decorator` vs `attribute/seamless`). `traigent/core/config_builder.py:174` vs `traigent/config/types.py:62`.
- Medium: Default execution mode mismatch (`edge_analytics` in decorator vs `cloud` in `OptimizedFunction`). `traigent/api/decorators.py:263` vs `traigent/core/optimized_function.py:313`.
- Medium: Docs show an `ObjectiveSchema(definitions=...)` constructor that does not exist. `docs/api-reference/decorator-reference.md:83` vs `traigent/core/objectives.py:188`.
- Medium: Evaluation guide uses `evaluator=` with `(output, expected)` signature, but SDK expects `custom_evaluator(func, config, example)` or `scoring_function`/`metric_functions`. `docs/guides/evaluation.md:53` vs `traigent/api/decorators.py:179`.
- Low: Validator accepts list of `Dataset` objects, but loader only accepts list of paths. `traigent/api/parameter_validator.py:25` vs `traigent/core/optimized_function.py:1915`.
- Low: `ConfigSpace` type alias in `traigent/api/types.py` conflicts with the `ConfigSpace` class and docs. `traigent/api/types.py:1532`.
- Low: `reps_per_trial` and `reps_aggregation` are documented as available but are hard-gated to enterprise. `docs/api-reference/complete-function-specification.md:90` vs `traigent/api/decorators.py:699`.
- Low: Dataset registry/root behavior (`TRAIGENT_DATASET_ROOT`, `TRAIGENT_DATASET_REGISTRY`) is implemented but not documented. `traigent/evaluators/base.py:109`, `traigent/evaluators/dataset_registry.py:1`.
- Low: Tuned-variable discovery helpers exist but are not integrated into decorator/TVL flows. `traigent/tuned_variables/discovery.py:1`.

## Opportunities / Improvements
- Unify config parsing/validation into a single pipeline shared by decorator and runtime `optimize()` (reduce drift between `api/` and `core/` builders).
- Honor `TVLOptions.apply_*` flags and apply `evaluation_set` consistently in runtime optimization.
- Fix constraint auto-naming by carrying a `var_names` map from normalized inline params or by mutating ParameterRange names before constraint normalization.
- Align TVL schema and examples: decide between `evaluation_set` vs `evaluation.workloads` and `dataset` vs `uri`, then update both parser and docs.
- Clarify DVC/params.yaml integration status (roadmap vs implemented) and link to the active path users should follow today.
- Document dataset registry/root configuration and evaluation-set semantics (including the seed behavior).
