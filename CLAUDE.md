# Traigent SDK Claude Instructions

## 🧠 Project Context
Traigent is a Python SDK for zero-code LLM optimization using decorators (`@traigent.optimize`). It intercepts calls, injects parameters, and optimizes against objectives (accuracy, cost, latency).

## 🏗️ Architecture & Patterns
- **Entry Point**: `traigent/api/decorators.py` (`@optimize`) wraps user functions in `OptimizedFunction` (`traigent/core/optimized_function.py`).
- **Orchestration**: `OptimizationOrchestrator` (`traigent/core/orchestrator.py`) manages the optimization loop and backend sync.
- **Integrations**: Adapters in `traigent/integrations/` (e.g., `LangChainAdapter`) handle framework-specific logic. **Do not inline framework logic**; use adapters.
- **Execution Modes**: `edge_analytics` (local + analytics), `mock` (testing), `cloud` (production).
- **Security**: `traigent/security/` handles JWT validation and encryption. **Never bypass auth in production code.**

## 📋 Claude Plans & IP Protection (CRITICAL)

**Implementation plans contain proprietary IP and MUST NOT be committed to version control.**

### Storage Location
- All Claude Code implementation plans are stored in `~/.claude/plans/` (user home directory)
- This directory is already excluded by `.gitignore` (line 253: `.claude/`)
- **NEVER** create plan files inside the repository directory structure
- **NEVER** commit plan files or reference them in committed code

### Plan Naming Convention
- Plans use auto-generated names like `async-sprouting-summit.md`, `bold-crimson-valley.md`
- Store plan path references only in conversation context, never in files

### When Creating Plans
When you create implementation plans using EnterPlanMode/ExitPlanMode:
1. Plans are automatically written to `~/.claude/plans/`
2. Verify the path starts with `/home/` or `~/` (outside repo)
3. If accidentally created in repo, immediately move to `~/.claude/plans/`

## ⚙️ Concurrency & Shared Resources (CRITICAL)

**Traigent uses dependency injection, multi-threading, and shared resource guardrails. ALL changes must consider these patterns.**

### Dependency Injection Pattern
- `OptimizedFunction` receives injected dependencies: `TraigentConfig`, `BudgetGuardrail`, `CloudClient`, etc.
- **NEVER** instantiate shared resources directly inside functions
- **ALWAYS** accept dependencies as parameters or use factory patterns
- Example: `OptimizationOrchestrator` injects budget guardrail into trial executor

```python
# ✅ CORRECT - Dependency injection
def run_trial(config: dict, budget_guardrail: BudgetGuardrail) -> TrialResult:
    budget_guardrail.check_before_trial()
    # ... trial logic

# ❌ WRONG - Direct instantiation creates multiple instances
def run_trial(config: dict) -> TrialResult:
    budget_guardrail = BudgetGuardrail()  # BREAKS shared state!
    budget_guardrail.check_before_trial()
```

### Multi-Threading Considerations
- Parallel trial execution uses `ThreadPoolExecutor` (see `traigent/core/orchestrator.py`)
- **All shared resources MUST be thread-safe** (use locks, thread-local storage, or immutable data)
- Budget guardrails use threading locks to prevent race conditions
- Example thread-safe patterns:
  - `BudgetGuardrail` uses `threading.Lock()` for spend tracking
  - Trial results are collected in thread-safe queues
  - Configuration sampling uses immutable config snapshots

```python
# ✅ CORRECT - Thread-safe shared state
class BudgetGuardrail:
    def __init__(self):
        self._lock = threading.Lock()
        self._spent = 0.0

    def add_cost(self, cost: float):
        with self._lock:
            self._spent += cost

# ❌ WRONG - Race condition
class BudgetGuardrail:
    def __init__(self):
        self._spent = 0.0

    def add_cost(self, cost: float):
        self._spent += cost  # Multiple threads can corrupt this!
```

### Budget Guardrail Integration
When adding features that consume resources (LLM calls, API requests):
1. **Always check budget before execution**: Call `budget_guardrail.check_before_trial()` or similar
2. **Track costs after execution**: Call `budget_guardrail.add_cost(trial_cost)`
3. **Handle budget exceeded**: Raise `BudgetExceededError` and stop gracefully
4. **Thread-safety**: Never bypass locks or create duplicate guardrail instances

Key files:
- `traigent/core/budget.py` - Budget guardrail implementation
- `traigent/core/orchestrator.py` - Budget integration in optimization loop
- `traigent/core/trial_lifecycle.py` - Cost tracking per trial

### Resource Injection Checklist
Before implementing ANY new feature that:
- Makes LLM/API calls
- Executes user functions
- Runs in parallel trials
- Shares state across trials

**Ask yourself:**
1. ✅ Does this respect the injected budget guardrail?
2. ✅ Is this thread-safe for parallel execution?
3. ✅ Am I using dependency injection (not creating new instances)?
4. ✅ Do I handle budget exceeded gracefully?

## 🔄 Cross-Project Communication: TraigentSchema DTOs (CRITICAL)

**We own the SDK (Traigent), Frontend (FE), and Backend (BE). Use TraigentSchema DTOs for ALL cross-project data exchange.**

### TraigentSchema DTOs
Located in `traigent/cloud/dtos.py`, these DTOs define the canonical data contracts between SDK ↔ Backend ↔ Frontend:

**Core DTOs:**
- `ExperimentDTO` - Experiment definition and configuration
- `ExperimentRunDTO` - Optimization run lifecycle and results
- `ConfigurationRunDTO` - Individual trial/configuration results
- `InfrastructureDTO` - Compute infrastructure metadata
- `ConfigurationsDTO` - Experiment configurations wrapper
- `MeasuresDict` - Type-safe measures dictionary with validation

**Why Use DTOs:**
1. **Type Safety**: Enforced validation (Python identifiers, numeric types, max keys)
2. **Schema Compliance**: Validates against `optigen_schemas` (if installed)
3. **Privacy Modes**: Built-in support for Edge Analytics (privacy-preserving defaults)
4. **Backward Compatibility**: Handles schema evolution across SDK/BE/FE versions
5. **Single Source of Truth**: Same DTOs used by SDK, Backend API, and Frontend API client

### When to Use TraigentSchema DTOs

**✅ ALWAYS use DTOs for:**
- SDK → Backend API submissions (experiments, trials, results)
- Backend → Frontend responses (experiment lists, run details, analytics)
- Frontend → Backend requests (filters, updates, actions)
- Any new feature that crosses SDK/BE/FE boundaries

**❌ NEVER:**
- Serialize raw Python objects (dicts, dataclasses) without DTO conversion
- Create ad-hoc JSON schemas that bypass DTOs
- Modify DTO fields without updating all 3 projects (SDK, BE, FE)

### DTO Usage Pattern

```python
# ✅ CORRECT - Use DTOs for SDK → Backend
from traigent.cloud.dtos import ExperimentRunDTO, ConfigurationRunDTO

# Create DTO with validation
experiment_run = ExperimentRunDTO(
    id=run_id,
    experiment_id=experiment_id,
    status="running",
    summary_stats={"accuracy": 0.95},
    metadata={"dataset_size": 100}
)

# Validate against schema (if optigen_schemas installed)
experiment_run.validate()  # Raises DTOSerializationError if invalid

# Convert to dict for API submission
payload = experiment_run.to_dict()
response = await cloud_client.submit_experiment_run(payload)

# ❌ WRONG - Ad-hoc dict without validation
payload = {
    "id": run_id,
    "experiment_id": experiment_id,
    "status": "running",  # Typo: should be "not_started", "running", "completed", etc.
    "metrics": {"accuracy": 0.95},  # Wrong key: should be "summary_stats"
}
response = await cloud_client.submit_experiment_run(payload)  # May fail silently
```

### MeasuresDict Validation

The `MeasuresDict` class enforces:
- **Max 50 keys** (prevent unbounded memory)
- **Python identifier keys** (e.g., `accuracy_score`, not `accuracy-score`)
- **Numeric values only** (int, float, None) - non-numeric logged as warnings (Phase 0)

```python
from traigent.cloud.dtos import MeasuresDict

# ✅ CORRECT
measures = MeasuresDict({
    "accuracy": 0.95,
    "latency_ms": 120.5,
    "cost_usd": 0.002,
})

# ❌ WRONG - Invalid key pattern (hyphen not allowed)
measures = MeasuresDict({
    "accuracy-score": 0.95,  # Raises ValueError: must match ^[a-zA-Z_][a-zA-Z0-9_]*$
})

# ⚠️ WARNING - Non-numeric value (allowed in Phase 0, rejected in v2.0)
measures = MeasuresDict({
    "model_name": "gpt-4",  # Logs warning: use metadata instead
})
```

### Adding New Cross-Project Features

When implementing features that span SDK/BE/FE (like example scoring):

1. **Check existing DTOs first**: Can you extend `ExperimentRunDTO.metadata` or `ConfigurationRunDTO.measures`?
2. **If new DTO needed**: Create in `traigent/cloud/dtos.py` with validation
3. **Backend models**: Create SQLAlchemy models that map to DTO structure
4. **Backend API**: Use DTOs in request/response schemas (FastAPI/Pydantic)
5. **Frontend client**: Generate TypeScript types from DTO schemas
6. **Validate end-to-end**: Ensure SDK DTO → Backend model → Frontend type consistency

**Example: Adding Example Scoring (from plan)**
- SDK computes content scores → add to `ConfigurationRunDTO.measures` (via `MeasuresDict`)
- Backend stores scores → `ExampleScore` model matches DTO structure
- Backend API returns scores → serialize to DTO format
- Frontend displays scores → TypeScript types generated from DTO schemas

### DTO Files to Know

**SDK:**
- [traigent/cloud/dtos.py](traigent/cloud/dtos.py) - All DTO definitions

**Backend (TraigentBackend):**
- `src/models/` - SQLAlchemy models (map to DTOs)
- `src/routes/` - FastAPI endpoints (use Pydantic models based on DTOs)
- `src/schemas/` - Pydantic request/response schemas

**Frontend:**
- `src/types/` - TypeScript types (generated from DTOs/API schemas)
- `src/api/` - API client methods (consume DTO-based endpoints)

### Schema Validation (Optional)

Install `optigen_schemas` for strict validation:
```bash
pip install 'traigent[validation]'
```

Control strictness with environment variable:
```bash
# Strict (default): Raise exceptions on validation failure
export TRAIGENT_STRICT_VALIDATION=true

# Lenient: Log warnings only (for development)
export TRAIGENT_STRICT_VALIDATION=false
```

## 🛠️ Development Workflow
- **Setup**: `make install-dev` (installs with `[all,dev,dspy,docs]` - all optional dependencies).
- **Testing**: `TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/` (Critical: Use mock mode to avoid API costs).
  - **Markers**: `pytest -m "unit"`, `pytest -m "integration"`, `pytest -m "security"`.
- **Linting**: `make lint` (runs Ruff, MyPy, Bandit). `make format` (Black, Isort).
- **Mock Mode**: Always use `TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true` for local development and tests.

## 📐 Code Formatting & Linting (CRITICAL)

**Before committing any Python code changes, ALWAYS run:**

```bash
make format && make lint
```

Or manually:

```bash
# Format code
isort traigent/ tests/ examples/
black traigent/ tests/ examples/

# Lint code (use these exact commands - CI flags differ)
ruff check traigent/ --fix
black --check traigent/
isort --check-only traigent/
```

### Formatting Rules
- **Black**: Line length 88, Python 3.8+ target
- **isort**: Profile "black", line length 88 (must match black)
- **Ruff**: For linting only, not formatting

### ⚠️ CI vs Local Commands
The CI workflow uses different flags than local development:
- **CI**: `ruff check traigent/ --output-format=github` (for GitHub annotations)
- **Local**: `ruff check traigent/ --fix` (to auto-fix issues)

**Never use these flag combinations** (they are incompatible):
- `--output-format=github --statistics` ❌ (ruff doesn't support this)

### Pre-commit Checklist
1. Run `make format` (runs isort + black)
2. Run `make lint` (runs ruff + mypy + bandit)
3. Verify: `black --check traigent/ && isort --check-only traigent/`

## 🚨 Critical Rules
1. **Format Before Commit**: Always run `make format` before committing Python changes.
2. **Cross-Repo Policy**: Do not modify external systems. File issues for backend/upstream changes.
3. **Async-First**: Cloud operations (`traigent/cloud/`) must be async. Avoid blocking sync wrappers.
4. **Secrets**: NEVER hardcode secrets. Use `traigent.utils.env_config` or `os.environ`.
5. **Type Hints**: Required for all public APIs. Run `make lint` to verify.
6. **Dependencies**: Add new deps to `pyproject.toml` under the appropriate optional group (e.g., `integrations`).
7. **SonarQube Compliance**: Avoid patterns that trigger SonarQube S2583 (see below).

## ⚠️ SonarQube S2583: Conditions Always True/False

SonarQube uses interprocedural analysis and flags conditions it determines are always true/false. **This is a CI blocker.**

### Patterns to AVOID

```python
# BAD - Combined condition with ternary re-checking same condition
if p <= 0 or p >= 1:
    return 0.0 if p <= 0 else 1.0  # S2583: "p <= 0" always true here

# BAD - Even separate ifs can be flagged if SonarQube determines callers never hit them
if p <= 0:
    return 0.0
if p >= 1:  # S2583: If all callers pass p in (0,1), this is "always false"
    return 1.0
```

### Correct Approach

For **internal helper functions** where all callers guarantee valid input:

```python
def _internal_func(p: float) -> float:
    """Args: p: Value strictly in (0, 1). Callers must validate."""
    # No boundary checks - documented precondition, callers guarantee validity
    ...
```

For **public API functions** that accept untrusted input, keep validation but use explicit ValueError:

```python
def public_func(p: float) -> float:
    """Args: p: Probability in [0, 1]."""
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    ...
```

## 🧩 Common Tasks
- **New Integration**: Create adapter in `traigent/integrations/`, register in `traigent/integrations/registry.py`.
- **New Metric**: Implement `BaseMetric` in `traigent/metrics/`, register in `metric_registry.py`.
- **Debug**: Set `TRAIGENT_LOG_LEVEL=DEBUG`.

## 🧪 Test Quality Guidelines (CRITICAL)

When writing tests, avoid these anti-patterns identified in our meta-analysis:

### Anti-Patterns to Avoid

1. **IT-VRO (Validator Reliance Only)**: Never rely solely on a validator abstraction without explicit assertions.

   ```python
   # BAD - relies only on validator
   validation = result_validator(scenario, result)
   assert validation.passed

   # GOOD - explicit assertions THEN validator
   assert len(result.trials) >= 1, "Should have trials"
   for trial in result.trials:
       assert trial.config, "Trial should have config"
   validation = result_validator(scenario, result)
   assert validation.passed
   ```

2. **IT-VTA (Vacuous Truth Assertions)**: Avoid assertions that are always true.

   ```python
   # BAD - always true
   assert len(result.trials) >= 0
   assert not isinstance(result, Exception)

   # GOOD - meaningful bounds
   assert len(result.trials) >= expected_min
   assert result.stop_reason == "max_trials"
   ```

3. **IT-CBM (Condition-Behavior Mismatch)**: Ensure test setup actually triggers the behavior under test.

   ```python
   # BAD - config space (2 items) smaller than max_trials (10)
   # max_trials never triggers; config_exhaustion triggers first
   scenario = TestScenario(max_trials=10, config_space={"model": ["a", "b"]})

   # GOOD - config space larger than max_trials
   scenario = TestScenario(max_trials=3, config_space={"model": ["a", "b", "c", "d", "e"]})
   ```

### Test Structure Best Practices

1. **Explicit over Implicit**: Always add explicit assertions for the specific behavior under test.
2. **Specific Values**: Assert exact expected values, not just types.
3. **Document Validator Contract**: Know what validators check and don't check.
4. **Unit Tests**: Prefer direct function calls with specific assertions over complex scenario runners.

### Reference
See `tests/optimizer_validation/META_ANALYSIS_REPORT.md` and `tests/optimizer_validation/INDEPENDENT_META_ANALYSIS.md` for detailed analysis of test quality issues.

## 📊 Statistical Rigor for Experiment Results

**NEVER claim a hypothesis is "confirmed" or "proven" without proper statistical validation.**

When reporting experimental results:

### 1. Single-run experiments are INCONCLUSIVE by default
- One random seed = anecdotal evidence, not proof
- Always label single-run results as "preliminary" or "suggestive"

### 2. Required for claiming statistical significance
- Multiple independent runs (minimum 5, preferably 10+)
- Appropriate statistical test (t-test, bootstrap CI, McNemar's, etc.)
- Reported p-value < 0.05 (or stated significance threshold)
- Effect size calculation (Cohen's d or equivalent)

### 3. Language guidelines
- ❌ "The hypothesis is confirmed"
- ❌ "Results prove that X outperforms Y"
- ❌ "We demonstrate that..."
- ✅ "Results suggest..." (single run)
- ✅ "Preliminary evidence indicates..." (single run)
- ✅ "With p < 0.05, we reject the null hypothesis..." (validated)

### 4. Always acknowledge limitations
- Sample size and its implications
- Number of random seeds used
- Whether null hypothesis testing was performed
- Potential confounds

## 🤖 AutoCoder Operational Rules

### 1. "RUN" means RUN - No modifications
- "Commence", "Run it", "Execute", "Go ahead" = RUN THE SCRIPT AS-IS
- Do NOT make any code changes before, during, or after execution
- Do NOT ask "Want me to fix this?" or present accept/reject prompts
- **After the run**: Inform about issues/improvements, but WAIT for user to explicitly initiate changes
- Key shift: **I inform, you initiate.** Modifications require explicit user initiation.

### 2. No hardcoded values in print statements
- All printed configuration info MUST come from actual variables
- BAD: `print("Models: gpt-3.5-turbo, gpt-4o-mini")`
- GOOD: `print(f"Models: {', '.join(config_space['model'])}")`

### 3. Don't change approved formats
- Once a format/output is tested and approved, DO NOT modify it
- If you need to add features, ADD NEW functions/files
- Never silently change existing approved behavior

### 4. Ask before expensive operations
- Before running anything that costs money (API calls, cloud resources)
- Confirm the exact code that will run
- Confirm cost limits are set correctly
- Do NOT make last-minute code changes before expensive runs

### 5. Single source of truth
- Configuration values should be defined ONCE
- All references (prints, logs, docs) must read from that source
- Never duplicate configuration values as strings elsewhere

### 6. Test runs must be representative
- Test runs should use the same code path as production runs
- If test passes, production should work identically
- Do NOT add features between test and production runs

### 7. Documentation must reference source of truth
- Docstrings, comments, README files must NOT hardcode config values
- BAD: "Supports models: gpt-3.5-turbo, gpt-4o" (hardcoded in docstring)
- GOOD: "Supports models: see SUPPORTED_MODELS constant" (references source)

### 8. Do NOT modify Traigent core code
- Never modify files under `traigent/` directory (the SDK core)
- OK to modify: `examples/`, `tests/`, `docs/`, `scripts/`, `paper_experiments/`
- If Traigent has a bug, work around it in customer code
- Known workarounds:
  - `cost_limit=`/`cost_approved=` decorator params don't work → use env vars instead
  - `evaluator=` param doesn't work → use `metric_functions=` instead

### 9. Mock mode status
- `.env` has `TRAIGENT_MOCK_MODE=false` (real scoring for RAG examples)
- If VS Code test discovery breaks, set it back to true
- Prompt the user: "Mock mode is currently OFF. Want me to enable it for tests?"

### 10. Never push to main
- NEVER run `git push` to the main branch
- Always create a feature/dev branch for changes
- Use pull requests to merge into main
- If on main, checkout a new branch before committing