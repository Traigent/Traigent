# Traigent Optimizer Validation Test Suite

A comprehensive test infrastructure for validating the Traigent optimizer decorator across all configuration combinations, ensuring correctness of the `OptimizationResult` object, and testing failure scenarios.

## Quick Start

```bash
# Run all validation tests (no tracing)
make test-validation

# Run with tracing (requires Docker)
make jaeger-start
make test-validation-traced
# View traces at http://localhost:16686
```

## Directory Structure

```
tests/optimizer_validation/
├── README.md                    # This file
├── conftest.py                  # Shared fixtures and scenario runner
├── data/                        # Test datasets
│   ├── simple.jsonl
│   ├── sentiment.jsonl
│   └── malformed.jsonl
├── dimensions/                  # Single-dimension tests
│   ├── test_injection_modes.py  # context, parameter, attribute, seamless
│   ├── test_execution_modes.py  # edge_analytics, privacy, hybrid, etc.
│   ├── test_objectives.py       # Single, multi, weighted objectives
│   ├── test_config_spaces.py    # Categorical, continuous, mixed
│   ├── test_constraints.py      # Config-only, config+metrics, compound
│   └── test_evaluators.py       # Default, custom, scoring functions
├── interactions/                # Cross-dimension tests
│   └── test_key_combinations.py # Strategic sampling of key combos
├── failures/                    # Failure scenario tests
│   ├── test_function_bugs.py    # Function raises, returns None, etc.
│   ├── test_evaluator_bugs.py   # Evaluator failures
│   ├── test_dataset_issues.py   # Empty, malformed datasets
│   └── test_invocation_failures.py  # Invalid config, objectives
├── specs/                       # Test specifications
│   ├── scenario.py              # TestScenario dataclass
│   ├── builders.py              # Scenario builder functions
│   ├── validators.py            # Result validation helpers
│   └── trace_expectations.py    # Trace assertion helpers
└── tracing/                     # OpenTelemetry tracing infrastructure
    ├── tracer.py                # Tracer configuration
    ├── exporters.py             # OTLP, in-memory, file exporters
    ├── capture.py               # TraceCapture fixture
    ├── analyzer.py              # TraceAnalyzer for assertions
    └── invariants.py            # Global trace invariants
```

## Running Tests

### Without Tracing (Fast)

```bash
# All validation tests
make test-validation

# Specific dimension
TRAIGENT_MOCK_MODE=true pytest tests/optimizer_validation/dimensions/test_injection_modes.py -v

# Only failure tests
make test-validation-failures

# Only unit tests
make test-validation-unit
```

### With Tracing (Jaeger)

```bash
# 1. Start Jaeger (one-time)
make jaeger-start

# 2. Run tests with tracing
make test-validation-traced

# 3. View traces
open http://localhost:16686

# 4. Stop Jaeger when done
make jaeger-stop
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIGENT_MOCK_MODE` | Enable mock mode (no API calls) | `true` (auto-set) |
| `TRAIGENT_TRACE_ENABLED` | Enable OpenTelemetry tracing | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Jaeger/OTLP endpoint | - |

## Writing Tests

### Basic Test with Scenario

```python
import pytest
from tests.optimizer_validation.specs import basic_scenario

@pytest.mark.unit
@pytest.mark.asyncio
async def test_my_feature(scenario_runner, result_validator):
    # Define scenario
    scenario = basic_scenario(
        name="my_test",
        injection_mode="context",
        max_trials=3,
    )

    # Run optimization
    func, result = await scenario_runner(scenario)

    # Validate result
    assert not isinstance(result, Exception)
    validation = result_validator(scenario, result)
    assert validation.passed, validation.summary()
```

### Test with Custom Config Space

```python
from tests.optimizer_validation.specs import TestScenario, ExpectedResult

scenario = TestScenario(
    name="custom_config",
    config_space={
        "model": ["gpt-3.5-turbo", "gpt-4"],
        "temperature": (0.0, 1.0),  # Continuous range
        "max_tokens": [100, 500, 1000],
    },
    objectives=["accuracy", "cost"],
    max_trials=5,
)
```

### Test with Constraints

```python
from tests.optimizer_validation.specs import TestScenario, ConstraintSpec

def cost_constraint(config, metrics=None):
    """Reject configs with high cost."""
    if metrics and metrics.get("cost", 0) > 0.5:
        return False
    return True

scenario = TestScenario(
    name="constrained",
    constraints=[
        ConstraintSpec(
            name="cost_limit",
            constraint_fn=cost_constraint,
            requires_metrics=True,
        )
    ],
    # ...
)
```

### Test with Trace Validation

```python
from tests.optimizer_validation.specs import TraceExpectations

@pytest.mark.asyncio
async def test_with_tracing(traced_scenario_runner, trace_validator):
    scenario = basic_scenario(
        name="traced_test",
        trace_expectations=TraceExpectations(
            min_trial_spans=2,
            required_spans=["optimization_session", "trial_execution"],
        ),
    )

    func, result, trace = await traced_scenario_runner(scenario)

    # Validate trace
    trace_result = trace_validator(scenario, trace, result)
    assert trace_result.passed, trace_result.errors
```

## Viewing Traces in Jaeger

### Starting Jaeger

```bash
make jaeger-start
# or manually:
docker run -d --name traigent-jaeger \
    -p 16686:16686 \
    -p 4317:4317 \
    -p 4318:4318 \
    jaegertracing/all-in-one:latest
```

### Jaeger UI

1. Open http://localhost:16686
2. Select **Service**: `traigent-optimizer`
3. Click **Find Traces**
4. Click on any trace to see the span hierarchy

### Trace Naming

Each trace now includes informative test metadata automatically:

**Operation Name Format:**

```text
optimization: <test_name>
```

For example:

- `optimization: test_injection_mode_basic[context]`
- `optimization: test_multi_objective_weighted`
- `optimization: test_constraint_enforcement`

This makes it easy to find traces for specific tests in the Jaeger UI.

**Test Metadata Attributes:**

Each optimization span includes these test-related attributes:

| Attribute | Description | Example |
| --------- | ----------- | ------- |
| `test.name` | Full test name with parameters | `test_injection_mode_basic[context]` |
| `test.description` | Test docstring (first line) | `Test each injection mode works...` |
| `test.module` | Python module path | `tests.optimizer_validation.dimensions.test_injection_modes` |
| `test.class` | Test class name (if applicable) | `TestInjectionModeMatrix` |
| `test.param.*` | Parametrized test values | `test.param.injection_mode: context` |

### Trace Hierarchy

```text
optimization: test_injection_mode_basic[context] (root)
├── Attributes:
│   ├── test.name: "test_injection_mode_basic[context]"
│   ├── test.description: "Test each injection mode works..."
│   ├── test.module: "tests.optimizer_validation.dimensions.test_injection_modes"
│   ├── test.class: "TestInjectionModeMatrix"
│   ├── test.param.injection_mode: "context"
│   ├── traigent.function_name: "test_func"
│   ├── traigent.max_trials: 3
│   ├── traigent.objectives: "accuracy,cost"
│   ├── optimization.trial_count: 3
│   ├── optimization.best_score: 0.95
│   └── optimization.stop_reason: "max_trials_reached"
│
├── trial_execution
│   ├── Attributes:
│   │   ├── trial.id: "abc123"
│   │   ├── trial.number: 1
│   │   ├── trial.config: {"model": "gpt-4", "temperature": 0.7}
│   │   ├── trial.status: "completed"
│   │   └── trial.metric.accuracy: 0.92
│   │
│   ├── example_evaluation
│   │   ├── example.id: "example_0"
│   │   ├── example.index: 0
│   │   ├── example.input: {"text": "Hello world"}
│   │   ├── example.expected_output: "Greeting response"
│   │   ├── example.actual_output: "Hello! How can I help?"
│   │   ├── example.success: true
│   │   ├── example.execution_time_ms: 150.5
│   │   └── example.metric.accuracy: 0.95
│   │
│   ├── example_evaluation
│   │   └── ...
│   └── ...
│
├── trial_execution
│   └── ...
└── ...
```

### Useful Jaeger Queries

- **Find slow trials**: Sort by duration, look for long `trial_execution` spans
- **Find failures**: Filter by `trial.status = "failed"` or `example.success = false`
- **Compare configs**: Look at `trial.config` across different trials
- **Analyze metrics**: Check `example.metric.*` attributes

## Available Fixtures

### `scenario_runner`

Executes a `TestScenario` and returns `(decorated_function, result_or_exception)`.

```python
async def test_example(scenario_runner):
    scenario = basic_scenario(name="test")
    func, result = await scenario_runner(scenario)
```

### `result_validator`

Validates an `OptimizationResult` against scenario expectations.

```python
def test_example(result_validator):
    validation = result_validator(scenario, result)
    assert validation.passed, validation.summary()
```

### `traced_scenario_runner`

Like `scenario_runner` but also captures traces.

```python
async def test_example(traced_scenario_runner):
    func, result, trace = await traced_scenario_runner(scenario)
```

### `trace_validator`

Validates captured traces against expectations.

```python
def test_example(trace_validator):
    validation = trace_validator(scenario, trace, result)
    assert validation.passed, validation.errors
```

### `dataset_factory`

Creates test datasets programmatically.

```python
def test_example(dataset_factory):
    path, dataset = dataset_factory("my_dataset", size=10)
```

## Scenario Builders

### `basic_scenario()`

Simple scenario for basic functionality tests.

```python
from tests.optimizer_validation.specs import basic_scenario

scenario = basic_scenario(
    name="my_test",
    injection_mode="context",  # or "parameter", "attribute", "seamless"
    execution_mode="edge_analytics",
    max_trials=3,
)
```

### `multi_objective_scenario()`

Scenario with multiple objectives.

```python
from tests.optimizer_validation.specs import multi_objective_scenario

scenario = multi_objective_scenario(
    name="multi_obj",
    objectives=["accuracy", "cost", "latency"],
    weights=[0.5, 0.3, 0.2],
)
```

### `constrained_scenario()`

Scenario with constraints.

```python
from tests.optimizer_validation.specs import constrained_scenario

scenario = constrained_scenario(
    name="constrained",
    constraint_fn=lambda config, metrics: config.get("temperature", 0) < 0.8,
)
```

### `failure_scenario()`

Scenario expecting failure.

```python
from tests.optimizer_validation.specs import failure_scenario

scenario = failure_scenario(
    name="should_fail",
    function_should_raise=ValueError,
    expected_error_type=ValueError,
)
```

## Global Trace Invariants

The trace analyzer automatically checks these invariants:

| Invariant | Description |
|-----------|-------------|
| `root_span_exists` | Must have `optimization_session` root span |
| `all_trials_have_parent` | Trial spans must have valid parent |
| `no_orphan_spans` | All spans connected to trace |
| `valid_timestamps` | `end_time >= start_time` |
| `trial_has_trial_id` | Every trial span has `trial.id` |
| `trial_has_config` | Every trial span has `trial.config` |
| `trial_has_status` | Every trial span has `trial.status` |
| `completed_trials_have_metrics` | Completed trials have metrics |
| `trial_count_matches_result` | Span count matches result |
| `config_matches_result` | Span config matches result config |

To skip specific invariants:

```python
scenario = basic_scenario(
    trace_expectations=TraceExpectations(
        skip_invariants=["trial_count_matches_result"],
    ),
)
```

## Troubleshooting

### Tests fail with "Dataset path must reside under..."

The test datasets must be in the project directory. Use the `dataset_factory` fixture or place files in `tests/optimizer_validation/data/`.

### No traces appearing in Jaeger

1. Ensure Jaeger is running: `docker ps | grep jaeger`
2. Check environment variables are set:
   ```bash
   TRAIGENT_TRACE_ENABLED=true
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   ```
3. Wait a few seconds for spans to flush

### Circular import errors with tracing

The tracing module uses lazy imports to avoid circular dependencies. If you see import errors, ensure you're importing from the correct locations.

### Mock mode not working

Ensure `TRAIGENT_MOCK_MODE=true` is set. The `conftest.py` sets this automatically for all validation tests.

## Contributing

### Adding a New Dimension Test

1. Create `tests/optimizer_validation/dimensions/test_<dimension>.py`
2. Use parametrization for different values
3. Use `scenario_runner` and `result_validator` fixtures
4. Add appropriate markers (`@pytest.mark.unit`, etc.)

### Adding a New Failure Test

1. Create test in `tests/optimizer_validation/failures/`
2. Use `failure_scenario()` builder
3. Assert on expected exception type or error message

### Adding Trace Expectations

1. Define expectations in `specs/trace_expectations.py`
2. Add to scenario: `trace_expectations=my_expectations()`
3. Use `traced_scenario_runner` and `trace_validator`

---

## Knowledge Graph System

The test suite includes a Knowledge Graph (KG) system for semantic analysis of test coverage. This enables automatic detection of coverage gaps and suggests missing test combinations.

### Quick Start

```bash
# Generate coverage data
cd tests/optimizer_validation/viewer
python knowledge_graph.py

# Start the viewer
python -m http.server 8765

# Open coverage analysis: http://127.0.0.1:8765/coverage_analysis.html
```

### Ontology Overview

The KG defines 10 dimensions that characterize optimizer behavior:

| Dimension | Values | Description |
|-----------|--------|-------------|
| InjectionMode | context, parameter, attribute, seamless | How configs are injected |
| ExecutionMode | edge_analytics, privacy, hybrid, cloud, standard | Where optimization runs |
| Algorithm | random, grid, optuna_tpe, optuna_cmaes, optuna_random | Search algorithm |
| ConfigSpaceType | categorical, continuous, mixed, nested, conditional | Parameter space type |
| ObjectiveConfig | single, multi_weighted, multi_pareto, constrained | Objective setup |
| StopCondition | max_trials, timeout, convergence, early_stopping | Stop criteria |
| ParallelMode | sequential, parallel_trials, parallel_examples | Parallelism level |
| ConstraintUsage | none, config_constraints, metric_constraints | Constraint types |
| FailureMode | none, partial_failure, full_failure, recovery | Error handling |
| Reproducibility | none, seeded, deterministic | Reproducibility level |

### Critical Dimension Pairs

The KG tracks pairwise coverage for these critical combinations:

1. **InjectionMode × ExecutionMode** - All injection methods work in all environments
2. **Algorithm × ConfigSpaceType** - Algorithms handle all parameter types
3. **ParallelMode × StopCondition** - Parallel execution stops correctly
4. **ObjectiveConfig × ConstraintUsage** - Constrained multi-objective works
5. **Algorithm × FailureMode** - Algorithm resilience to failures

### Programmatic Usage

```python
from pathlib import Path
from tests.optimizer_validation.viewer.knowledge_graph import TestKnowledgeGraph

# Build graph
kg = TestKnowledgeGraph()
kg.load_tests(Path("tests/optimizer_validation/dimensions"))

# Find gaps
gaps = kg.find_coverage_gaps()
for gap in gaps[:5]:
    print(f"Missing: {gap.dimension1}={gap.value1} × {gap.dimension2}={gap.value2}")

# Get suggestions
suggestions = kg.suggest_new_tests(max_suggestions=10)
```

### Documentation

- [Knowledge Graph Design](viewer/knowledge_graph_design.md) - Full ontology specification
- [Coverage Analysis](viewer/coverage_analysis.html) - Interactive visualization

---

## Interactive Test Viewer

The test viewer provides a web dashboard for exploring test results and evidence.

### Starting the Viewer

```bash
cd tests/optimizer_validation/viewer
python -m http.server 8765
# Open http://127.0.0.1:8765/
```

### Features

1. **Test Dashboard** (`index.html`)
   - Browse tests by dimension category
   - View test scenarios with expected/actual results
   - See validation checks and evidence
   - Filter by pass/fail status

2. **Coverage Analysis** (`coverage_analysis.html`)
   - Pairwise coverage heatmap
   - Coverage gap identification
   - Suggested new test combinations
   - Dimension coverage statistics

---

## Evidence Validation

The test suite includes tools for validating that all tests emit proper evidence.

### Evidence Schema

Test evidence follows a defined JSON schema. See [`specs/evidence_schema.json`](specs/evidence_schema.json) for the full specification.

Required sections in test evidence:

| Section             | Required Fields                                        |
| ------------------- | ------------------------------------------------------ |
| `scenario`          | `name`, `config_space`, `injection_mode`, `max_trials` |
| `expected`          | `outcome`                                              |
| `actual`            | `type`                                                 |
| `validation_checks` | Array of checks with `passed` boolean                  |

### Validation Tool

Use the validation tool to check all tests emit proper evidence:

```bash
# Generate a JSON report
TRAIGENT_MOCK_MODE=true pytest tests/optimizer_validation/ \
    --json-report --json-report-file=report.json

# Validate the report
python -m tests.optimizer_validation.tools.validate_evidence report.json

# Show warnings too
python -m tests.optimizer_validation.tools.validate_evidence report.json -v
```

Example output:

```text
============================================================
TEST EVIDENCE VALIDATION REPORT
============================================================
Total tests: 49
With evidence: 49
Without evidence: 0
Issues found: 0

✓ All tests have valid evidence
============================================================
```

### Emitting Evidence in Tests

Evidence is automatically emitted when tests call `result_validator`:

```python
async def test_example(self, scenario_runner, result_validator):
    scenario = TestScenario(...)
    func, result = await scenario_runner(scenario)

    # This call emits evidence to stdout (captured by pytest-json-report)
    validation = result_validator(scenario, result)
    assert validation.passed
```

Tests that don't call `result_validator` will not emit evidence.

See [`tools/README.md`](tools/README.md) for detailed documentation

---

## Gist Templates (Tooltip Summaries)

Gist templates provide brief, informative tooltips for tests in the viewer. They appear on hover over test names in the tree view, giving at-a-glance information about what the test does and what happened.

### What is a Gist?

A gist is a one-liner summary combining:
- Key test characteristics (config type, injection mode, etc.)
- Expected outcome
- Actual result

**Example gists:**
- `empty-dataset -> ConfigurationError | PASS`
- `categorical | 18 combos | PASS`
- `PASS | max_trials_reached | 3 trials`

### Defining a Gist Template

Add a `gist_template` field to your `TestScenario`:

```python
scenario = TestScenario(
    name="empty_dataset",
    description="Test with empty dataset file",
    gist_template="empty-dataset -> {error_type()} | {status()}",
    # ... other fields
)
```

### Available Template Functions

| Function | Pre-run | Post-run | Description |
|----------|---------|----------|-------------|
| `{status()}` | `--` | `PASS` / `FAIL` | Overall test result |
| `{outcome()}` | `{outcome}` | `SUCCESS` / `EXCEPTION` | Actual outcome |
| `{error_type()}` | `{error}` | `ValueError`, etc. | Exception type if failed |
| `{trial_count()}` | `?` | `3` | Number of trials executed |
| `{best_score()}` | `--` | `0.8542` | Best score achieved |
| `{stop_reason()}` | `--` | `max_trials_reached` | Why optimization stopped |
| `{duration()}` | `--` | `1.23s` | Total execution time |
| `{config_space_size()}` | from scenario | `4 combos` | Search space size |
| `{injection_mode()}` | from scenario | `context` | Injection mode used |
| `{algorithm()}` | from scenario | `random` | Optimizer algorithm |
| `{expected_outcome()}` | from scenario | `SUCCESS` / `FAILURE` | What the test expects |
| `{expected_trials()}` | from scenario | `1-5` or `>=1` | Expected trial range |

### Example Templates by Test Type

**Failure Tests:**
```python
gist_template="empty-dataset -> {error_type()} | {status()}"
# Pre-run:  "empty-dataset -> {error} | --"
# Post-run: "empty-dataset -> DatasetError | PASS"
```

**Stop Condition Tests:**
```python
gist_template="max_trials=3 -> {stop_reason()} @ {trial_count()} | {status()}"
# Pre-run:  "max_trials=3 -> -- @ ? | --"
# Post-run: "max_trials=3 -> max_trials_reached @ 3 | PASS"
```

**Multi-Objective Tests:**
```python
gist_template="accuracy+cost -> best={best_score()} | {status()}"
# Pre-run:  "accuracy+cost -> best=-- | --"
# Post-run: "accuracy+cost -> best=0.8542 | PASS"
```

**Algorithm Tests:**
```python
gist_template="{algorithm()} -> {trial_count()} trials | {status()}"
# Pre-run:  "{algo} -> ? trials | --"
# Post-run: "random -> 5 trials | PASS"
```

**Config Space Tests:**
```python
gist_template="categorical | {config_space_size()} | {status()}"
# Pre-run:  "categorical | {size} | --"
# Post-run: "categorical | 18 combos | PASS"
```

**Injection Mode Tests:**
```python
gist_template="{injection_mode()} mode -> {trial_count()} trials | {status()}"
# Pre-run:  "{mode} mode -> ? trials | --"
# Post-run: "context mode -> 3 trials | PASS"
```

### Auto-Generated Defaults

If no `gist_template` is specified, the viewer auto-generates one based on test file location:

| Test Location | Default Template |
|---------------|------------------|
| `failures/` | `{status()} \| {error_type()}` |
| `stop_conditions` | `{status()} \| {stop_reason()} \| {trial_count()} trials` |
| `objectives` | `{status()} \| best={best_score()}` |
| `config_spaces` | `{status()} \| {config_space_size()}` |
| `injection` | `{status()} \| {injection_mode()}` |
| `algorithm` | `{status()} \| {algorithm()} \| {trial_count()} trials` |
| default | `{status()} \| {trial_count()} trials` |

### Guidelines for Writing Good Gists

1. **Be concise** - Aim for under 50 characters
2. **Lead with context** - Start with what makes this test unique
3. **Include the key assertion** - What must be true for the test to pass?
4. **End with status** - Always include `{status()}` so the result is visible

**Good examples:**
- `empty-dataset -> ConfigurationError | {status()}`
- `3 weighted objectives -> best={best_score()} | {status()}`
- `{injection_mode()} + {algorithm()} | {status()}`

**Avoid:**
- Generic templates that don't distinguish tests
- Too many placeholders (keep it to 2-3 dynamic parts)
- Redundant information already visible in the test name
