# Optimizer Validation Tools

Tools for validating test evidence and ensuring schema compliance.

## Evidence Validation Tool

The `validate_evidence.py` tool checks that all tests emit complete evidence according to the defined schema.

### Usage

```bash
# Validate a pytest JSON report
python -m tests.optimizer_validation.tools.validate_evidence report.json

# Validate with verbose output (show warnings)
python -m tests.optimizer_validation.tools.validate_evidence report.json -v

# Output as JSON
python -m tests.optimizer_validation.tools.validate_evidence report.json --json

# Validate viewer data.js (when available)
python -m tests.optimizer_validation.tools.validate_evidence --viewer-data
```

### Generating a Report

To generate a pytest JSON report for validation:

```bash
TRAIGENT_MOCK_MODE=true pytest tests/optimizer_validation/ \
    --json-report \
    --json-report-file=report.json
```

### Example Output

```
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

If there are issues:

```
❌ ERRORS (2):
----------------------------------------

  test_example_missing_fields:
    • [missing] scenario.injection_mode: Missing scenario.injection_mode
    • [missing] expected.outcome: Missing expected.outcome
```

## Evidence Schema

The evidence schema is defined in `tests/optimizer_validation/specs/evidence_schema.json`.

### Required Sections

Every test evidence must include:

| Section | Required Fields |
|---------|-----------------|
| `type` | Must be `"TEST_EVIDENCE"` |
| `scenario` | `name`, `config_space`, `injection_mode`, `max_trials` |
| `expected` | `outcome` |
| `actual` | `type` |
| `validation_checks` | Array of checks |
| `passed` | Boolean |

### Recommended Fields

These fields are optional but recommended:

| Section | Recommended Fields |
|---------|-------------------|
| `scenario` | `description`, `execution_mode`, `objectives`, `timeout`, `dataset_size` |
| `expected` | `min_trials`, `max_trials` |
| `actual` | `trial_count`, `stop_reason` |

### Config Space Structure

Each parameter in `config_space` must have:

```json
{
  "parameter_name": {
    "type": "categorical|continuous|fixed",
    "values": [...],       // for categorical
    "range": [min, max],   // for continuous
    "cardinality": 2       // number of options or "∞"
  }
}
```

### Validation Checks Structure

Each check must have:

```json
{
  "check": "check_name",   // or "name": "check_name"
  "passed": true,
  "expected": "...",       // optional
  "actual": "...",         // optional
  "message": "..."         // optional
}
```

## Emitting Evidence in Tests

Evidence is emitted when tests call the `result_validator` fixture:

```python
async def test_example(self, scenario_runner, result_validator):
    scenario = TestScenario(...)
    func, result = await scenario_runner(scenario)

    # This call emits evidence to stdout
    validation = result_validator(scenario, result)
    assert validation.passed
```

Tests that don't call `result_validator` will not emit evidence.

## Weak Test Analyzer

The `weak_test_analyzer.py` tool flags weak or vulnerable tests based on the
meta-analysis root causes (IT-VRO, IT-CBM, IT-VTA). It runs static analysis on
test code and optionally dynamic analysis using a pytest JSON report.

### Usage

```bash
# Static analysis only
python -m tests.optimizer_validation.tools.weak_test_analyzer

# Static + dynamic analysis (use a pytest JSON report)
python -m tests.optimizer_validation.tools.weak_test_analyzer --report report.json

# Emit JSON output to a file
python -m tests.optimizer_validation.tools.weak_test_analyzer --json --output weak_test_report.json
```
