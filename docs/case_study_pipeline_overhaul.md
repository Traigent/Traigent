# Case Study Pipeline Overhaul – Technical Documentation

> **Status note**: This document describes the historical paper-experiments pipeline. Verify paths/dependencies against current `paper_experiments/` before running; regenerate configs if they changed.

Current repo status: only `paper_experiments/case_study_fever/` is checked in. The other case study sections below are retained for historical reference and may not match the current tree.

## Executive Summary

This document describes a comprehensive overhaul of the Traigent paper experiments pipeline, standardizing evaluation across seven academic case studies (FEVER, KILT, HotpotQA, SQuAD, Summarization, TriviaQA, Spider). Only FEVER remains in this repo; the rest are historical references. The refactoring introduces:

- **Mandatory mock mode** for automated testing to prevent accidental API costs
- **Unified evaluation pipeline** using `LocalEvaluator` + `metric_functions`
- **Telemetry-driven metrics** capturing token/cost/latency from real LLM calls
- **Simulator safeguards** preventing production leakage

**Impact**: Custom evaluators in this pipeline were deprecated in favor of the `metric_functions` interface. Set `TRAIGENT_MOCK_MODE=true` for automated tests to avoid API spend.

**Timeline**: Keep mock mode on by default; enable real providers only for intentional benchmarking.

---

## Table of Contents

1. [Breaking Changes Alert](#breaking-changes-alert)
2. [Context & Background](#context-and-background)
3. [Motivation & Goals](#motivation-and-goals)
4. [Technical Architecture](#technical-architecture)
   - [Architecture Overview](#architecture-overview)
   - [Component Interactions](#component-interactions)
   - [Metric Functions Interface](#metric-functions-interface)
5. [Implementation Details](#implementation-details)
   - [Case Study Changes](#detailed-impact-by-case-study)
   - [Code Examples](#code-example-before-vs-after)
6. [Migration Guide](#migration-guide-for-existing-users)
7. [Testing Strategy](#testing-strategy)
8. [Risk Assessment](#risk-and-validation-notes)
9. [Follow-Up Actions](#follow-up-recommendations)
10. [Appendices](#appendix-interface-specifications)

---

## Breaking Changes Alert

| Breaking Change | Impact | Action Required | Priority |
|----------------|---------|-----------------|----------|
| **TRAIGENT_MOCK_MODE** now mandatory | Automated tests will make real API calls without this flag | Set `export TRAIGENT_MOCK_MODE=true` in all CI/test environments | 🔴 **CRITICAL** |
| **Custom evaluators deprecated** | Scenario-specific evaluators no longer function | Migrate to `metric_functions` interface | 🔴 **HIGH** |
| **Telemetry format changes** | New trial histories include additional fields | Update analysis scripts to handle new schema | 🟡 **MEDIUM** |
| **SDK version requirements** | Requires `openai>=1.0.0`, `anthropic>=0.18.0` | Update `requirements.txt` or `pyproject.toml` | 🟡 **MEDIUM** |

### Required Environment Configuration

```bash
# Mandatory for all automated tests
export TRAIGENT_MOCK_MODE=true

# Required for real LLM calls (when mock mode is disabled)
export OPENAI_API_KEY="sk-..."        # For GPT models
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude models
```

**⚠️ Warning**: Forgetting `TRAIGENT_MOCK_MODE=true` will trigger real API calls and incur costs during CI/QA runs.

---

## Context and Background

### What Are Paper Experiments?

The `paper_experiments/` directory contains one active case study in this repo (FEVER). The other case studies listed below are historical references that may not exist in this tree:

| Case Study | Domain | Task Type | Primary Metrics |
|------------|--------|-----------|-----------------|
| **FEVER** | Fact Verification | Classification | FEVER score, precision, recall |
| **KILT** | Knowledge Retrieval | QA + Retrieval | KILT accuracy, retrieval metrics |
| **HotpotQA** | Multi-hop QA | Reasoning + Retrieval | EM, F1, retrieval recall |
| **SQuAD** | Reading Comprehension | Span Extraction | EM, F1, span coverage |
| **Summarization** | Text Generation | Abstractive Summary | ROUGE-1/2/L, factuality |
| **TriviaQA** | Open-domain QA | Retrieval + Generation | EM, F1, hallucination rate |
| **Spider** | Text-to-SQL | Code Generation | Execution accuracy, schema match |

### Purpose

Each case study simulates a real-world NLP workflow to evaluate quality/cost/latency trade-offs under controlled conditions. These benchmarks enable:

- Academic research on LLM optimization strategies
- Comparative analysis of different models and configurations
- Reproducible evaluation of optimization algorithms (Bayesian, Grid Search, Random, Optuna)

### Previous Architecture Issues

**Before this overhaul:**

- ❌ Custom evaluators per scenario created maintenance burden
- ❌ No unified telemetry capture for cost/latency metrics
- ❌ Simulator code could accidentally execute in production
- ❌ Inconsistent mock vs. real execution paths
- ❌ Manual credential validation prone to errors

---

## Motivation and Goals

### Primary Objectives

1. **🛡️ Prevent Accidental API Costs**
   - Enforce `TRAIGENT_MOCK_MODE` for all automated testing
   - Add credential validation before real LLM calls
   - Fail fast if simulators are invoked incorrectly

2. **📊 Standardize Evaluation**
   - Single evaluation pipeline using `LocalEvaluator`
   - Consistent `metric_functions` interface across all scenarios
   - Automatic telemetry capture for token/cost/latency

3. **🔧 Reduce Maintenance Burden**
   - Eliminate redundant custom evaluators
   - Co-locate mock/real execution paths in pipeline modules
   - Simplify dependency management

4. **📈 Improve Observability**
   - Capture comprehensive telemetry from all LLM calls
   - Enable detailed cost/performance analysis
   - Support historical comparison and regression detection

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Cost Prevention** | 100% of automated tests use mock mode | CI pipeline checks |
| **Evaluation Consistency** | All scenarios use `LocalEvaluator` | Code review |
| **Telemetry Completeness** | 100% of real calls capture token/cost/latency | Trial history validation |
| **Code Reduction** | -30% lines in evaluator modules | Git diff analysis |
| **Test Coverage** | >80% for new metric functions | `pytest --cov` |

---

## Technical Architecture

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Case Study Pipeline                          │
│                                                                   │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │ CLI Entry    │───────>│  Pipeline    │                       │
│  │ Point        │        │  Orchestrator│                       │
│  └──────────────┘        └──────┬───────┘                       │
│                                  │                                │
│                    ┌─────────────┴─────────────┐                │
│                    │                             │                │
│          ┌─────────▼─────────┐       ┌─────────▼─────────┐      │
│          │   Mock Mode       │       │   Real Mode       │      │
│          │   (Simulator)     │       │   (LLM Calls)     │      │
│          └─────────┬─────────┘       └─────────┬─────────┘      │
│                    │                             │                │
│                    │  ┌──────────────────────┐  │                │
│                    └─>│  Metric Functions    │<─┘                │
│                       │  (Unified Interface) │                   │
│                       └──────────┬───────────┘                   │
│                                  │                                │
│                       ┌──────────▼───────────┐                   │
│                       │  LocalEvaluator      │                   │
│                       │  (Telemetry Capture) │                   │
│                       └──────────┬───────────┘                   │
│                                  │                                │
│                       ┌──────────▼───────────┐                   │
│                       │  Trial History JSON  │                   │
│                       │  (Results + Metrics) │                   │
│                       └──────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

#### 1. Mock Mode Flow

```text
User Request
    │
    ├─> Check TRAIGENT_MOCK_MODE=true
    │
    ├─> Route to Simulator
    │   ├─> Validate model identifier
    │   ├─> Generate deterministic response
    │   └─> Return simulated metrics
    │
    ├─> Invoke metric_functions
    │   └─> Extract metrics from simulation
    │
    └─> Write to trial_history.json
```

#### 2. Real Mode Flow

```text
User Request
    │
    ├─> Check TRAIGENT_MOCK_MODE=false/unset
    │
    ├─> Validate API Credentials
    │   ├─> OpenAI key for gpt-* models
    │   └─> Anthropic key for claude-* models
    │
    ├─> Build Scenario-Specific Prompt
    │
    ├─> Invoke LLM SDK
    │   └─> Capture telemetry (via capture_langchain_response)
    │       ├─> prompt_tokens
    │       ├─> completion_tokens
    │       ├─> total_cost
    │       └─> response_time_ms
    │
    ├─> Invoke metric_functions
    │   └─> Merge LLM output + telemetry
    │
    └─> Write to trial_history.json
```

### Metric Functions Interface

**New Standard Interface:**

```python
from typing import Any, Callable, Dict, Optional

MetricFunction = Callable[
    [Any, Any, Any, Optional[Dict], Optional[Dict]],  # output, expected, example, config, llm_metrics
    float  # metric value
]

def build_scenario_metric_functions(
    *,
    mock_mode: bool
) -> Dict[str, MetricFunction]:
    """
    Returns a dictionary of metric functions compatible with LocalEvaluator.

    Args:
        mock_mode: If True, metrics derive from simulator output.
                   If False, metrics use real LLM telemetry.

    Returns:
        Dictionary mapping metric names to callable functions.
    """
    pass
```

**Key Properties:**

- ✅ **Unified Interface**: All scenarios expose the same function signature
- ✅ **Telemetry Integration**: `llm_metrics` parameter provides captured data
- ✅ **Mode Awareness**: `mock_mode` flag controls metric computation logic
- ✅ **Type Safety**: Strong typing for better IDE support and validation

---

## Implementation Details

### Detailed Impact by Case Study

Note: Only `case_study_fever/` is present in this repo. The other sections are historical and may not match current code.

#### 1. FEVER (`paper_experiments/case_study_fever/`)

**Changes:**

- `pipeline.py`: Wires `build_fever_metric_functions` and separates mock/real call paths with credential validation
- `metrics.py`: Returns FEVER score, latency, cost, and telemetry metrics per example
- `simulator.py`: Enforces mock-only execution and recognized models

**New Metrics Available:**

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `fever_score` | Quality | Simulator/LLM | Official FEVER evaluation score |
| `precision` | Quality | Simulator/LLM | Label prediction precision |
| `recall` | Quality | Simulator/LLM | Evidence retrieval recall |
| `cost_usd_per_1k` | Cost | Telemetry | Cost per 1K examples |
| `latency_p95_ms` | Performance | Telemetry | 95th percentile latency |
| `prompt_tokens` | Resource | Telemetry | Input token count |
| `completion_tokens` | Resource | Telemetry | Output token count |

#### 2. KILT (`paper_experiments/case_study_kilt/`)

**Changes:**

- Real pipeline builds retrieval prompts; mock pipeline uses deterministic simulator
- Metrics cover KILT accuracy, retrieval precision/recall, hallucination rate
- Simulator rejects non-mock use and unknown models

**Key Features:**

- Dual-mode retrieval evaluation
- Hallucination detection in mock and real modes
- Evidence quality scoring

#### 3. HotpotQA (`paper_experiments/case_study_rag/`)

**Changes:**

- Pipeline now consumes retrieved context when building prompts
- Delegates all metrics to `LocalEvaluator`
- Simulator locked to mock mode; real latency/cost derived from telemetry

**Multi-Hop Reasoning:**

- Tracks intermediate reasoning steps
- Measures retrieval effectiveness for each hop
- Captures supporting fact coverage

#### 4. SQuAD (`paper_experiments/case_study_squad/`)

**Changes:**

- Real pipeline extracts spans using the LLM
- Metric helper measures EM/F1/span coverage from telemetry
- Simulator requires mock mode and approved model names

**Span Extraction Metrics:**

- Exact Match (EM) score
- Token-level F1 score
- Answer span position accuracy

#### 5. Summarization (`paper_experiments/case_study_summarization/`)

**Changes:**

- New metric helper merges simulator ROUGE proxies with actual summaries
- Real pipeline builds summarization prompts and validates API keys
- Simulator mock-only with deterministic ROUGE scores

**Summary Quality Metrics:**

- ROUGE-1, ROUGE-2, ROUGE-L scores
- Factual consistency (when ground truth available)
- Compression ratio and abstractiveness

#### 6. TriviaQA (`paper_experiments/case_study_triviaqa/`)

**Changes:**

- Pipeline constructs retrieval/verifier-aware prompts
- Metrics capture EM/F1, retrieval recall, hallucination rate
- Simulator enforces mock mode and known models

**Retrieval + QA Pipeline:**

- Document retrieval quality
- Answer extraction accuracy
- Hallucination detection for unsupported answers

#### 7. Spider – Special Case (`paper_experiments/case_study_spider/`)

**Changes:**

- Already captured telemetry; this change enhances existing LLM client
- Merges telemetry with simulator metrics
- Trims redundant evaluator logic

**SQL Generation Metrics:**

- Execution accuracy (query produces correct result)
- Schema match accuracy (correct tables/columns)
- SQL syntax validity

**Spider as Reference Implementation:**

- Demonstrates complete telemetry integration
- Shows best practices for custom metric functions
- Provides template for other scenarios

---

## Code Example: Before vs After

### Before (Custom Evaluator)

```python
# Old approach: Custom evaluator per scenario
class FEVERCustomEvaluator:
    """Deprecated: Will be removed in next release"""

    def evaluate(self, func, config, example):
        # Tightly coupled to simulation logic
        simulation = simulate_case_study_execution(
            example.claim,
            config
        )

        # No telemetry capture
        # Cost/latency metrics hardcoded to 0
        return {
            "fever_score": simulation.metrics["fever_score"],
            "cost_usd_per_1k": 0.0,  # Always zero!
            "latency_p95_ms": 0.0,   # Always zero!
        }

# Usage in experiment
optimized_func = traigent.optimize(
    configuration_space=config_space,
    custom_evaluator=FEVERCustomEvaluator(),  # Manual wiring
)
```

### After (Metric Functions + LocalEvaluator)

```python
# New approach: Metric functions with telemetry
def build_fever_metric_functions(
    *,
    mock_mode: bool
) -> Dict[str, Callable[..., float]]:
    """
    Build FEVER metric functions compatible with LocalEvaluator.

    Args:
        mock_mode: If True, use simulator metrics.
                   If False, use real LLM telemetry.

    Returns:
        Dictionary of metric name -> metric function.
    """

    def _ensure_metrics(
        output: Any,
        expected: Any,
        example: Any,
        config: Optional[Dict] = None,
        llm_metrics: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Extract metrics from simulator or telemetry."""

        if mock_mode:
            # Mock mode: Use simulator output
            simulation = simulate_case_study_execution(
                example.claim,
                config or {}
            )
            return {
                "fever_score": simulation.metrics["fever_score"],
                "cost_usd_per_1k": simulation.metrics.get("cost", 0.0),
                "latency_p95_ms": simulation.metrics.get("latency", 0.0),
            }
        else:
            # Real mode: Use captured telemetry
            if llm_metrics is None:
                raise ValueError("llm_metrics required in real mode")

            # Compute FEVER score from LLM output
            fever_result = compute_fever_score(
                prediction=output,
                gold_label=expected.label,
                gold_evidence=expected.evidence
            )

            return {
                "fever_score": fever_result["score"],
                "cost_usd_per_1k": llm_metrics.get("total_cost", 0.0) * 1000,
                "latency_p95_ms": llm_metrics.get("response_time_ms", 0.0),
                "prompt_tokens": llm_metrics.get("prompt_tokens", 0),
                "completion_tokens": llm_metrics.get("completion_tokens", 0),
            }

    # Return individual metric extractors
    return {
        "fever_score": lambda **ctx: _ensure_metrics(**ctx)["fever_score"],
        "cost_usd_per_1k": lambda **ctx: _ensure_metrics(**ctx)["cost_usd_per_1k"],
        "latency_p95_ms": lambda **ctx: _ensure_metrics(**ctx)["latency_p95_ms"],
        "prompt_tokens": lambda **ctx: _ensure_metrics(**ctx).get("prompt_tokens", 0),
        "completion_tokens": lambda **ctx: _ensure_metrics(**ctx).get("completion_tokens", 0),
    }

# Usage in experiment (automatic wiring via pipeline)
optimized_func = build_fever_optimized_function(
    mock_mode=os.getenv("TRAIGENT_MOCK_MODE", "").lower() == "true"
)
# LocalEvaluator automatically wired with metric_functions
```

### Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Telemetry** | ❌ No real cost/latency capture | ✅ Automatic via `llm_metrics` |
| **Maintenance** | ❌ Separate evaluator classes | ✅ Simple function dictionaries |
| **Type Safety** | ❌ Weak typing | ✅ Strong type hints |
| **Mode Switching** | ❌ Manual logic | ✅ Automatic via `mock_mode` flag |
| **Extensibility** | ❌ Hard to add metrics | ✅ Just add to dictionary |
| **Testing** | ❌ Requires mock evaluator | ✅ Built-in mock mode |

---

## Migration Guide for Existing Users

### Step-by-Step Migration

#### Step 1: Update Environment Configuration

**In CI/CD pipelines:**

```bash
# .github/workflows/test.yml or similar
env:
  TRAIGENT_MOCK_MODE: "true"  # CRITICAL: Prevent API costs
  TRAIGENT_LOG_LEVEL: "INFO"
```

**For local development:**

```bash
# .env.test or test-specific configuration
export TRAIGENT_MOCK_MODE=true

# .env.production (when intentionally running real experiments)
export TRAIGENT_MOCK_MODE=false
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Step 2: Update Dependencies

```bash
# Update SDK versions
pip install --upgrade openai>=1.12 anthropic>=0.18

# Or in requirements.txt
openai>=1.12.0
anthropic>=0.18.0
```

#### Step 3: Remove Custom Evaluator Imports

**Before:**

```python
from paper_experiments.case_study_fever.evaluator import make_fever_evaluator

evaluator = make_fever_evaluator()
optimized_func = traigent.optimize(
    custom_evaluator=evaluator,
    ...
)
```

**After:**

```python
from paper_experiments.case_study_fever.pipeline import build_fever_optimized_function

optimized_func = build_fever_optimized_function(
    mock_mode=os.getenv("TRAIGENT_MOCK_MODE", "").lower() == "true"
)
# Evaluator automatically configured
```

#### Step 4: Validate Telemetry Output

After first optimization run:

```python
import json

# Load trial history
with open("trial_history.json") as f:
    trials = json.load(f)

# Verify telemetry fields present
for trial in trials:
    assert "latency_p95_ms" in trial["metrics"]
    assert "cost_usd_per_1k" in trial["metrics"]
    assert "prompt_tokens" in trial["metrics"]
    assert "completion_tokens" in trial["metrics"]

    # Real mode should have non-zero values
    if not mock_mode:
        assert trial["metrics"]["cost_usd_per_1k"] > 0
        assert trial["metrics"]["latency_p95_ms"] > 0
```

#### Step 5: Update Analysis Scripts

**Before:**

```python
# Old schema
trial = {
    "metrics": {
        "fever_score": 0.85,
        # cost/latency always 0
    }
}
```

**After:**

```python
# New schema with telemetry
trial = {
    "metrics": {
        "fever_score": 0.85,
        "cost_usd_per_1k": 0.12,
        "latency_p95_ms": 450.0,
        "prompt_tokens": 1200,
        "completion_tokens": 300,
    }
}
```

**Migration helper:**

```python
def normalize_trial_history(old_trial: Dict) -> Dict:
    """Backfill missing telemetry fields for old trials."""
    metrics = old_trial.get("metrics", {})

    # Add missing fields with default values
    metrics.setdefault("cost_usd_per_1k", 0.0)
    metrics.setdefault("latency_p95_ms", 0.0)
    metrics.setdefault("prompt_tokens", 0)
    metrics.setdefault("completion_tokens", 0)

    old_trial["metrics"] = metrics
    return old_trial
```

### Rollback Plan

If issues arise during migration:

1. **Temporary Escape Hatch** (deprecated, will be removed):

```python
# Emergency fallback to old evaluator
from paper_experiments.case_study_fever.evaluator import make_fever_evaluator

optimized_func = traigent.optimize(
    mock_mode_config={"override_evaluator": True},  # Temporary flag
    custom_evaluator=make_fever_evaluator(),
    ...
)
```

2. **Version Pinning**:

```bash
# Pin to pre-migration version
pip install traigent==1.0.0  # Before pipeline overhaul
```

3. **Git Branch**:

```bash
# Revert to pre-migration commit
git checkout tags/v1.0.0-pre-migration
```

**Note**: Escape hatches will be removed in next major release. Plan migration accordingly.

---

## Testing Strategy

### Unit Tests (Per Case Study)

#### Test 1: Simulator Mode Enforcement

```python
def test_simulator_requires_mock_mode():
    """Simulator should reject execution when TRAIGENT_MOCK_MODE is unset."""

    # Clear environment
    if "TRAIGENT_MOCK_MODE" in os.environ:
        del os.environ["TRAIGENT_MOCK_MODE"]

    with pytest.raises(RuntimeError, match="TRAIGENT_MOCK_MODE must be 'true'"):
        simulate_case_study_execution(
            claim="Test claim",
            config={"model": "gpt-4o"}
        )

def test_simulator_rejects_unknown_model():
    """Simulator should reject unknown model identifiers."""

    os.environ["TRAIGENT_MOCK_MODE"] = "true"

    with pytest.raises(ValueError, match="Unknown model"):
        simulate_case_study_execution(
            claim="Test claim",
            config={"model": "unknown-model-xyz"}
        )
```

#### Test 2: Real Pipeline Credential Validation

```python
def test_real_pipeline_requires_api_key():
    """Real pipeline should raise error when API key is missing."""

    # Clear API keys
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        if key in os.environ:
            del os.environ[key]

    os.environ["TRAIGENT_MOCK_MODE"] = "false"

    with pytest.raises(RuntimeError, match="API key required"):
        build_fever_optimized_function(mock_mode=False)
```

#### Test 3: Metric Functions Return Float

```python
def test_metric_functions_return_float():
    """Each metric function should return a float value."""

    os.environ["TRAIGENT_MOCK_MODE"] = "true"

    metrics = build_fever_metric_functions(mock_mode=True)

    # Simulate telemetry
    llm_metrics = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_cost": 0.0015,
        "response_time_ms": 250.0,
    }

    for metric_name, metric_func in metrics.items():
        result = metric_func(
            output="SUPPORTS",
            expected={"label": "SUPPORTS", "evidence": []},
            example={"claim": "Test"},
            config={"model": "gpt-4o"},
            llm_metrics=llm_metrics
        )

        assert isinstance(result, (int, float)), \
            f"{metric_name} must return numeric value, got {type(result)}"
```

### Integration Tests

#### Test 4: End-to-End Mock Mode

```bash
# Run complete optimization in mock mode
TRAIGENT_MOCK_MODE=true python paper_experiments/cli.py optimize \
    --scenario fever \
    --mock-mode on \
    --trials 2 \
    --algorithm grid \
    --output-dir /tmp/fever_mock_test

# Verify output
python -c "
import json
with open('/tmp/fever_mock_test/trial_history.json') as f:
    trials = json.load(f)

assert len(trials) == 2, 'Expected 2 trials'
for trial in trials:
    assert 'latency_p95_ms' in trial['metrics']
    assert 'cost_usd_per_1k' in trial['metrics']
    print(f'✓ Trial {trial[\"trial_id\"]} has complete telemetry')
"
```

#### Test 5: End-to-End Real Mode (Requires API Keys)

```bash
# Run with real LLM calls (manual verification only)
TRAIGENT_MOCK_MODE=false \
OPENAI_API_KEY="sk-..." \
python paper_experiments/cli.py optimize \
    --scenario fever \
    --mock-mode off \
    --trials 1 \
    --algorithm random \
    --output-dir /tmp/fever_real_test

# Verify non-zero metrics
python -c "
import json
with open('/tmp/fever_real_test/trial_history.json') as f:
    trial = json.load(f)[0]

assert trial['metrics']['cost_usd_per_1k'] > 0, 'Expected non-zero cost'
assert trial['metrics']['latency_p95_ms'] > 0, 'Expected non-zero latency'
print('✓ Real mode captured telemetry correctly')
"
```

### Regression / Benchmark Tests

#### Test 6: Historical Baseline Comparison

```python
def test_fever_mock_baseline_quality():
    """FEVER mock baseline should stay within ±5% of historical baseline."""

    HISTORICAL_BASELINE = 0.75  # From previous version
    TOLERANCE = 0.05

    os.environ["TRAIGENT_MOCK_MODE"] = "true"

    # Run optimization
    result = run_optimization(
        scenario="fever",
        trials=10,
        algorithm="random"
    )

    best_fever_score = max(
        trial["metrics"]["fever_score"]
        for trial in result["trials"]
    )

    assert abs(best_fever_score - HISTORICAL_BASELINE) < TOLERANCE, \
        f"Regression detected: {best_fever_score} vs {HISTORICAL_BASELINE}"
```

#### Test 7: Performance Benchmarks

```python
def test_spider_latency_benchmark():
    """Verify gpt-3.5-turbo is faster than gpt-4o-mini (Spider scenario)."""

    os.environ["TRAIGENT_MOCK_MODE"] = "true"

    latency_35 = get_average_latency(model="gpt-3.5-turbo", trials=5)
    latency_4o = get_average_latency(model="gpt-4o-mini", trials=5)

    assert latency_35 < latency_4o, \
        f"gpt-3.5-turbo should be faster: {latency_35}ms vs {latency_4o}ms"
```

---

## Risk and Validation Notes

### Critical Risks

#### Risk 1: Environment Sensitivity

**Risk**: Simulator gating relies on `TRAIGENT_MOCK_MODE`; misconfigured environments may incur real API charges.

**Mitigation:**

- ✅ Add `TRAIGENT_MOCK_MODE` validation in CI/CD pipeline entry points
- ✅ Implement `TRAIGENT_FAIL_ON_REAL_MODE` toggle for strict enforcement
- ✅ Add cost estimation warnings before real mode execution
- ✅ Document environment setup in all quickstart guides

**Validation:**

```python
# Pipeline entry point
if os.getenv("CI") == "true":  # Running in CI
    if os.getenv("TRAIGENT_MOCK_MODE", "").lower() != "true":
        raise EnvironmentError(
            "TRAIGENT_MOCK_MODE must be 'true' in CI environments. "
            "Set 'export TRAIGENT_MOCK_MODE=true' to prevent API costs."
        )
```

#### Risk 2: Backward Compatibility

**Risk**: Telemetry-driven metrics will diverge from historic logs; communicating this to analysts is mandatory.

**Mitigation:**

- ✅ Publish changelog with clear migration timeline
- ✅ Provide schema migration scripts
- ✅ Maintain parallel metrics during transition period (1 release cycle)
- ✅ Add version metadata to trial history files

**Example:**

```json
{
  "schema_version": "2.0",
  "migration_note": "Telemetry fields added in v2.0. See migration guide.",
  "trials": [...]
}
```

#### Risk 3: Performance Overhead

**Risk**: Capturing telemetry adds ~5–10ms per real LLM call; factor this into latency targets.

**Measurement:**

```python
# Telemetry overhead benchmark
import time

def benchmark_telemetry_overhead(iterations=100):
    """Measure telemetry capture overhead."""

    # Without telemetry
    start = time.time()
    for _ in range(iterations):
        call_llm_without_telemetry()
    baseline = (time.time() - start) / iterations

    # With telemetry
    start = time.time()
    for _ in range(iterations):
        call_llm_with_telemetry()
    with_telemetry = (time.time() - start) / iterations

    overhead = with_telemetry - baseline
    print(f"Telemetry overhead: {overhead*1000:.2f}ms per call")

    assert overhead < 0.015, "Telemetry overhead exceeds 15ms threshold"
```

**Mitigation:**

- Telemetry capture is async where possible
- Overhead included in reported latency metrics
- Performance budgets updated to account for measurement cost

#### Risk 4: Custom Metrics Loss

**Risk**: Teams relying on bespoke evaluator metrics must retrofit them into `metric_functions` to avoid silent drops.

**Detection:**

```python
# Audit script to detect custom metrics
def audit_custom_metrics():
    """Scan for custom evaluator usage."""

    deprecated_imports = [
        "make_fever_evaluator",
        "make_kilt_evaluator",
        # ... other deprecated evaluators
    ]

    for filepath in glob.glob("**/*.py", recursive=True):
        with open(filepath) as f:
            content = f.read()

        for deprecated in deprecated_imports:
            if deprecated in content:
                print(f"⚠️  {filepath} uses deprecated {deprecated}")
```

**Mitigation:**

- Automated deprecation warnings in code
- Migration guide with metric conversion examples
- One-release grace period with dual support

#### Risk 5: Dependency Drift

**Risk**: Pipelines now explicitly depend on latest OpenAI/Anthropic SDKs; pin versions in production lock files.

**Best Practice:**

```bash
# requirements.txt (development)
openai>=1.12.0
anthropic>=0.18.0

# requirements.lock (production)
openai==1.12.3
anthropic==0.18.1
# ... pinned transitive dependencies
```

---

## Follow-Up Recommendations

### Immediate Actions (Week 1)

1. **Update CI/CD Pipelines**
   - Add `TRAIGENT_MOCK_MODE=true` to all test environments
   - Implement pre-flight checks for environment configuration
   - Add telemetry validation to deployment gates

2. **Documentation Updates**
   - Update contributor guide with new evaluation patterns
   - Add troubleshooting section for common migration issues
   - Create video walkthrough of migration process

3. **Communication**
   - Publish changelog entry with migration timeline
   - Email all stakeholders about breaking changes
   - Schedule team training session

### Short-Term Actions (Month 1)

4. **Enhanced Testing**
   - Implement smoke tests for each scenario (mock + real mode)
   - Add telemetry validation to CI pipeline
   - Create regression test suite comparing v1 vs v2 outputs

5. **Safety Features**
   - Implement `TRAIGENT_FAIL_ON_REAL_MODE` toggle
   - Add cost estimation preview before real runs
   - Create audit log for all real API calls

6. **Monitoring**
   - Set up alerts for unexpected API usage
   - Track migration adoption metrics
   - Monitor performance overhead in production

### Long-Term Actions (Quarter 1)

7. **Deprecation Cleanup**
   - Remove legacy evaluator modules (after 1 release grace period)
   - Archive old trial histories with schema v1
   - Simplify codebase with unified patterns

8. **Advanced Features**
   - Add support for custom telemetry backends
   - Implement automatic baseline refresh
   - Create interactive migration assistant tool

9. **Knowledge Transfer**
   - Document lessons learned from migration
   - Create reference implementation guide
   - Publish best practices for future changes

---

## Appendix: Interface Specifications

### Metric Functions Interface

**Type Signature:**

```python
from typing import Any, Callable, Dict, Optional

MetricFunction = Callable[
    [
        Any,                     # output: LLM response or simulation result
        Any,                     # expected: Ground truth / gold standard
        Any,                     # example: Original input example
        Optional[Dict[str, Any]], # config: Model configuration
        Optional[Dict[str, Any]], # llm_metrics: Captured telemetry
    ],
    float  # Metric value (must be numeric)
]

MetricFunctionsDict = Dict[str, MetricFunction]
```

**Parameter Details:**

| Parameter | Type | Description | Mock Mode | Real Mode |
|-----------|------|-------------|-----------|-----------|
| `output` | `Any` | Model output (prediction, generation, etc.) | Simulator result | LLM response |
| `expected` | `Any` | Ground truth for comparison | Dataset label | Dataset label |
| `example` | `Any` | Original input example | Dataset row | Dataset row |
| `config` | `Optional[Dict]` | Model configuration (`model`, `temperature`, etc.) | Provided | Provided |
| `llm_metrics` | `Optional[Dict]` | Captured telemetry (see below) | `None` | Populated |

### LocalEvaluator Telemetry Schema

**Telemetry Dictionary Structure:**

```python
llm_metrics = {
    # Token counts
    "prompt_tokens": int,       # Input tokens sent to LLM
    "completion_tokens": int,   # Output tokens generated by LLM
    "total_tokens": int,        # prompt_tokens + completion_tokens

    # Cost metrics (USD)
    "input_cost": float,        # Cost for prompt tokens
    "output_cost": float,       # Cost for completion tokens
    "total_cost": float,        # input_cost + output_cost

    # Performance metrics
    "response_time_ms": float,  # End-to-end latency in milliseconds
    "tokens_per_second": float, # Derived throughput if response_time_ms is available

    # Model metadata
    "model": str,               # Model identifier (e.g., "gpt-4o")

    # Internal metrics object
    "_full_metrics": Any,       # ExampleMetrics object from extract_llm_metrics
}
```

**Telemetry Capture:**

Telemetry is captured via `capture_langchain_response` wrapper:

```python
from traigent.utils.langchain_interceptor import capture_langchain_response

# Capture immediately after an LLM call
response = llm_instance.invoke(prompt)
capture_langchain_response(response)

# Metrics are later extracted via extract_llm_metrics in the evaluators
```

### Mock-Mode Simulator API

**Function Signature:**

```python
def simulate_case_study_execution(
    *args,
    **kwargs
) -> SimulationResult:
    """
    Deterministic simulator for case study execution.

    Requirements:
        - os.getenv("TRAIGENT_MOCK_MODE") must equal "true"
        - config["model"] must be in approved list

    Raises:
        RuntimeError: If TRAIGENT_MOCK_MODE is not set
        ValueError: If model identifier is unknown

    Returns:
        SimulationResult with deterministic metrics
    """
    pass
```

**Approved Model Identifiers:**

| Provider | Models |
|----------|--------|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` |
| Anthropic | `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` |

**SimulationResult Schema:**

```python
@dataclass
class SimulationResult:
    """Result from mock simulation."""

    output: Any                    # Simulated model output
    metrics: Dict[str, float]      # Quality + cost/latency estimates
    deterministic: bool = True     # Always True for simulators

    # Quality metrics (scenario-specific)
    # Example for FEVER:
    # metrics = {
    #     "fever_score": 0.75,
    #     "precision": 0.80,
    #     "recall": 0.70,
    #     "cost_usd_per_1k": 0.10,
    #     "latency_p95_ms": 250.0,
    # }
```

---

## Summary

This pipeline overhaul represents a significant improvement in the Traigent paper experiments infrastructure:

✅ **Safety**: Mandatory mock mode prevents accidental API costs
✅ **Consistency**: Unified evaluation pipeline across all scenarios
✅ **Observability**: Comprehensive telemetry capture for all metrics
✅ **Maintainability**: Reduced code complexity and maintenance burden
✅ **Quality**: Enhanced testing and validation framework

**Next Steps for Teams:**

1. Set `TRAIGENT_MOCK_MODE=true` in all CI/CD pipelines **today**
2. Update dependencies to required SDK versions
3. Remove custom evaluator imports and migrate to new pipelines
4. Validate telemetry output in trial histories
5. Update analysis scripts for new schema format

**Questions or Issues?**

- 📧 Contact the Traigent team via team channels
- 🐛 Report bugs with `[Pipeline Overhaul]` tag
- 📚 See [CLAUDE.md](../CLAUDE.md) for additional SDK documentation

---

**Document Version**: 2.0
**Last Reviewed**: January 2025
**Next Review**: April 2025
