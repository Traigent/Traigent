# Multi-Agent Workflow Cost Tracking - Schema Review

**Date**: 2026-01-16
**Feature**: Multi-agent workflow cost tracking via `with_usage()` helper
**Status**: ✅ No schema changes required

## Executive Summary

The multi-agent workflow cost tracking feature is **fully compatible** with the existing Traigent cross-project data contracts. No changes are required to the backend schema, SDK DTOs, or frontend interfaces.

---

## Schema Components Reviewed

### 1. Configuration Run DTOs (`traigent/cloud/dtos.py`)

**Current Schema**:
```python
@dataclass
class ConfigurationRunDTO:
    id: str
    experiment_run_id: str
    trial_number: int
    configuration: dict[str, Any]
    measures: dict[str, Any]  # ✅ Flexible - accepts any metrics
    metadata: dict[str, Any]
```

**Assessment**: ✅ **Compatible**
- `measures` field is already `dict[str, Any]`
- Accepts arbitrary metric keys including `cost`, `input_tokens`, `output_tokens`
- No changes needed

---

### 2. Workflow Trace Schemas (`traigent/integrations/observability/workflow_traces.py`)

**Current Schema**:
```python
@dataclass
class SpanPayload:
    # ... other fields
    input_tokens: int = 0        # ✅ Already present
    output_tokens: int = 0       # ✅ Already present
    cost_usd: float = 0.0        # ✅ Already present
    metadata: dict[str, Any]
```

**Assessment**: ✅ **Compatible**
- Workflow trace spans already have dedicated token and cost fields
- These fields are serialized to backend in `to_dict()` (lines 159-161)
- Multi-agent workflows can report per-agent costs via these fields
- No changes needed

---

### 3. Metrics Data Model (`traigent/evaluators/metrics_tracker.py`)

**Current Schema**:
```python
@dataclass
class TokenMetrics:
    input_tokens: int = 0        # ✅ Already present
    output_tokens: int = 0       # ✅ Already present
    total_tokens: int = 0

@dataclass
class CostMetrics:
    input_cost: float = 0.0      # ✅ Already present
    output_cost: float = 0.0     # ✅ Already present
    total_cost: float = 0.0      # ✅ Already present

@dataclass
class ExampleMetrics:
    tokens: TokenMetrics
    cost: CostMetrics
    response: ResponseMetrics
    success: bool
    custom_metrics: dict[str, Any]
```

**Assessment**: ✅ **Compatible**
- All required fields for cost/token tracking are already present
- The `with_usage()` feature injects into these existing fields
- Backend serialization already handles these metrics
- No changes needed

---

### 4. Backend API Contract (`traigent/cloud/api_operations.py`)

**Current Implementation**:
```python
async def update_config_run_measures(
    self,
    config_run_id: str,
    metrics: dict[str, float],      # ✅ Accepts any metric keys
    execution_time: float | None = None,
) -> bool:
    # ... mapping logic

    # Include any metrics dynamically
    for key, value in metrics.items():
        if key not in mapped_metrics and value is not None:
            mapped_metrics[key] = ensure_numeric(value)

    # Send to backend
    measures_data = {
        "measures": {
            "metrics": mapped_metrics,   # ✅ Flexible dict
            "metadata": {}
        }
    }
```

**Backend Endpoint**: `PUT /configuration-runs/{config_run_id}/measures`

**Payload Structure**:
```json
{
  "measures": {
    "metrics": {
      "accuracy": 0.95,
      "cost": 0.00234,           // ✅ Accepted
      "input_tokens": 1500,      // ✅ Accepted
      "output_tokens": 750,      // ✅ Accepted
      "latency": 1.23,
      "custom_metric": 42.0      // ✅ Any key accepted
    },
    "metadata": {
      "execution_time": 1.23
    }
  }
}
```

**Assessment**: ✅ **Compatible**
- Backend API accepts arbitrary metric keys via `dict[str, float]`
- Dynamic mapping ensures all metrics are forwarded
- No schema validation enforces specific keys
- `cost`, `input_tokens`, `output_tokens` are treated like any other metric
- No changes needed

---

## Data Flow Verification

### SDK → Backend Flow

```
User Function
    ↓
with_usage(text, total_cost, input_tokens, output_tokens)
    ↓
Returns: {"text": "...", "__traigent_meta__": {...}}
    ↓
LocalEvaluator._extract_and_inject_traigent_meta()
    ↓
Injects into ExampleMetrics.cost and ExampleMetrics.tokens
    ↓
MetricsTracker.aggregate_metrics()
    ↓
Returns dict with "cost", "input_tokens", "output_tokens" keys
    ↓
ApiOperations.update_config_run_measures()
    ↓
Converts to: {"measures": {"metrics": {...}}}
    ↓
PUT /configuration-runs/{id}/measures
    ↓
Backend stores in ConfigurationRun.measures
```

✅ **Verified**: Full end-to-end compatibility with existing schema

---

## Cross-Project Contract Compatibility

### SDK ↔ Backend Contract

| Component | SDK Field | Backend Field | Status |
|-----------|-----------|---------------|--------|
| Cost | `ExampleMetrics.cost.total_cost` | `measures.metrics.cost` | ✅ Compatible |
| Input Tokens | `ExampleMetrics.tokens.input_tokens` | `measures.metrics.input_tokens` | ✅ Compatible |
| Output Tokens | `ExampleMetrics.tokens.output_tokens` | `measures.metrics.output_tokens` | ✅ Compatible |
| Workflow Spans | `SpanPayload.cost_usd` | `TraceSpan.cost_usd` | ✅ Compatible |
| Workflow Spans | `SpanPayload.input_tokens` | `TraceSpan.input_tokens` | ✅ Compatible |

### Backend ↔ Frontend Contract

The frontend consumes:
- `ConfigurationRun.measures.metrics.*` - All metrics as key-value pairs
- `TraceSpan.{cost_usd, input_tokens, output_tokens}` - Workflow trace data

✅ **No changes required** - Frontend already handles these fields

---

## External Schema Validation

### optigen_schemas Package

**Reference**: `traigent/cloud/dtos.py:17-24`
```python
try:
    from optigen_schemas.validator import SchemaValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False
```

**Status**: Optional dependency, not strictly enforced
**Impact**: None - the `measures` field is validated as `dict[str, Any]` generically

---

## Reserved Key Protocol

### `__traigent_meta__` Key

**Purpose**: Internal protocol for cost/token injection
**Scope**: SDK-only (not sent to backend)
**Format**:
```python
{
    "text": str,
    "__traigent_meta__": {
        "total_cost": float,
        "usage": {
            "input_tokens": int,
            "output_tokens": int
        }
    }
}
```

**Backend Impact**: ✅ **None**
- This key is stripped during metric extraction (line 823 in `local.py`)
- Only the injected metrics reach the backend
- No schema changes required

---

## Recommendations

### ✅ No Action Required

The multi-agent workflow cost tracking feature requires **no changes** to:
1. Backend database schema
2. Backend API contracts (`/configuration-runs`, `/trace-spans`)
3. SDK DTOs (`dtos.py`)
4. Frontend data models
5. Workflow trace schemas

### 📝 Documentation Recommendations

Consider documenting:
1. **For Users**: How `with_usage()` metrics flow to backend/frontend
2. **For Backend Team**: That `measures.metrics` may now include `input_tokens`, `output_tokens` from multi-agent workflows
3. **For Frontend Team**: No changes needed, existing visualizations will display new metrics automatically

### 🔄 Future Considerations

If stricter schema validation is added in the future:
- Ensure `measures.metrics` remains flexible for custom metrics
- Add `input_tokens`, `output_tokens`, `cost` to any whitelists
- Consider formalizing the `__traigent_meta__` protocol if exposing it publicly

---

## Conclusion

✅ **Schema Review Complete**: No changes required
✅ **Cross-Project Compatibility**: Fully maintained
✅ **Backward Compatibility**: 100% preserved

The multi-agent workflow cost tracking feature integrates seamlessly with the existing Traigent schema architecture.
