# DTO Contract Recommendations v2 - Addressing Schema Mismatches

**Date**: 2026-01-16
**Status**: Updated based on Codex feedback
**Previous**: `/tmp/dto_contract_recommendations.md` (v1 - had critical schema mismatches)

---

## Critical Issues Identified by Codex

### 1. **HIGH**: SDK vs Backend Measures Structure Mismatch

**Current State**:

| Component | Structure | Evidence |
|-----------|-----------|----------|
| **SDK sends** | `{"measures": {"metrics": {...}, "metadata": {...}}}` | `api_operations.py:612` |
| **Backend expects** | `{"measures": [{"metric1": val1, "metric2": val2}]}` | `configuration_run_service.py:184-191` |
| **Backend stores** | `measures = db.Column(JSONType, default=list)` | `configuration_run.py:36` |
| **Schema defines** | `"measures": {"type": "array", "items": {...}}` | `configuration_run_schema.json:79-92` |

**Problem**: SDK sends nested object, Backend expects array. **This is broken today.**

**Impact**: Backend's `_validate_measures_format()` (line 181) will return `None` for SDK payloads, causing silent failures.

---

### 2. **HIGH**: WorkflowCostSummary Recomputes Totals (Violates "No Transformations")

**v1 Recommendation** (dto_contract_recommendations.md:337):
```python
def aggregate_from_agents(self) -> 'WorkflowCostSummary':
    """Aggregate totals from agent breakdowns."""
    # Override any provided values with aggregated values
    self.total_input_tokens = sum(a.input_tokens for a in self.agent_breakdowns)
    ...
```

**Problem**: This is a **transformation**, not validation. It can mask client errors (e.g., client sends wrong total, backend silently fixes it).

**Correct Approach**: Validate that provided totals match computed totals, raise error if not.

---

### 3. **MEDIUM**: Key Pattern Not Enforced in SDK

**v1 Recommendation**:
- Schema: `"patternProperties": {"^[a-zA-Z_][a-zA-Z0-9_]*$": ...}` (line 53)
- Frontend: Validates key pattern (line 465)
- **SDK**: No key pattern validation ❌

**Problem**: SDK's `MeasuresDict` accepts any string keys (e.g., `"my-metric"`, `"123abc"`), which will fail schema validation downstream.

---

### 4. **MEDIUM**: Boolean/String Value Type Inconsistency

**v1 Recommendation**:
- `MeasureValue`: `number | string | boolean | null` (line 39)
- But `configuration_run_schema.json:51`: `"type": ["number", "null", "string"]` (no boolean)
- Other schemas: May differ

**Problem**: Inconsistent value types across schemas breaks "single source of truth."

---

### 5. **MEDIUM**: Backend MeasuresDict Not Enforced by Pydantic

**v1 Recommendation** (line 218):
```python
class MeasuresDict(dict):  # Plain dict subclass
    MAX_KEYS = 50
```

**Problem**: Pydantic won't invoke this unless you use `@field_validator` or `__get_validators__`. Validation is dead code.

---

### 6. **LOW**: maxProperties Scope

**v1 Recommendation** (line 52):
```json
"maxProperties": 50
```

**Problem**: Applies per measure object in the array, not globally. If you have 10 measure objects with 10 keys each = 100 keys total, no limit enforced.

---

## Resolved Canonical Structure

### Question 1: What is the canonical measures shape?

**Answer**: **Array of MeasureResult objects** (aligns with schema)

**Rationale**:
- Schema defines `"measures": {"type": "array", "items": {...}}`
- Backend stores `measures = db.Column(JSONType, default=list)`
- Backend service expects `list[dict[str, Any]]`
- **SDK is wrong** - it sends nested object, needs to be fixed

**Use Case**: Each configuration run evaluates multiple examples (e.g., 100 test cases). The measures array stores results per example or aggregated batches.

**Why Array?**:
- Flexibility: Can store per-example results or aggregated batches
- Extensibility: Can add different result types (e.g., per-agent breakdowns)
- Schema-compliant: Matches existing backend schema

**Canonical Structure**:
```json
{
  "measures": [
    {
      "accuracy": 0.95,
      "cost": 0.001,
      "input_tokens": 1500,
      "output_tokens": 750,
      "latency": 1.23
    }
  ]
}
```

For multi-agent workflows with per-agent breakdowns:
```json
{
  "measures": [
    {
      "workflow_total_cost": 0.009,
      "workflow_total_tokens": 450,
      "agent_breakdowns": [
        {
          "agent_id": "researcher",
          "agent_name": "Researcher",
          "input_tokens": 100,
          "output_tokens": 50,
          "total_tokens": 150,
          "input_cost": 0.001,
          "output_cost": 0.002,
          "total_cost": 0.003,
          "model_used": "gpt-4o-mini"
        },
        {
          "agent_id": "writer",
          "agent_name": "Writer",
          "input_tokens": 200,
          "output_tokens": 100,
          "total_tokens": 300,
          "input_cost": 0.002,
          "output_cost": 0.004,
          "total_cost": 0.006,
          "model_used": "gpt-4o"
        }
      ]
    }
  ]
}
```

---

### Question 2: Should key regex be enforced in SDK?

**Answer**: **Yes** (aligns with schema)

**Rationale**:
- Schema defines `"patternProperties": {"^[a-zA-Z_][a-zA-Z0-9_]*$": ...}`
- This pattern matches Python identifier syntax (sensible for metric names)
- Frontend validates this pattern
- **SDK must validate too** to prevent downstream failures

**Implementation**: Add to `MeasuresDict.__setitem__()`:
```python
import re

KEY_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

def __setitem__(self, key: str, value: Any) -> None:
    if not KEY_PATTERN.match(key):
        raise ValueError(
            f"Measure key '{key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ "
            "(Python identifier syntax)"
        )
    ...
```

---

### Question 3: Are boolean/string values allowed?

**Answer**: **Numeric only for measures, metadata for other types** (aligns with use case)

**Rationale**:
- Measures are **optimization targets** - must be numeric (int, float)
- Non-numeric data (model name, tags, etc.) belongs in `metadata` or `run_metadata`
- Current schema already specifies `"type": ["number", "null", "string"]` but string support is legacy
- **Tighten to numeric only** for new measures

**Canonical Value Types**:
```json
{
  "MeasureValue": {
    "oneOf": [
      {"type": "number"},
      {"type": "null"}
    ]
  }
}
```

**For non-numeric data**, use configuration run metadata:
```json
{
  "measures": [{"accuracy": 0.95, "cost": 0.001}],
  "metadata": {
    "model_name": "gpt-4o",
    "tags": ["production", "v2.0"],
    "experiment_notes": "First production run"
  }
}
```

---

## Corrected Recommendations

### 1. TraigentSchema - Measures Type Definitions

**File**: `traigent_schema/schemas/evaluation/measures_types.json` (NEW)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://traigent.ai/schemas/measures_types.json",
  "title": "Measures Type Definitions",
  "description": "Shared type definitions for measures validation across SDK, Backend, Frontend",
  "version": "2.0.0",

  "definitions": {
    "MeasureValue": {
      "description": "A single measure value (numeric only for optimization metrics)",
      "oneOf": [
        {"type": "number"},
        {"type": "null"}
      ]
    },

    "MeasureResult": {
      "description": "A single measure result object with cardinality limit and key pattern",
      "type": "object",
      "maxProperties": 50,
      "patternProperties": {
        "^[a-zA-Z_][a-zA-Z0-9_]*$": {
          "$ref": "#/definitions/MeasureValue"
        }
      },
      "additionalProperties": false,
      "x-notes": [
        "Keys must be Python identifier syntax (^[a-zA-Z_][a-zA-Z0-9_]*$)",
        "Values must be numeric (int, float) or null",
        "Maximum 50 keys per measure result object",
        "Non-numeric metadata belongs in run_metadata, not measures"
      ]
    },

    "MeasuresArray": {
      "description": "Array of measure result objects",
      "type": "array",
      "items": {
        "$ref": "#/definitions/MeasureResult"
      },
      "minItems": 1,
      "x-notes": [
        "Each element is a complete set of metrics for one evaluation",
        "Typically contains 1 element (aggregated metrics)",
        "May contain multiple elements for per-example or per-agent results"
      ]
    },

    "AgentCostBreakdown": {
      "description": "Per-agent cost breakdown with arithmetic validation",
      "type": "object",
      "required": [
        "agent_id",
        "agent_name",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "model_used"
      ],
      "properties": {
        "agent_id": {
          "type": "string",
          "minLength": 1,
          "description": "Unique identifier for the agent"
        },
        "agent_name": {
          "type": "string",
          "minLength": 1,
          "description": "Human-readable agent name"
        },
        "input_tokens": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of input tokens consumed"
        },
        "output_tokens": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of output tokens generated"
        },
        "total_tokens": {
          "type": "integer",
          "minimum": 0,
          "description": "Total tokens (must equal input_tokens + output_tokens)"
        },
        "input_cost": {
          "type": "number",
          "minimum": 0,
          "description": "Cost in USD for input tokens"
        },
        "output_cost": {
          "type": "number",
          "minimum": 0,
          "description": "Cost in USD for output tokens"
        },
        "total_cost": {
          "type": "number",
          "minimum": 0,
          "description": "Total cost in USD (must equal input_cost + output_cost)"
        },
        "model_used": {
          "type": "string",
          "minLength": 1,
          "description": "Model identifier (e.g., 'gpt-4o-mini')"
        }
      },
      "additionalProperties": false,
      "x-validation": {
        "arithmetic": [
          "total_tokens == input_tokens + output_tokens",
          "abs(total_cost - (input_cost + output_cost)) <= 0.0001"
        ]
      }
    }
  }
}
```

**Update**: `configuration_run_schema.json`

```json
{
  "measures": {
    "oneOf": [
      {
        "$ref": "measures_types.json#/definitions/MeasuresArray",
        "description": "Array of measure results"
      },
      {
        "type": "null"
      }
    ],
    "description": "Results from measure evaluations for this configuration"
  }
}
```

---

### 2. TraigentSDK - Fix Measures Structure

**File**: `traigent/cloud/api_operations.py` (MODIFY lines 611-618)

**BEFORE** (WRONG):
```python
# Build measures data in the correct Traigent format
measures_data = {"measures": {"metrics": mapped_metrics, "metadata": {}}}

# Add execution time if provided
if execution_time is not None:
    measures_data["measures"]["metadata"]["execution_time"] = float(execution_time)
```

**AFTER** (CORRECT):
```python
# Build measures data in schema-compliant array format
# Each configuration run has one aggregated measure result object
measure_result = dict(mapped_metrics)  # Copy to avoid mutation

measures_data = {"measures": [measure_result]}

# Add execution time to run_metadata (separate from measures)
if execution_time is not None:
    # Note: execution_time should be sent via separate metadata endpoint
    # For now, include in measure_result for backward compatibility
    logger.debug(
        f"execution_time should be sent via metadata endpoint, "
        f"not in measures (found: {execution_time})"
    )
```

---

### 3. TraigentSDK - Add Key Pattern Validation

**File**: `traigent/cloud/dtos.py` (MODIFY MeasuresDict)

**Add import**:
```python
import re
```

**Add class constant and validation**:
```python
class MeasuresDict(UserDict):
    """Type-safe measures dict with validation.

    Enforces:
    - Maximum 50 keys to prevent unbounded memory usage
    - String keys matching Python identifier pattern (^[a-zA-Z_][a-zA-Z0-9_]*$)
    - Numeric value types only (int, float, None) for optimization metrics
    """

    MAX_KEYS = 50
    KEY_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        if data:
            self._validate_dict(data)
            self.data.update(data)

    def _validate_dict(self, data: dict[str, Any]) -> None:
        """Validate measures format."""
        if len(data) > self.MAX_KEYS:
            raise ValueError(
                f"Measures cannot exceed {self.MAX_KEYS} keys, got {len(data)}"
            )

        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Measure key must be string, got {type(key).__name__}"
                )

            # NEW: Validate key pattern (Python identifier syntax)
            if not self.KEY_PATTERN.match(key):
                raise ValueError(
                    f"Measure key '{key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ "
                    f"(Python identifier syntax). "
                    f"Invalid characters: use underscores instead of hyphens/spaces."
                )

            # NEW: Tighten value types to numeric only (no string, no boolean)
            if not isinstance(value, (int, float, type(None))):
                raise TypeError(
                    f"Measure '{key}' must be numeric type (int, float, None), "
                    f"got {type(value).__name__}. "
                    f"Non-numeric data should be stored in configuration run metadata."
                )

    def __setitem__(self, key: str, value: Any) -> None:
        """Validate on assignment."""
        if len(self.data) >= self.MAX_KEYS and key not in self.data:
            raise ValueError(f"Measures cannot exceed {self.MAX_KEYS} keys")

        if not isinstance(key, str):
            raise TypeError(f"Key must be string, got {type(key).__name__}")

        # NEW: Validate key pattern
        if not self.KEY_PATTERN.match(key):
            raise ValueError(
                f"Measure key '{key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ "
                f"(Python identifier syntax)"
            )

        # NEW: Tighten value types to numeric only
        if not isinstance(value, (int, float, type(None))):
            raise TypeError(
                f"Value must be numeric type (int, float, None), "
                f"got {type(value).__name__}"
            )

        self.data[key] = value
```

---

### 4. TraigentSDK - Fix WorkflowCostSummary (Validate, Don't Transform)

**File**: `traigent/cloud/agent_dtos.py` (MODIFY WorkflowCostSummary)

**BEFORE** (WRONG - transforms data):
```python
@model_validator(mode='after')
def aggregate_from_agents(self) -> 'WorkflowCostSummary':
    """Aggregate totals from agent breakdowns."""
    # Override any provided values with aggregated values
    self.total_input_tokens = sum(a.input_tokens for a in self.agent_breakdowns)
    ...
```

**AFTER** (CORRECT - validates data):
```python
def __post_init__(self) -> None:
    """Validate and verify aggregation (no transformations)."""
    errors: list[str] = []

    if not self.workflow_id:
        errors.append("workflow_id cannot be empty")
    if not self.workflow_name:
        errors.append("workflow_name cannot be empty")
    if not self.agent_breakdowns:
        errors.append("agent_breakdowns cannot be empty")

    if errors:
        raise ValueError(
            f"WorkflowCostSummary validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    # Compute expected totals from agent breakdowns
    expected_input_tokens = sum(a.input_tokens for a in self.agent_breakdowns)
    expected_output_tokens = sum(a.output_tokens for a in self.agent_breakdowns)
    expected_total_tokens = sum(a.total_tokens for a in self.agent_breakdowns)
    expected_input_cost = sum(a.input_cost for a in self.agent_breakdowns)
    expected_output_cost = sum(a.output_cost for a in self.agent_breakdowns)
    expected_total_cost = sum(a.total_cost for a in self.agent_breakdowns)

    # Validate that provided totals match computed totals (no recomputation!)
    if self.total_input_tokens != expected_input_tokens:
        errors.append(
            f"total_input_tokens ({self.total_input_tokens}) must equal "
            f"sum of agent input_tokens ({expected_input_tokens})"
        )

    if self.total_output_tokens != expected_output_tokens:
        errors.append(
            f"total_output_tokens ({self.total_output_tokens}) must equal "
            f"sum of agent output_tokens ({expected_output_tokens})"
        )

    if self.total_tokens != expected_total_tokens:
        errors.append(
            f"total_tokens ({self.total_tokens}) must equal "
            f"sum of agent total_tokens ({expected_total_tokens})"
        )

    # Use floating point tolerance for costs
    if abs(self.total_input_cost - expected_input_cost) > 0.0001:
        errors.append(
            f"total_input_cost ({self.total_input_cost:.8f}) must equal "
            f"sum of agent input_costs ({expected_input_cost:.8f})"
        )

    if abs(self.total_output_cost - expected_output_cost) > 0.0001:
        errors.append(
            f"total_output_cost ({self.total_output_cost:.8f}) must equal "
            f"sum of agent output_costs ({expected_output_cost:.8f})"
        )

    if abs(self.total_cost - expected_total_cost) > 0.0001:
        errors.append(
            f"total_cost ({self.total_cost:.8f}) must equal "
            f"sum of agent total_costs ({expected_total_cost:.8f})"
        )

    if errors:
        raise ValueError(
            f"WorkflowCostSummary aggregation validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )
```

---

### 5. TraigentBackend - Fix Pydantic Integration

**File**: `src/models/measures_dtos.py` (NEW - v2 with proper Pydantic integration)

```python
"""Measures DTOs with strict validation matching SDK.

These models enforce the same validation rules as the SDK to maintain
DTO contract consistency across all layers.
"""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MeasuresDict(dict):
    """Type-safe measures dict with validation.

    Note: This is a dict subclass for JSON serialization compatibility.
    Validation is enforced via Pydantic field_validator.
    """

    MAX_KEYS = 50
    KEY_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    @classmethod
    def validate(cls, data: dict[str, Any]) -> 'MeasuresDict':
        """Validate measures dict (used by Pydantic)."""
        if not isinstance(data, dict):
            raise TypeError(f"Measures must be dict, got {type(data).__name__}")

        if len(data) > cls.MAX_KEYS:
            raise ValueError(
                f"Measures cannot exceed {cls.MAX_KEYS} keys, got {len(data)}"
            )

        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Measure key must be string, got {type(key).__name__}"
                )

            if not cls.KEY_PATTERN.match(key):
                raise ValueError(
                    f"Measure key '{key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
                )

            if not isinstance(value, (int, float, type(None))):
                raise TypeError(
                    f"Measure '{key}' must be numeric (int, float, None), "
                    f"got {type(value).__name__}"
                )

        return cls(data)


class AgentCostBreakdown(BaseModel):
    """Per-agent cost breakdown with arithmetic validation."""

    agent_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    input_cost: float = Field(..., ge=0)
    output_cost: float = Field(..., ge=0)
    total_cost: float = Field(..., ge=0)
    model_used: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def validate_arithmetic(self) -> 'AgentCostBreakdown':
        """Validate token and cost arithmetic (no transformations)."""
        import math

        errors = []

        # Validate token arithmetic
        expected_total_tokens = self.input_tokens + self.output_tokens
        if self.total_tokens != expected_total_tokens:
            errors.append(
                f"total_tokens ({self.total_tokens}) must equal "
                f"input_tokens ({self.input_tokens}) + "
                f"output_tokens ({self.output_tokens}) = {expected_total_tokens}"
            )

        # Validate cost arithmetic (with floating point tolerance)
        expected_total_cost = self.input_cost + self.output_cost
        if abs(self.total_cost - expected_total_cost) > 0.0001:
            errors.append(
                f"total_cost ({self.total_cost:.8f}) must equal "
                f"input_cost ({self.input_cost:.8f}) + "
                f"output_cost ({self.output_cost:.8f}) = {expected_total_cost:.8f}"
            )

        # Validate no NaN/Inf
        for field_name, value in [
            ("input_cost", self.input_cost),
            ("output_cost", self.output_cost),
            ("total_cost", self.total_cost),
        ]:
            if math.isnan(value) or math.isinf(value):
                errors.append(f"{field_name} cannot be NaN or Inf")

        if errors:
            raise ValueError(
                "AgentCostBreakdown validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return self
```

**Update**: `src/services/configuration_run_service.py`

```python
from src.models.measures_dtos import MeasuresDict

def _validate_measures_format(self, measures: list | dict[str, Any]) -> list | None:
    """Validate and normalize measures to schema-compliant format."""
    # Must be a list according to schema
    if not isinstance(measures, list):
        logger.error(f"Measures not in array format: {type(measures)}")
        return None

    # Validate each measure result object with strict validation
    validated_measures = []
    for i, measure_result in enumerate(measures):
        if not isinstance(measure_result, dict):
            logger.error(f"Measure result {i} is not a dictionary: {type(measure_result)}")
            return None

        try:
            # Use MeasuresDict.validate() for strict validation
            validated = MeasuresDict.validate(measure_result)
            validated_measures.append(dict(validated))
        except (ValueError, TypeError) as e:
            logger.error(f"Measure result {i} validation failed: {e}")
            return None

    return validated_measures
```

---

### 6. TraigentFrontend - TypeScript Interfaces

**File**: `src/types/measures.ts` (NEW - v2 with corrected structure)

```typescript
/**
 * Measures type definitions matching TraigentSchema v2.0.0.
 *
 * Key changes from v1:
 * - Measures is array, not nested object
 * - Values are numeric only (no boolean/string)
 * - Key pattern enforced (^[a-zA-Z_][a-zA-Z0-9_]*$)
 */

/**
 * A single measure value (numeric only for optimization metrics).
 */
export type MeasureValue = number | null;

/**
 * A single measure result object.
 *
 * Enforces:
 * - Maximum 50 keys
 * - Keys matching pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ (Python identifier)
 * - Numeric values only (int, float, null)
 */
export interface MeasureResult {
  [key: string]: MeasureValue;
}

/**
 * Array of measure result objects (canonical structure).
 */
export type MeasuresArray = MeasureResult[];

/**
 * Per-agent cost breakdown with arithmetic validation.
 */
export interface AgentCostBreakdown {
  agent_id: string;
  agent_name: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_cost: number;
  output_cost: number;
  total_cost: number;
  model_used: string;
}

/**
 * Validation utilities for measures.
 */
export class MeasuresValidator {
  static readonly MAX_KEYS = 50;
  static readonly KEY_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

  /**
   * Validate measure result object.
   *
   * @throws {Error} If validation fails
   */
  static validateMeasureResult(measures: Record<string, any>): MeasureResult {
    const keys = Object.keys(measures);

    // Check cardinality limit
    if (keys.length > this.MAX_KEYS) {
      throw new Error(
        `Measures cannot exceed ${this.MAX_KEYS} keys, got ${keys.length}`
      );
    }

    // Validate each key-value pair
    for (const [key, value] of Object.entries(measures)) {
      // Validate key format (Python identifier syntax)
      if (!this.KEY_PATTERN.test(key)) {
        throw new Error(
          `Measure key '${key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ ` +
          `(Python identifier syntax). Use underscores instead of hyphens/spaces.`
        );
      }

      // Validate value type (numeric only)
      if (value !== null && typeof value !== 'number') {
        throw new Error(
          `Measure '${key}' must be numeric type (number, null), ` +
          `got ${typeof value}. ` +
          `Non-numeric data should be stored in configuration run metadata.`
        );
      }

      // Validate numeric values (no NaN or Infinity)
      if (typeof value === 'number' && (!isFinite(value) || isNaN(value))) {
        throw new Error(`Measure '${key}' cannot be NaN or Infinity`);
      }
    }

    return measures as MeasureResult;
  }

  /**
   * Validate measures array.
   *
   * @throws {Error} If validation fails
   */
  static validateMeasuresArray(measures: any): MeasuresArray {
    if (!Array.isArray(measures)) {
      throw new Error(
        `Measures must be an array, got ${typeof measures}. ` +
        `Expected format: [{"accuracy": 0.95, "cost": 0.001}]`
      );
    }

    if (measures.length === 0) {
      throw new Error('Measures array cannot be empty');
    }

    return measures.map((result, index) => {
      try {
        return this.validateMeasureResult(result);
      } catch (error) {
        throw new Error(
          `Measure result ${index} validation failed: ${error.message}`
        );
      }
    });
  }

  /**
   * Validate agent cost breakdown arithmetic.
   *
   * @throws {Error} If validation fails
   */
  static validateAgentCostBreakdown(breakdown: AgentCostBreakdown): void {
    const errors: string[] = [];

    // Validate identifiers
    if (!breakdown.agent_id || breakdown.agent_id.trim() === '') {
      errors.push('agent_id cannot be empty');
    }
    if (!breakdown.agent_name || breakdown.agent_name.trim() === '') {
      errors.push('agent_name cannot be empty');
    }
    if (!breakdown.model_used || breakdown.model_used.trim() === '') {
      errors.push('model_used cannot be empty');
    }

    // Validate token arithmetic
    const expectedTotalTokens = breakdown.input_tokens + breakdown.output_tokens;
    if (breakdown.total_tokens !== expectedTotalTokens) {
      errors.push(
        `total_tokens (${breakdown.total_tokens}) must equal ` +
        `input_tokens (${breakdown.input_tokens}) + ` +
        `output_tokens (${breakdown.output_tokens}) = ${expectedTotalTokens}`
      );
    }

    // Validate cost arithmetic (with floating point tolerance)
    const expectedTotalCost = breakdown.input_cost + breakdown.output_cost;
    if (Math.abs(breakdown.total_cost - expectedTotalCost) > 0.0001) {
      errors.push(
        `total_cost (${breakdown.total_cost.toFixed(8)}) must equal ` +
        `input_cost (${breakdown.input_cost.toFixed(8)}) + ` +
        `output_cost (${breakdown.output_cost.toFixed(8)}) = ${expectedTotalCost.toFixed(8)}`
      );
    }

    if (errors.length > 0) {
      throw new Error(
        'AgentCostBreakdown validation failed:\n' +
        errors.map(e => `  - ${e}`).join('\n')
      );
    }
  }
}
```

---

## Migration Path

### Phase 0: Fix SDK Measures Structure (CRITICAL - Do First)

**Estimated**: 1 day

**Tasks**:
1. ✅ Update `api_operations.py` to send measures as array (lines 611-618)
2. ✅ Add key pattern validation to `MeasuresDict` (regex ^[a-zA-Z_][a-zA-Z0-9_]*$)
3. ✅ Tighten value types to numeric only (int, float, None)
4. ✅ Fix `WorkflowCostSummary` to validate, not transform
5. ✅ Update tests for new structure
6. ✅ Run SDK tests to verify

**Critical**: This fixes the **broken SDK → Backend contract** identified by Codex.

---

### Phase 1: Schema Updates

**Estimated**: 1 day

**Tasks**:
1. Create `measures_types.json` with corrected definitions
2. Update `configuration_run_schema.json` to reference shared types
3. Add `maxProperties`, `patternProperties`, numeric-only value types
4. Update other schemas that reference measures (metric_submission_schema.json, etc.)

---

### Phase 2: Backend Updates

**Estimated**: 2 days

**Tasks**:
1. Create `src/models/measures_dtos.py` with Pydantic models
2. Update `configuration_run_service.py` to use strict validation via `MeasuresDict.validate()`
3. Update error handling to return structured errors
4. Add unit tests for validation logic

---

### Phase 3: Frontend Updates

**Estimated**: 2 days

**Tasks**:
1. Create `src/types/measures.ts` with TypeScript interfaces (v2 structure)
2. Add `MeasuresValidator` utility class
3. Update API service to validate before sending
4. Update components to render array structure
5. Add unit tests for validation logic

---

### Phase 4: Integration Testing

**Estimated**: 1 day

**Tasks**:
1. Test SDK → Backend validation (array structure)
2. Test Frontend → Backend validation
3. Test error handling across layers
4. Verify consistent error messages
5. Test multi-agent workflow cost tracking end-to-end

---

## Summary of Changes from v1

| Issue | v1 Approach | v2 Fix |
|-------|-------------|--------|
| **Measures structure** | Kept nested object | Changed to array (matches schema) |
| **SDK sends** | `{"measures": {"metrics": {...}}}` | `{"measures": [{...}]}` |
| **WorkflowCostSummary** | Recomputes totals | Validates totals, raises error if mismatch |
| **Key pattern** | Schema/FE only | Added to SDK MeasuresDict |
| **Value types** | Boolean/string allowed | Numeric only (int, float, None) |
| **Backend MeasuresDict** | Plain dict subclass | Added `validate()` classmethod for Pydantic |
| **maxProperties scope** | Per object (correct) | Documented scope clearly |

---

## Estimated Timeline

- **Phase 0 (SDK fixes)**: 1 day ← **DO THIS FIRST**
- **Phase 1 (Schema)**: 1 day
- **Phase 2 (Backend)**: 2 days
- **Phase 3 (Frontend)**: 2 days
- **Phase 4 (Testing)**: 1 day

**Total**: 7 days (includes critical SDK fix)

---

## Conclusion

**Key Insight from Codex**: The SDK is currently sending the wrong structure to the Backend. This must be fixed **before** implementing cross-project DTO alignment.

**Proceed with**:
1. **Immediate**: Fix SDK measures structure (Phase 0)
2. **Then**: Implement cross-project DTO alignment (Phases 1-4)

This establishes proper DTO contract architecture where:
- Schema is single source of truth
- All layers use same structure and validation
- No transformations (validation only)
- Fail-fast at every layer
