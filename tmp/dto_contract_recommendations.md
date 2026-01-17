# DTO Contract Recommendations - Cross-Project Alignment

## Principle: Shared Types, Shared Validation

TraigentSchema serves as the single source of truth for DTO contracts. All projects (SDK, Backend, Frontend) should:
1. Use the same type definitions
2. Enforce the same validation rules
3. Avoid transformations (data flows through unchanged)
4. Fail fast with clear errors

## Current State: Inconsistent Validation

| Layer | Validation Approach | Problem |
|-------|---------------------|---------|
| **SDK** | Strict - MeasuresDict enforces 50-key limit, primitive types only, raises MetricExtractionError | ✅ Good |
| **Backend** | Lenient - converts invalid values to `None`, no cardinality limit, no type enforcement | ❌ Inconsistent |
| **Frontend** | Generic - no validation, renders any data type | ❌ No contract enforcement |

**Problem**: SDK is strict, but Backend accepts anything. If SDK validation is bypassed (direct API call), Backend accepts invalid data.

---

## Recommended Changes

### 1. TraigentSchema - Add Type Definitions

**File**: `/home/nimrodbu/Traigent_enterprise/TraigentSchema/traigent_schema/schemas/evaluation/measures_types.json`

Create shared type definitions for measures:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://traigent.ai/schemas/measures_types.json",
  "title": "Measures Type Definitions",
  "description": "Shared type definitions for measures validation across SDK, Backend, Frontend",

  "definitions": {
    "MeasureValue": {
      "description": "A single measure value (primitive types only)",
      "oneOf": [
        {"type": "number"},
        {"type": "string"},
        {"type": "boolean"},
        {"type": "null"}
      ]
    },

    "MeasureResult": {
      "description": "A single measure result object with cardinality limit",
      "type": "object",
      "maxProperties": 50,
      "patternProperties": {
        "^[a-zA-Z_][a-zA-Z0-9_]*$": {
          "$ref": "#/definitions/MeasureValue"
        }
      },
      "additionalProperties": false
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
          "description": "Total tokens (must equal input + output)"
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
          "description": "Total cost in USD (must equal input + output)"
        },
        "model_used": {
          "type": "string",
          "minLength": 1,
          "description": "Model identifier (e.g., 'gpt-4o-mini')"
        }
      },
      "additionalProperties": false
    },

    "WorkflowCostSummary": {
      "description": "Aggregated cost summary across multiple agents",
      "type": "object",
      "required": [
        "workflow_id",
        "workflow_name",
        "agent_breakdowns",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_input_cost",
        "total_output_cost",
        "total_cost"
      ],
      "properties": {
        "workflow_id": {
          "type": "string",
          "minLength": 1
        },
        "workflow_name": {
          "type": "string",
          "minLength": 1
        },
        "agent_breakdowns": {
          "type": "array",
          "minItems": 1,
          "items": {
            "$ref": "#/definitions/AgentCostBreakdown"
          }
        },
        "total_input_tokens": {
          "type": "integer",
          "minimum": 0
        },
        "total_output_tokens": {
          "type": "integer",
          "minimum": 0
        },
        "total_tokens": {
          "type": "integer",
          "minimum": 0
        },
        "total_input_cost": {
          "type": "number",
          "minimum": 0
        },
        "total_output_cost": {
          "type": "number",
          "minimum": 0
        },
        "total_cost": {
          "type": "number",
          "minimum": 0
        }
      },
      "additionalProperties": false
    }
  }
}
```

**Update**: `configuration_run_schema.json` to reference shared types:

```json
{
  "measures": {
    "type": "array",
    "description": "Array of measure results",
    "items": {
      "$ref": "measures_types.json#/definitions/MeasureResult"
    }
  }
}
```

---

### 2. TraigentBackend - Add Typed Models

**File**: `/home/nimrodbu/Traigent_enterprise/TraigentBackend/src/models/measures_dtos.py` (NEW)

Create Pydantic models matching SDK DTOs:

```python
"""Measures DTOs with strict validation.

These models match the SDK's DTOs and enforce the same validation rules
to maintain DTO contract consistency across all layers.
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator

class MeasuresDict(dict):
    """Type-safe measures dict with validation.

    Enforces:
    - Maximum 50 keys (cardinality limit)
    - String keys only
    - Primitive value types only (int, float, str, bool, None)
    """

    MAX_KEYS = 50

    def __init__(self, data: dict[str, Any] | None = None):
        if data:
            self._validate_dict(data)
        super().__init__(data or {})

    def _validate_dict(self, data: dict[str, Any]) -> None:
        """Validate measures format."""
        if len(data) > self.MAX_KEYS:
            raise ValueError(
                f"Measures cannot exceed {self.MAX_KEYS} keys, got {len(data)}"
            )

        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f"Measure key must be string, got {type(key).__name__}")

            if not isinstance(value, (int, float, str, bool, type(None))):
                raise TypeError(
                    f"Measure '{key}' must be primitive type "
                    f"(int, float, str, bool, None), got {type(value).__name__}"
                )

    def __setitem__(self, key: str, value: Any) -> None:
        """Validate on assignment."""
        if len(self) >= self.MAX_KEYS and key not in self:
            raise ValueError(f"Measures cannot exceed {self.MAX_KEYS} keys")

        if not isinstance(key, str):
            raise TypeError(f"Key must be string, got {type(key).__name__}")

        if not isinstance(value, (int, float, str, bool, type(None))):
            raise TypeError(
                f"Value must be primitive type (int, float, str, bool, None), "
                f"got {type(value).__name__}"
            )

        super().__setitem__(key, value)


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
        """Validate token and cost arithmetic."""
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


class WorkflowCostSummary(BaseModel):
    """Aggregated cost summary across multiple agents."""

    workflow_id: str = Field(..., min_length=1)
    workflow_name: str = Field(..., min_length=1)
    agent_breakdowns: list[AgentCostBreakdown] = Field(..., min_length=1)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_input_cost: float = Field(default=0.0, ge=0)
    total_output_cost: float = Field(default=0.0, ge=0)
    total_cost: float = Field(default=0.0, ge=0)

    @model_validator(mode='after')
    def aggregate_from_agents(self) -> 'WorkflowCostSummary':
        """Aggregate totals from agent breakdowns."""
        # Override any provided values with aggregated values
        self.total_input_tokens = sum(a.input_tokens for a in self.agent_breakdowns)
        self.total_output_tokens = sum(a.output_tokens for a in self.agent_breakdowns)
        self.total_tokens = sum(a.total_tokens for a in self.agent_breakdowns)
        self.total_input_cost = sum(a.input_cost for a in self.agent_breakdowns)
        self.total_output_cost = sum(a.output_cost for a in self.agent_breakdowns)
        self.total_cost = sum(a.total_cost for a in self.agent_breakdowns)

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
            # Use MeasuresDict for strict validation
            validated = MeasuresDict(measure_result)
            validated_measures.append(dict(validated))
        except (ValueError, TypeError) as e:
            logger.error(f"Measure result {i} validation failed: {e}")
            return None

    return validated_measures
```

---

### 3. TraigentFrontend - Add TypeScript Interfaces

**File**: `/home/nimrodbu/Traigent_enterprise/TraigentFrontend/src/types/measures.ts` (NEW)

```typescript
/**
 * Measures type definitions matching TraigentSchema and SDK DTOs.
 *
 * These types enforce the DTO contract on the frontend to catch
 * validation errors before sending to backend.
 */

/**
 * Primitive value types allowed in measures.
 */
export type MeasureValue = number | string | boolean | null;

/**
 * A single measure result object with cardinality limit.
 *
 * Enforces:
 * - Maximum 50 keys
 * - String keys matching pattern ^[a-zA-Z_][a-zA-Z0-9_]*$
 * - Primitive value types only
 */
export interface MeasureResult {
  [key: string]: MeasureValue;
}

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
 * Aggregated cost summary across multiple agents.
 */
export interface WorkflowCostSummary {
  workflow_id: string;
  workflow_name: string;
  agent_breakdowns: AgentCostBreakdown[];
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  total_input_cost: number;
  total_output_cost: number;
  total_cost: number;
}

/**
 * Validation utilities for measures.
 */
export class MeasuresValidator {
  static readonly MAX_KEYS = 50;

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
      // Validate key format
      if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(key)) {
        throw new Error(
          `Measure key '${key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$`
        );
      }

      // Validate value type
      const valueType = typeof value;
      if (
        valueType !== 'number' &&
        valueType !== 'string' &&
        valueType !== 'boolean' &&
        value !== null
      ) {
        throw new Error(
          `Measure '${key}' must be primitive type (number, string, boolean, null), ` +
          `got ${valueType}`
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

**Update**: `src/services/api.ts` to use typed validation:

```typescript
import { MeasuresValidator, MeasureResult } from '@/types/measures';

// In configuration run submission
export const submitConfigurationRun = async (data: {
  experiment_run_id: string;
  experiment_parameters: Record<string, any>;
  measures: MeasureResult[];
}) => {
  // Validate measures before sending
  for (let i = 0; i < data.measures.length; i++) {
    try {
      MeasuresValidator.validateMeasureResult(data.measures[i]);
    } catch (error) {
      throw new Error(
        `Measure result ${i} validation failed: ${error.message}`
      );
    }
  }

  // Send to backend
  return await api.post('/api/v1/configuration-runs', data);
};
```

---

## Benefits of This Approach

### 1. **True DTO Contract**
- SDK, Backend, Frontend all use the same type definitions
- Schema is single source of truth
- No transformations = data flows through unchanged

### 2. **Fail-Fast Validation**
- Invalid data caught at all layers
- Clear error messages with context
- No silent failures or data corruption

### 3. **Type Safety**
- Python: Pydantic models with runtime validation
- TypeScript: Compile-time type checking + runtime validation
- Schema: JSON Schema for API contract

### 4. **Maintainability**
- Types defined once in Schema
- Changes propagate to all layers
- Consistent validation logic

### 5. **Pre-Release Advantage**
- Breaking changes acceptable now
- Establish good patterns before GA
- Easier to maintain long-term

---

## Migration Path

### Phase 1: Schema Updates (1 day)
1. Create `measures_types.json` with shared type definitions
2. Update `configuration_run_schema.json` to reference shared types
3. Add `maxProperties: 50` cardinality limit

### Phase 2: Backend Updates (2 days)
1. Create `src/models/measures_dtos.py` with Pydantic models
2. Update `configuration_run_service.py` to use strict validation
3. Update error handling to return structured errors
4. Add unit tests for validation logic

### Phase 3: Frontend Updates (2 days)
1. Create `src/types/measures.ts` with TypeScript interfaces
2. Add `MeasuresValidator` utility class
3. Update API service to validate before sending
4. Add specialized rendering for multi-agent metrics
5. Add unit tests for validation logic

### Phase 4: Integration Testing (1 day)
1. Test SDK → Backend validation
2. Test Frontend → Backend validation
3. Test error handling across layers
4. Verify consistent error messages

---

## Open Questions

1. **Schema Versioning**: Should we version the schemas (v1, v2) for backward compatibility?
2. **Error Response Format**: Should Backend return structured validation errors matching SDK's exception format?
3. **Frontend Validation Mode**: Should FE validation be strict by default or configurable?
4. **Code Generation**: Should we generate Backend/Frontend types from Schema automatically?

---

## Recommendation

**Proceed with cross-project DTO alignment** because:
- You own all layers (SDK, Backend, Frontend)
- Pre-release = breaking changes acceptable
- Establishes proper architecture for long-term maintainability
- Prevents data corruption and silent failures
- Enforces true DTO contract across all layers

**Estimated effort**: 6 days total (1 Schema + 2 Backend + 2 Frontend + 1 Testing)

This is the right time to do it - before GA release!
