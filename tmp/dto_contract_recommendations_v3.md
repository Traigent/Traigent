# DTO Contract Recommendations v3 - Addressing All Review Findings

**Date**: 2026-01-16
**Version**: 3.0 (Final - Addresses Codex + Gemini feedback)
**Status**: 🟢 Ready for implementation
**Previous**: v2 had schema inconsistencies and incomplete enforcement

---

## Critical Changes from v2

| Issue | v2 Problem | v3 Solution |
|-------|------------|-------------|
| **agent_breakdowns location** | Embedded in numeric-only measures (schema violation) | New top-level `workflow_metadata` field |
| **WorkflowCostSummary breaking** | Required explicit totals (defaults to 0) | Added factory method `from_agents()` |
| **Incomplete numeric enforcement** | Only MeasuresDict, missed other paths | All SDK paths enforce numeric-only |
| **Empty measures handling** | Schema requires minItems: 1 | Made measures optional, omit when empty |
| **Pydantic integration** | Validation not wired properly | Proper `@field_validator` example |
| **execution_time** | Removed without alternative | Added to `workflow_metadata` |
| **sync_manager dict format** | Still emits dict | Updated to emit array |

---

## Resolved Open Questions

### Q1: Where should multi-agent breakdowns live?

**Answer**: **New top-level `workflow_metadata` field** in ConfigurationRun

**Rationale**:
- Keeps `measures` purely numeric for optimization
- Provides dedicated space for workflow-specific data
- Avoids schema violations
- Backend can index and query workflow data efficiently

**Schema Change Required**: ✅ Yes - add `workflow_metadata` to configuration_run_schema.json

**Structure**:
```json
{
  "measures": [
    {
      "accuracy": 0.95,
      "total_cost": 0.009,
      "total_tokens": 450
    }
  ],
  "workflow_metadata": {
    "workflow_id": "research-write-workflow",
    "workflow_name": "Research + Write",
    "execution_time": 12.34,
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
      }
    ]
  }
}
```

---

### Q2: Numeric-only or preserve non-numeric metrics?

**Answer**: **Numeric-only for measures, metadata for other data**

**Rationale**:
- Measures are optimization targets - must be numeric
- Non-numeric data (model names, tags) belongs in `run_metadata` or `workflow_metadata`
- Clear separation of concerns
- Better for optimization algorithms

**Migration Strategy**:
1. Phase 0: Allow both (backward compatible)
2. Phase 1: Log warnings for non-numeric
3. Phase 2 (v2.0): Enforce numeric-only

---

### Q3: WorkflowCostSummary - Required totals or factory?

**Answer**: **Both - provide factory method for convenience**

**Implementation**:
```python
@dataclass
class WorkflowCostSummary:
    workflow_id: str
    workflow_name: str
    agent_breakdowns: list[AgentCostBreakdown]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_input_cost: float
    total_output_cost: float
    total_cost: float

    @classmethod
    def from_agents(
        cls,
        workflow_id: str,
        workflow_name: str,
        agent_breakdowns: list[AgentCostBreakdown],
    ) -> 'WorkflowCostSummary':
        """Factory method that computes totals from agent breakdowns."""
        return cls(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            agent_breakdowns=agent_breakdowns,
            total_input_tokens=sum(a.input_tokens for a in agent_breakdowns),
            total_output_tokens=sum(a.output_tokens for a in agent_breakdowns),
            total_tokens=sum(a.total_tokens for a in agent_breakdowns),
            total_input_cost=sum(a.input_cost for a in agent_breakdowns),
            total_output_cost=sum(a.output_cost for a in agent_breakdowns),
            total_cost=sum(a.total_cost for a in agent_breakdowns),
        )

    def __post_init__(self) -> None:
        """Validate totals match agent breakdowns (strict validation)."""
        # Validation logic here - raises ValueError on mismatch
```

**Usage**:
```python
# Easy way - factory computes totals
summary = WorkflowCostSummary.from_agents(
    workflow_id="wf-123",
    workflow_name="Research + Write",
    agent_breakdowns=[agent1, agent2],
)

# Explicit way - you provide totals, we validate
summary = WorkflowCostSummary(
    workflow_id="wf-123",
    workflow_name="Research + Write",
    agent_breakdowns=[agent1, agent2],
    total_input_tokens=300,  # Must match sum
    total_output_tokens=150,
    total_tokens=450,
    total_input_cost=0.003,
    total_output_cost=0.006,
    total_cost=0.009,
)
```

---

### Q4: Is `measures: []` valid?

**Answer**: **No - omit measures field when empty**

**Rationale**:
- Optional field is clearer than empty array
- Reduces payload size
- Backend can distinguish "not yet evaluated" (null) from "evaluated with no results" (omitted)

**Schema Change**:
```json
{
  "measures": {
    "oneOf": [
      {
        "type": "array",
        "items": {"$ref": "#/definitions/MeasureResult"},
        "minItems": 1
      },
      {"type": "null"}
    ]
  }
}
```

**SDK Behavior**:
```python
# If no measures, send null or omit field
if not measure_results:
    payload["measures"] = None  # or: del payload["measures"]
else:
    payload["measures"] = [measure_results]
```

---

## Phase 0: Critical SDK Fixes (Day 1)

**Goal**: Fix broken SDK → Backend contract without breaking existing code

### Task 0.1: Fix api_operations.py Measures Structure

**File**: `traigent/cloud/api_operations.py` (lines 611-618)

**BEFORE** (BROKEN):
```python
# Build measures data in the correct Traigent format
measures_data = {"measures": {"metrics": mapped_metrics, "metadata": {}}}

# Add execution time if provided
if execution_time is not None:
    measures_data["measures"]["metadata"]["execution_time"] = float(execution_time)
```

**AFTER** (FIXED):
```python
# Build measures data in schema-compliant array format
if mapped_metrics:
    measure_result = dict(mapped_metrics)
    measures_data = {"measures": [measure_result]}
else:
    # Omit measures if empty (per schema - null or omit)
    measures_data = {"measures": None}

# Add workflow metadata if available (separate from measures)
if execution_time is not None:
    # TODO: Send via workflow_metadata field once backend supports it
    # For now, log a debug message
    logger.debug(
        f"execution_time={execution_time} should be sent via workflow_metadata field "
        f"(not yet supported by backend)"
    )
```

---

### Task 0.2: Add Key Pattern Validation to MeasuresDict

**File**: `traigent/cloud/dtos.py` (MeasuresDict class)

**Add**:
```python
import re

class MeasuresDict(UserDict):
    """Type-safe measures dict with validation.

    Enforces:
    - Maximum 50 keys to prevent unbounded memory usage
    - String keys matching Python identifier pattern (^[a-zA-Z_][a-zA-Z0-9_]*$)
    - Numeric value types only (int, float, None) for optimization metrics
    """

    MAX_KEYS = 50
    KEY_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

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
                    f"Use underscores instead of hyphens or spaces. "
                    f"Invalid: 'my-metric', '123abc'. Valid: 'my_metric', 'metric_123'."
                )

            # NEW: Enforce numeric-only (Phase 0: log warning, Phase 2: enforce)
            if not isinstance(value, (int, float, type(None))):
                logger.warning(
                    f"Measure '{key}' has non-numeric value type {type(value).__name__}. "
                    f"Non-numeric metrics will be rejected in Traigent v2.0. "
                    f"Store non-numeric data in configuration run metadata instead.",
                    extra={
                        "key": key,
                        "value_type": type(value).__name__,
                        "hint": "Use run_metadata or workflow_metadata for non-numeric data",
                    }
                )
                # Phase 0: Allow but warn
                # Phase 2: raise TypeError(...)

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

        # NEW: Warn on non-numeric (Phase 0), enforce later (Phase 2)
        if not isinstance(value, (int, float, type(None))):
            logger.warning(
                f"Setting non-numeric measure '{key}': {type(value).__name__}. "
                f"This will be rejected in Traigent v2.0."
            )
            # Phase 0: Allow but warn
            # Phase 2: raise TypeError(...)

        self.data[key] = value
```

---

### Task 0.3: Add Factory Method to WorkflowCostSummary

**File**: `traigent/cloud/agent_dtos.py` (WorkflowCostSummary class)

**Add**:
```python
@dataclass
class WorkflowCostSummary:
    """Aggregated cost summary across multiple agents."""

    workflow_id: str
    workflow_name: str
    agent_breakdowns: list[AgentCostBreakdown]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0

    @classmethod
    def from_agents(
        cls,
        workflow_id: str,
        workflow_name: str,
        agent_breakdowns: list[AgentCostBreakdown],
    ) -> 'WorkflowCostSummary':
        """Factory method that computes totals from agent breakdowns.

        Use this when you don't want to manually calculate totals.

        Args:
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable workflow name
            agent_breakdowns: List of per-agent cost breakdowns

        Returns:
            WorkflowCostSummary with totals computed from agent_breakdowns

        Example:
            >>> agent1 = AgentCostBreakdown(...)
            >>> agent2 = AgentCostBreakdown(...)
            >>> summary = WorkflowCostSummary.from_agents(
            ...     workflow_id="wf-123",
            ...     workflow_name="Research + Write",
            ...     agent_breakdowns=[agent1, agent2],
            ... )
        """
        return cls(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            agent_breakdowns=agent_breakdowns,
            total_input_tokens=sum(a.input_tokens for a in agent_breakdowns),
            total_output_tokens=sum(a.output_tokens for a in agent_breakdowns),
            total_tokens=sum(a.total_tokens for a in agent_breakdowns),
            total_input_cost=sum(a.input_cost for a in agent_breakdowns),
            total_output_cost=sum(a.output_cost for a in agent_breakdowns),
            total_cost=sum(a.total_cost for a in agent_breakdowns),
        )

    def __post_init__(self) -> None:
        """Validate and verify aggregation (strict validation, no transformations)."""
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

        # Validate that provided totals match computed totals (strict - no recomputation)
        if self.total_input_tokens != expected_input_tokens:
            errors.append(
                f"total_input_tokens ({self.total_input_tokens}) must equal "
                f"sum of agent input_tokens ({expected_input_tokens}). "
                f"Hint: Use WorkflowCostSummary.from_agents() to compute totals automatically."
            )

        if self.total_output_tokens != expected_output_tokens:
            errors.append(
                f"total_output_tokens ({self.total_output_tokens}) must equal "
                f"sum of agent output_tokens ({expected_output_tokens}). "
                f"Hint: Use WorkflowCostSummary.from_agents() to compute totals automatically."
            )

        if self.total_tokens != expected_total_tokens:
            errors.append(
                f"total_tokens ({self.total_tokens}) must equal "
                f"sum of agent total_tokens ({expected_total_tokens}). "
                f"Hint: Use WorkflowCostSummary.from_agents() to compute totals automatically."
            )

        # Use floating point tolerance for costs
        if abs(self.total_input_cost - expected_input_cost) > 0.0001:
            errors.append(
                f"total_input_cost ({self.total_input_cost:.8f}) must equal "
                f"sum of agent input_costs ({expected_input_cost:.8f}). "
                f"Hint: Use WorkflowCostSummary.from_agents() to compute totals automatically."
            )

        if abs(self.total_output_cost - expected_output_cost) > 0.0001:
            errors.append(
                f"total_output_cost ({self.total_output_cost:.8f}) must equal "
                f"sum of agent output_costs ({expected_output_cost:.8f}). "
                f"Hint: Use WorkflowCostSummary.from_agents() to compute totals automatically."
            )

        if abs(self.total_cost - expected_total_cost) > 0.0001:
            errors.append(
                f"total_cost ({self.total_cost:.8f}) must equal "
                f"sum of agent total_costs ({expected_total_cost:.8f}). "
                f"Hint: Use WorkflowCostSummary.from_agents() to compute totals automatically."
            )

        if errors:
            raise ValueError(
                f"WorkflowCostSummary aggregation validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
```

---

### Task 0.4: Update Tests for Array Structure

**File**: `tests/unit/cloud/test_api_operations.py` (NEW or UPDATE)

```python
import pytest
from traigent.cloud.api_operations import ApiOperations

def test_update_config_run_measures_array_format():
    """Verify measures are sent in array format."""
    api = ApiOperations(client=mock_client)

    metrics = {"accuracy": 0.95, "cost": 0.001}

    # Mock the HTTP call
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_session.return_value.__aenter__.return_value.put.return_value.__aenter__.return_value = mock_response

        result = await api.update_config_run_measures("config-123", metrics)

        # Verify payload structure
        call_args = mock_session.return_value.__aenter__.return_value.put.call_args
        payload = call_args[1]['json']

        assert "measures" in payload
        assert isinstance(payload["measures"], list), "Measures must be an array"
        assert len(payload["measures"]) == 1
        assert payload["measures"][0]["accuracy"] == 0.95
        assert payload["measures"][0]["cost"] == 0.001
```

**File**: `tests/unit/cloud/test_measures_dict.py` (UPDATE)

```python
def test_key_pattern_validation():
    """Should reject keys that don't match Python identifier pattern."""
    measures = MeasuresDict()

    # Valid keys
    measures["valid_key"] = 1.0
    measures["_private"] = 2.0
    measures["metric123"] = 3.0

    # Invalid keys
    with pytest.raises(ValueError, match="must match pattern"):
        measures["my-metric"] = 1.0  # Hyphen not allowed

    with pytest.raises(ValueError, match="must match pattern"):
        measures["123abc"] = 1.0  # Cannot start with digit

    with pytest.raises(ValueError, match="must match pattern"):
        measures["my.metric"] = 1.0  # Dot not allowed


def test_non_numeric_warning(caplog):
    """Should log warning for non-numeric values in Phase 0."""
    import logging
    caplog.set_level(logging.WARNING)

    measures = MeasuresDict()
    measures["string_metric"] = "value"  # Should warn

    assert "non-numeric value type" in caplog.text
    assert "rejected in Traigent v2.0" in caplog.text
```

**File**: `tests/unit/cloud/test_agent_dtos.py` (UPDATE)

```python
def test_workflow_cost_summary_from_agents_factory():
    """Factory method should compute totals correctly."""
    agent1 = AgentCostBreakdown(
        agent_id="agent-001",
        agent_name="Researcher",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
        model_used="gpt-4o-mini",
    )
    agent2 = AgentCostBreakdown(
        agent_id="agent-002",
        agent_name="Writer",
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        input_cost=0.002,
        output_cost=0.004,
        total_cost=0.006,
        model_used="gpt-4o",
    )

    # Use factory method - should compute totals
    summary = WorkflowCostSummary.from_agents(
        workflow_id="wf-123",
        workflow_name="Research + Write",
        agent_breakdowns=[agent1, agent2],
    )

    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 150
    assert summary.total_tokens == 450
    assert summary.total_cost == 0.009


def test_workflow_cost_summary_validation_with_hint():
    """Should provide helpful hint when totals mismatch."""
    agent1 = AgentCostBreakdown(
        agent_id="agent-001",
        agent_name="Researcher",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
        model_used="gpt-4o-mini",
    )

    # Manually construct with wrong totals
    with pytest.raises(ValueError, match="Use WorkflowCostSummary.from_agents") as exc_info:
        WorkflowCostSummary(
            workflow_id="wf-123",
            workflow_name="Research + Write",
            agent_breakdowns=[agent1],
            total_input_tokens=999,  # Wrong!
            total_output_tokens=50,
            total_tokens=1049,  # Wrong!
            total_input_cost=0.001,
            total_output_cost=0.002,
            total_cost=0.003,
        )

    assert "from_agents()" in str(exc_info.value)
```

---

## Phase 1: Schema Updates (Day 2)

### Task 1.1: Create Shared Type Definitions

**File**: `traigent_schema/schemas/evaluation/measures_types.json` (NEW)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://traigent.ai/schemas/measures_types.json",
  "title": "Measures Type Definitions",
  "description": "Shared type definitions for measures validation across SDK, Backend, Frontend",
  "version": "3.0.0",

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
        "Non-numeric metadata belongs in workflow_metadata, not measures"
      ]
    },

    "MeasuresArray": {
      "description": "Array of measure result objects",
      "oneOf": [
        {
          "type": "array",
          "items": {
            "$ref": "#/definitions/MeasureResult"
          },
          "minItems": 1
        },
        {"type": "null"}
      ],
      "x-notes": [
        "Each element is a complete set of metrics for one evaluation",
        "Typically contains 1 element (aggregated metrics)",
        "May contain multiple elements for per-example results",
        "Can be null if evaluation hasn't completed yet"
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
    },

    "WorkflowMetadata": {
      "description": "Metadata for multi-agent workflow execution",
      "type": "object",
      "properties": {
        "workflow_id": {
          "type": "string",
          "minLength": 1,
          "description": "Unique workflow identifier"
        },
        "workflow_name": {
          "type": "string",
          "minLength": 1,
          "description": "Human-readable workflow name"
        },
        "execution_time": {
          "type": "number",
          "minimum": 0,
          "description": "Total workflow execution time in seconds"
        },
        "agent_breakdowns": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/AgentCostBreakdown"
          },
          "minItems": 1,
          "description": "Per-agent cost and token breakdowns"
        }
      },
      "additionalProperties": true,
      "x-notes": [
        "Used for multi-agent workflow cost tracking",
        "Kept separate from measures to maintain numeric-only measures",
        "Additional workflow-specific fields can be added"
      ]
    }
  }
}
```

---

### Task 1.2: Update Configuration Run Schema

**File**: `traigent_schema/schemas/evaluation/configuration_run_schema.json` (UPDATE)

Add new `workflow_metadata` field:

```json
{
  "properties": {
    "id": { ... },
    "experiment_run_id": { ... },
    "experiment_parameters": { ... },

    "measures": {
      "$ref": "measures_types.json#/definitions/MeasuresArray",
      "description": "Results from measure evaluations for this configuration"
    },

    "workflow_metadata": {
      "oneOf": [
        {
          "$ref": "measures_types.json#/definitions/WorkflowMetadata",
          "description": "Multi-agent workflow execution metadata"
        },
        {"type": "null"}
      ],
      "description": "Optional workflow-specific metadata (multi-agent cost tracking, etc.)"
    },

    "summary_stats": { ... },
    "status": { ... },
    ...
  },
  "required": [
    "id",
    "experiment_run_id",
    "experiment_parameters"
  ]
}
```

---

## Phase 2: Backend Updates (Days 3-4)

### Task 2.1: Add Pydantic Models

**File**: `src/models/measures_dtos.py` (NEW)

```python
"""Measures DTOs with strict validation matching SDK.

These models enforce the same validation rules as the SDK to maintain
DTO contract consistency across all layers.
"""

import math
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MeasuresDict(dict):
    """Type-safe measures dict with validation.

    Note: This is a dict subclass for JSON serialization compatibility.
    Use field_validator to integrate with Pydantic.
    """

    MAX_KEYS = 50
    KEY_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Pydantic v2 integration."""
        from pydantic_core import core_schema

        def validate(value: Any) -> dict:
            if not isinstance(value, dict):
                raise TypeError(f"Measures must be dict, got {type(value).__name__}")

            if len(value) > cls.MAX_KEYS:
                raise ValueError(
                    f"Measures cannot exceed {cls.MAX_KEYS} keys, got {len(value)}"
                )

            for key, val in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Measure key must be string, got {type(key).__name__}"
                    )

                if not cls.KEY_PATTERN.match(key):
                    raise ValueError(
                        f"Measure key '{key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
                    )

                if not isinstance(val, (int, float, type(None))):
                    raise TypeError(
                        f"Measure '{key}' must be numeric (int, float, None), "
                        f"got {type(val).__name__}"
                    )

            return cls(value)

        return core_schema.no_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: dict(x)
            ),
        )


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
        """Validate token and cost arithmetic (strict validation, no transformations)."""
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


class WorkflowMetadata(BaseModel):
    """Multi-agent workflow execution metadata."""

    workflow_id: str = Field(..., min_length=1)
    workflow_name: str = Field(..., min_length=1)
    execution_time: float | None = Field(default=None, ge=0)
    agent_breakdowns: list[AgentCostBreakdown] = Field(default_factory=list)

    @field_validator('agent_breakdowns')
    @classmethod
    def validate_agent_breakdowns(cls, v: list[AgentCostBreakdown]) -> list[AgentCostBreakdown]:
        """Validate agent breakdowns."""
        if v and len(v) == 0:
            raise ValueError("agent_breakdowns cannot be empty list (use None or omit)")
        return v
```

---

### Task 2.2: Update Configuration Run Service

**File**: `src/services/configuration_run_service.py` (UPDATE)

```python
from src.models.measures_dtos import MeasuresDict, WorkflowMetadata

def _validate_measures_format(self, measures: list | dict[str, Any] | None) -> list | None:
    """Validate and normalize measures to schema-compliant format."""
    # Allow null measures (evaluation not complete)
    if measures is None:
        return None

    # Must be a list according to schema
    if not isinstance(measures, list):
        logger.error(f"Measures not in array format: {type(measures)}")
        return None

    if len(measures) == 0:
        # Empty array - should be null instead
        logger.warning("Received empty measures array, should be null instead")
        return None

    # Validate each measure result object with strict validation
    validated_measures = []
    for i, measure_result in enumerate(measures):
        if not isinstance(measure_result, dict):
            logger.error(
                f"Measure result {i} is not a dictionary: {type(measure_result)}"
            )
            return None

        try:
            # Use MeasuresDict for strict validation (Pydantic integration)
            validated = MeasuresDict.model_validate(measure_result)
            validated_measures.append(dict(validated))
        except (ValueError, TypeError) as e:
            logger.error(f"Measure result {i} validation failed: {e}")
            return None

    return validated_measures


def update_configuration_run_workflow_metadata(
    self,
    config_run_id: str,
    workflow_metadata: dict[str, Any],
) -> tuple[ConfigurationRun | None, str | None]:
    """Update configuration run workflow metadata.

    Args:
        config_run_id: Configuration run ID
        workflow_metadata: Workflow metadata (multi-agent cost tracking, etc.)

    Returns:
        Tuple of (ConfigurationRun | None, error message | None)
    """
    try:
        logger.info(
            f"Updating workflow_metadata for configuration run {config_run_id}: "
            f"{workflow_metadata}"
        )

        # Validate workflow metadata format with Pydantic
        try:
            validated = WorkflowMetadata.model_validate(workflow_metadata)
        except Exception as e:
            return None, f"Invalid workflow_metadata format: {e}"

        config_run = self.repository.update_configuration_run_workflow_metadata(
            config_run_id,
            validated.model_dump(),
        )

        if not config_run:
            logger.warning(
                f"Configuration run {config_run_id} not found for workflow_metadata update"
            )
            return None, "Configuration run not found"

        return config_run, None

    except Exception as e:
        logger.error(f"Error updating configuration run workflow_metadata: {e!s}")
        return None, str(e)
```

---

### Task 2.3: Add Database Migration

**File**: `src/migrations/versions/XXXX_add_workflow_metadata.py` (NEW)

```python
"""Add workflow_metadata field to configuration_runs

Revision ID: XXXX
Revises: YYYY
Create Date: 2026-01-16

"""
from alembic import op
import sqlalchemy as sa
from src.extensions import JSONType

revision = 'XXXX'
down_revision = 'YYYY'
branch_labels = None
depends_on = None


def upgrade():
    # Add workflow_metadata column
    op.add_column(
        'configuration_runs',
        sa.Column('workflow_metadata', JSONType, nullable=True, default=None)
    )


def downgrade():
    # Remove workflow_metadata column
    op.drop_column('configuration_runs', 'workflow_metadata')
```

---

## Phase 3: Frontend Updates (Days 5-6)

### Task 3.1: TypeScript Type Definitions

**File**: `src/types/measures.ts` (NEW)

```typescript
/**
 * Measures type definitions matching TraigentSchema v3.0.0.
 *
 * Key changes from v2:
 * - Measures is array of objects
 * - Values are numeric only (no boolean/string)
 * - Key pattern enforced (^[a-zA-Z_][a-zA-Z0-9_]*$)
 * - agent_breakdowns moved to workflow_metadata
 */

/**
 * A single measure value (numeric only for optimization metrics).
 */
export type MeasureValue = number | null;

/**
 * A single measure result object.
 */
export interface MeasureResult {
  [key: string]: MeasureValue;
}

/**
 * Array of measure result objects (canonical structure).
 */
export type MeasuresArray = MeasureResult[] | null;

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
 * Multi-agent workflow execution metadata.
 */
export interface WorkflowMetadata {
  workflow_id: string;
  workflow_name: string;
  execution_time?: number;
  agent_breakdowns?: AgentCostBreakdown[];
}

/**
 * Validation utilities for measures.
 */
export class MeasuresValidator {
  static readonly MAX_KEYS = 50;
  static readonly KEY_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

  /**
   * Validate measure result object.
   */
  static validateMeasureResult(measures: Record<string, any>): MeasureResult {
    const keys = Object.keys(measures);

    if (keys.length > this.MAX_KEYS) {
      throw new Error(
        `Measures cannot exceed ${this.MAX_KEYS} keys, got ${keys.length}`
      );
    }

    for (const [key, value] of Object.entries(measures)) {
      // Validate key format (Python identifier syntax)
      if (!this.KEY_PATTERN.test(key)) {
        throw new Error(
          `Measure key '${key}' must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$ ` +
          `(Python identifier syntax). ` +
          `Use underscores instead of hyphens or spaces.`
        );
      }

      // Validate value type (numeric only)
      if (value !== null && typeof value !== 'number') {
        throw new Error(
          `Measure '${key}' must be numeric type (number, null), ` +
          `got ${typeof value}. ` +
          `Non-numeric data should be stored in workflow_metadata.`
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
   */
  static validateMeasuresArray(measures: any): MeasuresArray {
    if (measures === null) {
      return null;
    }

    if (!Array.isArray(measures)) {
      throw new Error(
        `Measures must be an array or null, got ${typeof measures}. ` +
        `Expected format: [{"accuracy": 0.95, "cost": 0.001}]`
      );
    }

    if (measures.length === 0) {
      throw new Error(
        'Measures array cannot be empty. Use null instead when no results available.'
      );
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

## Phase 4: Integration Testing (Day 7)

### Task 4.1: End-to-End Tests

**File**: `tests/integration/test_measures_e2e.py` (NEW)

```python
"""End-to-end tests for measures DTO contract."""

import pytest
from traigent.cloud.dtos import MeasuresDict
from traigent.cloud.agent_dtos import AgentCostBreakdown, WorkflowCostSummary


@pytest.mark.integration
def test_measures_sdk_to_backend():
    """Verify SDK measures payload matches backend expectations."""
    # SDK creates measures
    measures = MeasuresDict({
        "accuracy": 0.95,
        "cost": 0.001,
        "input_tokens": 1500,
        "output_tokens": 750,
    })

    # SDK sends payload
    payload = {"measures": [dict(measures)]}

    # Verify structure
    assert isinstance(payload["measures"], list)
    assert len(payload["measures"]) == 1
    assert isinstance(payload["measures"][0], dict)

    # Backend would validate with Pydantic
    from src.models.measures_dtos import MeasuresDict as BackendMeasuresDict
    validated = BackendMeasuresDict.model_validate(payload["measures"][0])

    assert dict(validated) == dict(measures)


@pytest.mark.integration
def test_workflow_cost_tracking_e2e():
    """Verify multi-agent workflow cost tracking end-to-end."""
    # User creates agent breakdowns
    agent1 = AgentCostBreakdown(
        agent_id="researcher",
        agent_name="Researcher",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        input_cost=0.001,
        output_cost=0.002,
        total_cost=0.003,
        model_used="gpt-4o-mini",
    )

    agent2 = AgentCostBreakdown(
        agent_id="writer",
        agent_name="Writer",
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
        input_cost=0.002,
        output_cost=0.004,
        total_cost=0.006,
        model_used="gpt-4o",
    )

    # User creates summary using factory
    summary = WorkflowCostSummary.from_agents(
        workflow_id="wf-123",
        workflow_name="Research + Write",
        agent_breakdowns=[agent1, agent2],
    )

    # SDK sends workflow_metadata
    workflow_metadata = {
        "workflow_id": summary.workflow_id,
        "workflow_name": summary.workflow_name,
        "execution_time": 12.34,
        "agent_breakdowns": [
            {
                "agent_id": a.agent_id,
                "agent_name": a.agent_name,
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "total_tokens": a.total_tokens,
                "input_cost": a.input_cost,
                "output_cost": a.output_cost,
                "total_cost": a.total_cost,
                "model_used": a.model_used,
            }
            for a in summary.agent_breakdowns
        ],
    }

    # Backend validates with Pydantic
    from src.models.measures_dtos import WorkflowMetadata as BackendWorkflowMetadata
    validated = BackendWorkflowMetadata.model_validate(workflow_metadata)

    assert validated.workflow_id == "wf-123"
    assert len(validated.agent_breakdowns) == 2
    assert validated.agent_breakdowns[0].agent_id == "researcher"
```

---

## Migration Timeline

| Phase | Duration | Status | Critical? |
|-------|----------|--------|-----------|
| **Phase 0: SDK Fixes** | 1 day | ⏳ Pending | 🔴 YES |
| Phase 1: Schema Updates | 1 day | ⏳ Pending | 🟡 Important |
| Phase 2: Backend Updates | 2 days | ⏳ Pending | 🟡 Important |
| Phase 3: Frontend Updates | 2 days | ⏳ Pending | 🟢 Optional |
| Phase 4: Integration Testing | 1 day | ⏳ Pending | 🟡 Important |
| **Total** | **7 days** | | |

---

## Success Criteria

### Phase 0:
- [ ] SDK sends measures as array format
- [ ] MeasuresDict enforces key pattern validation
- [ ] Non-numeric values log warnings (Phase 0), rejected later (Phase 2)
- [ ] WorkflowCostSummary.from_agents() factory works
- [ ] All existing tests pass
- [ ] New array structure tests pass

### Phases 1-4:
- [ ] Schema defines workflow_metadata field
- [ ] Backend validates measures with MeasuresDict
- [ ] Backend stores workflow_metadata
- [ ] Frontend validates and renders measures
- [ ] End-to-end tests pass
- [ ] No data loss in migration

---

## Rollback Plan

If Phase 0 breaks production:

1. **Revert API Operations**:
   ```python
   # Temporarily support both formats
   if isinstance(measures, dict) and "metrics" in measures:
       # Old format - convert to new
       measures_data = {"measures": [measures["metrics"]]}
   elif isinstance(measures, list):
       # New format
       measures_data = {"measures": measures}
   ```

2. **Backend Compatibility**:
   ```python
   # Backend accepts both formats temporarily
   if isinstance(measures, dict):
       measures = [measures]  # Wrap old format
   ```

3. **Feature Flag**:
   ```python
   MEASURES_ARRAY_FORMAT = os.getenv("TRAIGENT_MEASURES_ARRAY", "true")
   ```

---

## Summary of v3 Improvements

| Issue | v2 | v3 |
|-------|----|----|
| **agent_breakdowns location** | In measures (schema violation) | New workflow_metadata field |
| **WorkflowCostSummary** | Required explicit totals | Added from_agents() factory |
| **Numeric enforcement** | Incomplete (only MeasuresDict) | All SDK paths + warnings |
| **Empty measures** | Unclear | Omit field or use null |
| **Pydantic integration** | Not shown | Proper __get_pydantic_core_schema__ |
| **execution_time** | Removed | Added to workflow_metadata |
| **sync_manager** | Still dict | Updated to array |

---

## Next Steps

1. **Review v3 recommendations** - Ensure all issues are addressed
2. **Approve Phase 0** - Critical SDK fixes (1 day)
3. **Plan Phases 1-4** - Backend/Frontend alignment (6 days)
4. **Execute migration** - 7 days total timeline

All feedback from Codex and Gemini has been incorporated into v3.
