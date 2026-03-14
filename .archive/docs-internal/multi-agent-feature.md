# Multi-Agent Parameter and Measure Mapping System

## Overview

This document specifies the robust agent-parameter and agent-measure mapping system that replaces fragile label-parsing with explicit configuration from the SDK.

**Status**: SDK Implementation Complete ✓
**Version**: 1.0
**Last Updated**: 2026-01-09

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| SDK Types | ✅ Complete | `AgentConfiguration`, `AgentDefinition`, etc. in `traigent/api/types.py` |
| Agent Inference | ✅ Complete | `traigent/api/agent_inference.py` |
| Parameter Ranges | ✅ Complete | `agent` param on `Range`, `Choices`, `IntRange` |
| Decorator | ✅ Complete | `agents`, `agent_prefixes`, `agent_measures`, `global_measures` params |
| TVL Support | ✅ Complete | `agent` field on tvars, flows through to orchestrator |
| Orchestrator | ✅ Complete | Builds and passes `agent_configuration` to backend |
| Backend | 🔲 Pending | Pass-through implementation needed |
| Frontend | 🔲 Pending | Use explicit config, add measure grouping |

---

## Problem Statement

Currently, the frontend uses label parsing (prefix matching like `trial_financial_agent_model`) to associate parameters and measures with agents. This approach is:

- **Fragile** - Depends on naming conventions being followed exactly
- **Error-prone** - Compound suffixes like `_prompt_template` can be misparsed
- **Limited** - Measures are NOT currently grouped by agent at all
- **Opaque** - No visibility into what the system detected

**Solution**: Replace label parsing with an explicit `agent_configuration` mapping provided by the SDK in `experiment_parameters`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SDK (Source of Truth)                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  @optimize      │───>│ Agent Inference │───>│ Orchestrator│ │
│  │  decorator      │    │ Builder         │    │             │ │
│  │  - agent="x"    │    │                 │    │             │ │
│  │  - agent_prefix │    │                 │    │             │ │
│  └─────────────────┘    └─────────────────┘    └──────┬──────┘ │
└──────────────────────────────────────────────────────┬┬────────┘
                                                       │││
                          agent_configuration          ││▼
┌──────────────────────────────────────────────────────┼┼────────┐
│                         Backend (Pass-through)       ││        │
│  Stores and returns agent_configuration unchanged    ││        │
└──────────────────────────────────────────────────────┼┼────────┘
                                                       ││
                          experiment_parameters        ││
                          .agent_configuration         ▼▼
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Consumer)                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ Config Resolver │───>│ Column Manager  │───>│ Table/Charts│ │
│  │ (explicit first)│    │ (group by agent)│    │             │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### AgentConfiguration Schema (v1.0)

```json
{
  "version": "1.0",
  "auto_inferred": false,
  "agents": {
    "<agent_id>": {
      "display_name": "Human Readable Name",
      "parameter_keys": ["param1", "param2"],
      "measure_ids": ["measure1", "measure2"],
      "primary_model": "param1",
      "order": 0,
      "agent_type": "llm",
      "meta": {
        "color": "#4299E1",
        "icon": "robot",
        "description": "Optional tooltip description"
      }
    }
  },
  "global": {
    "parameter_keys": ["global_param1"],
    "measure_ids": ["total_cost", "total_latency"],
    "order": 99
  }
}
```

### Field Definitions

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `version` | string | Yes | Schema version (currently "1.0") |
| `auto_inferred` | boolean | Yes | True if SDK-generated from patterns |
| `agents` | object | Yes | Map of agent_id to AgentDefinition |
| `global` | object | No | Global (non-agent-specific) config |

### AgentDefinition Fields

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `display_name` | string | Yes | Human-readable name for UI |
| `parameter_keys` | string[] | Yes | Parameter names belonging to this agent |
| `measure_ids` | string[] | Yes | Measure IDs belonging to this agent |
| `primary_model` | string | No | Key of the primary model parameter |
| `order` | number | No | Display order (lower = first) |
| `agent_type` | string | No | One of: "llm", "retriever", "router", "tool", "custom" |
| `meta` | object | No | UI metadata (color, icon, description) |

---

## SDK API

### Method 1: Explicit Agent on Parameters

```python
from traigent import optimize
from traigent.api.parameter_ranges import Range, Choices

@optimize(
    # Parameters with explicit agent assignment
    model=Choices(["gpt-4", "gpt-3.5"], agent="financial"),
    temperature=Range(0.0, 1.0, agent="financial"),
    legal_model=Choices(["claude-3"], agent="legal"),
    router_model=Choices(["gpt-4o-mini"], agent="router"),

    # Measure-to-agent mapping
    agent_measures={
        "financial": ["financial_accuracy", "financial_latency"],
        "legal": ["legal_accuracy", "legal_latency"],
        "router": ["routing_accuracy"],
    },
    global_measures=["total_accuracy", "total_cost"],
)
def multi_agent_pipeline(query: str, **config):
    ...
```

### Method 2: Prefix-Based with Explicit Prefixes

```python
@optimize(
    # Parameters follow naming convention
    financial_model=Choices(["gpt-4"]),
    financial_temperature=Range(0.0, 1.0),
    legal_model=Choices(["claude-3"]),
    max_retries=Range(1, 5),  # Global (no prefix match)

    # Tell SDK how to group by prefix
    agent_prefixes=["financial", "legal"],

    # Measure-to-agent mapping
    agent_measures={
        "financial": ["financial_accuracy"],
        "legal": ["legal_accuracy"],
    },
    global_measures=["total_cost"],
)
def pipeline(query: str, **config):
    ...
```

### Method 3: Fully Explicit Agent Definitions

```python
from traigent.api.types import AgentDefinition

@optimize(
    configuration_space={
        "model": ["gpt-4", "gpt-3.5"],
        "temperature": (0.0, 1.0),
    },
    agents={
        "financial": AgentDefinition(
            display_name="Financial Agent",
            parameter_keys=["model", "temperature"],
            measure_ids=["accuracy", "latency"],
            agent_type="llm",
        ),
    },
    global_measures=["total_cost"],
)
def pipeline(query: str, **config):
    ...
```

### Validation

The SDK raises `ValueError` if:
- `agent_prefixes` contains a prefix with no matching parameters
- `agent_measures` references an agent not defined elsewhere

```python
# This raises ValueError:
@optimize(
    model=Choices(["gpt-4"]),
    agent_prefixes=["nonexistent"],  # Error: no params start with "nonexistent_"
)
```

---

## Backend API Contract

### Session Creation (SDK → Backend)

**Endpoint**: `POST /api/sessions` (or equivalent session creation endpoint)

The SDK includes `agent_configuration` in `session_metadata` when creating a session:

```json
{
  "function_name": "multi_agent_pipeline",
  "search_space": {...},
  "session_metadata": {
    "evaluation_set": "test_queries",
    "agent_configuration": {
      "version": "1.0",
      "auto_inferred": false,
      "agents": {
        "financial": {
          "display_name": "Financial",
          "parameter_keys": ["financial_model", "financial_temperature"],
          "measure_ids": ["financial_accuracy"],
          "primary_model": "financial_model",
          "order": 0
        },
        "legal": {
          "display_name": "Legal",
          "parameter_keys": ["legal_model"],
          "measure_ids": ["legal_accuracy"],
          "order": 1
        }
      },
      "global": {
        "parameter_keys": [],
        "measure_ids": ["total_cost"],
        "order": 99
      }
    }
  }
}
```

### Experiment Runs Response

**Endpoint**: `GET /api/experiments/{id}/runs`

Backend returns `agent_configuration` in `experiment_parameters`:

```json
{
  "runs": [
    {
      "id": "run_001",
      "experiment_parameters": {
        "config": {...},
        "agent_configuration": {
          "version": "1.0",
          "agents": {...},
          "global": {...}
        }
      },
      "summary_stats": {...}
    }
  ]
}
```

### Backend Requirements

1. **Pass-through only** - Store and return `agent_configuration` without validation
2. **No schema enforcement** - Accept any valid JSON object
3. **Backward compatible** - Handle requests without `agent_configuration`
4. **Session-level storage** - `agent_configuration` is sent once at session creation, not per-trial
5. **Return in experiment_parameters** - Include `agent_configuration` when returning run/experiment data

### Backend Implementation Notes

The SDK sends `agent_configuration` in `session_metadata` during session creation:

```python
# SDK code (backend_session_manager.py:161-162)
if agent_configuration is not None:
    session_metadata["agent_configuration"] = agent_configuration.to_dict()
```

Backend should:
1. Accept `session_metadata.agent_configuration` in session creation
2. Store it associated with the session/experiment
3. Return it in `experiment_parameters.agent_configuration` for any run/experiment queries

### Pydantic Model (Backend)

```python
from pydantic import BaseModel, Field
from typing import Any

class AgentDefinition(BaseModel):
    display_name: str
    parameter_keys: list[str]
    measure_ids: list[str]
    primary_model: str | None = None
    order: int | None = None
    agent_type: str | None = None
    meta: dict[str, Any] | None = None

class GlobalConfiguration(BaseModel):
    parameter_keys: list[str]
    measure_ids: list[str]
    order: int | None = None

class AgentConfiguration(BaseModel):
    version: str = "1.0"
    agents: dict[str, AgentDefinition]
    global_: GlobalConfiguration | None = Field(None, alias="global")
    auto_inferred: bool = False

    class Config:
        populate_by_name = True
```

---

## Frontend Implementation

### Type Definitions

```typescript
// src/api/models/AgentConfiguration.ts

export type AgentType = "llm" | "retriever" | "router" | "tool" | "custom";

export interface AgentMeta {
  color?: string;
  icon?: string;
  description?: string;
}

export interface AgentDefinition {
  display_name: string;
  parameter_keys: string[];
  measure_ids: string[];
  primary_model?: string;
  order?: number;
  agent_type?: AgentType;
  meta?: AgentMeta;
}

export interface GlobalConfiguration {
  parameter_keys: string[];
  measure_ids: string[];
  order?: number;
}

export interface AgentConfiguration {
  version: "1.0";
  agents: Record<string, AgentDefinition>;
  global?: GlobalConfiguration;
  auto_inferred?: boolean;
}

export enum ConfigurationSource {
  EXPLICIT = "explicit",
  AUTO_INFERRED = "auto_inferred",
  LABEL_PARSED = "label_parsed",
  NONE = "none"
}
```

### Configuration Resolution

```typescript
// src/utils/agentParameterUtils.ts

export interface AgentConfigResult {
  config: AgentConfiguration | null;
  source: ConfigurationSource;
  warnings: string[];
}

export function getAgentConfiguration(run: ConfigurationRun): AgentConfigResult {
  const warnings: string[] = [];

  // Priority 1: Explicit configuration from SDK
  const explicit = run.experiment_parameters?.agent_configuration;
  if (explicit && Object.keys(explicit.agents || {}).length > 0) {
    // Validate version
    if (explicit.version && explicit.version !== "1.0") {
      warnings.push(`Unknown version: ${explicit.version}`);
    }

    return {
      config: explicit,
      source: explicit.auto_inferred
        ? ConfigurationSource.AUTO_INFERRED
        : ConfigurationSource.EXPLICIT,
      warnings
    };
  }

  // Priority 2: Legacy label parsing (deprecated)
  const inferred = inferFromLabels(run);
  if (inferred) {
    return {
      config: inferred,
      source: ConfigurationSource.LABEL_PARSED,
      warnings: ["Using legacy label parsing - consider updating SDK"]
    };
  }

  // No multi-agent configuration
  return { config: null, source: ConfigurationSource.NONE, warnings };
}
```

### Column Manager Updates

```typescript
// src/hooks/useColumnManager.tsx

// Group BOTH parameters AND measures by agent
const agentConfig = getAgentConfiguration(run);

if (agentConfig.config) {
  // Sort agents by order
  const sortedAgents = getSortedAgents(agentConfig.config);

  // Assign agent groups to parameter columns
  parameterColumns.forEach(col => {
    col.agentGroup = findAgentForParameter(col.key, agentConfig.config);
  });

  // Assign agent groups to measure columns
  measureColumns.forEach(col => {
    col.agentGroup = findAgentForMeasure(col.id, agentConfig.config);
  });
}
```

### Visual Design

| Category | Separator | Style |
| -------- | --------- | ----- |
| Params to Measures | Blue solid (3px) | `#3B82F6` |
| Agent within Params | Purple dashed (2px) | `#8B5CF6` |
| Agent within Measures | Teal dashed (2px) | `#14B8A6` |

**Single-Agent Display**:
```
| ID | Status | Model | Temp | || Score | Accuracy | Cost |
                              ↑ blue solid (params→measures)
```

**Multi-Agent Display**:
```
| ID | Status | Model | ¦ Fin Model | Fin Temp | ¦ Legal Model | || Total | ¦ Fin Acc | ¦ Legal Acc |
               ↑ purple   ↑ purple                ↑ blue solid   ↑ teal    ↑ teal
```

---

## TVL Spec Format

### Flat with Agent Attribute (Implemented ✓)

The `agent` field on tvars is fully supported and flows through to `agent_configuration`:

```yaml
tvl_version: "0.9"

tvars:
  - name: financial_model
    type: enum[str]
    domain: ["gpt-4", "gpt-3.5"]
    agent: financial  # Maps to agent "financial"

  - name: financial_temperature
    type: float
    domain:
      range: [0.0, 1.0]
    agent: financial

  - name: legal_model
    type: enum[str]
    domain: ["claude-3"]
    agent: legal

  - name: max_retries
    type: int
    domain: [1, 5]
    # No agent field = global parameter

objectives:
  - name: accuracy
    direction: maximize
```

### TVL Data Flow

```
TVL spec (agent: "x" on tvar)
    ↓
TVarDecl.agent (parsed in spec_loader)
    ↓
_extract_parameter_agents_from_tvars()
    ↓
TVLSpecArtifact.parameter_agents
    ↓
runtime_overrides()["tvl_parameter_agents"]
    ↓
algorithm_kwargs["tvl_parameter_agents"]
    ↓
orchestrator (merged with ParameterRange agents)
    ↓
build_agent_configuration(parameter_agents=merged)
    ↓
AgentConfiguration sent to backend
```

### Merge Behavior

When both TVL spec and decorator specify agents, **decorator takes priority**:

```python
# TVL spec has: agent: "tvl_agent" on "model" parameter
# Decorator has: model=Choices(["gpt-4"], agent="decorator_agent")
# Result: model → "decorator_agent" (decorator wins)
```

This allows users to override TVL defaults at runtime.

### Hierarchical Format (Future)

A hierarchical `agents:` section is planned but not yet implemented:

```yaml
# FUTURE: Not yet implemented
agents:
  financial:
    display_name: "Financial Agent"
    type: llm
    tvars:
      - name: model
        type: enum[str]
        domain: ["gpt-4"]
    measures:
      - accuracy
```

---

## Migration Guide

### For SDK Users

**Before (prefix convention only)**:
```python
@optimize(
    financial_agent_model=Choices(["gpt-4"]),
    financial_agent_temperature=Range(0.0, 1.0),
)
```

**After (explicit agent)**:
```python
@optimize(
    financial_model=Choices(["gpt-4"], agent="financial"),
    financial_temperature=Range(0.0, 1.0, agent="financial"),
    agent_measures={"financial": ["accuracy"]},
)
```

### For Frontend

1. Check for `experiment_parameters.agent_configuration` first
2. Fall back to label parsing only if not present
3. Log warning when using label parsing fallback

---

## Rollout Plan

| Phase | Component | Status | Changes |
| ----- | --------- | ------ | ------- |
| 1 | SDK | ✅ Done | Add types, agent param, inference builder, TVL support |
| 2 | Backend | 🔲 Next | Pass-through agent_configuration in session_metadata |
| 3 | Frontend | 🔲 Pending | Use explicit config, add measure grouping |
| 4 | Deprecate | 🔲 Future | Remove label parsing fallback |

---

## SDK File Reference

Key files for understanding the implementation:

| File | Purpose |
| ---- | ------- |
| `traigent/api/types.py` | `AgentConfiguration`, `AgentDefinition`, `GlobalConfiguration` types |
| `traigent/api/agent_inference.py` | `build_agent_configuration()`, `extract_parameter_agents()` |
| `traigent/api/parameter_ranges.py` | `Range`, `Choices`, `IntRange` with `agent` parameter |
| `traigent/api/decorators.py` | `@optimize` with `agents`, `agent_prefixes`, etc. |
| `traigent/tvl/models.py` | `TVarDecl` with `agent` field |
| `traigent/tvl/spec_loader.py` | `_extract_parameter_agents_from_tvars()`, `TVLSpecArtifact.parameter_agents` |
| `traigent/core/orchestrator.py` | Builds and stores `_agent_configuration` |
| `traigent/core/optimized_function.py` | Passes `tvl_parameter_agents` through runtime overrides |
| `traigent/core/backend_session_manager.py` | Includes `agent_configuration` in session_metadata |

---

## Testing

### SDK Unit Tests

```python
def test_explicit_agent_assignment():
    """Test agent assignment via Range(..., agent='x')."""
    config_space = {
        "model": Choices(["gpt-4"], agent="financial"),
        "temp": Range(0.0, 1.0, agent="financial"),
        "legal_model": Choices(["claude"], agent="legal"),
    }
    result = build_agent_configuration(
        configuration_space=config_space,
        parameter_agents={"model": "financial", "temp": "financial", "legal_model": "legal"},
    )

    assert result is not None
    assert "financial" in result.agents
    assert "legal" in result.agents


def test_invalid_prefix_raises_error():
    """Invalid prefixes should raise ValueError."""
    with pytest.raises(ValueError, match="no parameters start with"):
        build_agent_configuration(
            configuration_space={"model": Choices(["gpt-4"])},
            agent_prefixes=["nonexistent"],
        )
```

### Frontend Tests

```typescript
describe('getAgentConfiguration', () => {
  it('should use explicit config when present', () => {
    const run = {
      experiment_parameters: {
        agent_configuration: {
          version: "1.0",
          agents: { financial: { display_name: "Financial", parameter_keys: ["model"], measure_ids: [] } }
        }
      }
    };

    const result = getAgentConfiguration(run);
    expect(result.source).toBe(ConfigurationSource.EXPLICIT);
  });

  it('should fall back to label parsing when no config', () => {
    const run = { experiment_parameters: {} };
    const result = getAgentConfiguration(run);
    expect(result.source).toBe(ConfigurationSource.LABEL_PARSED);
  });
});
```

---

## Appendix: Full Example Payload

```json
{
  "trial_id": "trial_123",
  "config": {
    "financial_model": "gpt-4",
    "financial_temperature": 0.3,
    "financial_prompt_template": "analyze",
    "legal_model": "gpt-4-turbo",
    "legal_temperature": 0.2,
    "router_model": "gpt-4o-mini",
    "max_retries": 3
  },
  "metrics": {
    "financial_accuracy": 0.92,
    "financial_latency": 150,
    "legal_accuracy": 0.88,
    "legal_latency": 200,
    "routing_accuracy": 0.95,
    "total_accuracy": 0.90,
    "total_cost": 0.12,
    "total_latency": 450
  },
  "agent_configuration": {
    "version": "1.0",
    "auto_inferred": false,
    "agents": {
      "financial": {
        "display_name": "Financial Agent",
        "parameter_keys": [
          "financial_model",
          "financial_temperature",
          "financial_prompt_template"
        ],
        "measure_ids": ["financial_accuracy", "financial_latency"],
        "primary_model": "financial_model",
        "order": 1,
        "agent_type": "llm",
        "meta": {
          "color": "#4299E1",
          "description": "Handles financial analysis queries"
        }
      },
      "legal": {
        "display_name": "Legal Agent",
        "parameter_keys": ["legal_model", "legal_temperature"],
        "measure_ids": ["legal_accuracy", "legal_latency"],
        "primary_model": "legal_model",
        "order": 2,
        "agent_type": "llm",
        "meta": {
          "color": "#48BB78",
          "description": "Handles legal document review"
        }
      },
      "router": {
        "display_name": "Router",
        "parameter_keys": ["router_model"],
        "measure_ids": ["routing_accuracy"],
        "order": 0,
        "agent_type": "router",
        "meta": {
          "color": "#ED8936",
          "description": "Routes queries to appropriate agent"
        }
      }
    },
    "global": {
      "parameter_keys": ["max_retries"],
      "measure_ids": ["total_accuracy", "total_cost", "total_latency"],
      "order": 99
    }
  }
}
```
