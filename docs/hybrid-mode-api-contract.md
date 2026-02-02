# Traigent Hybrid Mode - API Contract

This document defines the REST API contract for external services integrating with Traigent's Hybrid Mode optimization.

## Machine-Readable Specifications

For code generation and validation, use these schema files:

| Format | File | Description |
|--------|------|-------------|
| **OpenAPI 3.0** | [hybrid-mode-openapi.yaml](./hybrid-mode-openapi.yaml) | Full API specification with examples |
| **JSON Schema** | [hybrid-mode-schemas.json](./hybrid-mode-schemas.json) | DTO definitions for validation |

### Using the OpenAPI Spec

Generate client code:
```bash
# Python client
openapi-generator generate -i docs/hybrid-mode-openapi.yaml -g python -o client/

# TypeScript client
openapi-generator generate -i docs/hybrid-mode-openapi.yaml -g typescript-fetch -o client/
```

Validate requests/responses:
```python
from jsonschema import validate
import json

with open("docs/hybrid-mode-schemas.json") as f:
    schemas = json.load(f)

# Validate an execute response
validate(response_data, schemas["definitions"]["ExecuteResponse"])
```

---

## Overview

Hybrid Mode allows Traigent to optimize external agentic services by:
1. Discovering tunable parameters via REST API
2. Executing the agent with different configurations
3. Evaluating outputs to measure quality
4. Finding optimal configurations based on objectives

## Base URL

All endpoints use the prefix `/traigent/v1/`.

Example: `http://your-service:8080/traigent/v1/capabilities`

---

## Endpoints

### GET /traigent/v1/capabilities

**Required**: Yes

Returns service capabilities for the initial handshake.

#### Response (200 OK)

```json
{
  "version": "1.0",
  "supports_evaluate": true,
  "supports_keep_alive": false,
  "supports_streaming": false,
  "max_batch_size": 100,
  "max_payload_bytes": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | API version (currently "1.0") |
| `supports_evaluate` | boolean | Yes | Whether `/evaluate` endpoint is available |
| `supports_keep_alive` | boolean | No | Whether session heartbeat is supported |
| `supports_streaming` | boolean | No | Whether streaming responses are supported |
| `max_batch_size` | integer | No | Maximum inputs per execute request (default: 100) |
| `max_payload_bytes` | integer | No | Maximum request payload size (null = unlimited) |

---

### GET /traigent/v1/config-space

**Required**: Yes

Returns tunable variable definitions. Traigent uses these to understand what parameters it can optimize.

#### Response (200 OK)

```json
{
  "schema_version": "0.9",
  "capability_id": "my_agent",
  "tunables": [
    {
      "name": "model",
      "type": "enum",
      "domain": {"values": ["gpt-4o", "claude-3-sonnet", "llama-70b"]},
      "default": "gpt-4o"
    },
    {
      "name": "temperature",
      "type": "float",
      "domain": {"range": [0.0, 2.0], "resolution": 0.1},
      "default": 0.7
    },
    {
      "name": "max_tokens",
      "type": "int",
      "domain": {"range": [100, 4096]},
      "default": 1024
    },
    {
      "name": "use_cache",
      "type": "bool",
      "default": true
    }
  ],
  "constraints": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | Yes | TVL schema version (currently "0.9") |
| `capability_id` | string | Yes | Unique identifier for this capability |
| `tunables` | array | Yes | List of tunable variable definitions |
| `constraints` | object | No | Structural and behavioral constraints |

#### Tunable Definition

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Variable name (valid Python identifier) |
| `type` | string | Yes | One of: `enum`, `float`, `int`, `bool`, `str` |
| `domain` | object | Conditional | Domain specification (required for enum, float, int, str) |
| `default` | any | No | Default value |
| `agent` | string | No | Agent name for multi-agent grouping |
| `is_tool` | boolean | No | True if this tunable configures an MCP tool |
| `constraints` | array | No | Conditional constraints (e.g., `"requires model == gpt-4"`) |

#### Tunable Types

**Enum** - Discrete choices:
```json
{
  "name": "model",
  "type": "enum",
  "domain": {"values": ["option_a", "option_b", "option_c"]},
  "default": "option_a"
}
```

**Float** - Continuous range:
```json
{
  "name": "temperature",
  "type": "float",
  "domain": {
    "range": [0.0, 2.0],
    "resolution": 0.1
  },
  "default": 0.7
}
```

**Integer** - Discrete range:
```json
{
  "name": "max_tokens",
  "type": "int",
  "domain": {"range": [100, 4096]},
  "default": 1024
}
```

**Boolean**:
```json
{
  "name": "use_cache",
  "type": "bool",
  "default": true
}
```

**String** - Discrete string values:
```json
{
  "name": "prompt_style",
  "type": "str",
  "domain": {"values": ["concise", "detailed", "technical"]},
  "default": "concise"
}
```

---

### POST /traigent/v1/execute

**Required**: Yes

Execute the agent with a specific configuration on a batch of inputs.

#### Request

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "capability_id": "my_agent",
  "config": {
    "model": "gpt-4o",
    "temperature": 0.5,
    "max_tokens": 1024,
    "use_cache": true
  },
  "inputs": [
    {
      "input_id": "ex_001",
      "data": {"query": "What is machine learning?"}
    },
    {
      "input_id": "ex_002",
      "data": {"query": "Explain neural networks"}
    }
  ],
  "session_id": null,
  "batch_options": {
    "parallelism": 1,
    "fail_fast": false,
    "timeout_per_item_ms": 0
  },
  "timeout_ms": 30000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | No | Idempotency key (UUID). Auto-generated if not provided |
| `capability_id` | string | Yes | Identifier for the capability to invoke |
| `config` | object | Yes | Configuration parameters (tunable values) |
| `inputs` | array | Yes | List of input examples to process |
| `session_id` | string | No | Session ID for stateful agents |
| `batch_options` | object | No | Batch execution control |
| `timeout_ms` | integer | No | Request timeout in milliseconds (default: 30000) |

**Input Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_id` | string | Yes | Unique identifier for this input |
| `data` | object | Yes | Input data (structure defined by your agent) |

**Batch Options**:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `parallelism` | integer | 1 | Maximum concurrent executions |
| `fail_fast` | boolean | false | Stop batch on first failure |
| `timeout_per_item_ms` | integer | 0 | Per-item timeout (0 = use request timeout) |

#### Response (200 OK)

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "execution_id": "exec_123456",
  "status": "completed",
  "outputs": [
    {
      "input_id": "ex_001",
      "output": {
        "response": "Machine learning is a subset of AI...",
        "model_used": "gpt-4o-2024-01-01"
      },
      "cost_usd": 0.002,
      "latency_ms": 150
    },
    {
      "input_id": "ex_002",
      "output": {
        "response": "Neural networks are computing systems...",
        "model_used": "gpt-4o-2024-01-01"
      },
      "cost_usd": 0.003,
      "latency_ms": 180
    }
  ],
  "operational_metrics": {
    "total_cost_usd": 0.005,
    "cost_usd": 0.005,
    "latency_ms": 330,
    "tokens_used": 450,
    "cache_hits": 0,
    "retry_count": 0
  },
  "quality_metrics": null,
  "session_id": null,
  "error": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | Yes | Echoed request ID for correlation |
| `execution_id` | string | Yes | Unique ID for this execution (used in evaluate) |
| `status` | string | Yes | `"completed"`, `"partial"`, or `"failed"` |
| `outputs` | array | Yes | Per-input results |
| `operational_metrics` | object | Yes | Aggregate operational metrics |
| `quality_metrics` | object | No | Quality metrics (if using combined mode) |
| `session_id` | string | No | Session ID for stateful agents |
| `error` | object | No | Error details if status is failed/partial |

**Output Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_id` | string | Yes | Matching input identifier |
| `output` | object | Yes | Agent output (structure defined by your agent) |
| `cost_usd` | number | No | Cost for this input |
| `latency_ms` | number | No | Latency for this input |

**Operational Metrics** (recommended):
| Field | Type | Description |
|-------|------|-------------|
| `total_cost_usd` | number | Total cost in USD |
| `cost_usd` | number | Alias for total_cost_usd |
| `latency_ms` | number | Total latency in milliseconds |

You can add any custom operational metrics:
```json
{
  "operational_metrics": {
    "total_cost_usd": 0.005,
    "latency_ms": 330,
    "tokens_input": 100,
    "tokens_output": 350,
    "api_calls": 2,
    "cache_hit_rate": 0.25,
    "retry_count": 0,
    "model_version": "gpt-4o-2024-01-01"
  }
}
```

---

### POST /traigent/v1/evaluate

**Required**: No (only if `supports_evaluate: true` in capabilities)

Evaluate outputs against expected targets to measure quality.

#### Request

```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "capability_id": "my_agent",
  "execution_id": "exec_123456",
  "evaluations": [
    {
      "input_id": "ex_001",
      "output": {"response": "Machine learning is a subset of AI..."},
      "target": {"expected": "Machine learning is a branch of artificial intelligence..."}
    },
    {
      "input_id": "ex_002",
      "output": {"response": "Neural networks are computing systems..."},
      "target": {"expected": "Neural networks are computer systems inspired by the brain..."}
    }
  ],
  "config": null,
  "session_id": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | No | Idempotency key (UUID) |
| `capability_id` | string | Yes | Identifier for the evaluation capability |
| `execution_id` | string | No | Reference to previous execute (for caching) |
| `evaluations` | array | Yes | List of output+target pairs |
| `config` | object | No | Optional config for evaluation parameters |
| `session_id` | string | No | Session ID for stateful agents |

**Evaluation Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_id` | string | Yes | Matching input identifier |
| `output` | object | Yes | Agent output to evaluate |
| `target` | object | Yes | Expected/reference output |

#### Response (200 OK)

```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "results": [
    {
      "input_id": "ex_001",
      "metrics": {
        "accuracy": 0.92,
        "relevance": 0.88,
        "fluency": 0.95,
        "semantic_similarity": 0.87,
        "hallucination_score": 0.02
      }
    },
    {
      "input_id": "ex_002",
      "metrics": {
        "accuracy": 0.89,
        "relevance": 0.91,
        "fluency": 0.93,
        "semantic_similarity": 0.85,
        "hallucination_score": 0.05
      }
    }
  ],
  "aggregate_metrics": {
    "accuracy": {"mean": 0.905, "std": 0.021, "n": 2},
    "relevance": {"mean": 0.895, "std": 0.021, "n": 2},
    "fluency": {"mean": 0.94, "std": 0.014, "n": 2},
    "semantic_similarity": {"mean": 0.86, "std": 0.014, "n": 2},
    "hallucination_score": {"mean": 0.035, "std": 0.021, "n": 2}
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | Yes | Echoed request ID |
| `status` | string | Yes | `"completed"`, `"partial"`, or `"failed"` |
| `results` | array | Yes | Per-example evaluation results |
| `aggregate_metrics` | object | Yes | Aggregated statistics per metric |

**Result Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_id` | string | Yes | Matching input identifier |
| `metrics` | object | Yes | Quality metrics for this example |

**Aggregate Metric**:
| Field | Type | Description |
|-------|------|-------------|
| `mean` | number | Mean value across all examples |
| `std` | number | Standard deviation |
| `n` | integer | Number of examples |

**Custom Quality Metrics** - add any metrics specific to your use case:
```json
{
  "results": [
    {
      "input_id": "ex_001",
      "metrics": {
        "accuracy": 0.92,
        "semantic_similarity": 0.87,
        "hallucination_score": 0.02,
        "citation_coverage": 0.75,
        "safety_score": 1.0,
        "factual_correctness": 0.90,
        "response_relevance": 0.88,
        "instruction_following": 0.95
      }
    }
  ]
}
```

---

### GET /traigent/v1/health

**Required**: No

Health check endpoint for monitoring.

#### Response (200 OK)

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "details": {
    "database": "connected",
    "cache": "connected"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | string | Yes | `"healthy"`, `"degraded"`, or `"unhealthy"` |
| `version` | string | No | Service version |
| `uptime_seconds` | number | No | Service uptime |
| `details` | object | No | Additional health details |

---

### POST /traigent/v1/keep-alive

**Required**: No (only if `supports_keep_alive: true`)

Session heartbeat for stateful agents.

#### Request

```json
{
  "session_id": "session_abc123"
}
```

#### Response (200 OK)

```json
{
  "status": "alive",
  "session_id": "session_abc123"
}
```

---

## Error Responses

All endpoints should return appropriate HTTP status codes:

| Status | Description |
|--------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 401 | Unauthorized (authentication required) |
| 404 | Not Found (unknown capability) |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

Error response format:

```json
{
  "error": {
    "code": "INVALID_CONFIG",
    "message": "Unknown tunable: invalid_param",
    "details": {
      "invalid_params": ["invalid_param"]
    }
  }
}
```

---

## Data Types Summary

### Status Values

| Value | Description |
|-------|-------------|
| `completed` | All items processed successfully |
| `partial` | Some items failed, others succeeded |
| `failed` | All items failed |

### Tunable Types

| Type | Domain Format | Example |
|------|---------------|---------|
| `enum` | `{"values": [...]}` | `["a", "b", "c"]` |
| `float` | `{"range": [min, max], "resolution": step}` | `[0.0, 1.0]` |
| `int` | `{"range": [min, max]}` | `[0, 100]` |
| `bool` | N/A | `true` / `false` |
| `str` | `{"values": [...]}` | `["opt1", "opt2"]` |

---

## Idempotency

All POST requests support idempotency via the `request_id` field:

- If a request with the same `request_id` is received again, the service may return the cached response
- This enables safe retries without duplicate processing
- Generate `request_id` as a UUID

---

## Related Documentation

- [Client Integration Guide](./hybrid-mode-client-guide.md) - Implementation patterns and examples
- [Flask Demo](../examples/hybrid_mode_demo/) - Working example implementation
- [TraigentService Wrapper](../traigent/wrapper/) - Decorator-based SDK for building services
