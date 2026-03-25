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
execute_response_schema = {
    "$schema": schemas.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
    "$ref": "#/definitions/ExecuteResponse",
    "definitions": schemas["definitions"],
}
validate(response_data, execute_response_schema)
```

### Publishing and Versioning (Swagger)

Use `docs/hybrid-mode-openapi.yaml` as the single source of truth and publish it via Swagger UI (or Redoc) instead of ad-hoc zip bundles.

- Keep `info.version` in the OpenAPI file under semantic versioning (`MAJOR.MINOR.PATCH`).
- Publish each released spec from git tags (for example `v1.0.0`, `v1.0.1`).
- Run validation in CI before publish:
  - `npx @apidevtools/swagger-cli validate docs/hybrid-mode-openapi.yaml`
  - `npx @redocly/cli lint docs/hybrid-mode-openapi.yaml`

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

## Authentication (Optional)

If your service requires authentication, accept an `Authorization` header (for example, bearer token).

Configure the Traigent SDK client with:
- `hybrid_api_auth_header="Bearer <token>"`

For current contract compatibility, keep auth policy consistent across all endpoints. RBAC and stronger policy models can be layered later without changing endpoint shapes.

---

## Privacy-Preserving Mode (Default)

Traigent Hybrid Mode is designed with privacy as the default. **Only configuration values and metrics are observed during optimization** - the actual content of inputs, outputs, and targets is never transmitted to Traigent.

### How It Works

1. **Inputs**: Only `example_id` is required. Your service stores input data locally and looks it up by ID.
2. **Outputs**: Return `output_id` instead of `output` content. Your service stores outputs locally.
3. **Evaluation**: Use `output_id` and `target_id` instead of actual content. Your service performs evaluation locally.

### What Traigent Observes

| Data | Observed by Traigent |
|------|---------------------|
| Tunable definitions (config-space) | ✓ Yes |
| Configuration values per trial | ✓ Yes |
| Operational metrics (cost, latency, tokens) | ✓ Yes |
| Quality metrics (accuracy, relevance, etc.) | ✓ Yes |
| Input data content | ✗ No (only IDs) |
| Output data content | ✗ No (only IDs) |
| Target/expected output content | ✗ No (only IDs) |

### Session and ID Management

For privacy-preserving mode, your service must:

1. **Generate unique IDs**: Create stable identifiers for inputs, outputs, and targets
2. **Track session context**: Use `session_id` to scope outputs to specific optimization runs
3. **Store data locally**: Maintain a local mapping of IDs to actual content
4. **Handle ID collisions**: Different optimization runs may reuse example IDs but produce different outputs - use `session_id` to distinguish

**Example ID format**:

```text
example_id: "example_001"                  # Stable across runs
output_id: "out_example_001_run_abc123"    # Scoped to run
target_id: "target_001"                    # Stable across runs
```

### Full-Content Datasets (Optional)

If privacy is not a concern, you can send full content instead of IDs:
- Include `data` in `ExampleItem`
- Include `output` in `OutputItem`
- Include `output` and `target` in `EvaluationItem`

Both modes can be mixed — for example, send input `data` but return only `output_id`.

---

## Evaluation Mode Selection: Two-Phase vs Combined

The SDK automatically selects how evaluation happens based on the execute response. Your service controls this by what it returns in `quality_metrics` and declares in `supports_evaluate`.

| Execute Response | Capabilities | SDK Behavior |
|------------------|-------------|--------------|
| `quality_metrics` is non-null (contains data) | Any | **Combined Mode** — SDK uses the returned quality metrics directly. `/evaluate` is NOT called. |
| `quality_metrics` is `null` or absent | `supports_evaluate: true` | **Two-Phase Mode** — SDK calls `POST /evaluate` separately to obtain quality metrics. |
| `quality_metrics` is `null` or absent | `supports_evaluate: false` | **Execute-only** — Only operational metrics (cost, latency) are collected. No quality evaluation. |

### Combined Mode

Return quality metrics directly in the execute response to skip the separate evaluate call:

```json
{
  "status": "completed",
  "outputs": [
    {
      "example_id": "ex_001",
      "output_id": "out_001_run_abc",
      "cost_usd": 0.005,
      "metrics": {"accuracy": 0.92, "relevance": 0.88}
    }
  ],
  "operational_metrics": {"total_cost_usd": 0.005},
  "quality_metrics": {"accuracy": 0.92, "relevance": 0.88}
}
```

### Two-Phase Mode

Return `quality_metrics: null` (or omit it) and set `supports_evaluate: true` in capabilities. The SDK will call `/evaluate` after `/execute`:

```json
{
  "status": "completed",
  "outputs": [
    {
      "example_id": "ex_001",
      "output_id": "out_001_run_abc",
      "cost_usd": 0.005
    }
  ],
  "operational_metrics": {"total_cost_usd": 0.005},
  "quality_metrics": null
}
```

The SDK then sends the outputs to `POST /evaluate` for quality scoring.

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
  "max_payload_bytes": null,
  "tunable_ids": ["my_agent"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | API version (currently "1.0") |
| `supports_evaluate` | boolean | No | Whether `/evaluate` endpoint is available (default: `true`) |
| `supports_keep_alive` | boolean | No | Whether session heartbeat is supported |
| `supports_streaming` | boolean | No | Whether streaming responses are supported |
| `max_batch_size` | integer | No | Maximum inputs per execute request (default: 100) |
| `max_payload_bytes` | integer | No | Maximum request payload size (null = unlimited) |
| `tunable_ids` | array[string] | No | Optional list of tunable IDs supported by this service |

Notes:
- `tunable_ids` is optional and useful for multi-tunable services.
- The canonical tunable used for execution/evaluation remains `tunable_id` returned by `GET /traigent/v1/config-space`.

---

### GET /traigent/v1/config-space

**Required**: Yes

Returns tunable variable definitions. Traigent uses these to understand what parameters it can optimize.

#### Response (200 OK)

```json
{
  "schema_version": "0.9",
  "tunable_id": "my_agent",
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
  "objectives": [
    {"name": "accuracy", "direction": "maximize", "weight": 2.0},
    {"name": "cost", "direction": "minimize", "weight": 1.0}
  ],
  "exploration": {
    "strategy": "nsga2",
    "budgets": {"max_trials": 50, "max_spend_usd": 10.0}
  },
  "promotion_policy": {
    "dominance": "epsilon_pareto",
    "alpha": 0.05,
    "min_effect": {"accuracy": 0.02}
  },
  "defaults": {"model": "gpt-4o"},
  "measures": ["accuracy", "cost", "latency"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | Yes | TVL schema version (currently "0.9") |
| `tunable_id` | string | Yes | Unique identifier for this tunable |
| `tunables` | array | Yes | List of tunable variable definitions |
| `constraints` | object \| array | No | Structural and behavioral constraints |
| `objectives` | array | No | Optional objective definitions (TVL 0.9 JSON format) |
| `exploration` | object | No | Optional exploration strategy, budgets, and convergence |
| `promotion_policy` | object | No | Optional promotion policy (epsilon-Pareto) |
| `defaults` | object | No | Optional default configuration values |
| `measures` | array[string] | No | Optional declared metric names produced by service |

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
  "tunable_id": "my_agent",
  "config": {
    "model": "gpt-4o",
    "temperature": 0.5,
    "max_tokens": 1024,
    "use_cache": true
  },
  "examples": [
    {
      "example_id": "ex_001",
      "data": {"query": "What is machine learning?"}
    },
    {
      "example_id": "ex_002",
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
| `tunable_id` | string | Yes | Identifier for the tunable to invoke |
| `config` | object | Yes | Configuration parameters (tunable values) |
| `examples` | array | Yes | List of input examples to process |
| `session_id` | string | No | Session ID for stateful agents |
| `batch_options` | object | No | Batch execution control |
| `timeout_ms` | integer | No | Request timeout in milliseconds (default: 30000) |

**Example Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `example_id` | string | Yes | Unique identifier for this input |
| `data` | object | No | Input data (optional in privacy-preserving mode) |

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
      "example_id": "ex_001",
      "output": {
        "response": "Machine learning is a subset of AI...",
        "model_used": "gpt-4o-2024-01-01"
      },
      "cost_usd": 0.002,
      "latency_ms": 150
    },
    {
      "example_id": "ex_002",
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
| `example_id` | string | Yes | Matching input identifier |
| `output` | object | No | Agent output (optional in privacy-preserving mode) |
| `output_id` | string | No | Output identifier (for privacy-preserving mode) |
| `cost_usd` | number | No | Cost for this input |
| `latency_ms` | number | No | Latency for this input |
| `metrics` | object | No | Per-input quality metrics (combined execute+evaluate mode) |
| `error` | string | No | Error message for this input when execution partially fails |

**Operational Metrics**:
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
  "tunable_id": "my_agent",
  "execution_id": "exec_123456",
  "evaluations": [
    {
      "example_id": "ex_001",
      "output": {"response": "Machine learning is a subset of AI..."},
      "target": {"expected": "Machine learning is a branch of artificial intelligence..."}
    },
    {
      "example_id": "ex_002",
      "output": {"response": "Neural networks are computing systems..."},
      "target": {"expected": "Neural networks are computer systems inspired by the brain..."}
    }
  ],
  "config": null,
  "session_id": null,
  "timeout_ms": 30000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | No | Idempotency key (UUID) |
| `tunable_id` | string | Yes | Identifier for the evaluation tunable |
| `execution_id` | string | No | Reference to previous execute (for caching) |
| `evaluations` | array | No | List of output+target pairs (optional when using execution_id-only workflows) |
| `config` | object | No | Optional config for evaluation parameters |
| `session_id` | string | No | Session ID for stateful agents |
| `timeout_ms` | integer | No | Optional request timeout in milliseconds (wrapper returns `408` on timeout) |

**Evaluation Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `example_id` | string | Yes | Matching input identifier |
| `output` | object | No | Agent output (optional in privacy-preserving mode) |
| `output_id` | string | No | Output identifier (for privacy-preserving mode) |
| `target` | object | No | Expected/reference output (optional in privacy-preserving mode) |
| `target_id` | string | No | Target identifier (for privacy-preserving mode) |

#### Response (200 OK)

```json
{
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "results": [
    {
      "example_id": "ex_001",
      "metrics": {
        "accuracy": 0.92,
        "relevance": 0.88,
        "fluency": 0.95,
        "semantic_similarity": 0.87,
        "hallucination_score": 0.02
      }
    },
    {
      "example_id": "ex_002",
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
| `error` | object | No | Error details for partial/failed evaluations (`code`, `message`, `failed_examples`) |

**Result Item**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `example_id` | string | Yes | Matching input identifier |
| `metrics` | object | Yes | Quality metrics for this example |
| `error` | string | No | Error message for this example when evaluation partially fails |

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
      "example_id": "ex_001",
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

#### Response (404 Not Found)

```json
{
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "Session not found: session_abc123"
  }
}
```

---

## Error Responses

All endpoints should return structured errors with explicit retry behavior:

| Status | Meaning | Client behavior |
|--------|---------|-----------------|
| 400 | Bad request / validation failure | Do not retry until payload is fixed |
| 401 | Unauthorized | Refresh/reconfigure credentials, then retry |
| 404 | Not found (route/session) | Do not retry blindly; re-discover or recreate state |
| 408 | Request timeout (`timeout_ms` budget exceeded) | Safe to retry with backoff and/or higher timeout if idempotent |
| 429 | Rate limited | Retry with backoff, honor `Retry-After` when present |
| 500 | Internal server error | Retry with bounded backoff; alert on repeated failures |
| 503 | Temporary service unavailability | Retry with backoff, honor `Retry-After` when present |

Wrapper behavior notes:
- Wrapper server enforces `timeout_ms` and emits `408 REQUEST_TIMEOUT` for `POST /execute` and optional `timeout_ms` on `POST /evaluate`.
- Services can intentionally emit `401`, `429`, or `503` by raising wrapper error types (`UnauthorizedError`, `RateLimitError`, `ServiceUnavailableError`).
- OpenAPI operation-level response lists are normative for each endpoint.

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

## Observability and Tracing

For distributed tracing, clients may send W3C trace headers:

- `traceparent`
- `tracestate`

Wrapper behavior:
- Incoming trace headers are echoed in responses.
- Wrapper adds `x-traigent-request-id` when a request includes/produces `request_id`.
- `request_id` is required for business-level correlation; trace headers are complementary for span-level correlation.

Transport note:
- Production contract target is `HTTPS + HTTP/2`.
- Traigent HTTP transport supports strict enforcement via:
  - `create_transport(..., require_http2=True)` (rejects non-HTTPS endpoints and non-HTTP/2 responses)
- Contract v1 remains synchronous request/response; async job APIs and gRPC are future roadmap items.

---

## Data Types Summary

### Stop Reasons

When optimization completes, the SDK reports a `stop_reason` indicating why it stopped:

| Value | Description |
|-------|-------------|
| `max_trials_reached` | Maximum number of trials completed |
| `max_samples_reached` | Maximum dataset samples consumed |
| `cost_limit` | Budget limit exceeded |
| `timeout` | Wall-clock time limit exceeded |
| `plateau` | Convergence criterion met (no improvement) |
| `optimizer` | Optimizer signaled completion |
| `condition` | Generic stop condition triggered |
| `user_cancelled` | User cancelled the run |
| `error` | Optimization failed due to an exception |
| `vendor_error` | Provider error (rate limit, quota, or service failure) |
| `network_error` | Connectivity failure |

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

All `GET` endpoints (`/capabilities`, `/config-space`, `/health`) must be side-effect free and idempotent by contract.

`POST /execute` and `POST /evaluate` support idempotency via the `request_id` field:

- If a request with the same `request_id` and the same payload is received again, the service should return the cached response.
- If the same `request_id` is reused with a different payload, the service should return `400 INVALID_REQUEST`.
- This enables safe retries without duplicate processing
- Generate `request_id` as a UUID

`POST /keep-alive` is session-heartbeat semantics and does not use `request_id`.

Wrapper behavior:
- `TraigentService` implements in-process idempotency caches for `execute` and `evaluate`.
- If your server does not implement idempotency, retrying the same request can run it twice and duplicate side effects/costs.

Optional extension:
- `OPTIONS` can be implemented for method discovery/CORS convenience, but it is not required by the core hybrid contract.

---

## Related Documentation

- [Client Integration Guide](./hybrid-mode-client-guide.md) - Implementation patterns and examples
- [TraigentService Wrapper](../traigent/wrapper/) - Decorator-based SDK for building services
