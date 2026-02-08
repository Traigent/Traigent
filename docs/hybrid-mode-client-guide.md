# Traigent Hybrid Mode - Client Integration Guide

This guide explains how to integrate your external agentic service with Traigent for optimization.

## Overview

Traigent's Hybrid Mode allows you to optimize any external service by implementing a simple REST API. Traigent will:

1. **Discover** your tunable parameters automatically
2. **Execute** your agent with different configurations
3. **Evaluate** outputs against targets
4. **Optimize** to find the best configuration

## Privacy-Preserving Mode (Default)

Traigent Hybrid Mode is designed with **privacy as the default**. Only configuration values and metrics are observed during optimization - your actual data content is never transmitted to Traigent.

### What Gets Transmitted

| Data Type | Transmitted | Description |
|-----------|-------------|-------------|
| Tunable definitions | Yes | Your config-space response |
| Configuration values | Yes | Tunable values per trial |
| Operational metrics | Yes | Cost, latency, tokens, etc. |
| Quality metrics | Yes | Accuracy, relevance, etc. |
| Input data content | **No** | Only input IDs |
| Output data content | **No** | Only output IDs |
| Target data content | **No** | Only target IDs |

### How to Implement Privacy-Preserving Mode

**Execute endpoint**: Return `output_id` instead of `output` content:

```python
@app.route("/traigent/v1/execute", methods=["POST"])
def execute():
    outputs = []
    for inp in inputs:
        result = process_input(inp["input_id"])  # Look up by ID locally
        output_id = generate_output_id(inp["input_id"], session_id)
        store_output_locally(output_id, result)  # Store locally

        outputs.append({
            "input_id": inp["input_id"],
            "output_id": output_id,  # Only transmit ID, not content
            "cost_usd": 0.005,
        })

    return jsonify({...})
```

**Evaluate endpoint**: Accept `output_id` and `target_id` instead of content:

```python
@app.route("/traigent/v1/evaluate", methods=["POST"])
def evaluate():
    results = []
    for eval_item in evaluations:
        # Look up data locally using IDs
        output = get_output_locally(eval_item["output_id"])
        target = get_target_locally(eval_item["target_id"])

        metrics = compute_metrics(output, target)
        results.append({
            "input_id": eval_item["input_id"],
            "metrics": metrics,
        })

    return jsonify({...})
```

### Session and ID Management

For privacy-preserving mode, your service must:

1. **Generate stable input IDs**: These should be consistent across optimization runs
2. **Scope output IDs to sessions**: Include session_id to prevent collisions between runs
3. **Store data locally**: Maintain a mapping of IDs to actual content

```python
def generate_output_id(input_id: str, session_id: str) -> str:
    return f"out_{input_id}_{session_id}"

# Storage example
output_storage: dict[str, dict] = {}

def store_output_locally(output_id: str, data: dict):
    output_storage[output_id] = data

def get_output_locally(output_id: str) -> dict:
    return output_storage[output_id]
```

## Quick Start

### Option 1: Implement REST API Directly

If using Flask, FastAPI, or another framework, implement these endpoints:

```
GET  /traigent/v1/capabilities   - Required
GET  /traigent/v1/config-space   - Required
POST /traigent/v1/execute        - Required
POST /traigent/v1/evaluate       - Optional
GET  /traigent/v1/health         - Optional
```

See the [API Contract](./hybrid-mode-api-contract.md) for detailed specifications.

### Option 2: Use the TraigentService Wrapper

The SDK provides a decorator-based wrapper for building services:

```python
from traigent.wrapper import TraigentService

app = TraigentService(capability_id="my_agent")

@app.tunables
def config_space():
    return {
        "model": {"type": "enum", "values": ["gpt-4", "claude-3"]},
        "temperature": {"type": "float", "range": [0.0, 1.0]},
    }

@app.execute
async def generate_response(input_id: str, data: dict, config: dict) -> dict:
    result = await my_agent_logic(data["query"], **config)
    return {
        "output": result,
        "cost_usd": 0.005,
        "metrics": {"tokens_used": 150}  # Custom operational metrics
    }

@app.evaluate
async def score_output(output: dict, target: dict, config: dict) -> dict:
    return {
        "accuracy": compute_accuracy(output, target),
        "safety_score": check_safety(output),  # Custom quality metrics
    }

if __name__ == "__main__":
    app.run(port=8080)
```

---

## Defining Tunables

Tunables are the parameters Traigent will optimize. Define them in your config-space response.

### Enum (Discrete Choices)

Use for model selection, prompt templates, or discrete options:

```json
{
  "name": "model",
  "type": "enum",
  "domain": {"values": ["gpt-4o", "claude-3-sonnet", "llama-70b"]},
  "default": "gpt-4o"
}
```

### Float (Continuous Range)

Use for temperature, top_p, or other continuous parameters:

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

The `resolution` field is optional and specifies the step size for sampling.

### Integer (Discrete Range)

Use for max_tokens, retry counts, or other integer parameters:

```json
{
  "name": "max_tokens",
  "type": "int",
  "domain": {"range": [100, 4096]},
  "default": 1024
}
```

### Boolean

Use for feature flags:

```json
{
  "name": "use_caching",
  "type": "bool",
  "default": true
}
```

### Multi-Agent Grouping

For services with multiple agents, use the `agent` field:

```json
{
  "tunables": [
    {
      "name": "model",
      "type": "enum",
      "domain": {"values": ["gpt-4o", "claude-3"]},
      "agent": "planner"
    },
    {
      "name": "model",
      "type": "enum",
      "domain": {"values": ["gpt-4o-mini", "claude-3-haiku"]},
      "agent": "executor"
    }
  ]
}
```

---

## Returning Custom Metrics

### Operational Metrics (Execute Response)

Track costs, latency, and any operational data in the execute response:

```json
{
  "operational_metrics": {
    "total_cost_usd": 0.005,
    "latency_ms": 150,
    "tokens_input": 100,
    "tokens_output": 200,
    "api_calls": 1,
    "cache_hit_rate": 0.5,
    "retry_count": 0,
    "model_version": "gpt-4o-2024-01-01"
  }
}
```

**Required metrics:**
- `total_cost_usd` - Total cost in USD (or `cost_usd` as alias)
- `latency_ms` - Total latency in milliseconds

**Recommended metrics:**
- `tokens_input` - Input token count
- `tokens_output` - Output token count
- `cache_hit_rate` - Cache hit ratio (0.0-1.0)
- `retry_count` - Number of retries

**Custom metrics:**
Add any metrics specific to your service. Traigent can optimize for any numeric metric.

### Quality Metrics (Evaluate Response)

Measure output quality with any metrics you need:

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
        "instruction_following": 0.95
      }
    }
  ],
  "aggregate_metrics": {
    "accuracy": {"mean": 0.92, "std": 0.05, "n": 100},
    "hallucination_score": {"mean": 0.02, "std": 0.01, "n": 100}
  }
}
```

**Common quality metrics:**
- `accuracy` - Overall correctness
- `semantic_similarity` - Embedding-based similarity to target
- `hallucination_score` - Presence of fabricated information (lower is better)
- `citation_coverage` - How well citations support claims
- `safety_score` - Adherence to safety guidelines
- `factual_correctness` - Factual accuracy
- `instruction_following` - How well instructions were followed

---

## Implementation Examples

### Flask Example

See the complete [Flask demo](../examples/hybrid_mode_demo/app.py):

```python
from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)

TUNABLES = [
    {"name": "model", "type": "enum", "domain": {"values": ["fast", "accurate"]}},
    {"name": "temperature", "type": "float", "domain": {"range": [0.0, 1.0]}},
]

@app.route("/traigent/v1/capabilities", methods=["GET"])
def capabilities():
    return jsonify({
        "version": "1.0",
        "supports_evaluate": True,
    })

@app.route("/traigent/v1/config-space", methods=["GET"])
def config_space():
    return jsonify({
        "schema_version": "0.9",
        "capability_id": "my_agent",
        "tunables": TUNABLES,
    })

@app.route("/traigent/v1/execute", methods=["POST"])
def execute():
    data = request.get_json()
    config = data.get("config", {})
    inputs = data.get("inputs", [])

    outputs = []
    total_cost = 0.0

    for inp in inputs:
        # Your agent logic here
        result = process_input(inp["data"], config)
        cost = calculate_cost(config)
        total_cost += cost

        outputs.append({
            "input_id": inp["input_id"],
            "output": result,
            "cost_usd": cost,
        })

    return jsonify({
        "request_id": data.get("request_id", str(uuid.uuid4())),
        "execution_id": str(uuid.uuid4()),
        "status": "completed",
        "outputs": outputs,
        "operational_metrics": {
            "total_cost_usd": total_cost,
            "latency_ms": 150,
        },
    })
```

### FastAPI Example

```python
from fastapi import FastAPI
from pydantic import BaseModel
import uuid

app = FastAPI()

class ExecuteRequest(BaseModel):
    request_id: str | None = None
    capability_id: str
    config: dict
    inputs: list[dict]

@app.get("/traigent/v1/capabilities")
async def capabilities():
    return {
        "version": "1.0",
        "supports_evaluate": True,
    }

@app.get("/traigent/v1/config-space")
async def config_space():
    return {
        "schema_version": "0.9",
        "capability_id": "my_agent",
        "tunables": [
            {"name": "model", "type": "enum", "domain": {"values": ["fast", "accurate"]}},
        ],
    }

@app.post("/traigent/v1/execute")
async def execute(req: ExecuteRequest):
    outputs = []
    for inp in req.inputs:
        result = await process_input(inp["data"], req.config)
        outputs.append({
            "input_id": inp["input_id"],
            "output": result,
            "cost_usd": 0.005,
        })

    return {
        "request_id": req.request_id or str(uuid.uuid4()),
        "execution_id": str(uuid.uuid4()),
        "status": "completed",
        "outputs": outputs,
        "operational_metrics": {"total_cost_usd": 0.005 * len(outputs)},
    }
```

---

## Testing Your Implementation

### Using curl

```bash
# Test capabilities
curl http://localhost:8080/traigent/v1/capabilities

# Test config space
curl http://localhost:8080/traigent/v1/config-space

# Test execute
curl -X POST http://localhost:8080/traigent/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "capability_id": "my_agent",
    "config": {"model": "fast", "temperature": 0.5},
    "inputs": [
      {"input_id": "ex_001", "data": {"query": "Hello"}}
    ]
  }'

# Test evaluate
curl -X POST http://localhost:8080/traigent/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-002",
    "capability_id": "my_agent",
    "evaluations": [
      {
        "input_id": "ex_001",
        "output": {"response": "Hi there!"},
        "target": {"expected": "Hello!"}
      }
    ]
  }'
```

### Using the Test Client

See the [test client](../examples/hybrid_mode_demo/test_client.py) for a complete testing script.

### Using Traigent SDK

```python
from traigent.hybrid import create_transport, HybridExecuteRequest
import asyncio

async def test_connection():
    transport = create_transport(
        transport_type="http",
        base_url="http://localhost:8080",
    )

    # Test capabilities
    caps = await transport.capabilities()
    print(f"Connected: version={caps.version}")

    # Test config space
    config_space = await transport.discover_config_space()
    print(f"Tunables: {[t.name for t in config_space.tunables]}")

    # Test execute
    response = await transport.execute(
        HybridExecuteRequest(
            capability_id="my_agent",
            config={"model": "fast"},
            inputs=[{"input_id": "ex_001", "data": {"query": "test"}}],
        )
    )
    print(f"Status: {response.status}")

asyncio.run(test_connection())
```

---

## Integration with Traigent Optimization

Once your service implements the API, optimize it with Traigent:

```python
import traigent

@traigent.optimize(
    execution_mode="hybrid_api",
    hybrid_api_endpoint="http://your-service:8080",
    capability_id="my_agent",
    eval_dataset=my_dataset,

    # Provide tunables/configuration space for optimizer search
    configuration_space={
        "model": ["fast", "accurate", "balanced"],
        "temperature": (0.0, 1.0),
    },

    # Optimization settings
    max_trials=50,
    cost_limit=10.0,
    hybrid_api_batch_size=10,
    hybrid_api_batch_parallelism=2,

    # Objectives (use your custom metric names)
    objectives=["accuracy", "cost", "latency"],
)
def my_agent(_query: str):
    return ""  # Execution is delegated to the external hybrid API

result = my_agent.optimize()
print(f"Best config: {result.best_config}")
print(f"Best metrics: {result.best_metrics}")
```

### Optimization Objectives

Traigent can optimize for any metrics you return:

```python
@traigent.optimize(
    # Minimize cost and latency, maximize accuracy
    objectives={
        "accuracy": "maximize",
        "total_cost_usd": "minimize",
        "hallucination_score": "minimize",
    },
    # Or use shorthand (all minimize by default except accuracy)
    objectives=["accuracy", "cost", "latency"],
)
```

---

## Best Practices

### 1. Idempotency

Use `request_id` for safe retries:

```python
@app.route("/traigent/v1/execute", methods=["POST"])
def execute():
    request_id = request.json.get("request_id")

    # Check cache for existing result
    cached = get_cached_result(request_id)
    if cached:
        return jsonify(cached)

    # Process and cache result
    result = process_request(request.json)
    cache_result(request_id, result)
    return jsonify(result)
```

### 2. Error Handling

Return appropriate HTTP status codes and structured errors:

```python
@app.route("/traigent/v1/execute", methods=["POST"])
def execute():
    try:
        data = request.get_json()
        if not data.get("inputs"):
            return jsonify({
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "inputs field is required"
                }
            }), 400

        result = process_request(data)
        return jsonify(result)

    except RateLimitError:
        return jsonify({
            "error": {"code": "RATE_LIMITED", "message": "Too many requests"}
        }), 429
    except Exception as e:
        return jsonify({
            "error": {"code": "INTERNAL_ERROR", "message": str(e)}
        }), 500
```

### 3. Partial Results

Return partial results when some inputs fail:

```python
def execute():
    outputs = []
    errors = []

    for inp in inputs:
        try:
            result = process_input(inp)
            outputs.append({"input_id": inp["input_id"], "output": result})
        except Exception as e:
            errors.append({"input_id": inp["input_id"], "error": str(e)})

    return jsonify({
        "status": "partial" if errors else "completed",
        "outputs": outputs,
        "error": {"failed_inputs": errors} if errors else None,
    })
```

### 4. Timeout Handling

Respect the `timeout_ms` parameter:

```python
import asyncio

@app.route("/traigent/v1/execute", methods=["POST"])
async def execute():
    data = request.get_json()
    timeout_ms = data.get("timeout_ms", 30000)
    timeout_sec = timeout_ms / 1000

    try:
        result = await asyncio.wait_for(
            process_request(data),
            timeout=timeout_sec
        )
        return jsonify(result)
    except asyncio.TimeoutError:
        return jsonify({
            "status": "failed",
            "error": {"code": "TIMEOUT", "message": f"Request timed out after {timeout_ms}ms"}
        }), 504
```

---

## Troubleshooting

### Connection Refused

```
Error: Connection refused to http://localhost:8080
```

**Solution**: Ensure your server is running and listening on the correct port.

### Invalid Config

```
Error: Unknown tunable parameter: invalid_param
```

**Solution**: Only use parameters defined in your config-space response.

### Missing Metrics

```
Warning: Missing required metric: total_cost_usd
```

**Solution**: Include `total_cost_usd` in your operational_metrics response.

### Timeout

```
Error: Request timed out after 30000ms
```

**Solution**: Increase `timeout_ms` in the request or optimize your service.

---

## Related Documentation

- [API Contract](./hybrid-mode-api-contract.md) - Detailed endpoint specifications
- [Flask Demo](../examples/hybrid_mode_demo/) - Working example implementation
- [TraigentService Wrapper](../traigent/wrapper/) - Decorator-based SDK
