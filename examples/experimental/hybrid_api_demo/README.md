# Traigent Hybrid API Demo

> **Experimental** — This example demonstrates the `hybrid_api` execution mode, which is functional but may change in future releases.

A minimal Flask application demonstrating how to implement the Traigent Hybrid API for external agentic services.

## Overview

This demo shows how to create a REST API that Traigent can optimize. It implements:

- **Tunables**: Configuration parameters that Traigent will optimize
- **Execute**: Generate responses using your agent with provided configuration
- **Evaluate**: Score outputs against expected targets (optional)

## Quick Start

### 1. Install Dependencies

```bash
pip install flask requests
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:8080`.

### 3. Test the API

In another terminal:

```bash
python test_mastra_js_api.py
```

Or test manually with curl:

```bash
# Check capabilities
curl http://localhost:8080/traigent/v1/capabilities

# Get tunable definitions
curl http://localhost:8080/traigent/v1/config-space

# Execute with configuration
curl -X POST http://localhost:8080/traigent/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test-001",
    "tunable_id": "demo_agent",
    "config": {"model": "accurate", "temperature": 0.7},
    "examples": [{"example_id": "ex_001", "data": {"query": "What is AI?"}}]
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/traigent/v1/capabilities` | GET | Service capabilities (features supported) |
| `/traigent/v1/config-space` | GET | Tunable variable definitions |
| `/traigent/v1/execute` | POST | Generate responses with configuration |
| `/traigent/v1/evaluate` | POST | Score outputs against targets |
| `/traigent/v1/health` | GET | Health check |

## Tunable Types

The demo defines four types of tunables:

```python
TUNABLES = [
    # Enum: Discrete choices
    {
        "name": "model",
        "type": "enum",
        "domain": {"values": ["fast", "balanced", "accurate"]},
        "default": "balanced",
    },
    # Float: Continuous range with resolution
    {
        "name": "temperature",
        "type": "float",
        "domain": {"range": [0.0, 1.0], "resolution": 0.1},
        "default": 0.5,
    },
    # Integer: Discrete range
    {
        "name": "max_retries",
        "type": "int",
        "domain": {"range": [0, 5]},
        "default": 2,
    },
    # Boolean
    {
        "name": "use_cache",
        "type": "bool",
        "default": True,
    },
]
```

## Custom Metrics

### Operational Metrics (in execute response)

Track costs, latency, and custom operational data:

```python
"operational_metrics": {
    # Required metrics
    "total_cost_usd": 0.005,
    "latency_ms": 150,
    # Custom metrics (add any you want to track/optimize)
    "tokens_used": 350,
    "examples_processed": 2,
    "model_tier": "accurate",
    "cache_enabled": True,
}
```

### Quality Metrics (in evaluate response)

Measure output quality with custom metrics:

```python
"metrics": {
    # Standard quality metrics
    "accuracy": 0.92,
    "relevance": 0.88,
    "fluency": 0.95,
    # Custom quality metrics
    "response_length": 150,
    "contains_keywords": 1.0,
}
```

## Integrating with Traigent

Once your server is running, optimize it with Traigent:

```python
import traigent

@traigent.optimize(
    execution_mode="hybrid_api",
    hybrid_api_endpoint="http://localhost:8080",
    tunable_id="demo_agent",
    batch_size=10,
    max_trials=20,
)
def my_agent():
    pass  # Not used in hybrid mode - the endpoint handles execution

result = traigent.run(my_agent, dataset=my_dataset)
print(f"Best config: {result.best_config}")
print(f"Best metrics: {result.best_metrics}")
```

## Customization

### Adding Custom Tunables

Edit the `TUNABLES` list in `app.py` to add your own parameters:

```python
TUNABLES = [
    {
        "name": "my_custom_param",
        "type": "float",
        "domain": {"range": [0.0, 100.0], "resolution": 1.0},
        "default": 50.0,
    },
    # ... more tunables
]
```

### Implementing Real Agent Logic

Replace the mock implementation in the `execute()` function with your actual agent:

```python
@app.route("/traigent/v1/execute", methods=["POST"])
def execute():
    data = request.get_json()
    config = data.get("config", {})
    examples = data.get("examples", [])

    outputs = []
    for inp in examples:
        # Call your actual agent/LLM here
        result = my_real_agent(
            query=inp["data"]["query"],
            model=config.get("model"),
            temperature=config.get("temperature"),
        )
        outputs.append({
            "example_id": inp["example_id"],
            "output": {"response": result.text},
            "cost_usd": result.cost,
            "latency_ms": result.latency,
        })

    return jsonify({...})
```

### Adding Custom Quality Metrics

Replace the mock evaluation in the `evaluate()` function with real metrics:

```python
@app.route("/traigent/v1/evaluate", methods=["POST"])
def evaluate():
    # Use semantic similarity, ROUGE, LLM-as-judge, etc.
    accuracy = compute_semantic_similarity(output_text, target_text)
    hallucination = detect_hallucination(output_text)
    safety = check_safety_guidelines(output_text)

    metrics = {
        "accuracy": accuracy,
        "hallucination_score": hallucination,
        "safety_score": safety,
    }
    ...
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Flask application with all endpoints |
| `test_mastra_js_api.py` | Test script to verify all endpoints |
| `requirements.txt` | Python dependencies |

## Next Steps

1. Read the [API Contract](../../../docs/hybrid-mode-api-contract.md) for detailed specifications
2. See the [Client Integration Guide](../../../docs/hybrid-mode-client-guide.md) for implementation patterns
3. Use the [TraigentService wrapper](../../../traigent/wrapper/) for a decorator-based approach
