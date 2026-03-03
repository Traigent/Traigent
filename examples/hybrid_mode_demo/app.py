"""Traigent Hybrid Mode Demo - Flask Agent Service.

A minimal Flask application demonstrating how to implement
the Traigent Hybrid API for external agentic services.

This demo shows:
- How to define tunable variables (configuration space)
- How to implement the execute endpoint with custom operational metrics
- How to implement the evaluate endpoint with custom quality metrics

Usage:
    # Start the server
    python app.py

    # Or with custom port
    FLASK_PORT=8080 python app.py

    # Test with curl
    curl http://localhost:8080/traigent/v1/capabilities
    curl http://localhost:8080/traigent/v1/config-space
"""

import os
import time
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)

# ============================================================
# STEP 1: Define Your Tunables (Configuration Space)
# ============================================================
# Tunables are the parameters Traigent will optimize.
# Define them using TVL 0.9 format.

TUNABLES = [
    {
        "name": "model",
        "type": "enum",
        "domain": {"values": ["fast", "balanced", "accurate"]},
        "default": "balanced",
    },
    {
        "name": "temperature",
        "type": "float",
        "domain": {"range": [0.0, 1.0], "resolution": 0.1},
        "default": 0.5,
    },
    {
        "name": "max_retries",
        "type": "int",
        "domain": {"range": [0, 5]},
        "default": 2,
    },
    {
        "name": "use_cache",
        "type": "bool",
        "default": True,
    },
]


# ============================================================
# STEP 2: Implement Capabilities Endpoint
# ============================================================
# Returns service features for the handshake with Traigent SDK.


@app.route("/traigent/v1/capabilities", methods=["GET"])
def capabilities():
    """Return service capabilities for handshake.

    Traigent SDK calls this first to discover what features
    your service supports.
    """
    return jsonify(
        {
            "version": "1.0",
            "supports_evaluate": True,  # Set False if you don't implement /evaluate
            "supports_keep_alive": False,  # Set True for stateful agents
            "supports_streaming": False,
            "max_batch_size": 100,
            "tunable_ids": ["demo_agent"],
        }
    )


# ============================================================
# STEP 3: Implement Config Space Endpoint
# ============================================================
# Returns the tunable variable definitions.


@app.route("/traigent/v1/config-space", methods=["GET"])
def config_space():
    """Return tunable variable definitions.

    Traigent SDK uses this to understand what parameters
    it can optimize for your service.
    """
    return jsonify(
        {
            "schema_version": "0.9",
            "tunable_id": "demo_agent",
            "tunables": TUNABLES,
            "constraints": {},
        }
    )


# ============================================================
# STEP 4: Implement Execute (Generate Response) Endpoint
# ============================================================
# This is where your agent logic goes.


@app.route("/traigent/v1/execute", methods=["POST"])
def execute():
    """Execute agent with tunable configuration.

    This is the main endpoint where your agent processes inputs
    using the provided configuration.

    Request format:
        {
            "request_id": "uuid",
            "tunable_id": "demo_agent",
            "config": {"model": "fast", "temperature": 0.5, ...},
            "inputs": [{"input_id": "ex_001", "data": {"query": "..."}}]
        }

    Response format:
        {
            "request_id": "uuid",
            "execution_id": "uuid",
            "status": "completed",
            "outputs": [...],
            "operational_metrics": {"total_cost_usd": ..., ...}
        }
    """
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    missing_fields = [field for field in ("config", "inputs") if field not in data]
    if missing_fields:
        return (
            jsonify(
                {
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                    "message": "Request body failed validation",
                }
            ),
            400,
        )

    config = data.get("config")
    inputs = data.get("inputs")
    if not isinstance(config, dict):
        return (
            jsonify(
                {
                    "error": "Field 'config' must be an object",
                    "message": "Request body failed validation",
                }
            ),
            400,
        )
    if not isinstance(inputs, list) or len(inputs) == 0:
        return (
            jsonify(
                {
                    "error": "Field 'inputs' must be a non-empty array",
                    "message": "Request body failed validation",
                }
            ),
            400,
        )

    for idx, inp in enumerate(inputs):
        if not isinstance(inp, dict) or not inp.get("input_id"):
            return (
                jsonify(
                    {
                        "error": f"Input at index {idx} must include non-empty 'input_id'",
                        "message": "Request body failed validation",
                    }
                ),
                400,
            )

    request_id = data.get("request_id", str(uuid.uuid4()))
    execution_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # YOUR AGENT LOGIC HERE
    # --------------------------------------------------------
    # Replace this with your actual agent implementation.
    # This example shows a mock implementation.

    outputs = []
    total_cost = 0.0
    total_tokens = 0
    start_time = time.time()

    for inp in inputs:
        input_id = inp.get("input_id")

        # Extract config values
        model = config.get("model", "balanced")

        # In production, this would call your actual LLM/agent
        query = inp.get("data", {}).get("query", "")

        # Calculate cost based on model tier
        cost_per_query = {
            "fast": 0.001,
            "balanced": 0.005,
            "accurate": 0.02,
        }.get(model, 0.005)

        # Simulate token usage
        tokens = len(query.split()) * 10 + 50

        total_cost += cost_per_query
        total_tokens += tokens

        # Build per-input output
        outputs.append(
            {
                "input_id": input_id,
                "output_id": f"out_{input_id}_{execution_id}",
                "cost_usd": cost_per_query,
                "latency_ms": 50 + (100 if model == "accurate" else 0),
            }
        )

    elapsed_ms = (time.time() - start_time) * 1000

    # --------------------------------------------------------
    # RETURN OPERATIONAL METRICS
    # --------------------------------------------------------
    # These metrics help Traigent optimize for cost, latency, etc.
    # You can add any custom metrics you want to track.

    return jsonify(
        {
            "request_id": request_id,
            "execution_id": execution_id,
            "status": "completed",
            "outputs": outputs,
            "operational_metrics": {
                # Required metrics
                "total_cost_usd": total_cost,
                "cost_usd": total_cost,  # Alias for compatibility
                "latency_ms": elapsed_ms,
                # Custom operational metrics (optional)
                # Add any metrics you want to track/optimize
                "tokens_used": total_tokens,
                "examples_processed": len(outputs),
                "model_tier": model,
                "cache_enabled": config.get("use_cache", True),
            },
            "quality_metrics": None,  # Set if using combined mode
            "session_id": data.get("session_id"),
            "error": None,
        }
    )


# ============================================================
# STEP 5: Implement Evaluate (Score Outputs) Endpoint
# ============================================================
# This is where your evaluation logic goes.
# Optional - only needed if supports_evaluate=True in capabilities.


@app.route("/traigent/v1/evaluate", methods=["POST"])
def evaluate():
    """Evaluate outputs against expected targets.

    This endpoint scores the quality of your agent's outputs
    by comparing them to expected results.

    Request format:
        {
            "request_id": "uuid",
            "tunable_id": "demo_agent",
            "evaluations": [
                {"input_id": "ex_001", "output": {...}, "target": {...}}
            ]
        }

    Response format:
        {
            "request_id": "uuid",
            "status": "completed",
            "results": [{"input_id": "ex_001", "metrics": {"accuracy": 0.9}}],
            "aggregate_metrics": {"accuracy": {"mean": 0.9, "std": 0.05, "n": 10}}
        }
    """
    data = request.get_json()

    request_id = data.get("request_id", str(uuid.uuid4()))
    evaluations = data.get("evaluations", [])

    # --------------------------------------------------------
    # YOUR EVALUATION LOGIC HERE
    # --------------------------------------------------------
    # Replace this with your actual evaluation implementation.
    # This example shows a mock implementation.

    results = []
    all_metrics = {
        "accuracy": [],
        "relevance": [],
        "fluency": [],
    }

    for eval_item in evaluations:
        input_id = eval_item.get("input_id")
        output = eval_item.get("output", {})
        target = eval_item.get("target", {})

        # Extract text for comparison - handle both dict and string formats
        if isinstance(output, dict):
            output_text = str(output.get("response", ""))
        else:
            output_text = str(output or "")

        if isinstance(target, dict):
            target_text = str(target.get("expected", target.get("answer", "")))
        else:
            target_text = str(target or "")

        # --------------------------------------------------------
        # Calculate quality metrics
        # In production, use real evaluation metrics like:
        # - Semantic similarity (cosine similarity of embeddings)
        # - ROUGE/BLEU scores for text comparison
        # - LLM-as-judge for quality assessment
        # --------------------------------------------------------

        # Mock accuracy based on length similarity
        if target_text:
            length_ratio = min(len(output_text) / len(target_text), 1.0)
            accuracy = 0.7 + (0.3 * length_ratio)
        else:
            accuracy = 0.5

        # Mock relevance and fluency
        relevance = 0.8 + (0.2 * accuracy)
        fluency = 0.9  # Assume good fluency for demo

        metrics = {
            # Standard quality metrics
            "accuracy": round(accuracy, 3),
            "relevance": round(relevance, 3),
            "fluency": round(fluency, 3),
            # Custom quality metrics (optional)
            # Add any metrics specific to your use case
            "response_length": len(output_text),
            "contains_keywords": 1.0 if "model" in output_text.lower() else 0.0,
        }

        results.append(
            {
                "input_id": input_id,
                "metrics": metrics,
            }
        )

        # Collect for aggregation
        for key in ["accuracy", "relevance", "fluency"]:
            all_metrics[key].append(metrics[key])

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------
    # Compute mean, std, n for each metric

    def compute_stats(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = variance**0.5
        else:
            std = 0.0
        return {"mean": round(mean, 4), "std": round(std, 4), "n": n}

    aggregate_metrics = {
        key: compute_stats(values) for key, values in all_metrics.items()
    }

    return jsonify(
        {
            "request_id": request_id,
            "execution_id": data.get("execution_id"),
            "status": "completed",
            "results": results,
            "aggregate_metrics": aggregate_metrics,
        }
    )


# ============================================================
# STEP 6: Health Check (Optional)
# ============================================================
# Useful for monitoring and load balancer health checks.


@app.route("/traigent/v1/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "version": "1.0.0",
            "tunable_id": "demo_agent",
        }
    )


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 8080))
    print(f"Starting Traigent Demo Agent on port {port}")
    print()
    print("Endpoints:")
    print(f"  GET  http://localhost:{port}/traigent/v1/capabilities")
    print(f"  GET  http://localhost:{port}/traigent/v1/config-space")
    print(f"  POST http://localhost:{port}/traigent/v1/execute")
    print(f"  POST http://localhost:{port}/traigent/v1/evaluate")
    print(f"  GET  http://localhost:{port}/traigent/v1/health")
    print()
    print("Test with: python test_mastra_js_api.py")
    print()
    app.run(host="0.0.0.0", port=port, debug=True)
