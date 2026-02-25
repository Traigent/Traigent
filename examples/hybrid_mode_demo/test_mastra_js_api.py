"""API contract tests for the JS-Mastra demo server.

Validates all Traigent Hybrid API endpoints:
  /health, /capabilities, /config-space, /execute, /evaluate

Usage:
    # Start the JS-Mastra demo server first:
    cd JS-Mastra-APIs-Validation && npm run api:dev

    # Then run this test:
    .venv/bin/python examples/hybrid_mode_demo/test_mastra_js_api.py

    # Or specify a different URL:
    .venv/bin/python examples/hybrid_mode_demo/test_mastra_js_api.py http://localhost:3000

Environment variables:
    MASTRA_API_URL       - Server URL (default: http://localhost:8080)
    MASTRA_AUTH_TOKEN    - Bearer token for auth (default: none)
"""

import json
import os
import sys
import uuid
from collections import Counter

import requests

BASE_URL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("MASTRA_API_URL", "http://localhost:8080")
)
_auth_token = os.environ.get("MASTRA_AUTH_TOKEN", "")
HEADERS = {"Content-Type": "application/json"}
if _auth_token:
    HEADERS["Authorization"] = f"Bearer {_auth_token}"


def print_response(name: str, response):
    """Pretty print a response."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print("=" * 60)
    print(f"Status: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))


def test_capabilities():
    """Test capabilities endpoint."""
    print("\n>>> Testing GET /traigent/v1/capabilities")
    response = requests.get(f"{BASE_URL}/traigent/v1/capabilities", headers=HEADERS)
    print_response("Capabilities", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "supports_evaluate" in data, "Missing supports_evaluate"
    assert "version" in data, "Missing version (required by ServiceCapabilities spec)"

    print("\n  Capabilities test passed!")
    return data


def test_config_space(tunable_id: str):
    """Test config space endpoint with tunable_id query param."""
    print(f"\n>>> Testing GET /traigent/v1/config-space?tunable_id={tunable_id}")
    response = requests.get(
        f"{BASE_URL}/traigent/v1/config-space",
        params={"tunable_id": tunable_id},
        headers=HEADERS,
    )
    print_response("Config Space", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert (
        "schema_version" in data
    ), "Missing schema_version (required by ConfigSpaceResponse spec)"
    assert (
        "tunable_id" in data
    ), "Missing tunable_id (required by ConfigSpaceResponse spec)"
    assert "tunables" in data, "Missing tunables in response"
    assert len(data["tunables"]) > 0, "No tunables defined"

    # Check tunable structure
    tunable = data["tunables"][0]
    assert "name" in tunable, "Tunable missing name"
    assert "type" in tunable, "Tunable missing type"

    print("\n  Config space test passed!")
    print(f"  Found {len(data['tunables'])} tunables:")
    for t in data["tunables"]:
        print(f"    - {t['name']} ({t['type']})")
    return data


def _build_config_from_tunables(tunables: list) -> dict:
    """Build a valid config by picking the first valid value for each tunable."""
    config = {}
    for t in tunables:
        name = t["name"]
        domain = t.get("domain", {})
        if "values" in domain and domain["values"]:
            config[name] = domain["values"][0]
        elif "range" in domain and len(domain["range"]) == 2:
            lo, hi = domain["range"]
            # Pick midpoint for float, low for int
            if t.get("type") == "float":
                config[name] = round((lo + hi) / 2, 2)
            else:
                config[name] = int(lo)
        elif "default" in t:
            config[name] = t["default"]
    return config


def test_execute(tunable_id: str, config: dict):
    """Test execute endpoint with dataset input_ids (privacy-preserving)."""
    print("\n>>> Testing POST /traigent/v1/execute")

    # input_ids are server-specific; these match the zap_agent benchmark dataset.
    # TODO: replace with dynamic discovery once traigent-api#43 is resolved.
    sent_request_id = f"test-{uuid.uuid4().hex[:8]}"
    request_data = {
        "request_id": sent_request_id,
        "tunable_id": tunable_id,
        "config": config,
        "inputs": [
            {"input_id": "q001"},
            {"input_id": "q051"},
        ],
    }

    response = requests.post(
        f"{BASE_URL}/traigent/v1/execute",
        json=request_data,
        headers=HEADERS,
    )
    print_response("Execute", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "completed", f"Expected completed, got {data['status']}"
    assert (
        "execution_id" in data
    ), "Missing execution_id (required by ExecuteResponse spec)"
    assert "request_id" in data, "Missing request_id (required by ExecuteResponse spec)"
    assert data["request_id"] == sent_request_id, (
        f"request_id not echoed correctly: sent {sent_request_id!r}, "
        f"got {data['request_id']!r}"
    )
    assert "outputs" in data, "Missing outputs"
    assert "operational_metrics" in data, "Missing operational_metrics"

    # Check outputs — server returns output_id (privacy-preserving)
    assert len(data["outputs"]) == 2, f"Expected 2 outputs, got {len(data['outputs'])}"
    for output in data["outputs"]:
        assert "input_id" in output, "Output missing input_id"
        assert "output_id" in output, "Output missing output_id"
        assert "cost_usd" in output, "Output missing cost_usd"

    # Verify input_id preservation (order-independent, catches duplicates)
    # See traigent-api#44 for spec wording on exact preservation requirement.
    sent_ids = Counter(inp["input_id"] for inp in request_data["inputs"])
    received_ids = Counter(out["input_id"] for out in data["outputs"])
    assert (
        sent_ids == received_ids
    ), f"input_id mismatch: sent {dict(sent_ids)}, got {dict(received_ids)}"

    # Validate quality_metrics shape when present (combined mode)
    if data.get("quality_metrics") is not None:
        assert isinstance(
            data["quality_metrics"], dict
        ), "Combined mode quality_metrics must be a dict (per ExecuteResponse spec)"

    # Check operational metrics
    metrics = data["operational_metrics"]
    assert "total_cost_usd" in metrics, "Missing total_cost_usd"
    assert "latency_ms" in metrics, "Missing latency_ms"

    print("\n  Execute test passed!")
    print(f"  Processed {len(data['outputs'])} inputs")
    print(f"  Total cost: ${metrics['total_cost_usd']:.4f}")
    print(f"  Latency: {metrics['latency_ms']:.2f}ms")

    return data


def test_evaluate(execute_data: dict, tunable_id: str):
    """Test evaluate endpoint by chaining output_ids from execute."""
    print("\n>>> Testing POST /traigent/v1/evaluate")

    # Build evaluations from execute outputs — chain output_id values
    evaluations = []
    for out in execute_data["outputs"]:
        evaluations.append(
            {
                "input_id": out["input_id"],
                "output_id": out["output_id"],
            }
        )

    request_data = {
        "request_id": "test-002",
        "tunable_id": tunable_id,
        "execution_id": execute_data["execution_id"],
        "evaluations": evaluations,
    }

    response = requests.post(
        f"{BASE_URL}/traigent/v1/evaluate",
        json=request_data,
        headers=HEADERS,
    )
    print_response("Evaluate", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "completed", f"Expected completed, got {data['status']}"
    assert "results" in data, "Missing results"
    assert (
        "aggregate_metrics" in data
    ), "Missing aggregate_metrics (required by EvaluateResponse spec)"

    # Check results
    assert len(data["results"]) == 2, f"Expected 2 results, got {len(data['results'])}"
    for result in data["results"]:
        assert "input_id" in result, "Result missing input_id"
        assert "metrics" in result, "Result missing metrics"

    # Check aggregate metrics — validate required fields per AggregateMetricStats
    agg = data["aggregate_metrics"]
    for metric_name, stats in agg.items():
        assert (
            "mean" in stats
        ), f"{metric_name} missing 'mean' (required by AggregateMetricStats)"
        assert (
            "std" in stats
        ), f"{metric_name} missing 'std' (required by AggregateMetricStats)"
        assert (
            "n" in stats
        ), f"{metric_name} missing 'n' (required by AggregateMetricStats)"

    print("\n  Evaluate test passed!")
    print(f"  Evaluated {len(data['results'])} examples")
    for metric, stats in agg.items():
        print(f"  {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    return data


def test_health():
    """Test health endpoint."""
    print("\n>>> Testing GET /traigent/v1/health")
    response = requests.get(f"{BASE_URL}/traigent/v1/health", headers=HEADERS)
    print_response("Health", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy", f"Expected healthy, got {data['status']}"

    print("\n  Health test passed!")
    return data


def test_error_format(tunable_id: str):
    """F5: Negative-path coverage — validate error response matches ErrorResponse.yaml."""
    print("\n>>> Testing POST /traigent/v1/execute (bad request)")

    # Send intentionally invalid request (missing required 'config' field)
    bad_request = {
        "request_id": f"err-{uuid.uuid4().hex[:8]}",
        "tunable_id": tunable_id,
        "inputs": [{"input_id": "q001"}],
        # 'config' deliberately omitted
    }

    response = requests.post(
        f"{BASE_URL}/traigent/v1/execute",
        json=bad_request,
        headers=HEADERS,
    )
    print_response("Error Format", response)

    # Server should return 4xx (400 or 422)
    assert (
        400 <= response.status_code < 500
    ), f"Expected 4xx for bad request, got {response.status_code}"
    data = response.json()

    # Validate ErrorResponse shape per spec
    assert (
        "error" in data or "message" in data
    ), "Error response must contain 'error' or 'message' field (per ErrorResponse spec)"

    print("\n  Error format test passed!")
    return data


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  JS-Mastra API Contract Tests")
    print("=" * 60)
    print(f"\nTesting server at: {BASE_URL}")

    try:
        # Test all endpoints
        caps = test_capabilities()
        tunable_id = caps["tunable_ids"][0]
        config_space = test_config_space(tunable_id)
        config = _build_config_from_tunables(config_space["tunables"])
        execute_result = test_execute(tunable_id, config)

        # Mode-aware /evaluate call per overview.md detection rules:
        #   quality_metrics non-null           => combined mode (skip /evaluate)
        #   quality_metrics null + supports_evaluate=true  => two-phase (call /evaluate)
        #   quality_metrics null + supports_evaluate=false => execute-only (skip)
        if execute_result.get("quality_metrics") is not None:
            print(
                "\nSKIP /evaluate: combined mode (quality_metrics in /execute response)"
            )
        elif not caps.get("supports_evaluate", True):
            print("\nSKIP /evaluate: execute-only mode (supports_evaluate=false)")
        else:
            test_evaluate(execute_result, tunable_id)

        test_error_format(tunable_id)
        test_health()

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your server is correctly implementing the Traigent Hybrid API.")
        print("You can now integrate it with Traigent for optimization.")
        print()

    except requests.ConnectionError:
        print("\n" + "=" * 60)
        print("  CONNECTION ERROR")
        print("=" * 60)
        print(f"\nCould not connect to server at {BASE_URL}")
        print("Make sure the server is running:")
        print("  cd JS-Mastra-APIs-Validation && npm run api:dev")
        print()
        sys.exit(1)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("  TEST FAILED")
        print("=" * 60)
        print(f"\nAssertion error: {e}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
