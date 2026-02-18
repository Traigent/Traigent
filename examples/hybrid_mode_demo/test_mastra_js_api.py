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
"""

import json
import sys

import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"


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
    response = requests.get(f"{BASE_URL}/traigent/v1/capabilities")
    print_response("Capabilities", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "supports_evaluate" in data, "Missing supports_evaluate"

    print("\n  Capabilities test passed!")
    return data


def test_config_space(tunable_id: str):
    """Test config space endpoint with tunable_id query param."""
    print(f"\n>>> Testing GET /traigent/v1/config-space?tunable_id={tunable_id}")
    response = requests.get(
        f"{BASE_URL}/traigent/v1/config-space",
        params={"tunable_id": tunable_id},
    )
    print_response("Config Space", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
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

    request_data = {
        "request_id": "test-001",
        "tunable_id": tunable_id,
        "config": config,
        "inputs": [
            {"input_id": "case_001"},
            {"input_id": "case_051"},
        ],
    }

    response = requests.post(
        f"{BASE_URL}/traigent/v1/execute",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )
    print_response("Execute", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "completed", f"Expected completed, got {data['status']}"
    assert "outputs" in data, "Missing outputs"
    assert "operational_metrics" in data, "Missing operational_metrics"

    # Check outputs — server returns output_id (privacy-preserving)
    assert len(data["outputs"]) == 2, f"Expected 2 outputs, got {len(data['outputs'])}"
    for output in data["outputs"]:
        assert "input_id" in output, "Output missing input_id"
        assert "output_id" in output, "Output missing output_id"
        assert "cost_usd" in output, "Output missing cost_usd"

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
        evaluations.append({
            "input_id": out["input_id"],
            "output_id": out["output_id"],
        })

    request_data = {
        "request_id": "test-002",
        "tunable_id": tunable_id,
        "execution_id": execute_data["execution_id"],
        "evaluations": evaluations,
    }

    response = requests.post(
        f"{BASE_URL}/traigent/v1/evaluate",
        json=request_data,
        headers={"Content-Type": "application/json"},
    )
    print_response("Evaluate", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "completed", f"Expected completed, got {data['status']}"
    assert "results" in data, "Missing results"
    assert "aggregate_metrics" in data, "Missing aggregate_metrics"

    # Check results
    assert len(data["results"]) == 2, f"Expected 2 results, got {len(data['results'])}"
    for result in data["results"]:
        assert "input_id" in result, "Result missing input_id"
        assert "metrics" in result, "Result missing metrics"

    # Check aggregate metrics
    agg = data["aggregate_metrics"]
    assert "accuracy" in agg, "Missing accuracy aggregate"
    assert "mean" in agg["accuracy"], "Missing mean in accuracy"

    print("\n  Evaluate test passed!")
    print(f"  Evaluated {len(data['results'])} examples")
    for metric, stats in agg.items():
        print(f"  {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    return data


def test_health():
    """Test health endpoint."""
    print("\n>>> Testing GET /traigent/v1/health")
    response = requests.get(f"{BASE_URL}/traigent/v1/health")
    print_response("Health", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "healthy", f"Expected healthy, got {data['status']}"

    print("\n  Health test passed!")
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
        test_evaluate(execute_result, tunable_id)
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
