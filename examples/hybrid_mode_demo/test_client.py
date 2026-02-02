"""Test client for the Traigent Hybrid Mode demo.

This script tests all endpoints of the demo Flask server.

Usage:
    # First start the server in another terminal:
    python app.py

    # Then run this test:
    python test_client.py

    # Or specify a different URL:
    python test_client.py http://your-server:8080
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
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))


def test_capabilities():
    """Test capabilities endpoint."""
    print("\n>>> Testing GET /traigent/v1/capabilities")
    response = requests.get(f"{BASE_URL}/traigent/v1/capabilities")
    print_response("Capabilities", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["version"] == "1.0", f"Expected version 1.0, got {data['version']}"
    assert "supports_evaluate" in data, "Missing supports_evaluate"

    print("\n  Capabilities test passed!")
    return data


def test_config_space():
    """Test config space endpoint."""
    print("\n>>> Testing GET /traigent/v1/config-space")
    response = requests.get(f"{BASE_URL}/traigent/v1/config-space")
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


def test_execute():
    """Test execute endpoint."""
    print("\n>>> Testing POST /traigent/v1/execute")

    request_data = {
        "request_id": "test-001",
        "capability_id": "demo_agent",
        "config": {
            "model": "accurate",
            "temperature": 0.7,
            "max_retries": 2,
            "use_cache": True,
        },
        "inputs": [
            {"input_id": "ex_001", "data": {"query": "What is artificial intelligence?"}},
            {"input_id": "ex_002", "data": {"query": "Explain machine learning"}},
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

    # Check outputs
    assert len(data["outputs"]) == 2, f"Expected 2 outputs, got {len(data['outputs'])}"
    for output in data["outputs"]:
        assert "input_id" in output, "Output missing input_id"
        assert "output" in output, "Output missing output"
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


def test_evaluate(execution_id: str):
    """Test evaluate endpoint."""
    print("\n>>> Testing POST /traigent/v1/evaluate")

    request_data = {
        "request_id": "test-002",
        "capability_id": "demo_agent",
        "execution_id": execution_id,
        "evaluations": [
            {
                "input_id": "ex_001",
                "output": {"response": "AI is artificial intelligence that simulates human thinking"},
                "target": {"expected": "Artificial Intelligence is the simulation of human intelligence"},
            },
            {
                "input_id": "ex_002",
                "output": {"response": "ML is a subset of AI using statistical methods"},
                "target": {"expected": "Machine Learning is a type of AI that learns from data"},
            },
        ],
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
    print("  Traigent Hybrid Mode Demo - Test Client")
    print("=" * 60)
    print(f"\nTesting server at: {BASE_URL}")

    try:
        # Test all endpoints
        test_capabilities()
        test_config_space()
        execute_result = test_execute()
        test_evaluate(execute_result["execution_id"])
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
        print("  python app.py")
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
