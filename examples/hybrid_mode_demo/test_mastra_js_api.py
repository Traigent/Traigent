"""API contract tests for a Traigent Hybrid API server.

Validates all Traigent Hybrid API endpoints:
  /health, /capabilities, /config-space, /execute, /evaluate

Server-agnostic: discovers tunable_id and config from the server itself.
Only the auth token and test input_ids are hardcoded.

Usage:
    .venv/bin/python examples/hybrid_mode_demo/test_mastra_js_api.py <base_url>

    # Example:
    .venv/bin/python examples/hybrid_mode_demo/test_mastra_js_api.py https://755e-46-116-17-120.ngrok-free.app
"""

import json
import sys

import requests

# --- Only these are hardcoded ---
AUTH_TOKEN = "Bearer QYG7VHh32VMZg7hLVBRvPEbTgFtco6dU"
TEST_INPUT_IDS = [
    "no-filter-single-search-trashcan-blue",
    "product-search-specific-model",
    "consultant-fridge",
]
# --------------------------------

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://ai.bazak.ai"
HEADERS = {
    "Authorization": AUTH_TOKEN,
    "Content-Type": "application/json",
    "User-Agent": "Traigent-SDK/1.0",
}


def print_response(name: str, response):
    """Pretty print a response."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print("=" * 60)
    print(f"Status: {response.status_code}")
    print("Response:")
    try:
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.JSONDecodeError:
        print(response.text[:500])


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


def test_capabilities():
    """Test capabilities endpoint. Returns discovered data."""
    print("\n>>> Testing GET /traigent/v1/capabilities")
    response = requests.get(f"{BASE_URL}/traigent/v1/capabilities", headers=HEADERS)
    print_response("Capabilities", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "supports_evaluate" in data, "Missing supports_evaluate"
    assert "tunable_ids" in data, "Missing tunable_ids"
    assert len(data["tunable_ids"]) > 0, "No tunable_ids returned"

    print("\n  Capabilities test passed!")
    return data


def test_config_space(tunable_id: str):
    """Test config space endpoint. Returns discovered tunables."""
    print(f"\n>>> Testing GET /traigent/v1/config-space?tunable_id={tunable_id}")
    response = requests.get(
        f"{BASE_URL}/traigent/v1/config-space",
        params={"tunable_id": tunable_id},
        headers=HEADERS,
    )
    print_response("Config Space", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "tunables" in data, "Missing tunables in response"
    assert len(data["tunables"]) > 0, "No tunables defined"

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
            if t.get("type") == "float":
                config[name] = round((lo + hi) / 2, 2)
            else:
                config[name] = int(lo)
        elif "default" in t:
            config[name] = t["default"]
    return config


def test_execute(tunable_id: str, config: dict):
    """Test execute endpoint with all input_ids in one request."""
    print("\n>>> Testing POST /traigent/v1/execute")

    request_data = {
        "request_id": "test-001",
        "tunable_id": tunable_id,
        "config": config,
        "inputs": [{"input_id": iid} for iid in TEST_INPUT_IDS],
    }

    response = requests.post(
        f"{BASE_URL}/traigent/v1/execute",
        json=request_data,
        headers=HEADERS,
        timeout=900,
    )
    print_response("Execute", response)

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}"
    )
    data = response.json()
    assert data["status"] == "completed", (
        f"Expected completed, got {data['status']}"
    )
    assert "outputs" in data, "Missing outputs"
    assert len(data["outputs"]) >= 1, "No outputs returned"

    for output in data["outputs"]:
        assert "input_id" in output, "Output missing input_id"
        assert "output_id" in output, "Output missing output_id"

    metrics = data.get("operational_metrics", {})
    print(f"\n  Execute test passed!")
    print(f"  Processed {len(data['outputs'])} inputs")
    print(f"  Total cost: ${metrics.get('total_cost_usd', 0):.4f}")
    print(f"  Latency: {metrics.get('latency_ms', 0):.0f}ms")

    return data


def test_evaluate(execute_data: dict, tunable_id: str):
    """Test evaluate endpoint by chaining output_ids from execute."""
    print("\n>>> Testing POST /traigent/v1/evaluate")

    evaluations = [
        {"input_id": out["input_id"], "output_id": out["output_id"]}
        for out in execute_data["outputs"]
    ]

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
        timeout=900,
    )
    print_response("Evaluate", response)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "completed", f"Expected completed, got {data['status']}"
    assert "results" in data, "Missing results"
    assert "aggregate_metrics" in data, "Missing aggregate_metrics"

    assert len(data["results"]) == len(evaluations), (
        f"Expected {len(evaluations)} results, got {len(data['results'])}"
    )
    for result in data["results"]:
        assert "input_id" in result, "Result missing input_id"
        assert "metrics" in result, "Result missing metrics"

    agg = data["aggregate_metrics"]
    assert "accuracy" in agg, "Missing accuracy aggregate"
    assert "mean" in agg["accuracy"], "Missing mean in accuracy"

    print("\n  Evaluate test passed!")
    print(f"  Evaluated {len(data['results'])} examples")
    for metric, stats in agg.items():
        print(f"  {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    return data


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  Traigent Hybrid API Contract Tests")
    print("=" * 60)
    print(f"\nServer:    {BASE_URL}")
    print(f"Input IDs: {TEST_INPUT_IDS}")

    try:
        test_health()

        caps = test_capabilities()
        tunable_id = caps["tunable_ids"][0]
        print(f"\n  Using tunable_id: {tunable_id}")

        config_space = test_config_space(tunable_id)
        config = _build_config_from_tunables(config_space["tunables"])
        print(f"\n  Using config: {json.dumps(config)}")

        execute_result = test_execute(tunable_id, config)
        test_evaluate(execute_result, tunable_id)

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
        print()

    except requests.ConnectionError:
        print("\n" + "=" * 60)
        print("  CONNECTION ERROR")
        print("=" * 60)
        print(f"\nCould not connect to server at {BASE_URL}")
        sys.exit(1)

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("  TEST FAILED")
        print("=" * 60)
        print(f"\nAssertion error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()