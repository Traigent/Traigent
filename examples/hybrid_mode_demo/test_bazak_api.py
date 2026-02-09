#!/usr/bin/env python3
"""Transparent API validation test for BazakDemo × Traigent hybrid protocol.

Unlike run_bazak_optimization.py (black-box demo), this script exercises
each BazakDemo API endpoint directly using the SDK's HTTPTransport and
protocol objects, validating request/response format compliance at every step.

Output shows the actual request/response payloads for each endpoint,
then a validation summary at the end.

Flow:
  Test 1: GET  /health         → status=healthy, version present
  Test 2: GET  /capabilities   → supports_evaluate, capability_ids
  Test 3: GET  /config-space   → 4 tunables, correct names/types/domains
  Test 4: POST /execute        → 3 inputs, outputs with output_id, metrics
  Test 5: POST /evaluate       → output_ids from Test 4, accuracy metrics
  Test 6: E2E  HybridAPIEvaluator → 1 trial via SDK, meaningful metrics

Prerequisites:
  - BazakDemo server running at http://localhost:8080
    (cd BazakDemo_Apis_Mastra_JS && node src/api/server.js)
  - OPENAI_API_KEY set in BazakDemo's .env

Usage:
    .venv/bin/python examples/hybrid_mode_demo/test_bazak_api.py
"""

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Final

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from traigent.evaluators import HybridAPIEvaluator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.hybrid.http_transport import HTTPTransport
from traigent.hybrid.protocol import (
    ConfigSpaceResponse,
    HybridEvaluateRequest,
    HybridEvaluateResponse,
    HybridExecuteRequest,
    HybridExecuteResponse,
    ServiceCapabilities,
)

SERVER_URL: Final[str] = os.getenv("BAZAK_BASE_URL", "http://localhost:8080")
CAPABILITY_ID: Final[str] = os.getenv("BAZAK_CAPABILITY_ID", "child-age-agent")

# Test inputs — 3 cases from the BazakDemo dataset
TEST_INPUTS: Final[list[dict[str, Any]]] = [
    {"input_id": "case_001"},
    {"input_id": "case_002"},
    {"input_id": "case_003"},
]

# Known tunables from BazakDemo's tunables.json
EXPECTED_TUNABLES: Final[dict[str, str]] = {
    "model": "enum",
    "temperature": "float",
    "system_prompt_version": "enum",
    "max_retries": "int",
}

# A valid config to use for execute/evaluate
TEST_CONFIG: Final[dict[str, Any]] = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "system_prompt_version": "v1",
    "max_retries": 0,
}

# Shared state between tests (execute → evaluate chain)
_shared: dict[str, Any] = {}

W = 70  # Output width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json(obj: Any, indent: int = 2) -> str:
    """Pretty-print a dict/list as JSON."""
    return json.dumps(obj, indent=indent, default=str)


def _section(title: str) -> None:
    """Print a section header."""
    print()
    print(f"{'─' * W}")
    print(f"  {title}")
    print(f"{'─' * W}")


def _kv(label: str, value: Any) -> None:
    """Print a key-value line."""
    print(f"  {label}: {value}")


def _block(label: str, data: Any) -> None:
    """Print a labeled JSON block."""
    print(f"  {label}:")
    for line in _json(data).splitlines():
        print(f"    {line}")


def assert_field(data: dict, key: str, expected_type: type | None = None) -> Any:
    """Assert a field exists and optionally check its type."""
    if key not in data:
        raise AssertionError(
            f"Missing required field: '{key}'. Got keys: {list(data.keys())}"
        )
    value = data[key]
    if expected_type is not None and not isinstance(value, expected_type):
        raise AssertionError(
            f"Field '{key}' should be {expected_type.__name__}, "
            f"got {type(value).__name__}: {value!r}"
        )
    return value


class TestResult:
    """Result of a single test."""

    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail


# ---------------------------------------------------------------------------
# Test 1: GET /health
# ---------------------------------------------------------------------------
async def test_health(transport: HTTPTransport) -> TestResult:
    """GET /traigent/v1/health → status=healthy, version present."""
    _section("GET /traigent/v1/health")

    resp = await transport.health_check()

    _block("Response", {
        "status": resp.status,
        "version": resp.version,
        "uptime_seconds": resp.uptime_seconds,
        "details": resp.details,
    })

    assert resp.status == "healthy", f"Expected status='healthy', got '{resp.status}'"
    assert resp.version is not None, "Missing version in health response"
    return TestResult("GET /health", True)


# ---------------------------------------------------------------------------
# Test 2: GET /capabilities
# ---------------------------------------------------------------------------
async def test_capabilities(transport: HTTPTransport) -> TestResult:
    """GET /traigent/v1/capabilities → supports_evaluate, version."""
    _section("GET /traigent/v1/capabilities")

    # Clear cache so we actually hit the endpoint
    transport._capabilities = None

    caps = await transport.capabilities()
    assert isinstance(
        caps, ServiceCapabilities
    ), f"Expected ServiceCapabilities, got {type(caps)}"

    _block("Response", {
        "version": caps.version,
        "supports_evaluate": caps.supports_evaluate,
        "supports_keep_alive": caps.supports_keep_alive,
        "supports_streaming": caps.supports_streaming,
        "max_batch_size": caps.max_batch_size,
        "max_payload_bytes": caps.max_payload_bytes,
    })

    assert (
        caps.supports_evaluate is True
    ), f"BazakDemo should support evaluate, got supports_evaluate={caps.supports_evaluate}"
    assert caps.version is not None, "Missing version in capabilities"

    return TestResult("GET /capabilities", True)


# ---------------------------------------------------------------------------
# Test 3: GET /config-space
# ---------------------------------------------------------------------------
async def test_config_space(transport: HTTPTransport) -> TestResult:
    """GET /traigent/v1/config-space → 4 tunables with correct types."""
    _section("GET /traigent/v1/config-space")

    cs_response = await transport.discover_config_space()
    assert isinstance(
        cs_response, ConfigSpaceResponse
    ), f"Expected ConfigSpaceResponse, got {type(cs_response)}"

    # Show the full config space
    tunables = cs_response.tvars
    _kv("schema_version", cs_response.schema_version)
    _kv("capability_id", cs_response.capability_id)
    if cs_response.defaults:
        _block("defaults", cs_response.defaults)
    if cs_response.objectives:
        _block("objectives", cs_response.objectives)

    print()
    print(f"  tunables ({len(tunables)}):")
    for t in tunables:
        print(f"    {t.name} ({t.type}):")
        print(f"      domain:  {_json(t.domain, indent=0)}")
        if t.default is not None:
            print(f"      default: {t.default}")
        if t.constraints:
            print(f"      constraints: {t.constraints}")

    # Show what optimizers receive after conversion
    config_space = cs_response.to_traigent_config_space()
    print()
    _block("to_traigent_config_space()", config_space)

    # Validate tunable count
    assert len(tunables) == len(EXPECTED_TUNABLES), (
        f"Expected {len(EXPECTED_TUNABLES)} tunables, got {len(tunables)}: "
        f"{[t.name for t in tunables]}"
    )

    # Validate each tunable name + type
    found = {t.name: t for t in tunables}
    for name, expected_type in EXPECTED_TUNABLES.items():
        assert name in found, f"Missing tunable '{name}'. Found: {list(found.keys())}"
        actual_type = found[name].type
        assert (
            actual_type == expected_type
        ), f"Tunable '{name}': expected type='{expected_type}', got '{actual_type}'"

    # Validate domains have correct structure
    model_tvar = found["model"]
    model_values = model_tvar.domain.get("values", [])
    assert (
        "gpt-4o" in model_values and "gpt-4o-mini" in model_values
    ), f"model domain should contain gpt-4o and gpt-4o-mini, got {model_values}"

    temp_tvar = found["temperature"]
    temp_range = temp_tvar.domain.get("range", [])
    assert (
        len(temp_range) == 2
    ), f"temperature domain should have range [low, high], got {temp_range}"

    # Validate config space conversion
    assert len(config_space) == len(
        EXPECTED_TUNABLES
    ), f"Config space should have {len(EXPECTED_TUNABLES)} entries, got {len(config_space)}"

    return TestResult("GET /config-space", True)


# ---------------------------------------------------------------------------
# Test 4: POST /execute
# ---------------------------------------------------------------------------
async def test_execute(transport: HTTPTransport) -> TestResult:
    """POST /traigent/v1/execute → 3 outputs with output_id, cost, latency."""
    _section("POST /traigent/v1/execute")

    request = HybridExecuteRequest(
        capability_id=CAPABILITY_ID,
        config=TEST_CONFIG,
        inputs=TEST_INPUTS,
        timeout_ms=60000,
    )

    # Show the request payload
    req_dict = request.to_dict()
    _block("Request", req_dict)

    assert "request_id" in req_dict, "Request missing request_id"
    assert req_dict["capability_id"] == CAPABILITY_ID
    assert len(req_dict["inputs"]) == len(TEST_INPUTS)

    # Execute
    resp = await transport.execute(request)
    assert isinstance(
        resp, HybridExecuteResponse
    ), f"Expected HybridExecuteResponse, got {type(resp)}"

    # Show response header
    print()
    _kv("status", resp.status)
    _kv("execution_id", resp.execution_id)
    _kv("request_id", resp.request_id)
    if resp.session_id:
        _kv("session_id", resp.session_id)

    # Show each output
    print()
    print(f"  outputs ({len(resp.outputs)}):")
    for i, output in enumerate(resp.outputs):
        print(f"    [{i}] input_id={output.get('input_id')}")
        print(f"        output_id={output.get('output_id')}")
        # Show all fields except the ones we already printed
        extra = {
            k: v
            for k, v in output.items()
            if k not in ("input_id", "output_id")
        }
        if extra:
            for k, v in extra.items():
                val_str = str(v)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"        {k}={val_str}")

    # Show operational metrics
    print()
    _block("operational_metrics", resp.operational_metrics)
    if resp.quality_metrics:
        _block("quality_metrics", resp.quality_metrics)

    total_cost = resp.get_total_cost()
    _kv("total_cost", f"${total_cost:.6f}")

    # Validate
    assert resp.status in (
        "completed",
        "partial",
    ), f"Expected status completed/partial, got '{resp.status}'"
    assert resp.execution_id, "Missing execution_id in response"
    assert resp.request_id, "Missing request_id in response"

    assert len(resp.outputs) == len(
        TEST_INPUTS
    ), f"Expected {len(TEST_INPUTS)} outputs, got {len(resp.outputs)}"

    for output in resp.outputs:
        input_id = assert_field(output, "input_id", str)
        output_id = assert_field(output, "output_id", str)
        assert output_id, f"output_id is empty for input_id={input_id}"
        assert (
            "cost_usd" in output or "tokens_used" in output
        ), f"Output for {input_id} missing cost/token info"

    op_metrics = resp.operational_metrics
    assert op_metrics, "Missing operational_metrics"
    assert (
        "total_cost_usd" in op_metrics or "cost_usd" in op_metrics
    ), f"operational_metrics missing cost. Keys: {list(op_metrics.keys())}"
    assert total_cost > 0, f"Total cost should be > 0, got {total_cost}"

    # Save for Test 5 (evaluate needs output_ids)
    _shared["execution_id"] = resp.execution_id
    _shared["outputs"] = resp.outputs
    _shared["session_id"] = resp.session_id

    return TestResult("POST /execute", True)


# ---------------------------------------------------------------------------
# Test 5: POST /evaluate
# ---------------------------------------------------------------------------
async def test_evaluate(transport: HTTPTransport) -> TestResult:
    """POST /traigent/v1/evaluate → accuracy metrics from output_ids."""
    _section("POST /traigent/v1/evaluate")

    if "outputs" not in _shared:
        return TestResult(
            "POST /evaluate", False, "Skipped: Test 4 (execute) must pass first"
        )

    # Build evaluations from execute outputs
    evaluations = []
    for output in _shared["outputs"]:
        evaluations.append(
            {
                "input_id": output["input_id"],
                "output_id": output["output_id"],
            }
        )

    request = HybridEvaluateRequest(
        capability_id=CAPABILITY_ID,
        execution_id=_shared["execution_id"],
        evaluations=evaluations,
        session_id=_shared.get("session_id"),
    )

    # Show the request payload
    req_dict = request.to_dict()
    _block("Request", req_dict)

    assert "request_id" in req_dict
    assert req_dict["capability_id"] == CAPABILITY_ID
    assert req_dict["execution_id"] == _shared["execution_id"]
    assert len(req_dict["evaluations"]) == len(TEST_INPUTS)

    # Evaluate
    resp = await transport.evaluate(request)
    assert isinstance(
        resp, HybridEvaluateResponse
    ), f"Expected HybridEvaluateResponse, got {type(resp)}"

    # Show response header
    print()
    _kv("status", resp.status)
    _kv("request_id", resp.request_id)

    # Show per-example results
    print()
    print(f"  results ({len(resp.results)}):")
    for i, result in enumerate(resp.results):
        input_id = result.get("input_id", "?")
        metrics = result.get("metrics", {})
        print(f"    [{i}] input_id={input_id}")
        for mk, mv in metrics.items():
            print(f"        {mk}={mv}")

    # Show aggregate metrics
    print()
    _block("aggregate_metrics", resp.aggregate_metrics)

    # Validate
    assert resp.status == "completed", f"Expected status=completed, got '{resp.status}'"
    assert len(resp.results) == len(
        TEST_INPUTS
    ), f"Expected {len(TEST_INPUTS)} results, got {len(resp.results)}"

    for result in resp.results:
        input_id = assert_field(result, "input_id", str)
        metrics = assert_field(result, "metrics", dict)
        assert (
            "accuracy" in metrics
        ), f"Result for {input_id} missing 'accuracy' metric. Keys: {list(metrics.keys())}"
        acc = metrics["accuracy"]
        assert acc in (
            0,
            1,
            0.0,
            1.0,
        ), f"accuracy for {input_id} should be 0 or 1, got {acc}"

    agg = resp.aggregate_metrics
    assert (
        "accuracy" in agg
    ), f"aggregate_metrics missing 'accuracy'. Keys: {list(agg.keys())}"
    acc_agg = agg["accuracy"]
    assert (
        "mean" in acc_agg
    ), f"aggregate_metrics.accuracy missing 'mean'. Keys: {list(acc_agg.keys())}"
    assert "n" in acc_agg, "aggregate_metrics.accuracy missing 'n'"
    assert acc_agg["n"] == len(
        TEST_INPUTS
    ), f"aggregate_metrics.accuracy.n should be {len(TEST_INPUTS)}, got {acc_agg['n']}"

    return TestResult("POST /evaluate", True)


# ---------------------------------------------------------------------------
# Test 6: E2E through HybridAPIEvaluator
# ---------------------------------------------------------------------------
async def test_e2e_evaluator() -> TestResult:
    """Full E2E: 1 trial via HybridAPIEvaluator, meaningful metrics."""
    _section("E2E: HybridAPIEvaluator (SDK integration)")

    evaluator = HybridAPIEvaluator(
        api_endpoint=SERVER_URL,
        capability_id=CAPABILITY_ID,
        batch_size=5,
        auto_discover_tvars=True,
    )

    _kv("api_endpoint", SERVER_URL)
    _kv("capability_id", CAPABILITY_ID)
    _kv("config", _json(TEST_CONFIG, indent=0))

    async with evaluator:
        # Discover config space
        config_space = await evaluator.discover_config_space()
        assert len(config_space) > 0, "Config space is empty"
        print()
        _block("discovered config_space", config_space)

        # Build small dataset
        dataset = Dataset(
            [
                EvaluationExample(input_data={"input_id": "case_001"}),
                EvaluationExample(input_data={"input_id": "case_002"}),
                EvaluationExample(input_data={"input_id": "case_003"}),
            ]
        )

        # Run evaluation
        eval_result = await evaluator.evaluate(
            func=lambda: None,
            config=TEST_CONFIG,
            dataset=dataset,
        )

        # Show result
        metrics = eval_result.aggregated_metrics
        print()
        _block("aggregated_metrics", metrics)
        _kv("total_examples", eval_result.total_examples)

        # Validate
        assert (
            "accuracy" in metrics
        ), f"E2E result missing 'accuracy'. Keys: {list(metrics.keys())}"

        accuracy = metrics["accuracy"]
        assert isinstance(
            accuracy, int | float
        ), f"accuracy should be numeric, got {type(accuracy)}"

    return TestResult("E2E HybridAPIEvaluator", True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
async def run_all_tests() -> list[TestResult]:
    """Run all API validation tests in sequence."""
    results: list[TestResult] = []

    transport = HTTPTransport(
        base_url=SERVER_URL,
        timeout=60.0,
    )

    async with transport:
        # Tests 1-5: Direct transport tests (sequential — execute feeds evaluate)
        tests = [
            ("Test 1", test_health),
            ("Test 2", test_capabilities),
            ("Test 3", test_config_space),
            ("Test 4", test_execute),
            ("Test 5", test_evaluate),
        ]
        for label, test_fn in tests:
            try:
                result = await test_fn(transport)
                results.append(result)
            except Exception as e:
                results.append(
                    TestResult(label, False, f"{type(e).__name__}: {e}")
                )
                if label == "Test 4":
                    # Test 5 depends on Test 4
                    results.append(
                        TestResult(
                            "POST /evaluate", False, "Skipped: Test 4 (execute) failed"
                        )
                    )
                    break

    # Test 6: E2E through SDK (separate transport)
    try:
        result = await test_e2e_evaluator()
        results.append(result)
    except Exception as e:
        tb = traceback.format_exc()
        results.append(
            TestResult(
                "E2E HybridAPIEvaluator", False, f"{type(e).__name__}: {e}\n{tb}"
            )
        )

    return results


def main() -> None:
    """Run API validation and print results."""
    print("=" * W)
    print("  BazakDemo API Validation — Traigent Hybrid Protocol")
    print("=" * W)
    print(f"  Server:     {SERVER_URL}")
    print(f"  Capability: {CAPABILITY_ID}")
    print(f"  Inputs:     {len(TEST_INPUTS)} examples")
    print(f"  Config:     {_json(TEST_CONFIG, indent=0)}")

    results = asyncio.run(run_all_tests())

    # --- Validation summary at the end ---
    print()
    print("=" * W)
    print("  VALIDATION SUMMARY")
    print("=" * W)
    for r in results:
        icon = "PASS" if r.passed else "FAIL"
        line = f"  [{icon}] {r.name}"
        if r.detail:
            line += f"  — {r.detail}"
        print(line)

    print(f"{'─' * W}")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    status = "ALL PASSED" if passed == total else f"{total - passed} FAILED"
    print(f"  {passed}/{total} tests — {status}")
    print("=" * W)

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
