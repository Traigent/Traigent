#!/usr/bin/env python3
"""Run Traigent optimization against the Hybrid Mode demo server.

This script demonstrates:
1. Starting the Flask demo server
2. Using HybridAPIEvaluator to execute trials against external service
3. Running optimization with different configurations
4. Reporting results to the Traigent backend (visible in FE)

Usage:
    # From the project root, load env and run:
    source walkthrough/examples/real/.env
    python examples/hybrid_mode_demo/run_optimization.py

Environment variables:
    TRAIGENT_API_KEY - API key for Traigent backend (required for FE visibility)
    TRAIGENT_API_URL - Backend URL (default: http://localhost:5000/api/v1)
"""

import asyncio
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Add traigent to path if not installed
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from traigent.evaluators import HybridAPIEvaluator
from traigent.evaluators.base import Dataset, EvaluationExample

SERVER_URL = "http://localhost:8080"
STARTUP_TIMEOUT = 10
REQUEST_HEADERS = {"User-Agent": "Traigent-SDK/1.0"}


def wait_for_server(url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Wait for server to become ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"{url}/traigent/v1/health",
                timeout=1,
                headers=REQUEST_HEADERS,
            )
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def create_evaluation_dataset() -> list[EvaluationExample]:
    """Create evaluation dataset."""
    examples = [
        {
            "input_id": "ex_001",
            "query": "What is artificial intelligence?",
            "expected": "AI is the simulation of human intelligence by machines.",
        },
        {
            "input_id": "ex_002",
            "query": "Explain machine learning in simple terms.",
            "expected": "Machine learning is a type of AI that learns from data.",
        },
        {
            "input_id": "ex_003",
            "query": "What is deep learning?",
            "expected": "Deep learning uses neural networks with many layers.",
        },
        {
            "input_id": "ex_004",
            "query": "How does natural language processing work?",
            "expected": "NLP helps computers understand human language.",
        },
        {
            "input_id": "ex_005",
            "query": "What is a neural network?",
            "expected": "A neural network is a computing system inspired by the brain.",
        },
    ]
    return [
        EvaluationExample(
            input_data={"query": ex["query"], "input_id": ex["input_id"]},
            expected_output=ex["expected"],
        )
        for ex in examples
    ]


def generate_configurations(config_space: dict) -> list[dict]:
    """Generate diverse configuration grid from config space."""
    keys = list(config_space.keys())
    values = []

    for key in keys:
        domain = config_space[key]
        if isinstance(domain, list):
            values.append(domain)
        elif isinstance(domain, dict):
            if "low" in domain and "high" in domain:
                # Numeric range - sample 2 points (low and high)
                low, high = domain["low"], domain["high"]
                if domain.get("type") == "int":
                    values.append([int(low), int(high)])
                else:
                    values.append([low, high])
            else:
                values.append([domain.get("default", 0)])
        else:
            values.append([domain])

    # Generate full grid (should be manageable: 3*2*2*2 = 24)
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo, strict=True)))

    # Limit to 12 for demo
    return configs[:12]


async def run_optimization() -> None:
    """Run the optimization against the external service."""
    print("\n" + "=" * 60)
    print("  Traigent Hybrid Mode Optimization")
    print("=" * 60)

    # Create evaluator pointing to Flask server
    evaluator = HybridAPIEvaluator(
        api_endpoint=SERVER_URL,
        tunable_id="demo_agent",
        batch_size=5,
        auto_discover_tvars=True,
    )

    async with evaluator:
        # Step 1: Discover configuration space from external service
        print("\n1. Discovering tunables from external service...")
        config_space = await evaluator.discover_config_space()
        print(f"   Found {len(config_space)} tunables:")
        for name, domain in config_space.items():
            print(f"   - {name}: {domain}")

        # Step 2: Generate configurations to try
        print("\n2. Generating configuration grid...")
        configurations = generate_configurations(config_space)
        print(f"   Will try {len(configurations)} configurations")

        # Step 3: Create evaluation dataset
        print("\n3. Preparing evaluation dataset...")
        dataset = create_evaluation_dataset()
        print(f"   Dataset has {len(dataset)} examples")

        # Step 4: Run trials
        print("\n4. Running optimization trials...")
        print("   " + "-" * 56)

        results = []
        best_config = None
        best_accuracy = -1

        for i, config in enumerate(configurations):
            print(f"\n   Trial {i+1}/{len(configurations)}")
            config_str = json.dumps(config)
            if len(config_str) > 60:
                config_str = config_str[:60] + "..."
            print(f"   Config: {config_str}")

            # Execute trial via HybridAPIEvaluator
            eval_result = await evaluator.evaluate(
                func=lambda: None,  # Not used in hybrid mode
                config=config,
                dataset=Dataset(dataset),
            )

            # Extract metrics
            metrics = eval_result.aggregated_metrics
            accuracy = metrics.get("accuracy", 0)
            cost = metrics.get("cost", 0)
            latency = metrics.get("latency", 0)

            print(
                f"   Results: accuracy={accuracy:.3f}, cost=${cost:.4f}, latency={latency:.1f}ms"
            )

            results.append(
                {
                    "trial": i + 1,
                    "config": config,
                    "metrics": metrics,
                    "successful": eval_result.successful_examples,
                    "total": eval_result.total_examples,
                }
            )

            # Track best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config

        # Step 5: Print results
        print("\n" + "=" * 60)
        print("  Optimization Results")
        print("=" * 60)

        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        print("\nBest Metrics:")
        best_result = next(r for r in results if r["config"] == best_config)
        for key, value in best_result["metrics"].items():
            if isinstance(value, float):
                if "cost" in key:
                    print(f"  {key}: ${value:.6f}")
                elif "latency" in key:
                    print(f"  {key}: {value:.2f}ms")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\nTrials Summary:")
        print(f"  Total trials: {len(results)}")
        print(f"  Best accuracy: {best_accuracy:.4f}")

        # Step 6: Report backend status
        api_key = os.environ.get("TRAIGENT_API_KEY")
        backend_url = os.environ.get("TRAIGENT_API_URL", "http://localhost:5000/api/v1")

        if api_key:
            print("\n5. Backend Integration")
            print(f"   Backend URL: {backend_url}")
            print(f"   API Key: ***{api_key[-4:]}")
            print("\n   To see results in the Traigent frontend, use the full")
            print(
                "   @traigent.optimize decorator with execution_mode='edge_analytics'."
            )
            print("   This demo shows direct HybridAPIEvaluator usage.")
        else:
            print("\nNote: Set TRAIGENT_API_KEY to enable backend integration.")


async def main() -> None:
    """Run the complete hybrid mode optimization demo."""
    print("=" * 60)
    print("  Traigent Hybrid Mode Demo - Full Optimization")
    print("=" * 60)
    print()

    # Check for environment variables
    print("Environment:")
    api_key = os.environ.get("TRAIGENT_API_KEY")
    backend_url = os.environ.get("TRAIGENT_API_URL", "http://localhost:5000/api/v1")
    print(f"  TRAIGENT_API_URL: {backend_url}")
    print(f"  TRAIGENT_API_KEY: {'***' + api_key[-4:] if api_key else 'Not set'}")

    # Start the Flask server
    print("\nStarting Flask server...")
    server_process = subprocess.Popen(
        [sys.executable, str(SCRIPT_DIR / "app.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "FLASK_PORT": "8080"},
    )

    try:
        # Wait for server to be ready
        print(f"Waiting for server at {SERVER_URL}...")
        if not wait_for_server(SERVER_URL):
            print("\nERROR: Server failed to start within timeout")
            server_process.terminate()
            stdout, _ = server_process.communicate(timeout=2)
            print("Server output:", stdout)
            sys.exit(1)

        print("Server is ready!")

        # Run optimization
        await run_optimization()

        print("\n" + "=" * 60)
        print("  Demo completed successfully!")
        print("=" * 60)

    finally:
        # Stop the server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("Server stopped.")


if __name__ == "__main__":
    # Load .env from walkthrough if available
    env_file = PROJECT_ROOT / "walkthrough" / "examples" / "real" / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
        print()

    asyncio.run(main())
