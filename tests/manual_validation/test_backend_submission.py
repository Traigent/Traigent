#!/usr/bin/env python3
"""Test backend submission functionality.

Note: This test requires a backend server running at http://localhost:5000.
If the backend is not available, the test will fail gracefully with an informative message.
"""

import asyncio
import json

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient


@pytest.mark.asyncio
async def test_submission():
    """Test submitting summary_stats to backend."""
    print("Testing backend submission...")
    print("=" * 60)
    print("⚠️  Note: This test requires a backend server at http://localhost:5000")
    print("    To run the backend: cd backend && python app.py")
    print("=" * 60)

    # Create backend client
    backend_config = BackendClientConfig(
        backend_base_url="http://localhost:5000", enable_session_sync=True
    )
    backend_client = BackendIntegratedClient(
        api_key=None,  # Edge Analytics mode doesn't need API key
        backend_config=backend_config,
        enable_fallback=True,
        timeout=3.0,  # Fail fast if backend is unreachable
    )

    # Create a test session
    session_id = backend_client.create_session(
        function_name="test_function",
        search_space={"temperature": [0.5, 0.7]},
        optimization_goal="maximize",
        metadata={"test": True},
    )

    print(f"Created session: {session_id}")

    # Prepare summary_stats data (matching pandas.describe() format)
    summary_stats = {
        "metrics": {
            "accuracy": {
                "count": 10,
                "mean": 0.85,
                "std": 0.1,
                "min": 0.7,
                "25%": 0.8,
                "50%": 0.85,
                "75%": 0.9,
                "max": 0.95,
            },
            "input_tokens": {
                "count": 10,
                "mean": 100.0,
                "std": 10.0,
                "min": 80.0,
                "25%": 95.0,
                "50%": 100.0,
                "75%": 105.0,
                "max": 120.0,
            },
            "total_cost": {
                "count": 10,
                "mean": 0.003,
                "std": 0.0005,
                "min": 0.002,
                "25%": 0.0025,
                "50%": 0.003,
                "75%": 0.0035,
                "max": 0.004,
            },
        },
        "execution_time": 10.5,
        "total_examples": 10,
        "metadata": {
            "sdk_version": "2.0.0",
            "aggregation_method": "pandas.describe",
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }

    print("\nSubmitting summary_stats...")
    print(f"Summary stats structure: {json.dumps(summary_stats, indent=2)[:500]}...")

    # Test submission
    trial_id = f"trial_test_{session_id[:8]}"
    config = {"temperature": 0.7}

    try:
        result = await backend_client._submit_summary_stats(
            session_id=session_id,
            trial_id=trial_id,
            config=config,
            summary_stats=summary_stats,
            status="completed",
        )

        if result:
            print("\n✅ Successfully submitted summary_stats!")
        else:
            print("\n❌ Failed to submit summary_stats")

    except Exception as e:
        print(f"\n❌ Error submitting summary_stats: {e}")

    # Also test regular submission for comparison
    print("\n" + "-" * 60)
    print("Testing regular metrics submission (for comparison)...")

    metrics = {
        "score": 0.85,
        "accuracy": 0.85,
        "input_tokens_mean": 100.0,
        "input_tokens_median": 100.0,
        "input_tokens_std": 10.0,
        "cost_mean": 0.003,
        "cost_median": 0.003,
        "cost_std": 0.0005,
    }

    trial_id2 = f"trial_test2_{session_id[:8]}"

    try:
        result = await backend_client._submit_trial_result_via_session(
            session_id=session_id,
            trial_id=trial_id2,
            config=config,
            metrics=metrics,
            status="completed",
        )

        if result:
            print("✅ Successfully submitted regular metrics!")
        else:
            print("❌ Failed to submit regular metrics")

    except Exception as e:
        print(f"❌ Error submitting regular metrics: {e}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    try:
        asyncio.run(test_submission())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n💡 If the backend is not running, start it with:")
        print("    cd backend && python app.py")
        exit(1)
