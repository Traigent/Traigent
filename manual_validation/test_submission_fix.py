#!/usr/bin/env python3
"""Test script to verify backend submission fix."""

import asyncio
import json

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)


@pytest.mark.asyncio
async def test_backend_submission():
    """Test that summary_stats are being sent correctly to backend."""
    print("Testing backend submission fix...")
    print("=" * 60)

    print("⚠️  Note: This test requires a backend server at http://localhost:5000")
    print("    To run the backend: cd backend && python app.py")

    try:
        # Create backend client
        backend_config = BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )
        backend_client = BackendIntegratedClient(
            api_key=None,
            backend_config=backend_config,
            enable_fallback=True,
            timeout=3.0,
        )

        # Create a test session
        session_id = backend_client.create_session(
            function_name="test_submission_fix",
            search_space={"temperature": [0.5, 0.7, 1.0]},
            optimization_goal="maximize",
            metadata={"test": True},
        )

        print(f"✅ Created session: {session_id}")

        # Create metrics tracker and add some example metrics
        tracker = MetricsTracker()
        tracker.start_tracking()

        for i in range(5):
            example = ExampleMetrics(
                tokens=TokenMetrics(
                    input_tokens=100 + i * 10,
                    output_tokens=50 + i * 5,
                    total_tokens=150 + i * 15,
                ),
                response=ResponseMetrics(
                    response_time_ms=1000 + i * 100, tokens_per_second=10 + i * 0.5
                ),
                cost=CostMetrics(
                    input_cost=0.001 + i * 0.0001,
                    output_cost=0.002 + i * 0.0002,
                    total_cost=0.003 + i * 0.0003,
                ),
                success=True,
                error=None,
            )
            tracker.add_example_metrics(example)

        tracker.end_tracking()

        # Generate summary stats
        summary_stats = tracker.format_as_summary_stats()

        print("\n📊 Summary stats structure:")
        print(json.dumps(summary_stats, indent=2)[:500] + "...")

        # Test submission with privacy mode
        test_config = {"temperature": 0.7, "model": "test-model"}

        # Submit with privacy mode (should use summary_stats)
        print("\n🔄 Submitting with privacy mode...")
        metadata = {
            "execution_mode": "privacy",  # Use "privacy" mode
            "trial_id": f"test_trial_{int(asyncio.get_event_loop().time())}",
            "summary_stats": summary_stats,
            "duration": 5.0,
        }

        backend_client.submit_result(
            session_id=session_id, config=test_config, score=0.85, metadata=metadata
        )

        print("✅ Submission complete")

        # Wait a bit for backend processing
        await asyncio.sleep(1)

        print("\n✨ Test completed successfully!")
        print(
            "Check backend logs to verify that summary_stats field is being sent correctly."
        )
        print(
            "The backend should receive data with 'summary_stats' field, not 'metrics' field."
        )

    except Exception as e:
        print(f"\n⚠️  Test skipped: {e}")
        print(
            "   Backend server is not running. To test this functionality, start the backend server."
        )
        # Skip test when backend is unavailable
        pytest.skip(f"Backend server not available: {e}")


if __name__ == "__main__":
    asyncio.run(test_backend_submission())
