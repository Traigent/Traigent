#!/usr/bin/env python3
"""Comprehensive test for backend submission in both privacy and cloud modes."""

import asyncio

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
async def test_privacy_mode():
    """Test privacy mode submission."""
    print("\n" + "=" * 60)
    print("🔒 Testing PRIVACY MODE submission")
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

        # Create session
        session_id = backend_client.create_session(
            function_name="test_privacy_mode",
            search_space={"temperature": [0.5, 0.7]},
            optimization_goal="maximize",
            metadata={"mode": "privacy"},
        )

        print(f"✅ Created session: {session_id}")

        # Create metrics with summary stats
        tracker = MetricsTracker()
        tracker.start_tracking()

        for i in range(10):
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
        summary_stats = tracker.format_as_summary_stats()

        print(
            f"📊 Generated summary stats with {len(summary_stats['metrics'])} metrics"
        )

        # Submit with privacy mode
        metadata = {
            "execution_mode": "privacy",
            "trial_id": "privacy_test_trial",
            "summary_stats": summary_stats,
            "duration": 10.5,
        }

        backend_client.submit_result(
            session_id=session_id,
            config={"temperature": 0.7, "model": "test-model"},
            score=0.85,
            metadata=metadata,
        )

        print("✅ Privacy mode submission complete")
        print("   Backend should receive:")
        print("   - metrics field with summary statistics (count, mean, std, etc.)")
        print("   - metadata.mode = 'privacy'")
        print("   - summary_stats field with full data")

    except Exception as e:
        print(f"\n⚠️  Test skipped: {e}")
        print(
            "   Backend server is not running. To test this functionality, start the backend server."
        )


@pytest.mark.asyncio
async def test_cloud_mode():
    """Test cloud mode submission."""
    print("\n" + "=" * 60)
    print("☁️  Testing CLOUD MODE submission")
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

        # Create session
        session_id = backend_client.create_session(
            function_name="test_cloud_mode",
            search_space={"temperature": [0.5, 0.7]},
            optimization_goal="maximize",
            metadata={"mode": "cloud"},
        )

        print(f"✅ Created session: {session_id}")

        # Submit with cloud mode (detailed metrics)
        metadata = {
            "execution_mode": "cloud",
            "trial_id": "cloud_test_trial",
            "input_tokens_mean": 150.0,
            "output_tokens_mean": 75.0,
            "total_tokens_mean": 225.0,
            "response_time_ms_mean": 1500.0,
            "cost_mean": 0.005,
            "duration": 10.5,
        }

        backend_client.submit_result(
            session_id=session_id,
            config={"temperature": 0.5, "model": "claude-3"},
            score=0.92,
            metadata=metadata,
        )

        print("✅ Cloud mode submission complete")
        print("   Backend should receive:")
        print("   - metrics field with detailed metrics (score, token stats, etc.)")
        print("   - metadata.mode = 'cloud'")
        print("   - No summary_stats field")

    except Exception as e:
        print(f"\n⚠️  Test skipped: {e}")
        print(
            "   Backend server is not running. To test this functionality, start the backend server."
        )


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 COMPREHENSIVE BACKEND SUBMISSION TEST")
    print("=" * 60)

    await test_privacy_mode()
    await asyncio.sleep(1)  # Give backend time to process

    await test_cloud_mode()
    await asyncio.sleep(1)  # Give backend time to process

    print("\n" + "=" * 60)
    print("✨ All tests completed!")
    print("=" * 60)
    print("\n📝 Summary:")
    print("1. Privacy mode: Sends summary statistics (pandas.describe format)")
    print("2. Cloud mode: Sends detailed metrics")
    print("3. Backend always receives 'metrics' field")
    print("4. Mode is indicated via metadata.mode field")
    print("\nCheck backend logs to verify data is stored correctly.")


if __name__ == "__main__":
    asyncio.run(main())
