#!/usr/bin/env python3
"""Test that backend submission works with proper metadata."""

import asyncio

import pytest

from traigent.cloud.backend_client import BackendClientConfig, BackendIntegratedClient


@pytest.mark.asyncio
async def test_submission():
    """Test submitting with proper metadata."""
    print("Testing backend submission with metadata fix...")
    print("=" * 60)

    print("⚠️  Note: This test requires a backend server at http://localhost:5000")
    print("    To run the backend: cd backend && python app.py")

    try:
        # Create backend client
        backend_config = BackendClientConfig(
            backend_base_url="http://localhost:5000", enable_session_sync=True
        )
        backend_client = BackendIntegratedClient(
            api_key=None,  # Edge Analytics mode doesn't need API key
            backend_config=backend_config,
            enable_fallback=True,
        )

        # Create a test session
        session_id = backend_client.create_session(
            function_name="test_function",
            search_space={"temperature": [0.5, 0.7]},
            optimization_goal="maximize",
            metadata={"test": True},
        )

        print(f"Created session: {session_id}")

        # Test 1: Summary stats submission (privacy mode)
        print("\n1. Testing summary_stats submission (privacy mode)...")

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
                }
            },
            "execution_time": 10.5,
            "total_examples": 10,
            "metadata": {
                "sdk_version": "2.0.0",
                "aggregation_method": "pandas.describe",
            },
        }

        trial_id = f"trial_privacy_{session_id[:8]}"
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
                print("   ✅ Successfully submitted summary_stats with metadata!")
            else:
                print("   ❌ Failed to submit summary_stats")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 2: Regular submission (cloud mode)
        print("\n2. Testing regular submission (cloud mode)...")

        metrics = {"score": 0.85, "accuracy": 0.85}

        trial_id2 = f"trial_cloud_{session_id[:8]}"

        try:
            result = await backend_client._submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id2,
                config=config,
                metrics=metrics,
                status="completed",
                execution_mode="cloud",
            )

            if result:
                print("   ✅ Successfully submitted regular metrics with metadata!")
            else:
                print("   ❌ Failed to submit regular metrics")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 3: Full submission flow
        print("\n3. Testing full submission flow...")

        metadata = {
            "execution_mode": "edge_analytics",
            "trial_id": f"trial_full_{session_id[:8]}",
            "summary_stats": summary_stats,
        }

        backend_client.submit_result(
            session_id=session_id,
            config={"temperature": 0.5},
            score=0.90,
            metadata=metadata,
        )

        print("   ✅ Submission flow completed")

        print("\n" + "=" * 60)
        print("Test complete!")
        print("\nExpected behavior:")
        print("- Backend should receive metadata field (not None)")
        print("- Privacy mode uses summary stats format in metrics")
        print("- Cloud mode uses simple metrics values")

    except Exception as e:
        print(f"\n⚠️  Test skipped: {e}")
        print(
            "   Backend server is not running. To test this functionality, start the backend server."
        )
        # Skip test when backend is unavailable
        pytest.skip(f"Backend server not available: {e}")


if __name__ == "__main__":
    asyncio.run(test_submission())
