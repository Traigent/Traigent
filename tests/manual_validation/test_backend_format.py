#!/usr/bin/env python3
"""Test different submission formats to understand what backend expects."""

import asyncio
import json

import aiohttp
import pytest


@pytest.mark.asyncio
async def test_formats():
    """Test different submission formats."""

    # First create a session
    session_data = {
        "problem_statement": "test_formats",
        "dataset": {"examples": [], "metadata": {}},
        "search_space": {"temperature": [0.5, 0.7]},
        "optimization_config": {
            "algorithm": "grid",
            "max_trials": 2,
            "optimization_goal": "maximize",
        },
        "metadata": {},
    }

    # Add timeout to prevent hanging
    timeout = aiohttp.ClientTimeout(total=5)

    print("⚠️  Note: This test requires a backend server at http://localhost:5000")
    print("    To run the backend: cd backend && python app.py")

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Create session
            async with session.post(
                "http://localhost:5000/api/v1/sessions", json=session_data
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    session_id = result.get("session_id")
                    print(f"Created session: {session_id}")
                else:
                    print(f"Failed to create session: {await response.text()}")
                    return

            # Test different submission formats
            test_configs = [
                {
                    "name": "Format 1: summary_stats only",
                    "data": {
                        "trial_id": "test_trial_1",
                        "config": {"temperature": 0.5},
                        "summary_stats": {
                            "metrics": {
                                "accuracy": {
                                    "count": 5,
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
                            "total_examples": 5,
                            "metadata": {
                                "sdk_version": "2.0.0",
                                "aggregation_method": "pandas.describe",
                            },
                        },
                        "status": "COMPLETED",
                        "metadata": {"mode": "privacy", "execution_mode": "privacy"},
                    },
                },
                {
                    "name": "Format 2: metrics + metadata with mode=privacy",
                    "data": {
                        "trial_id": "test_trial_2",
                        "config": {"temperature": 0.7},
                        "metrics": {
                            "accuracy": {
                                "count": 5,
                                "mean": 0.85,
                                "std": 0.1,
                                "min": 0.7,
                                "25%": 0.8,
                                "50%": 0.85,
                                "75%": 0.9,
                                "max": 0.95,
                            }
                        },
                        "status": "COMPLETED",
                        "metadata": {
                            "mode": "privacy",
                            "execution_mode": "privacy",
                            "aggregation_method": "pandas.describe",
                            "total_examples": 5,
                        },
                    },
                },
                {
                    "name": "Format 3: Both metrics and summary_stats",
                    "data": {
                        "trial_id": "test_trial_3",
                        "config": {"temperature": 0.9},
                        "metrics": {"score": 0.85},
                        "summary_stats": {
                            "metrics": {
                                "accuracy": {
                                    "count": 5,
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
                            "total_examples": 5,
                        },
                        "status": "COMPLETED",
                        "metadata": {"mode": "privacy", "execution_mode": "privacy"},
                    },
                },
            ]

            for test_config in test_configs:
                print(f"\n🧪 Testing: {test_config['name']}")
                print(f"Data: {json.dumps(test_config['data'], indent=2)[:300]}...")

                async with session.post(
                    f"http://localhost:5000/api/v1/sessions/{session_id}/results",
                    json=test_config["data"],
                ) as response:
                    if response.status in [200, 201]:
                        print("✅ Success!")
                    else:
                        error = await response.text()
                        print(f"❌ Failed: {response.status} - {error[:200]}")

    except TimeoutError:
        print("\n⚠️  Test timed out after 5 seconds")
        print("   The backend server is likely not running at http://localhost:5000")
        print("   To run the backend: cd backend && python app.py")
    except aiohttp.ClientConnectorError as e:
        print(f"\n⚠️  Could not connect to backend: {e}")
        print("   Please ensure the backend server is running at http://localhost:5000")
        print("   To run the backend: cd backend && python app.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_formats())
