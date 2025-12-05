#!/usr/bin/env python3
"""Local Integration Test Script for TraiGent + OptiGen.

This script tests the integration between TraiGent SDK and the local OptiGen
backend environment. It validates the complete flow from SDK configuration
to data storage and retrieval.
"""

import os
import sys
import time
from typing import Any, Dict

import requests


def print_status(message: str, status: str = "INFO") -> None:
    """Print colored status messages."""
    colors = {
        "INFO": "\033[34m",  # Blue
        "SUCCESS": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "RESET": "\033[0m",  # Reset
    }

    color = colors.get(status, colors["INFO"])
    reset = colors["RESET"]
    print(f"{color}[{status}]{reset} {message}")


def check_backend_health() -> bool:
    """Check if the local OptiGen backend is healthy and responding."""
    backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
    try:
        print_status("Checking backend health...")
        response = requests.get(f"{backend_url}/health", timeout=5)

        if response.status_code == 200:
            print_status("✅ Backend is healthy and responding", "SUCCESS")
            return True
        else:
            print_status(f"❌ Backend returned status {response.status_code}", "ERROR")
            return False

    except requests.exceptions.ConnectionError:
        print_status(f"❌ Cannot connect to backend at {backend_url}", "ERROR")
        print_status("Make sure the local testing environment is running:", "INFO")
        print_status(
            "  cd llm-evaluation-installer && ./scripts/start-local-test.sh", "INFO"
        )
        return False
    except Exception as e:
        print_status(f"❌ Error checking backend health: {e}", "ERROR")
        return False


def check_feature_flags() -> Dict[str, Any]:
    """Check backend feature flag configuration."""
    try:
        print_status("Checking feature flag configuration...")
        backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
        response = requests.get(f"{backend_url}/api/test/features/status", timeout=5)

        if response.status_code == 200:
            features = response.json()
            print_status("✅ Feature flags retrieved successfully", "SUCCESS")

            # Check critical flags for Edge Analytics mode
            critical_flags = {
                "edge_analytics_only": True,
                "api_key_management": True,
                "cloud_features": False,
                "evaluation_services": False,
            }

            all_good = True
            for flag, expected in critical_flags.items():
                actual = features.get(flag, not expected)
                if actual == expected:
                    print_status(f"  ✅ {flag}: {actual}", "SUCCESS")
                else:
                    print_status(
                        f"  ❌ {flag}: {actual} (expected {expected})", "ERROR"
                    )
                    all_good = False

            if all_good:
                print_status(
                    "✅ All critical feature flags are correctly configured", "SUCCESS"
                )
            else:
                print_status("❌ Some feature flags are misconfigured", "ERROR")

            return features

        else:
            print_status(
                f"❌ Could not retrieve feature flags (status {response.status_code})",
                "ERROR",
            )
            return {}

    except Exception as e:
        print_status(f"❌ Error checking feature flags: {e}", "ERROR")
        return {}


def check_api_key_management() -> bool:
    """Test if API key management endpoints are accessible."""
    try:
        print_status("Checking API key management...")

        # This should work in Edge Analytics mode
        backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
        response = requests.get(
            f"{backend_url}/api/test/features/api-keys-only", timeout=5
        )

        if response.status_code == 200:
            print_status("✅ API key management is enabled and accessible", "SUCCESS")
            return True
        elif response.status_code == 501:
            print_status("❌ API key management is disabled", "ERROR")
            print_status(
                "Check ENABLE_API_KEY_MANAGEMENT=true in backend config", "WARNING"
            )
            return False
        else:
            print_status(
                f"❌ Unexpected response from API key test: {response.status_code}",
                "ERROR",
            )
            return False

    except Exception as e:
        print_status(f"❌ Error testing API key management: {e}", "ERROR")
        return False


def check_traigent_import() -> bool:
    """Check if TraiGent SDK can be imported."""
    try:
        print_status("Checking TraiGent SDK import...")
        import traigent

        print_status(
            "✅ TraiGent SDK imported successfully (version available)", "SUCCESS"
        )

        # Check if we can access Edge Analytics mode configuration
        config = traigent.TraigentConfig.edge_analytics_mode()
        print_status("✅ Edge Analytics mode configuration created", "SUCCESS")
        print_status(f"  Storage path: {config.get_local_storage_path()}", "INFO")
        print_status(f"  Execution mode: {config.execution_mode}", "INFO")

        return True
    except ImportError as e:
        print_status(f"❌ Cannot import TraiGent SDK: {e}", "ERROR")
        print_status("Make sure TraiGent is installed: pip install -e .", "INFO")
        return False
    except Exception as e:
        print_status(f"❌ Error testing TraiGent: {e}", "ERROR")
        return False


def check_database_connection() -> bool:
    """Check if the database is accessible through the backend."""
    try:
        print_status("Checking database connection...")

        # Try a simple endpoint that requires database access
        backend_url = os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000")
        response = requests.get(f"{backend_url}/api/health", timeout=5)

        if response.status_code == 200:
            print_status("✅ Database connection appears to be working", "SUCCESS")
            return True
        else:
            print_status(
                f"❌ Database connection issue (status {response.status_code})", "ERROR"
            )
            return False

    except Exception as e:
        print_status(f"❌ Error checking database connection: {e}", "ERROR")
        return False


def check_frontend_access() -> bool:
    """Check if the frontend is accessible."""
    try:
        print_status("Checking frontend accessibility...")
        response = requests.get("http://localhost:3000", timeout=5)

        if response.status_code == 200:
            print_status(
                "✅ Frontend is accessible at http://localhost:3000", "SUCCESS"
            )
            return True
        else:
            print_status(f"❌ Frontend returned status {response.status_code}", "ERROR")
            return False

    except requests.exceptions.ConnectionError:
        print_status("❌ Cannot connect to frontend at http://localhost:3000", "ERROR")
        return False
    except Exception as e:
        print_status(f"❌ Error checking frontend: {e}", "ERROR")
        return False


def run_integration_test() -> bool:
    """Run basic integration test if API key is available."""
    api_key = os.getenv("TRAIGENT_API_KEY")

    if not api_key:
        print_status(
            "⚠️  TRAIGENT_API_KEY not set - skipping integration test", "WARNING"
        )
        print_status("To test SDK integration:", "INFO")
        print_status("1. Register at http://localhost:3000", "INFO")
        print_status("2. Generate an API key", "INFO")
        print_status("3. Set: export TRAIGENT_API_KEY=your-key", "INFO")
        print_status("4. Run this test again", "INFO")
        return False

    try:
        print_status("Running basic SDK integration test...")

        # Import TraiGent and configure for Edge Analytics mode
        import traigent

        config = traigent.TraigentConfig.edge_analytics_mode()
        traigent.initialize(
            api_key=api_key,
            api_url=os.environ.get("TRAIGENT_BACKEND_URL", "http://localhost:5000"),
            config=config,
        )

        print_status("✅ SDK configured successfully", "SUCCESS")
        print_status("✅ Integration test passed", "SUCCESS")

        return True

    except Exception as e:
        print_status(f"❌ Integration test failed: {e}", "ERROR")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("🧪 TraiGent + OptiGen Local Integration Test")
    print("=" * 60)
    print()

    tests = [
        ("Backend Health", check_backend_health),
        ("Feature Flags", lambda: bool(check_feature_flags())),
        ("API Key Management", check_api_key_management),
        ("TraiGent SDK", check_traigent_import),
        ("Database Connection", check_database_connection),
        ("Frontend Access", check_frontend_access),
        ("SDK Integration", run_integration_test),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print_status(f"❌ Test failed with exception: {e}", "ERROR")
            results[test_name] = False

        time.sleep(0.5)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1

    print(f"\nTests Passed: {passed}/{total}")

    if passed == total:
        print_status("🎉 All tests passed! Your local environment is ready.", "SUCCESS")
        print()
        print("Next Steps:")
        print("1. Navigate to http://localhost:3000")
        print("2. Register/login to create an account")
        print("3. Generate an API key for SDK usage")
        print("4. Run: python examples/local_mode_example.py")
        print("5. View results in the webapp")
        return True
    else:
        print_status(
            f"❌ {total - passed} tests failed. Please fix issues before proceeding.",
            "ERROR",
        )
        print()
        print("Common fixes:")
        print("• Ensure Docker services are running")
        print("• Check environment variables in .env file")
        print("• Verify feature flags are correctly set")
        print("• Make sure TraiGent SDK is installed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
