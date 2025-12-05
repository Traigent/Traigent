"""Comprehensive test suite for all security modes."""

import os
import subprocess
import sys


def run_tests_in_environment(environment: str, env_vars: dict):
    """Run tests in specific environment."""
    print(f"\n{'='*60}")
    print(f"Testing in {environment.upper()} mode")
    print(f"{'='*60}\n")

    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)

    # Run tests
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/security/test_jwt_integration.py",
        "tests/security/test_jwt_validator_secure.py",
        "-v",
        "--tb=short",
        "-q",
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Parse results
    output = result.stdout + result.stderr
    passed = failed = 0

    for line in output.split("\n"):
        if "passed" in line and "failed" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "passed" in part and i > 0:
                    passed = int(parts[i - 1])
                if "failed" in part and i > 0:
                    failed = int(parts[i - 1])

    print(f"Results for {environment}:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")

    if result.returncode != 0 and failed > 0:
        print(f"\nErrors in {environment} mode:")
        print(result.stderr[:500])

    return passed, failed


def main():
    """Run comprehensive security tests in all modes."""
    print("=" * 60)
    print("COMPREHENSIVE SECURITY TEST SUITE")
    print("Testing all security components in all modes")
    print("=" * 60)

    environments = {
        "production": {
            "TRAIGENT_ENVIRONMENT": "production",
            "TRAIGENT_JWKS_URL": "https://test.example.com/jwks",
            "TRAIGENT_JWT_ISSUER": "test_issuer",
            "TRAIGENT_JWT_AUDIENCE": "test_audience",
            "TRAIGENT_MASTER_PASSWORD": "prod_secure_password_123",
        },
        "staging": {
            "TRAIGENT_ENVIRONMENT": "staging",
            "TRAIGENT_JWKS_URL": "https://staging.example.com/jwks",
            "TRAIGENT_JWT_ISSUER": "staging_issuer",
            "TRAIGENT_JWT_AUDIENCE": "staging_audience",
            "TRAIGENT_MASTER_PASSWORD": "staging_password_123",
        },
        "development": {
            "TRAIGENT_ENVIRONMENT": "development",
            "TRAIGENT_MASTER_PASSWORD": "dev_password_123",
        },
    }

    total_passed = 0
    total_failed = 0

    for env_name, env_vars in environments.items():
        passed, failed = run_tests_in_environment(env_name, env_vars)
        total_passed += passed
        total_failed += failed

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")

    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED IN ALL MODES!")
        return 0
    else:
        print(f"\n❌ {total_failed} tests failed across environments")
        return 1


if __name__ == "__main__":
    sys.exit(main())
