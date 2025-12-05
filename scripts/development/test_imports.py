#!/usr/bin/env python
"""Test script to verify all security module imports are working correctly."""

# pylint: disable=unused-import
# ruff: noqa: F401

import sys


def test_imports():
    """Test all security module imports."""

    print("=" * 60)
    print("Testing Security Module Imports")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("=" * 60)

    errors = []

    # Test security modules
    print("\n📦 Testing security modules...")

    try:
        from traigent.security.jwt_validator import (
            JWTValidationError,  # noqa: F401 - Import test only
            JWTValidationResult,
            SecureJWTValidator,
            ValidationMode,
            get_secure_jwt_validator,
        )

        print("✅ jwt_validator imports OK")
    except ImportError as e:
        print(f"❌ jwt_validator import failed: {e}")
        errors.append(("jwt_validator", e))

    try:
        from traigent.security.auth import (
            MultiFactorAuth,
            OIDCAuthProvider,
            SAMLAuthProvider,
            SMSAuthProvider,
            TOTPAuthProvider,
            User,
        )

        print("✅ auth imports OK")
    except ImportError as e:
        print(f"❌ auth import failed: {e}")
        errors.append(("auth", e))

    try:
        from traigent.security.credentials import (
            CredentialType,
            EnhancedCredentialStore,
            SecureCredential,
            SecurityLevel,
        )

        print("✅ credentials imports OK")
    except ImportError as e:
        print(f"❌ credentials import failed: {e}")
        errors.append(("credentials", e))

    try:
        from traigent.security.rate_limiter import (
            RateLimitStrategy,
            SecureAuthenticationRateLimiter,
            SecureRateLimiter,
            SecureRateLimitResult,
        )

        print("✅ rate_limiter imports OK")
    except ImportError as e:
        print(f"❌ rate_limiter import failed: {e}")
        errors.append(("rate_limiter", e))

    try:
        from traigent.security.audit import (
            AuditLogger,
            ComplianceReporter,
            EventProcessor,
            SecurityMonitor,
        )

        print("✅ audit imports OK")
    except ImportError as e:
        print(f"❌ audit import failed: {e}")
        errors.append(("audit", e))

    try:
        from traigent.security.tenant import (
            Tenant,
            TenantContext,
            TenantManager,
            TenantStatus,
        )

        print("✅ tenant imports OK")
    except ImportError as e:
        print(f"❌ tenant import failed: {e}")
        errors.append(("tenant", e))

    # Test related modules
    print("\n📦 Testing related modules...")

    try:
        from traigent.cloud.auth import APIKey

        print("✅ cloud.auth.APIKey imports OK")
    except ImportError as e:
        print(f"❌ cloud.auth.APIKey import failed: {e}")
        errors.append(("cloud.auth.APIKey", e))

    try:
        from traigent.config.api_keys import APIKeyManager

        print("✅ config.api_keys.APIKeyManager imports OK")
    except ImportError as e:
        print(f"❌ config.api_keys.APIKeyManager import failed: {e}")
        errors.append(("config.api_keys.APIKeyManager", e))

    # Test the test files
    print("\n🧪 Testing test file imports...")

    try:
        import tests.unit.test_jwt_validator

        print("✅ test_jwt_validator imports OK")
    except ImportError as e:
        print(f"❌ test_jwt_validator import failed: {e}")
        errors.append(("test_jwt_validator", e))

    try:
        import tests.unit.security.test_security_authentication

        print("✅ test_security_authentication imports OK")
    except ImportError as e:
        print(f"❌ test_security_authentication import failed: {e}")
        errors.append(("test_security_authentication", e))

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} import(s) failed:")
        for module, error in errors:
            print(f"  - {module}: {error}")
        print("\nTo fix import errors:")
        print("1. Make sure you're using the correct virtual environment")
        print("2. Check if the module files exist")
        print("3. Verify the class/function names in the module")
        return False
    else:
        print("🎉 All imports successful!")
        print("\nYou can now run tests with:")
        print("  pytest tests/unit/test_jwt_validator.py -v")
        print("  pytest tests/unit/security/test_security_authentication.py -v")
        print("  pytest tests/security/ -v")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
