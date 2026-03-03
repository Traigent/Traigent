#!/usr/bin/env python3
"""Test backend configuration changes and validate no hardcoded URLs."""

import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_backend_config():
    """Test the new backend configuration system."""
    from traigent.config.backend_config import BackendConfig

    print("\n=== Testing Backend Configuration ===\n")

    # Test 1: Default configuration (no env vars set)
    print("1. Testing default configuration (no env vars):")
    # Clear any existing env vars
    for var in [
        "TRAIGENT_BACKEND_URL",
        "TRAIGENT_API_URL",
        "TRAIGENT_API_KEY",
        "TRAIGENT_DEFAULT_LOCAL_URL",
        "TRAIGENT_ENV",
    ]:
        os.environ.pop(var, None)

    config = BackendConfig.get_config_summary()
    print(f"   Backend URL: {config['backend_url']}")
    print(f"   Backend API URL: {config['backend_api_url']}")
    print(f"   API Key Configured: {config['api_key_configured']}")
    print(f"   Is Local: {config['is_local']}")
    print(f"   Requires Auth: {config['requires_auth']}")
    print(f"   Environment: {config['environment']}")
    print(f"   Configured Via: {config['configured_via']}")
    assert (
        config["backend_url"] == "http://localhost:5000"
    ), "Should default to local backend"
    assert config["is_local"], "Default should be local"
    assert config["configured_via"] == "default"
    assert (
        config["backend_api_url"] == "http://localhost:5000/api/v1"
    ), "Default API base should include versioned path"
    print("   ✅ Default configuration working correctly\n")

    # Test 2: Local development environment
    print("2. Testing local development environment:")
    os.environ["TRAIGENT_ENV"] = "development"
    config = BackendConfig.get_config_summary()
    print(f"   Backend URL: {config['backend_url']}")
    print(f"   Backend API URL: {config['backend_api_url']}")
    print(f"   Is Local: {config['is_local']}")
    print(f"   Requires Auth: {config['requires_auth']}")
    print(f"   Configured Via: {config['configured_via']}")
    assert (
        config["backend_url"] == "http://localhost:5000"
    ), "Should use local URL in dev"
    assert config["is_local"], "Should detect local backend"
    assert config["configured_via"] == "default"
    assert (
        config["backend_api_url"] == "http://localhost:5000/api/v1"
    ), "Local API base should reflect default path"
    print("   ✅ Local development configuration working correctly\n")

    # Test 3: Primary environment variable
    print("3. Testing TRAIGENT_BACKEND_URL environment variable:")
    os.environ["TRAIGENT_BACKEND_URL"] = "http://custom.backend:8080"
    config = BackendConfig.get_config_summary()
    print(f"   Backend URL: {config['backend_url']}")
    assert (
        config["backend_url"] == "http://custom.backend:8080"
    ), "Should use TRAIGENT_BACKEND_URL"
    assert (
        config["backend_api_url"] == "http://custom.backend:8080/api/v1"
    ), "API base should derive from backend origin"
    assert config["configured_via"] == "TRAIGENT_BACKEND_URL"
    print("   ✅ TRAIGENT_BACKEND_URL working correctly\n")

    # Test 3b: TRAIGENT_API_URL host override
    print("3b. Testing TRAIGENT_API_URL host override:")
    os.environ.pop("TRAIGENT_BACKEND_URL", None)
    os.environ["TRAIGENT_API_URL"] = "api.example.com"
    config = BackendConfig.get_config_summary()
    print(f"   Backend URL: {config['backend_url']}")
    assert (
        config["backend_url"] == "https://api.example.com"
    ), "Host-only API URL should resolve to https origin"
    assert config["configured_via"] == "TRAIGENT_API_URL"
    assert (
        config["backend_api_url"] == "https://api.example.com/api/v1"
    ), "API base should include default path"
    print("   ✅ TRAIGENT_API_URL host override working correctly\n")

    # Test 3c: TRAIGENT_API_URL with explicit path
    print("3c. Testing TRAIGENT_API_URL path override:")
    os.environ["TRAIGENT_API_URL"] = "http://localhost:5000/api/v1"
    config = BackendConfig.get_config_summary()
    print(f"   Backend URL: {config['backend_url']}")
    assert (
        config["backend_url"] == "http://localhost:5000"
    ), "API URL with path should set backend origin"
    assert (
        config["backend_api_url"] == "http://localhost:5000/api/v1"
    ), "API base should match explicit path"
    print("   ✅ TRAIGENT_API_URL path override working correctly\n")

    # Test 4: API key configuration
    print("4. Testing API key configuration:")
    os.environ["TRAIGENT_API_KEY"] = "traigent-key-12345"
    config = BackendConfig.get_config_summary()
    print(f"   API Key Configured: {config['api_key_configured']}")
    print(f"   API Key Prefix: {config['api_key_prefix']}")
    print(f"   API Key Env: {config['api_key_env']}")
    assert config["api_key_configured"], "Should have API key"
    assert config["api_key_prefix"] == "traigent...", "Should show prefix"
    assert config["api_key_env"] == "TRAIGENT_API_KEY"

    print("   ✅ API key configuration working correctly\n")

    # Test 5: Test with traigent.initialize()
    print("5. Testing traigent.initialize() with new config:")
    import traigent

    # Clear env vars and set specific ones
    for var in [
        "TRAIGENT_BACKEND_URL",
        "TRAIGENT_API_URL",
        "TRAIGENT_API_KEY",
        "TRAIGENT_DEFAULT_LOCAL_URL",
        "TRAIGENT_ENV",
    ]:
        os.environ.pop(var, None)

    os.environ["TRAIGENT_BACKEND_URL"] = "http://localhost:5000"
    os.environ["TRAIGENT_API_KEY"] = "test-api-key"

    # Initialize without explicit URL (should use env var)
    result = traigent.initialize()
    print(f"   Initialize result: {result}")

    # Check that global config was set correctly
    from traigent.api.functions import _GLOBAL_CONFIG

    print(f"   Global config backend URL: {_GLOBAL_CONFIG.get('traigent_api_url')}")
    assert (
        _GLOBAL_CONFIG.get("traigent_api_url") == "http://localhost:5000/api/v1"
    ), "Should use env var"
    print("   ✅ traigent.initialize() using centralized config correctly\n")

    print("=== All Backend Configuration Tests Passed ===\n")


def test_backend_client_config():
    """Test that backend client uses centralized config."""
    print("\n=== Testing Backend Client Configuration ===\n")

    # Set up environment for local backend
    os.environ["TRAIGENT_BACKEND_URL"] = "http://localhost:5000"
    os.environ["TRAIGENT_API_KEY"] = "test-key"
    os.environ["TRAIGENT_ENV"] = "development"

    # Import after setting env vars
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    # Test BackendClientConfig uses centralized config when not provided
    print("Testing BackendClientConfig defaults to centralized config:")
    config = BackendClientConfig()
    print(f"   Backend URL: {config.backend_base_url}")
    assert config.backend_base_url == "http://localhost:5000", "Should use env var"
    assert (
        config.api_base_url == "http://localhost:5000/api/v1"
    ), "API base should include default path"
    print("   ✅ BackendClientConfig using centralized configuration\n")

    # Test BackendIntegratedClient creation
    print("Testing BackendIntegratedClient with centralized config:")
    client = BackendIntegratedClient(
        api_key="test-key", backend_config=config, enable_fallback=True
    )
    print(f"   Client backend URL: {client.backend_config.backend_base_url}")
    assert (
        client.backend_config.backend_base_url == "http://localhost:5000"
    ), "Should use configured URL"
    assert (
        client.backend_config.api_base_url == "http://localhost:5000/api/v1"
    ), "Client should derive API base URL"
    print("   ✅ BackendIntegratedClient using centralized configuration\n")

    print("=== Backend Client Configuration Test Passed ===\n")


if __name__ == "__main__":
    try:
        test_backend_config()
        test_backend_client_config()
        print("\n✅ All tests passed! Backend configuration is working correctly.\n")
    except Exception as e:
        print(f"\n❌ Test failed: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
