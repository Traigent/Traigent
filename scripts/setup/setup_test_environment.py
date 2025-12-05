#!/usr/bin/env python3
"""Setup script for TraiGent test environment.

This script prepares the test environment by creating necessary directories,
fixture files, and installing optional dependencies needed for testing.
"""

import os
from pathlib import Path


def generate_rsa_keypair():
    """Generate an RSA keypair for tests without storing static secrets."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError as exc:
        raise RuntimeError(
            "cryptography is required to generate test RSA keys. Install it with `pip install cryptography`."
        ) from exc

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem.decode(), public_pem.decode()


def create_directory_structure():
    """Create necessary test directories."""
    directories = [
        "tests/fixtures",
        "tests/fixtures/jwt",
        "tests/mcp/fixtures",
    ]

    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {path}")
        else:
            print(f"ℹ️  Directory already exists: {path}")


def create_test_env_file():
    """Create test environment file with mock credentials."""
    test_env_file = Path(".env.test")

    env_content = """# Test environment variables
# DO NOT USE REAL API KEYS IN THIS FILE
TRAIGENT_API_KEY=test-api-key-12345
OPENAI_API_KEY=test-openai-key-67890
ANTHROPIC_API_KEY=test-anthropic-key-abcdef
TRAIGENT_TEST_MODE=true
TRAIGENT_BACKEND_URL=http://localhost:5000
TRAIGENT_DISABLE_ANALYTICS=true
TRAIGENT_LOG_LEVEL=ERROR
"""

    if not test_env_file.exists():
        with open(test_env_file, "w") as f:
            f.write(env_content)
        print(f"✅ Created {test_env_file}")
    else:
        print(f"ℹ️  Test environment file already exists: {test_env_file}")


def create_jwt_test_fixtures():
    """Create mock JWT keys for testing."""
    jwt_test_dir = Path("tests/fixtures/jwt")

    try:
        private_key, public_key = generate_rsa_keypair()
    except RuntimeError as exc:
        print(f"⚠️  {exc}")
        print(
            "   Skipping JWT fixture generation. Run this script again after installing dependencies or provide your own test keys."
        )
        return

    # Create key files
    private_key_path = jwt_test_dir / "test_private_key.pem"
    public_key_path = jwt_test_dir / "test_public_key.pem"

    if not private_key_path.exists():
        with open(private_key_path, "w") as f:
            f.write(private_key)
        print(f"✅ Created test private key: {private_key_path}")

    if not public_key_path.exists():
        with open(public_key_path, "w") as f:
            f.write(public_key)
        print(f"✅ Created test public key: {public_key_path}")


def create_mcp_fixtures():
    """Create MCP test fixtures."""
    mcp_fixtures_dir = Path("tests/mcp/fixtures")

    # Create a sample MCP validation fixture
    validation_fixture = {
        "agent_spec": {
            "name": "test_agent",
            "type": "conversational",
            "model": "gpt-3.5-turbo",
        },
        "expected_validation": {"is_valid": True, "errors": []},
    }

    fixture_path = mcp_fixtures_dir / "sample_validation.json"
    if not fixture_path.exists():
        import json

        with open(fixture_path, "w") as f:
            json.dump(validation_fixture, f, indent=2)
        print(f"✅ Created MCP validation fixture: {fixture_path}")


def check_dependencies():
    """Check and suggest installation of optional test dependencies."""
    print("\n📦 Checking optional test dependencies...")

    optional_deps = {
        "pyjwt": "JWT token handling",
        "cryptography": "Cryptographic operations",
        "mcp": "Model Context Protocol (optional)",
        "pytest-asyncio": "Async test support",
        "pytest-mock": "Mocking support",
        "pytest-cov": "Coverage reporting",
    }

    missing_deps = []
    for package, description in optional_deps.items():
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - {description}")
        except ImportError:
            missing_deps.append(package)
            print(f"❌ {package} - {description} (not installed)")

    if missing_deps:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_deps)}")
        print("\nTo install missing dependencies, run:")
        print(f"pip install {' '.join(missing_deps)}")

        # Check if they're in requirements-dev.txt
        if Path("requirements/requirements-dev.txt").exists():
            print("\nOr install all dev dependencies:")
            print("pip install -r requirements/requirements-dev.txt")


def main():
    """Main setup function."""
    print("🔧 Setting up TraiGent test environment...\n")

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    # Run setup steps
    create_directory_structure()
    create_test_env_file()
    create_jwt_test_fixtures()
    create_mcp_fixtures()
    check_dependencies()

    print("\n✅ Test environment setup completed!")
    print("\n📝 Next steps:")
    print("1. Install any missing dependencies shown above")
    print("2. Copy .env.test to .env if you don't have a .env file")
    print("3. Run tests with: pytest tests/unit -v")
    print("4. For VSCode: Reload window for changes to take effect")

    # Create a marker file to indicate setup has been run
    marker_file = Path(".test_environment_setup")
    with open(marker_file, "w") as f:
        f.write(
            f"Test environment setup completed on {os.environ.get('USER', 'unknown')}@{os.uname().nodename}\n"
        )


if __name__ == "__main__":
    main()
