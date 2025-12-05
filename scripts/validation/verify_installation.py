#!/usr/bin/env python
"""
Verify TraiGent Installation Script

This script verifies that all TraiGent dependencies are installed correctly
and that the SDK is ready to use.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependency(module_name: str, import_path: str = None) -> bool:
    """Check if a dependency is installed."""
    import_name = import_path or module_name
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def main():
    """Run installation verification checks."""
    print("🔍 TraiGent Installation Verification")
    print("=" * 50)

    all_ok = True

    # Core TraiGent
    print("\n📦 Core Package:")
    if check_dependency("traigent"):
        print("  ✅ traigent")
    else:
        print("  ❌ traigent - Core package not installed!")
        all_ok = False

    # Required Dependencies
    print("\n📚 Required Dependencies:")
    required_deps = [
        ("langchain", "langchain"),
        ("langchain-community", "langchain_community"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-chroma", "langchain_chroma"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pydantic", "pydantic"),
    ]

    for package_name, import_name in required_deps:
        if check_dependency(package_name, import_name):
            print(f"  ✅ {package_name}")
        else:
            print(
                f"  ❌ {package_name} - Missing! Install with: pip install {package_name}"
            )
            all_ok = False

    # Optional Dependencies
    print("\n📋 Optional Dependencies:")
    optional_deps = [
        ("mlflow", "mlflow"),
        ("wandb", "wandb"),
        ("streamlit", "streamlit"),
    ]

    for package_name, import_name in optional_deps:
        if check_dependency(package_name, import_name):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ⚠️  {package_name} - Optional, not installed")

    # Test basic imports
    print("\n🧪 Testing Basic Imports:")
    try:
        # Import checks only
        from dotenv import load_dotenv  # noqa: F401 - Import check only
        from langchain_chroma import Chroma  # noqa: F401 - Import check only
        from langchain_openai import ChatOpenAI  # noqa: F401 - Import check only

        import traigent

        print("  ✅ All basic imports work!")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        all_ok = False

    # Check environment variables
    print("\n🔐 Environment Variables:")
    env_vars = [
        "OPENAI_API_KEY",
        "TRAIGENT_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    for var in env_vars:
        if os.environ.get(var):
            print(f"  ✅ {var} is set")
        else:
            print(f"  ⚠️  {var} not set (may be required for some features)")

    # Test mock mode
    print("\n🎭 Testing Mock Mode:")
    try:
        os.environ["TRAIGENT_MOCK_MODE"] = "true"
        import traigent

        # Create a simple test function
        @traigent.optimize(
            configuration_space={"temperature": [0.1, 0.5, 0.9]},
            objectives=["accuracy"],
        )
        def test_function(input: str) -> str:
            return f"Test output for: {input}"

        print("  ✅ Mock mode setup works!")

        # Clean up
        del os.environ["TRAIGENT_MOCK_MODE"]
    except Exception as e:
        print(f"  ❌ Mock mode error: {e}")
        all_ok = False

    # Final summary
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ Installation verified successfully!")
        print("\n🚀 You're ready to use TraiGent!")
        print("\nQuick test command:")
        print("  TRAIGENT_MOCK_MODE=true python examples/core/hello-world/run.py")
        return 0
    else:
        print("❌ Some issues found. Please fix them before proceeding.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements/requirements-integrations.txt")
        print("  pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
