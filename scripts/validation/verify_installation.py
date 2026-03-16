#!/usr/bin/env python
"""Verify the documented Traigent source installation."""

from __future__ import annotations

import importlib
import os
import sys
from importlib import metadata


def check_distribution(distribution_name: str) -> tuple[bool, str | None]:
    """Return whether a distribution is installed and its version if available."""
    try:
        return True, metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return False, None


def check_import(module_name: str) -> bool:
    """Return True when a module import succeeds."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def main() -> int:
    """Run installation verification checks."""
    print("Traigent Installation Verification")
    print("=" * 50)
    print('Expected install path: pip install -e ".[recommended]"')

    all_ok = True

    print("\nCore Package:")
    traigent_installed, traigent_version = check_distribution("traigent")
    if traigent_installed:
        print(f"  OK traigent ({traigent_version})")
    else:
        print("  FAIL traigent distribution is not installed")
        all_ok = False

    print("\nRecommended Dependencies:")
    required_deps = [
        ("langchain", "langchain"),
        ("langchain-community", "langchain_community"),
        ("langchain-openai", "langchain_openai"),
        ("langchain-chroma", "langchain_chroma"),
        ("openai", "openai"),
        ("python-dotenv", "dotenv"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
        ("httpx", "httpx"),
        ("mcp", "mcp"),
        ("pydantic-ai", "pydantic_ai"),
    ]

    for package_name, import_name in required_deps:
        if check_import(import_name):
            print(f"  OK {package_name}")
        else:
            print(f"  FAIL {package_name} missing or not importable")
            all_ok = False

    print("\nOptional Integrations:")
    for package_name, import_name in [("mlflow", "mlflow"), ("wandb", "wandb")]:
        if check_import(import_name):
            print(f"  OK {package_name}")
        else:
            print(f"  WARN {package_name} not importable")

    print("\nBasic Import Checks:")
    try:
        from dotenv import load_dotenv  # noqa: F401
        from langchain_chroma import Chroma  # noqa: F401
        from langchain_openai import ChatOpenAI  # noqa: F401

        import traigent

        print(
            f"  OK traigent imported from {traigent.__file__} with version "
            f"{traigent.get_version_info()['version']}"
        )
    except Exception as exc:
        print(f"  FAIL import check failed: {exc}")
        all_ok = False

    print("\nEnvironment Variables:")
    print("  INFO API keys are optional for this verifier; mock/offline mode is used.")
    for var in ["OPENAI_API_KEY", "TRAIGENT_API_KEY", "ANTHROPIC_API_KEY"]:
        if os.environ.get(var):
            print(f"  OK {var} is set")
        else:
            print(f"  WARN {var} not set")

    print("\nMock/Offline Validation:")
    old_mock = os.environ.get("TRAIGENT_MOCK_LLM")
    old_offline = os.environ.get("TRAIGENT_OFFLINE_MODE")
    try:
        os.environ["TRAIGENT_MOCK_LLM"] = "true"
        os.environ["TRAIGENT_OFFLINE_MODE"] = "true"
        import traigent

        @traigent.optimize(
            configuration_space={"temperature": [0.1, 0.5, 0.9]},
            objectives=["accuracy"],
        )
        def test_function(user_input: str) -> str:
            return f"Test output for: {user_input}"

        result = test_function("hello")
        print(f"  OK decorated function returned: {result}")
    except Exception as exc:
        print(f"  FAIL mock/offline validation failed: {exc}")
        all_ok = False
    finally:
        if old_mock is None:
            os.environ.pop("TRAIGENT_MOCK_LLM", None)
        else:
            os.environ["TRAIGENT_MOCK_LLM"] = old_mock
        if old_offline is None:
            os.environ.pop("TRAIGENT_OFFLINE_MODE", None)
        else:
            os.environ["TRAIGENT_OFFLINE_MODE"] = old_offline

    print("\n" + "=" * 50)
    if all_ok:
        print("Installation verified successfully.")
        print("\nQuick test command:")
        print(
            "  TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true "
            "python examples/quickstart/01_simple_qa.py"
        )
        return 0

    print("Installation verification failed.")
    print("\nReinstall the documented source bundle:")
    print('  pip install -e ".[recommended]"')
    return 1


if __name__ == "__main__":
    sys.exit(main())
