#!/usr/bin/env python3
"""Check environment and dependencies for TraiGent walkthrough."""

import importlib.util
import os
import sys

# Colors for output
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"{GREEN}✓ Python {version.major}.{version.minor}.{version.micro}{NC}")
        return True
    else:
        print(f"{RED}✗ Python {version.major}.{version.minor} (need 3.8+){NC}")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print(f"{GREEN}✓ {package_name} installed{NC}")
        return True
    else:
        print(f"{YELLOW}⚠ {package_name} not installed{NC}")
        return False


def check_api_keys():
    """Check for API keys."""
    keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "TRAIGENT_API_KEY": "TraiGent Cloud",
    }

    found_any = False
    for key, name in keys.items():
        if os.getenv(key):
            print(f"{GREEN}✓ {name} API key found{NC}")
            found_any = True
        else:
            print(f"{YELLOW}⚠ {name} API key not set{NC}")

    return found_any


def check_traigent_installation():
    """Check if TraiGent is properly installed."""
    try:
        import traigent

        print(f"{GREEN}✓ TraiGent SDK installed{NC}")

        # Check version if available
        if hasattr(traigent, "__version__"):
            print(f"  Version: {traigent.__version__}")

        return True
    except ImportError:
        print(f"{RED}✗ TraiGent SDK not installed{NC}")
        print("  Run: pip install -e .")
        return False


def main():
    print(f"{CYAN}{'='*50}{NC}")
    print(f"{CYAN}TraiGent Environment Check{NC}")
    print(f"{CYAN}{'='*50}{NC}\n")

    all_good = True

    print("Python Environment:")
    if not check_python_version():
        all_good = False
    print()

    print("Core Dependencies:")
    if not check_traigent_installation():
        all_good = False
    print()

    print("Optional Integrations:")
    check_package("langchain")
    check_package("langchain_openai", "langchain_openai")
    check_package("langchain_anthropic", "langchain_anthropic")
    check_package("langchain_community", "langchain_community")
    check_package("openai")
    check_package("anthropic")
    print()

    print("API Keys:")
    has_keys = check_api_keys()
    print()

    if all_good:
        print(f"{GREEN}{'='*50}{NC}")
        print(f"{GREEN}✅ Environment ready for TraiGent walkthrough!{NC}")

        if not has_keys:
            print(
                f"\n{YELLOW}Note: No API keys found. You can still use MOCK mode.{NC}"
            )
            print(f"Set API keys for real optimization:{NC}")
            print("  export OPENAI_API_KEY='your-key'")
            print("  export ANTHROPIC_API_KEY='your-key'")
    else:
        print(f"{RED}{'='*50}{NC}")
        print(f"{RED}⚠ Some requirements are missing.{NC}")
        print("\nTo fix:")
        print("1. cd to TraiGent root directory")
        print("2. pip install -r requirements/requirements.txt")
        print("3. pip install -r requirements/requirements-integrations.txt")
        print("4. pip install -e .")

    print(f"{CYAN}{'='*50}{NC}")


if __name__ == "__main__":
    main()
