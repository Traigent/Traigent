#!/usr/bin/env python
"""
Traigent Quick Start Script

One-command setup for new users to get started with Traigent.
Handles dependency checking, environment setup, and runs a demo.
"""

import os
import subprocess
import sys
from pathlib import Path


# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(message):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_info(message):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.ENDC}")


def check_python_version():
    """Check if Python version is suitable."""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(
            f"Python {version.major}.{version.minor}.{version.micro} is supported"
        )
        return True
    else:
        print_error(
            f"Python {version.major}.{version.minor} is not supported. Please use Python 3.11+"
        )
        return False


def check_virtual_env():
    """Check if we're in a virtual environment."""
    print_header("Checking Virtual Environment")
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print_success("Running in virtual environment")
        return True
    else:
        print_warning("Not in a virtual environment")
        print_info("Recommended: Create a virtual environment with:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        response = input("\nContinue anyway? (y/n): ")
        return response.lower() == "y"


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")

    print_info("Installing Traigent SDK with the documented source install...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".[recommended]", "--quiet"],
            check=True,
            capture_output=True,
        )
        print_success("Traigent SDK installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error("Failed to install Traigent SDK")
        print(e.stderr.decode() if e.stderr else "")
        return False


def verify_imports():
    """Verify all required imports work."""
    print_header("Verifying Imports")

    required_imports = [
        ("traigent", "Traigent SDK"),
        ("langchain", "LangChain"),
        ("langchain_openai", "LangChain OpenAI"),
        ("langchain_chroma", "LangChain Chroma"),
        ("openai", "OpenAI"),
        ("dotenv", "python-dotenv"),
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("httpx", "httpx"),
        ("mcp", "MCP"),
    ]

    all_good = True
    for module, name in required_imports:
        try:
            __import__(module)
            print_success(f"{name} imported successfully")
        except ImportError:
            print_error(f"Cannot import {name} ({module})")
            all_good = False

    return all_good


def setup_environment():
    """Set up environment variables."""
    print_header("Setting Up Environment")

    env_file = Path(".env")

    if env_file.exists():
        print_info(".env file already exists")
    else:
        print_info("Creating .env template for real credentials...")
        env_content = """# Traigent Environment Variables
# Set this to sync optimization results to the Traigent portal.
# TRAIGENT_API_KEY=your-traigent-key

# Add provider keys for real LLM calls from your own application code.
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
# TRAIGENT_BACKEND_URL=http://localhost:5000
"""
        env_file.write_text(env_content)
        print_success(".env file created")

    if os.environ.get("TRAIGENT_API_KEY"):
        print_success("TRAIGENT_API_KEY detected; portal sync is available")
    else:
        print_info("Set TRAIGENT_API_KEY in .env or your shell to sync results")
    return True


def run_demo():
    """Run a simple demo to verify everything works."""
    print_header("Running Demo")

    try:
        print_info("Running packaged quickstart with default optimization settings...")
        result = subprocess.run(
            [sys.executable, "-m", "traigent.examples.quickstart"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            print(result.stdout)
            print_success("Demo completed successfully!")
            return True
        else:
            print_error("Demo failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print_error("Demo timed out")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("🎉 Setup Complete!")

    print(f"{Colors.BOLD}Quick commands to try:{Colors.ENDC}")
    print("\n1. Run examples:")
    print("   python examples/core/rag-optimization/run.py")
    print("   python examples/core/multi-objective-tradeoff/run.py")

    print("\n2. Verify installation:")
    print("   python scripts/validation/verify_installation.py")

    print("\n3. Explore documentation:")
    print("   - README.md - Main documentation")
    print("   - docs/evaluation_guide.md - Understanding evaluation")
    print("   - examples/ - Working examples")

    print("\n4. When ready for real usage:")
    print("   - Add your API keys to .env")
    print("   - Run with real LLM providers")

    print(f"\n{Colors.BOLD}{Colors.GREEN}Happy optimizing! 🚀{Colors.ENDC}")


def main():
    """Main entry point."""
    print_header("🚀 Traigent Quick Start")
    print("This script will set up Traigent and verify everything works.\n")

    # Check Python version
    if not check_python_version():
        return 1

    # Check virtual environment
    if not check_virtual_env():
        return 1

    # Install dependencies
    if not install_dependencies():
        print_error("Failed to install dependencies")
        print_info('Try running: pip install -e ".[recommended]"')
        return 1

    # Verify imports
    if not verify_imports():
        print_error("Some imports failed")
        print_info(
            'Try reinstalling the documented bundle: pip install -e ".[recommended]"'
        )
        return 1

    # Set up environment
    if not setup_environment():
        return 1

    # Run demo
    if not run_demo():
        print_warning("Demo didn't complete, but setup is done")

    # Print next steps
    print_next_steps()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
