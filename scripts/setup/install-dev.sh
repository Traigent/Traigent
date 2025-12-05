#!/bin/bash
# Development installation script for TraiGent SDK

set -e

echo "🚀 Setting up TraiGent SDK for development..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version OK: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install package in development mode with all dependencies
echo "📦 Installing TraiGent SDK in development mode..."
pip install -e ".[dev,integrations,bayesian,docs]"

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Run initial tests
echo "🧪 Running initial tests..."
pytest tests/ -v

# Check code quality
echo "🔍 Running code quality checks..."
black --check traigent tests examples
isort --check-only traigent tests examples
flake8 traigent tests examples
mypy traigent

echo ""
echo "✅ Development setup complete!"
echo ""
echo "🔍 Verifying tokencost integration..."
python scripts/verify_tokencost_integration.py
echo ""
echo "🎯 Next steps:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run tests: pytest"
echo "  3. Check code style: black traigent tests"
echo "  4. Try the example: python examples/basic_optimization.py"
echo "  5. Verify tokencost: python scripts/verify_tokencost_integration.py"
echo "  6. Start developing! 🚀"
echo ""
echo "📚 Documentation:"
echo "  • README.md - Getting started"
echo "  • docs/CONTRIBUTING.md - Contributing guidelines"
echo "  • docs/ARCHITECTURE.md - System architecture"
echo "  • docs/SPRINT_PLAN.md - Development roadmap"
echo "  • docs/README.md - Complete documentation index"
echo ""
