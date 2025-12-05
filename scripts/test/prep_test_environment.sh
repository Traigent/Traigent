#!/bin/bash
# Prepare Test Environment - Clean caches and reinstall package
# Usage: ./scripts/test/prep_test_environment.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧹 Preparing Test Environment - Traigent SDK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Clean Python caches
echo "📦 Step 1/6: Cleaning Python cache files..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true
echo "   ✅ Python cache cleaned"
echo ""

# Step 2: Clean pytest cache
echo "🧪 Step 2/6: Cleaning pytest cache..."
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
rm -f .coverage 2>/dev/null || true
rm -rf htmlcov/ 2>/dev/null || true
rm -f coverage.xml 2>/dev/null || true
echo "   ✅ Pytest cache cleaned"
echo ""

# Step 3: Clean linter/type checker caches
echo "🔍 Step 3/6: Cleaning linter and type checker caches..."
rm -rf .mypy_cache/ 2>/dev/null || true
rm -rf .ruff_cache/ 2>/dev/null || true
rm -rf .tox/ 2>/dev/null || true
echo "   ✅ Linter caches cleaned"
echo ""

# Step 4: Clean build artifacts
echo "🏗️  Step 4/6: Cleaning build artifacts..."
rm -rf build/ 2>/dev/null || true
rm -rf dist/ 2>/dev/null || true
rm -rf *.egg-info/ 2>/dev/null || true
rm -rf .eggs/ 2>/dev/null || true
echo "   ✅ Build artifacts cleaned"
echo ""

# Step 5: Verify virtual environment
echo "🐍 Step 5/6: Verifying virtual environment..."
if [ ! -d ".venv" ]; then
    echo "   ⚠️  .venv directory not found!"
    echo "   Creating virtual environment with uv..."
    uv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment exists"
fi

# Check if we're in the virtual environment
if [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo "   ⚠️  Virtual environment not activated"
    echo "   Please run: source .venv/bin/activate"
    echo ""
fi
echo ""

# Step 6: Reinstall package in editable mode
echo "📦 Step 6/6: Reinstalling package in editable mode..."
if command -v uv &> /dev/null; then
    echo "   Using uv (faster)..."
    uv pip install -e ".[test]" --quiet
else
    echo "   Using pip..."
    pip install -e ".[test]" --quiet
fi
echo "   ✅ Package reinstalled"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Test Environment Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "You can now run tests:"
echo "  pytest tests/                          # Run all tests"
echo "  pytest tests/unit/                     # Run unit tests only"
echo "  pytest tests/ -v --cov=traigent        # Run with coverage"
echo "  pytest tests/ -k test_optimization     # Run specific tests"
echo ""
echo "Or use VS Code Test Explorer to run tests interactively."
echo ""
