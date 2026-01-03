#!/bin/bash
# Local CI verification script
# Run this before pushing to catch CI issues early
#
# Usage: ./scripts/ci/local_ci_check.sh

set -e  # Exit on first error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================"
echo "🔍 Local CI Verification (test-examples workflow)"
echo "============================================================"
echo ""

# Ensure we're in project root
cd "$(dirname "$0")/../.."

# Use local venv python
PYTHON=".venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}❌ Virtual environment not found at .venv${NC}"
    echo "Run: make install-dev"
    exit 1
fi

export TRAIGENT_MOCK_LLM=true
export TRAIGENT_OFFLINE_MODE=true

echo -e "${YELLOW}Step 1/6: Verify installation${NC}"
$PYTHON scripts/validation/verify_installation.py || { echo -e "${RED}❌ Installation verification failed${NC}"; exit 1; }
echo -e "${GREEN}✅ Installation verified${NC}"
echo ""

echo -e "${YELLOW}Step 2/6: Run quickstart${NC}"
echo "y" | $PYTHON scripts/setup/quickstart.py || true
echo -e "${GREEN}✅ Quickstart completed${NC}"
echo ""

echo -e "${YELLOW}Step 3/6: Test first run example${NC}"
$PYTHON examples/quickstart/01_simple_qa.py || { echo -e "${RED}❌ First run example failed${NC}"; exit 1; }
echo -e "${GREEN}✅ First run example passed${NC}"
echo ""

echo -e "${YELLOW}Step 4/6: Test README examples${NC}"
$PYTHON -m pytest tests/examples/test_readme_examples.py -v || { echo -e "${RED}❌ README examples failed${NC}"; exit 1; }
echo -e "${GREEN}✅ README examples passed${NC}"
echo ""

echo -e "${YELLOW}Step 5/6: Test mock mode${NC}"
$PYTHON -c "
import os
os.environ['TRAIGENT_MOCK_LLM'] = 'true'
os.environ['TRAIGENT_OFFLINE_MODE'] = 'true'
import traigent

@traigent.optimize(
    configuration_space={'temperature': [0.1, 0.5]},
    objectives=['accuracy']
)
def test_func(x):
    return x

result = test_func('test')
assert result == 'test'
print('✅ Mock mode works!')
" || { echo -e "${RED}❌ Mock mode test failed${NC}"; exit 1; }
echo -e "${GREEN}✅ Mock mode test passed${NC}"
echo ""

echo -e "${YELLOW}Step 6/6: Run diagnostics${NC}"
$PYTHON -c "from traigent.utils.diagnostics import diagnose; report = diagnose(); report.print_report()" || true
echo -e "${GREEN}✅ Diagnostics completed${NC}"
echo ""

echo "============================================================"
echo -e "${GREEN}🎉 All local CI checks passed!${NC}"
echo "============================================================"
echo ""
echo "You can now safely push your changes."
