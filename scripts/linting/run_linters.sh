#!/bin/bash

# Script to run all linters on the TraiGent project
# This will help you see all problems from different linters

# Navigate to project root from scripts/linting directory
cd "$(dirname "$0")/../.."

echo "========================================="
echo "Running All Linters on TraiGent Project"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories to check
DIRS="traigent tests examples"

# Run Flake8
echo -e "${YELLOW}Running Flake8...${NC}"
flake8 $DIRS --config=.flake8 --format='%(path)s:%(row)d:%(col)d: [FLAKE8] %(code)s %(text)s' 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Flake8: No issues found${NC}"
else
    echo -e "${RED}✗ Flake8: Issues found (see above)${NC}"
fi
echo ""

# Run MyPy (if available)
if command -v mypy &> /dev/null; then
    echo -e "${YELLOW}Running MyPy...${NC}"
    mypy $DIRS --config-file=mypy.ini 2>/dev/null | sed 's/^/[MYPY] /'
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ MyPy: No type issues found${NC}"
    else
        echo -e "${RED}✗ MyPy: Type issues found (see above)${NC}"
    fi
    echo ""
fi

# Run Pylint (if available)
if command -v pylint &> /dev/null; then
    echo -e "${YELLOW}Running Pylint...${NC}"
    pylint $DIRS --max-line-length=100 --disable=C0111,C0103,R0903,R0913,W0212 --msg-template='%(path)s:%(line)d:%(column)d: [PYLINT] %(msg_id)s %(msg)s' 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Pylint: No issues found${NC}"
    else
        echo -e "${RED}✗ Pylint: Issues found (see above)${NC}"
    fi
    echo ""
fi

# Run Ruff (if available)
if command -v ruff &> /dev/null; then
    echo -e "${YELLOW}Running Ruff...${NC}"
    ruff check $DIRS --config=pyproject.toml 2>/dev/null | sed 's/^/[RUFF] /'
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Ruff: No issues found${NC}"
    else
        echo -e "${RED}✗ Ruff: Issues found (see above)${NC}"
    fi
    echo ""
fi

# Summary
echo "========================================="
echo "Linting Complete!"
echo "========================================="
echo ""
echo "To see problems in VS Code:"
echo "1. Open the Problems panel: View → Problems (or Ctrl+Shift+M)"
echo "2. Reload VS Code window: Ctrl+Shift+P → 'Developer: Reload Window'"
echo "3. Check that Python extension is using correct interpreter:"
echo "   - Bottom left corner should show: Python 3.x.x ('traigent_test_env')"
echo ""
echo "For real-time linting in VS Code:"
echo "- Problems should appear automatically as you type"
echo "- Look for squiggly lines under problematic code"
echo "- Hover over squiggly lines to see error details"
