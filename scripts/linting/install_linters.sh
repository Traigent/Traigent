#!/bin/bash

# Script to install all linters for VS Code integration
echo "========================================="
echo "Installing Linters for TraiGent Project"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating venv...${NC}"
    # Navigate to project root from scripts/linting directory
    cd "$(dirname "$0")/../.."
    source venv/bin/activate
fi

# Function to install a package
install_package() {
    local package=$1
    local name=$2
    echo -e "${YELLOW}Installing $name...${NC}"

    # Try to use pip from virtual env directly
    if [ -f "venv/bin/pip3" ]; then
        venv/bin/pip3 install --upgrade $package 2>/dev/null
    elif [ -f "venv/bin/python" ]; then
        venv/bin/python -m pip install --upgrade $package 2>/dev/null
    else
        python -m pip install --upgrade $package 2>/dev/null
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name installed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to install $name${NC}"
    fi
    echo ""
}

# Install linters
install_package "flake8" "Flake8"
install_package "mypy" "MyPy"
install_package "pylint" "Pylint"
install_package "ruff" "Ruff"
install_package "black" "Black (formatter)"
install_package "isort" "isort (import sorter)"

# Verify installations
echo "========================================="
echo "Verifying Installations"
echo "========================================="
echo ""

# Check each tool
for tool in flake8 mypy pylint ruff black isort; do
    if [ -f "traigent_test_env/bin/$tool" ]; then
        echo -e "${GREEN}✓ $tool is installed${NC}"
    else
        echo -e "${RED}✗ $tool is NOT installed${NC}"
    fi
done

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Restart VS Code or reload the window (Ctrl+Shift+P → 'Developer: Reload Window')"
echo "2. Ensure Python interpreter is set to traigent_test_env"
echo "3. Open the Problems panel (View → Problems or Ctrl+Shift+M)"
echo "4. Linters should now show all problems in your code"
echo ""
echo "To run all linters manually, use: ./run_linters.sh"
