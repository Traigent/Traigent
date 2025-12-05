#!/bin/bash
# Script to update vulnerable packages identified in security audit

echo "Updating vulnerable packages to secure versions..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "traigent_test_env" ]; then
    source traigent_test_env/bin/activate
fi

# Update vulnerable packages to secure versions
echo "Updating requests to latest secure version..."
pip install --upgrade "requests>=2.31.0"

echo "Updating urllib3 to latest secure version..."
pip install --upgrade "urllib3>=1.26.18"

echo "Updating pyyaml to latest secure version..."
pip install --upgrade "pyyaml>=6.0.1"

echo "Updating cryptography to latest..."
pip install --upgrade "cryptography>=41.0.0"

echo "Updating other security-related packages..."
pip install --upgrade "pyjwt>=2.8.0"
pip install --upgrade "passlib>=1.7.4"

# Check for other outdated packages
echo ""
echo "Checking for other outdated packages..."
pip list --outdated

echo ""
echo "Vulnerable packages updated. Please run tests to ensure compatibility."
echo "Run: pytest tests/ -v"
