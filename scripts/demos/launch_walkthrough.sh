#!/bin/bash

# Traigent Interactive Walkthrough Launcher
# This script launches the interactive walkthrough system

# Colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║     🚀 Traigent Interactive Walkthrough Launcher 🚀     ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "walkthrough/walkthrough.sh" ]; then
    echo -e "${YELLOW}⚠️  Please run this script from the Traigent root directory${NC}"
    exit 1
fi

# Launch the walkthrough
echo -e "${GREEN}Launching Traigent Interactive Walkthrough...${NC}"
echo ""
cd walkthrough && ./walkthrough.sh
