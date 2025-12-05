#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TERM="${TERM:-xterm-256color}"

mkdir -p output

echo "=================================="
echo "  Traigent Demo Generator"
echo "=================================="
echo ""

chmod +x scripts/*.sh 2>/dev/null || true

# Step 1: Generate cast files
echo "Step 1: Generating cast files..."
python3 scripts/generate-cast.py

# Step 2: Convert to SVG (if svg-term available)
if command -v svg-term &> /dev/null; then
    echo ""
    echo "Step 2: Converting to SVG..."
    for f in output/*.cast; do
        name=$(basename "$f" .cast)
        svg-term --in "$f" --out "output/${name}.svg" --window --width 100 --height 30
        echo "  output/${name}.svg"
    done
else
    echo ""
    echo "Note: svg-term not found. Install with: npm install -g svg-term-cli"
fi

echo ""
echo "Done! Files in output/"
ls -la output/
