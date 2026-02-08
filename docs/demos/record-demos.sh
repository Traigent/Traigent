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

# Resolve svg-term binary (global install or local node_modules install)
SVG_TERM_BIN=""
if command -v svg-term &> /dev/null; then
    SVG_TERM_BIN="$(command -v svg-term)"
elif [[ -x "$SCRIPT_DIR/node_modules/.bin/svg-term" ]]; then
    SVG_TERM_BIN="$SCRIPT_DIR/node_modules/.bin/svg-term"
fi

# Step 2: Convert to SVG (if svg-term available)
if [[ -n "$SVG_TERM_BIN" ]]; then
    echo ""
    echo "Step 2: Converting to SVG..."
    for f in output/*.cast; do
        name=$(basename "$f" .cast)
        "$SVG_TERM_BIN" --in "$f" --out "output/${name}.svg" --window --width 100 --height 30
        echo "  output/${name}.svg"
    done

    echo ""
    echo "Step 3: Generating still thumbnails..."
    python3 scripts/extract-still.py \
      --input output/optimize.svg \
      --output output/optimize-still.svg \
      --top-contains "@traigent.optimize" \
      --require "configuration_space" \
      --require "objectives=[" \
      --require "eval_dataset="

    python3 scripts/extract-still.py \
      --input output/hooks.svg \
      --output output/hooks-still.svg \
      --top-contains "@traigent.optimize" \
      --require "configuration_space" \
      --require "objectives=[" \
      --require "eval_dataset="

    python3 scripts/extract-still.py \
      --input output/github-hooks.svg \
      --output output/github-hooks-still.svg \
      --top-contains "@traigent.optimize" \
      --require "configuration_space" \
      --require "objectives=[" \
      --require "eval_dataset="
else
    echo ""
    echo "Note: svg-term not found. Install with: npm install --prefix docs/demos svg-term-cli"
fi

echo ""
echo "Done! Files in output/"
ls -la output/
