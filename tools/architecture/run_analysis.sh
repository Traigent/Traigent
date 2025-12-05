#!/bin/bash
# TraiGent Architecture Analysis Runner
#
# Runs the complete architecture analysis pipeline:
# 1. Main architecture analysis (diagrams, complexity, class hierarchy)
# 2. Call graph analysis
# 3. Priority issue identification
# 4. Focused diagram generation
# 5. SVG rendering
# 6. Threshold checking
# 7. Baseline comparison
#
# Usage:
#   ./tools/architecture/run_analysis.sh [options]
#
# Options:
#   --quick         Skip SVG rendering and call graph
#   --with-coverage Run with test coverage (slower)
#   --update-baseline Update baseline after analysis
#   --verbose       Show detailed output
#   --help          Show this help

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$SCRIPT_DIR/output"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# Default options
QUICK=false
WITH_COVERAGE=false
UPDATE_BASELINE=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --with-coverage)
            WITH_COVERAGE=true
            shift
            ;;
        --update-baseline)
            UPDATE_BASELINE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            head -25 "$0" | tail -22
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use system Python if venv not available
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

echo "🏗️  TraiGent Architecture Analysis"
echo "=================================="
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/svg"
mkdir -p "$OUTPUT_DIR/focused"

# Step 1: Main architecture analysis
echo "📊 Step 1/7: Running main architecture analysis..."
if $VERBOSE; then
    $VENV_PYTHON "$SCRIPT_DIR/generate_diagrams.py" --all-features --format all --verbose
else
    $VENV_PYTHON "$SCRIPT_DIR/generate_diagrams.py" --all-features --format all
fi

# Step 2: Call graph analysis (skip if quick mode)
if ! $QUICK; then
    echo ""
    echo "🔗 Step 2/7: Generating call graph..."
    if $VERBOSE; then
        $VENV_PYTHON "$SCRIPT_DIR/generate_call_graph.py" --verbose
    else
        $VENV_PYTHON "$SCRIPT_DIR/generate_call_graph.py"
    fi
else
    echo ""
    echo "⏭️  Step 2/7: Skipping call graph (quick mode)"
fi

# Step 3: Priority analysis
echo ""
echo "🎯 Step 3/7: Identifying priority issues..."
$VENV_PYTHON "$SCRIPT_DIR/analyze_priorities.py"

# Step 4: Focused diagrams
echo ""
echo "🔍 Step 4/7: Generating focused diagrams..."
$VENV_PYTHON "$SCRIPT_DIR/generate_focused_diagrams.py" --output-dir "$OUTPUT_DIR/focused"

# Step 5: SVG rendering (skip if quick mode)
if ! $QUICK; then
    echo ""
    echo "🖼️  Step 5/7: Rendering SVG diagrams..."

    # Render main DOT files
    for dotfile in "$OUTPUT_DIR"/*.dot; do
        if [ -f "$dotfile" ]; then
            name=$(basename "$dotfile" .dot)
            dot -Tsvg "$dotfile" -o "$OUTPUT_DIR/svg/${name}.svg" 2>/dev/null || true
            if $VERBOSE; then
                echo "   ✅ ${name}.svg"
            fi
        fi
    done

    # Render focused DOT files
    for dotfile in "$OUTPUT_DIR/focused"/*.dot; do
        if [ -f "$dotfile" ]; then
            name=$(basename "$dotfile" .dot)
            dot -Tsvg "$dotfile" -o "$OUTPUT_DIR/svg/${name}.svg" 2>/dev/null || true
            if $VERBOSE; then
                echo "   ✅ ${name}.svg"
            fi
        fi
    done

    svg_count=$(ls -1 "$OUTPUT_DIR/svg"/*.svg 2>/dev/null | wc -l || echo "0")
    echo "   Generated $svg_count SVG files"
else
    echo ""
    echo "⏭️  Step 5/7: Skipping SVG rendering (quick mode)"
fi

# Step 6: Threshold checking
echo ""
echo "🔒 Step 6/7: Checking architecture thresholds..."
$VENV_PYTHON "$SCRIPT_DIR/check_thresholds.py" \
    --max-complexity 50 \
    --max-lines 2500 \
    --output "$OUTPUT_DIR/threshold_report.md"

# Step 7: Baseline comparison
echo ""
echo "📈 Step 7/7: Comparing against baseline..."
$VENV_PYTHON "$SCRIPT_DIR/compare_baseline.py" \
    --output "$OUTPUT_DIR/drift_report.md"

# Optional: Update baseline
if $UPDATE_BASELINE; then
    echo ""
    echo "📌 Updating baseline metrics..."
    cp "$OUTPUT_DIR/architecture_data.json" "$SCRIPT_DIR/baseline_metrics.json"
    echo "   Baseline updated"
fi

# Optional: Coverage analysis
if $WITH_COVERAGE; then
    echo ""
    echo "🧪 Running coverage analysis (this may take a while)..."
    "${PROJECT_ROOT}/.venv/bin/pytest" tests/unit/ \
        --cov=traigent \
        --cov-report=json:"$OUTPUT_DIR/coverage.json" \
        -q --tb=no || true

    $VENV_PYTHON "$SCRIPT_DIR/generate_diagrams.py" \
        --coverage "$OUTPUT_DIR/coverage.json" \
        --all-features
fi

# Summary
echo ""
echo "=================================="
echo "✨ Analysis Complete!"
echo ""
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "Key reports:"
echo "   - architecture_report.md    Full architecture report"
echo "   - priority_issues.md        High-priority issues"
echo "   - hot_paths.md              Most-called functions"
echo "   - drift_report.md           Changes from baseline"
echo "   - threshold_report.md       Threshold violations"
echo ""
if ! $QUICK; then
    echo "Diagrams (SVG):"
    echo "   - svg/complexity_heatmap.svg"
    echo "   - svg/class_hierarchy.svg"
    echo "   - svg/call_graph_focused.svg"
    echo "   - svg/top_classes.svg"
fi
