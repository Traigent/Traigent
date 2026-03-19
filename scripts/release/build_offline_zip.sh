#!/usr/bin/env bash
# Build a fully self-contained offline zip for Traigent SDK.
# Usage:
#   ./scripts/release/build_offline_zip.sh              # builds for current Python
#   ./scripts/release/build_offline_zip.sh 3.11 3.12    # builds for specific versions
#
# Output: dist/traigent-<version>-offline-py<pyver>-linux_x86_64.zip
#
# The zip contains:
#   wheels/          — traigent wheel + all dependency wheels
#   install.sh       — one-liner installer (creates venv, installs everything offline)
#   README-INSTALL   — human-readable instructions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERSION=$(python3 -c "
import re
with open('$PROJECT_ROOT/pyproject.toml') as f:
    m = re.search(r'^version\s*=\s*\"(.+?)\"', f.read(), re.M)
    print(m.group(1))
")

# Default: build for current Python only
if [ $# -eq 0 ]; then
    PYTHON_VERSIONS=("$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')")
else
    PYTHON_VERSIONS=("$@")
fi

echo "=== Traigent SDK v${VERSION} — Offline Release Builder ==="
echo "Target Python versions: ${PYTHON_VERSIONS[*]}"
echo "Platform: linux-x86_64"
echo ""

mkdir -p "$PROJECT_ROOT/dist"

for PYVER in "${PYTHON_VERSIONS[@]}"; do
    PYBIN="python${PYVER}"

    # Verify the Python version exists
    if ! command -v "$PYBIN" &>/dev/null; then
        echo "ERROR: $PYBIN not found — skipping"
        continue
    fi

    echo "--- Building for Python ${PYVER} ---"

    WORK_DIR=$(mktemp -d)
    STAGE_DIR="$WORK_DIR/traigent-${VERSION}-py${PYVER}"
    WHEELS_DIR="$STAGE_DIR/wheels"
    mkdir -p "$WHEELS_DIR"

    # 1. Build the traigent wheel
    echo "[1/4] Building traigent wheel..."
    (cd "$PROJECT_ROOT" && "$PYBIN" -m build --wheel --outdir "$WHEELS_DIR" 2>&1 | tail -1)

    # 2. Download all dependency wheels (core + all extras)
    echo "[2/3] Downloading dependency wheels (full bundle)..."
    "$PYBIN" -m pip download \
        --only-binary=:all: \
        --dest "$WHEELS_DIR" \
        "traigent[recommended] @ file://$WHEELS_DIR/$(ls "$WHEELS_DIR"/traigent-*.whl | head -1 | xargs basename)" 2>&1 | tail -5

    # 3. Create install.sh
    echo "[3/3] Creating install script & README..."
    cat > "$STAGE_DIR/install.sh" << 'INSTALL_EOF'
#!/usr/bin/env bash
# Traigent SDK — Offline Installer
# Just run: bash install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/wheels"
VENV_DIR="${1:-.venv}"

echo "=== Traigent SDK Offline Installer ==="

# Find Python
for PY in python3.13 python3.12 python3.11 python3; do
    if command -v "$PY" &>/dev/null; then
        PYTHON="$PY"
        break
    fi
done

if [ -z "${PYTHON:-}" ]; then
    echo "ERROR: Python 3.11+ not found. Please install Python first."
    exit 1
fi

PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using: $PYTHON (Python $PYVER)"

# Check minimum version
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"' || {
    echo "ERROR: Python 3.11+ required"
    exit 1
}

# Create venv
echo "Creating virtual environment in $VENV_DIR..."
"$PYTHON" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip from local wheels only (no network)
pip install --quiet --no-index --find-links="$WHEELS_DIR" --upgrade pip 2>/dev/null || true

# Install everything from local wheels
echo "Installing traigent + dependencies from local wheels..."
pip install --no-index --find-links="$WHEELS_DIR" "traigent[recommended]"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To activate:  source $VENV_DIR/bin/activate"
echo "To verify:    python -c 'import traigent; print(traigent.__version__)'"
echo ""
INSTALL_EOF
    chmod +x "$STAGE_DIR/install.sh"

    # 4. Create README
    cat > "$STAGE_DIR/README-INSTALL.txt" << 'README_EOF'
Traigent SDK — Offline Installation & Quickstart
=================================================

INSTALL (2 minutes)
-------------------
Prerequisites: Python 3.11+ on Linux.

  1. unzip traigent-*.zip
  2. cd traigent-*/
  3. bash install.sh
  4. source .venv/bin/activate

Verify:
  python -c "import traigent; print(traigent.__version__)"


QUICKSTART
----------
Create a file called "my_app.py":

    import traigent
    from traigent import Range, Choices

    @traigent.optimize(
        eval_dataset=[
            {"input": {"query": "What is Python?"}, "expected": "A programming language"},
            {"input": {"query": "What is 2+2?"},    "expected": "4"},
        ],
        model=Choices(["gpt-4o-mini", "gpt-4o"]),
        temperature=Range(0.0, 1.0),
        objectives=["accuracy"],
    )
    def answer(query: str) -> str:
        # Your LLM call goes here
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[{"role": "user", "content": query}],
        )
        return response.choices[0].message.content

    # Dry-run (no LLM calls, validates the pipeline):
    results = answer.optimize(max_trials=3, mock=True)
    print("Best config:", results.best_config)

Run it:
    python my_app.py


ENVIRONMENT VARIABLES
---------------------
Required for real (non-mock) optimization:

  export OPENAI_API_KEY="sk-..."           # or whichever LLM provider
  export TRAIGENT_API_KEY="your-key"       # from https://portal.traigent.ai
  export TRAIGENT_BACKEND_URL="https://portal.traigent.ai"  # (default)

For mock/dry-run mode, no API keys are needed.


WHAT'S INCLUDED
---------------
  wheels/             Pre-built Python wheels (traigent + all dependencies)
  install.sh          Automated installer
  README-INSTALL.txt  This file

Included packages: LangChain, OpenAI, Anthropic, Groq, Google GenAI,
scikit-learn, scipy, numpy, pandas, matplotlib, plotly, MLflow, W&B,
Optuna, PydanticAI, and more (262 packages total).


SUPPORT
-------
  Docs:    https://docs.traigent.ai
  Portal:  https://portal.traigent.ai
  Email:   support@traigent.ai
README_EOF

    # 5. Create the zip
    ZIP_NAME="traigent-${VERSION}-offline-py${PYVER}-linux_x86_64.zip"
    echo "[zip] Creating $ZIP_NAME..."
    (cd "$WORK_DIR" && zip -r "$PROJECT_ROOT/dist/$ZIP_NAME" "traigent-${VERSION}-py${PYVER}" -x '*.pyc' 2>&1 | tail -1)

    # Cleanup
    rm -rf "$WORK_DIR"

    ZIP_SIZE=$(du -h "$PROJECT_ROOT/dist/$ZIP_NAME" | cut -f1)
    echo "    => dist/$ZIP_NAME ($ZIP_SIZE)"
    echo ""
done

echo "=== Done! Release zips are in dist/ ==="
ls -lh "$PROJECT_ROOT/dist/"*.zip 2>/dev/null
