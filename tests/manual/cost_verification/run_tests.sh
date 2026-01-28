#!/bin/bash
# Helper script to run cost verification tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading $PROJECT_ROOT/.env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Also try the real examples .env
if [ -f "$PROJECT_ROOT/walkthrough/examples/real/.env" ]; then
    echo "Loading $PROJECT_ROOT/walkthrough/examples/real/.env"
    set -a
    source "$PROJECT_ROOT/walkthrough/examples/real/.env"
    set +a
fi

# Check for required environment variables
echo ""
echo "Environment check:"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:+set (${#OPENAI_API_KEY} chars)}"
echo "  ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:+set (${#ANTHROPIC_API_KEY} chars)}"
echo "  TRAIGENT_API_KEY: ${TRAIGENT_API_KEY:+set (${#TRAIGENT_API_KEY} chars)}"
echo "  TRAIGENT_BACKEND_URL: ${TRAIGENT_BACKEND_URL:-http://localhost:5000}"
echo "  LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:+set}"
echo ""

# Change to project root for imports
cd "$PROJECT_ROOT"

# Run the verification script
python3 -m tests.manual.cost_verification.verify_cost_tracking "$@"
