#!/usr/bin/env bash
# Pre-commit hook: prevent staging gitignored files
set -e

IGNORED=$(git diff --cached --name-only --diff-filter=ACM | git check-ignore --stdin 2>/dev/null || true)

if [ -n "$IGNORED" ]; then
    echo "ERROR: The following staged files are gitignored:"
    echo "$IGNORED"
    echo ""
    echo "Remove them with: git reset HEAD <file>"
    exit 1
fi
