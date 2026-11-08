#!/usr/bin/env bash
# Guardrail: public-surface tests must assert contracts, not tautologies.
set -euo pipefail

TARGETS=(
  "tests/integration"
  "tests/unit/api"
  "tests/unit/integrations"
  "tests/unit/optimizers"
  "tests/unit/utils"
)

MISSING=()
for target in "${TARGETS[@]}"; do
  if [ ! -d "$target" ]; then
    MISSING+=("$target")
  fi
done

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "ERROR: target directory not found: ${MISSING[*]}" >&2
  echo "Run from the repository root, or update TARGETS in this guard if a directory was renamed." >&2
  exit 1
fi

EXISTING_TARGETS=("${TARGETS[@]}")

PATTERNS=(
  '^[[:space:]]*assert .*\bor[[:space:]]+True\b'
  '^[[:space:]]*assert .*is None[[:space:]]+or[[:space:]].*is not None'
  '^[[:space:]]*assert .*is not None[[:space:]]+or[[:space:]].*is None'
  'implementation[- ]dependent'
)

VIOLATIONS=()
for pattern in "${PATTERNS[@]}"; do
  rc=0
  # Exit 0 = matches, 1 = no match; anything else is a search error and must fail.
  if command -v rg >/dev/null 2>&1; then
    MATCHES="$(rg -n -i --glob '*.py' "$pattern" "${EXISTING_TARGETS[@]}")" || rc=$?
  else
    MATCHES="$(grep -R -n -i -E "$pattern" "${EXISTING_TARGETS[@]}" --include='*.py')" || rc=$?
  fi
  if [ "$rc" -gt 1 ]; then
    echo "ERROR: search for pattern '$pattern' failed (exit $rc)" >&2
    exit 1
  fi
  while IFS= read -r match; do
    [ -z "$match" ] && continue
    VIOLATIONS+=("$match")
  done <<< "$MATCHES"
done

if [ "${#VIOLATIONS[@]}" -gt 0 ]; then
  echo "ERROR: Public-surface tests contain tautological or implementation-dependent assertions."
  echo "Replace them with documented contract checks, or move the test out of public contract coverage."
  printf '%s\n' "${VIOLATIONS[@]}"
  exit 1
fi

exit 0
