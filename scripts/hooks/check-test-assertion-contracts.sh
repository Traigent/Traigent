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

EXISTING_TARGETS=()
for target in "${TARGETS[@]}"; do
  if [ -d "$target" ]; then
    EXISTING_TARGETS+=("$target")
  fi
done

if [ "${#EXISTING_TARGETS[@]}" -eq 0 ]; then
  exit 0
fi

PATTERNS=(
  '^[[:space:]]*assert .*\bor[[:space:]]+True\b'
  '^[[:space:]]*assert .*is None[[:space:]]+or[[:space:]].*is not None'
  '^[[:space:]]*assert .*is not None[[:space:]]+or[[:space:]].*is None'
  'implementation[- ]dependent'
)

VIOLATIONS=()
for pattern in "${PATTERNS[@]}"; do
  if command -v rg >/dev/null 2>&1; then
    MATCHES="$(rg -n -i --glob '*.py' "$pattern" "${EXISTING_TARGETS[@]}" || true)"
  else
    MATCHES="$(grep -R -n -i -E "$pattern" "${EXISTING_TARGETS[@]}" --include='*.py' || true)"
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
