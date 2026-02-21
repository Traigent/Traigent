#!/usr/bin/env bash
# Guardrail: prevent strict=False in runtime post-call cost accounting paths.
set -euo pipefail

PATTERN='cost_from_tokens\([^)]*strict=False'
TARGET_DIR="traigent"
ALLOWED_FILE="traigent/utils/cost_calculator.py"
ALLOWED_FUNCTION="get_model_pricing_per_1k"

if command -v rg >/dev/null 2>&1; then
  MATCHES="$(rg -n "$PATTERN" "$TARGET_DIR" --glob '*.py' || true)"
else
  MATCHES="$(grep -R -n -E "$PATTERN" "$TARGET_DIR" --include='*.py' || true)"
fi

if [ -z "$MATCHES" ]; then
  exit 0
fi

BOUNDS="$(
  awk -v fn="$ALLOWED_FUNCTION" '
    $0 ~ ("^def " fn "\\(") { start = NR }
    start && /^def / && NR > start { end = NR - 1; print start ":" end; exit }
    END {
      if (start && !end) {
        print start ":" NR
      }
    }
  ' "$ALLOWED_FILE"
)"

if [ -z "$BOUNDS" ]; then
  echo "ERROR: could not resolve allowed function bounds in $ALLOWED_FILE"
  exit 1
fi

ALLOWED_START="${BOUNDS%%:*}"
ALLOWED_END="${BOUNDS##*:}"
VIOLATIONS=()

while IFS= read -r match; do
  [ -z "$match" ] && continue

  file="${match%%:*}"
  remainder="${match#*:}"
  line_no="${remainder%%:*}"

  if [ "$file" = "$ALLOWED_FILE" ] && [ "$line_no" -ge "$ALLOWED_START" ] && [ "$line_no" -le "$ALLOWED_END" ]; then
    continue
  fi

  VIOLATIONS+=("$match")
done <<< "$MATCHES"

if [ "${#VIOLATIONS[@]}" -gt 0 ]; then
  echo "ERROR: Disallowed cost_from_tokens(..., strict=False) usage detected."
  echo "Allowed location: $ALLOWED_FILE::$ALLOWED_FUNCTION only."
  printf '%s\n' "${VIOLATIONS[@]}"
  exit 1
fi

exit 0
