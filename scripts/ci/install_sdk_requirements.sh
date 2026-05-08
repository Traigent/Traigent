#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

schema_requirement="${TRAIGENT_SCHEMA_REQUIREMENT:-traigent-schema>=4.1.0,<5.0.0}"
install_schema="${TRAIGENT_INSTALL_SCHEMA:-}"

for arg in "$@"; do
  case "$arg" in
    *internal_schema*) install_schema="1" ;;
  esac
done

if [ "$install_schema" = "1" ]; then
  python -m pip install "$schema_requirement"
fi

if [ "$#" -gt 0 ]; then
  python -m pip install "$@"
fi
