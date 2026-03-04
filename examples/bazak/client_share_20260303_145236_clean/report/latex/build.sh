#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error CLIENT_VALIDATION_REPORT.tex >/tmp/bazak_latex_build.log
pdflatex -interaction=nonstopmode -halt-on-error CLIENT_VALIDATION_REPORT.tex >>/tmp/bazak_latex_build.log
echo "Built: $(pwd)/CLIENT_VALIDATION_REPORT.pdf"
