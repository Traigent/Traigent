# Code Quality Scripts

This directory contains helper scripts for generating repo-local quality reports.

## Scripts

### `generate_quality_report.py`

Generates a Markdown quality report under `reports/code-quality/` using the tooling
available in the current Python environment. The report currently summarizes:

- `flake8`
- `ruff`
- `mypy`

## Usage

Run the report generator from the repository root:

```bash
python scripts/quality/generate_quality_report.py
```

The script uses the active interpreter, so prefer running it from the checked-in
`.venv` or another environment with the Traigent dev dependencies installed.
