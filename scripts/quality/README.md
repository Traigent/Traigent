# Code Quality Scripts

This directory contains scripts for maintaining code quality standards across the TraiGent SDK.

## Scripts

### check_code_quality.py

Performs various code quality checks:
- **Model name typos**: Ensures no typos like "o4-mini" instead of "gpt-4o-mini"
- **Debug print statements**: Verifies no debug prints in production code
- **Hardcoded API keys**: Checks for accidentally committed API keys
- **Logging imports**: Ensures proper logging setup in files that use logging
- **Integration detection**: Validates that integration detection works properly

## Usage

Run the code quality checks:
```bash
python scripts/quality/check_code_quality.py
```

The script will exit with code 0 if all checks pass, or 1 if any checks fail.

## Integration with CI/CD

These checks should be run as part of the continuous integration pipeline to ensure code quality standards are maintained.
