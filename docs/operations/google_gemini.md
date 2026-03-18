# Google Gemini Operations Guide

Validate Gemini integration behavior against the current plugin and model
discovery surfaces shipped in this repository.

## Environment
```bash
# Required for real runs
export GOOGLE_API_KEY="<key>"   # or GEMINI_API_KEY

# Mock mode (dry-run, no SDK needed)
# export GEMINI_MOCK=true
```

## Dependencies
```bash
pip install -e ".[integrations]"  # or: pip install -r requirements/requirements-integrations.txt
```

## Validation Checks
```bash
pytest -q tests/unit/integrations/test_cloud_plugins.py -o addopts='' -k gemini
pytest -q tests/unit/integrations/test_model_discovery.py -o addopts='' -k GeminiDiscovery
```

These checks cover provider-specific parameter mappings and Gemini model
discovery. This repository does not currently ship a standalone Gemini example
script.
