# Google Gemini Operations Guide

Validate Gemini integration behavior against the current plugin and model
discovery surfaces shipped in this repository.

## Environment
```bash
# Required for real runs
export GOOGLE_API_KEY="<key>"   # or GEMINI_API_KEY

# Local mock mode: export TRAIGENT_MOCK_LLM=true, or call
# traigent.testing.enable_mock_mode_for_quickstart() in tutorial code.
# Provider-specific *_MOCK env vars are ignored.
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
