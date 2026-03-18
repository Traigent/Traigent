# Azure OpenAI Operations Guide

Validate Azure OpenAI integration behavior against the current plugin and model
discovery surfaces shipped in this repository.

## Environment
```bash
# Required for real runs
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<key>"
# Optional: API version override (default: 2024-02-15-preview)
# export AZURE_OPENAI_API_VERSION=2024-10-21

# Mock mode (dry-run, no SDK needed)
# export AZURE_OPENAI_MOCK=true

# Note: use `--provider azure` in the experiment scripts below.
```

## Dependencies
```bash
pip install -e ".[integrations]"  # or: pip install -r requirements/requirements-integrations.txt
```

## Validation Checks
```bash
pytest -q tests/unit/integrations/test_cloud_plugins.py -o addopts='' -k azure
pytest -q tests/unit/integrations/test_model_discovery.py -o addopts='' -k AzureOpenAIDiscovery
```

These checks verify that Azure-specific parameter overrides and model-discovery
helpers still match the current code. This repository does not currently ship a
standalone Azure example script.

Note: In Azure OpenAI, `model` is treated as the deployment name.
