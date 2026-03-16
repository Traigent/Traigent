# AWS Bedrock Operations Guide

This guide outlines the current Bedrock validation surfaces that ship in this
repository.

## Prerequisites
- Bedrock enabled in your AWS account and region (e.g., `us-east-1`).
- Model access approved for the Claude family you intend to use (e.g., Sonnet/Haiku).
- IAM user/role with permissions:
  - `bedrock:InvokeModel`
  - `bedrock:InvokeModelWithResponseStream`
  - Optional: `sts:AssumeRole` (CI usage)

## Local Setup
```bash
# Choose a supported region
export AWS_REGION=us-east-1

# Use either a profile...
export AWS_PROFILE=traigent-bedrock
# ...or explicit keys (avoid in shared shells)
# export AWS_ACCESS_KEY_ID=...
# export AWS_SECRET_ACCESS_KEY=...

# Optional explicit model override
# export BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Optional: Mock Bedrock to dry-run without AWS credentials
# export BEDROCK_MOCK=true
```

## Dependency Installation
Ensure you installed integration dependencies:
```bash
pip install -e ".[integrations]"  # or: pip install -r requirements/requirements-integrations.txt
```

## Validation Checks
```bash
pytest -q tests/unit/integrations/test_cloud_plugins.py -o addopts='' -k bedrock
pytest -q tests/unit/integrations/test_bedrock_client.py -o addopts=''
```

## Minimal Real Run

With AWS credentials configured, you can exercise the shipped Bedrock example:

```bash
python examples/integrations/bedrock/bedrock_hello.py
```

If the Bedrock path is selected, the pipeline will route through `traigent.integrations.bedrock_client.BedrockChatClient`.

## Troubleshooting
- `AccessDeniedException`: Verify model access is granted in Bedrock console and IAM policy includes `InvokeModel*`.
- `RegionError`: Confirm `AWS_REGION` is a Bedrock-enabled region and matches your model access region.
- `ImportError: boto3`: Run `pip install boto3 botocore` or install the `integrations` extra.

## Notes
- Costs: Start with small max tokens and low trial counts. Use AWS Budgets/Cost Anomaly detection alerts.
- Reproducibility: Use `RandomSamplerPlan` to fix evaluation subsets across runs.
