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

# Local mock mode: export TRAIGENT_MOCK_LLM=true, or call
# traigent.testing.enable_mock_mode_for_quickstart() in tutorial code.
# Provider-specific *_MOCK env vars are ignored.
```

## Credential Boundary

Bedrock credentials are AWS/customer credentials. The SDK's optional Bedrock
client creates a local `boto3`/`aioboto3` Bedrock Runtime client and relies on
the AWS SDK credential chain on the machine where the optimization runs. The
SDK does not use a Traigent API key as an AWS credential and does not send AWS
credentials to the Traigent backend.

If you use `AWS_BEARER_TOKEN_BEDROCK`, treat it as a short-lived AWS/provider
credential with the same boundary. It belongs in your local AWS credential
environment or tooling, not in Traigent backend configuration.

For the broader credential and telemetry boundary, see the
[Credential & data trust model](../security/trust_model.md).

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

Bedrock coverage includes the native `BedrockChatClient` plus LangChain
`ChatBedrock` and `ChatBedrockConverse` interception. In mock mode,
`TRAIGENT_MOCK_LLM` routes Bedrock calls through `MockAdapter`; in real mode,
responses are captured for token/cost/latency extraction.

## Troubleshooting
- `AccessDeniedException`: Verify model access is granted in Bedrock console and IAM policy includes `InvokeModel*`.
- `RegionError`: Confirm `AWS_REGION` is a Bedrock-enabled region and matches your model access region.
- `ImportError: boto3`: Run `pip install boto3 botocore` or install the `integrations` extra.

## Notes
- Costs: Start with small max tokens and low trial counts. Use AWS Budgets/Cost Anomaly detection alerts.
- Reproducibility: Use `RandomSamplerPlan` to fix evaluation subsets across runs.
