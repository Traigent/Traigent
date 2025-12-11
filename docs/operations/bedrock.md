#+ AWS Bedrock Operations Guide (Experiments)

This guide outlines a minimal setup to run paper experiments against AWS Bedrock without deploying any server components.

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

# Tell pipelines to use Bedrock
export TRAIGENT_LLM_PROVIDER=bedrock
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

## Smoke Tests

Run the KILT case study with Bedrock:
```bash
# Ensure a single trial and a tiny dataset subset if needed to keep costs low
python paper_experiments/case_study_kilt/run_case_study.py \
  --real-mode on --provider bedrock \
  --model claude-3-sonnet \
  --temperature 0.3 \
  --trials 1 --parallel-trials 1

Run the FEVER case study with Bedrock:
```bash
python paper_experiments/case_study_fever/run_case_study.py \
  --mock-mode off --provider bedrock \
  --trials 1 --parallel-trials 1
```

Run the Spider case study with Bedrock:
```bash
python paper_experiments/case_study_spider/run_case_study.py \
  --mock-mode off --provider bedrock \
  --trials 1 --parallel-trials 1
```
```

If the Bedrock path is selected, the pipeline will route through `traigent.integrations.bedrock_client.BedrockChatClient`.

## Troubleshooting
- `AccessDeniedException`: Verify model access is granted in Bedrock console and IAM policy includes `InvokeModel*`.
- `RegionError`: Confirm `AWS_REGION` is a Bedrock-enabled region and matches your model access region.
- `ImportError: boto3`: Run `pip install boto3 botocore` or install `requirements-integrations.txt` bundle.

## Notes
- Costs: Start with small max tokens and low trial counts. Use AWS Budgets/Cost Anomaly detection alerts.
- Reproducibility: Use `RandomSamplerPlan` to fix evaluation subsets across runs.
