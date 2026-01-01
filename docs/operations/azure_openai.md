# Azure OpenAI Operations Guide (Experiments)

Run paper experiments against Azure OpenAI using a small adapter with mock support.

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

## Smoke Tests
```bash
python paper_experiments/case_study_kilt/run_case_study.py \
  --real-mode on --provider azure \
  --model gpt-4o-mini \
  --temperature 0.3 --trials 1 --parallel-trials 1

python paper_experiments/case_study_fever/run_case_study.py \
  --mock-mode off --provider azure \
  --trials 1 --parallel-trials 1

python paper_experiments/case_study_spider/run_case_study.py \
  --mock-mode off --provider azure \
  --trials 1 --parallel-trials 1
```

Note: In Azure, "model" is treated as the deployment name.
