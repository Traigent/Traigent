#+ Google Gemini Operations Guide (Experiments)

Run paper experiments against Google AI Studio (Gemini) via a small adapter with mock mode.

## Environment
```bash
# Provider switch
export TRAIGENT_LLM_PROVIDER=google

# Required for real runs
export GOOGLE_API_KEY="<key>"

# Mock mode (dry-run, no SDK needed)
# export GEMINI_MOCK=true
```

## Smoke Tests
```bash
python paper_experiments/case_study_kilt/run_case_study.py \
  --real-mode on --provider google \
  --model gemini-1.5-flash \
  --temperature 0.3 --trials 1 --parallel-trials 1

python paper_experiments/case_study_fever/run_case_study.py \
  --mock-mode off --provider google \
  --trials 1 --parallel-trials 1

python paper_experiments/case_study_spider/run_case_study.py \
  --mock-mode off --provider google \
  --trials 1 --parallel-trials 1
```

