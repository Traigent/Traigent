# CI Pipeline Example with DVC + Traigent

This example shows how to integrate Traigent optimization into a
[DVC](https://dvc.org/) (Data Version Control) pipeline for reproducible,
CI-driven LLM tuning.

## Pipeline stages

| Stage | Script | Purpose |
|---|---|---|
| `prepare_dataset` | `scripts/auto_tune/prepare_dataset.py` | Split raw data into train/test |
| `optimize` | `traigent.cli optimize` | Run Traigent optimization loop |
| `evaluate` | `scripts/auto_tune/evaluate_performance.py` | Compare against baseline |
| `report` | `scripts/auto_tune/generate_report.py` | Generate performance report |

## Quick start

```bash
# Install DVC and Traigent
pip install dvc "traigent[recommended]"

# Copy pipeline files to your project root
cp dvc.yaml params.yaml /path/to/your/project/

# Initialize DVC and run
cd /path/to/your/project
dvc init
dvc repro
```

## Configuration

Edit `params.yaml` to control:

- **prepare**: dataset split ratio and random seed
- **optimize**: search strategy (`grid`, `random`, `bayesian`), trial budget, model list
- **evaluate**: improvement threshold and confidence level
- **report**: output format and recommendation count

Environment-specific overrides for `staging` and `production` are at the
bottom of `params.yaml`.

## CI integration

Add `dvc repro` as a step in your CI workflow. DVC caches intermediate
outputs, so only stages with changed dependencies re-run.

```yaml
# .github/workflows/optimize.yml
- name: Run optimization pipeline
  run: dvc repro
  env:
    TRAIGENT_API_KEY: ${{ secrets.TRAIGENT_API_KEY }}
```
