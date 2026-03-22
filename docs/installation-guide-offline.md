# Traigent SDK v0.10.0 — Offline Installation Guide

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Linux x86_64 |
| **Python** | 3.11, 3.12, or 3.13 |
| **Disk** | ~1 GB free space |
| **Permissions** | Write access to the installation directory |

No internet connection is required. Everything is bundled in the zip.

---

## Step 1: Extract

```bash
unzip traigent-0.10.0-offline-py3.12-linux_x86_64.zip
cd traigent-0.10.0-py3.12/
```

## Step 2: Install

```bash
bash install.sh
```

This creates a Python virtual environment (`.venv/`) and installs Traigent with all dependencies from the bundled wheels. No network calls are made.

To install to a custom location:

```bash
bash install.sh /opt/traigent/venv
```

## Step 3: Activate

```bash
source .venv/bin/activate
```

## Step 4: Verify

```bash
python -c "import traigent; print(traigent.__version__)"
```

Expected output:
```
traigent v0.10.0
```

---

## Environment Variables

Set these before running optimizations:

```bash
# LLM provider key (at least one required for real optimization)
export OPENAI_API_KEY="sk-..." # pragma: allowlist secret
export ANTHROPIC_API_KEY="sk-ant-..." # pragma: allowlist secret

# Traigent platform (optional — for tracking results in the portal)
export TRAIGENT_API_KEY="your-api-key" # pragma: allowlist secret
export TRAIGENT_BACKEND_URL="https://portal.traigent.ai"
```

For **mock/dry-run mode** (no LLM calls, no API keys needed), no environment variables are required.

---

## Quick Test — Mock Mode

Create `test_traigent.py`:

```python
import os

os.environ["TRAIGENT_MOCK_LLM"] = "true"

import traigent
from traigent import Range, Choices

@traigent.optimize(
    eval_dataset=[
        {"input": {"query": "What is Python?"}, "expected": "A programming language"},
        {"input": {"query": "What is 2+2?"},    "expected": "4"},
        {"input": {"query": "Capital of France?"}, "expected": "Paris"},
    ],
    model=Choices(["gpt-4o-mini", "gpt-4o"]),
    temperature=Range(0.0, 1.0),
    objectives=["accuracy"],
)
def answer(query: str) -> str:
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content

# Mock mode — validates the full pipeline without making LLM calls
results = answer.optimize(max_trials=3)
print("Best config:", results.best_config)
```

Run it:

```bash
python test_traigent.py
```

This runs 3 mock trials to validate the pipeline works end-to-end without any API keys or network.

---

## Real Optimization

Once the mock test passes, switch to real mode:

```python
# Set your API keys first, then:
results = answer.optimize(
    max_trials=10,
    algorithm="bayesian",   # or "grid", "random", "optuna"
)

print("Best config:", results.best_config)
print("Best score:", results.best_score)

# Apply the best configuration
answer.apply_best_config()
```

---

## What's Included

The zip bundles 262 Python packages including:

| Category | Packages |
|----------|----------|
| **LLM Providers** | OpenAI, Anthropic, Groq, Google GenAI |
| **Frameworks** | LangChain, LangGraph, PydanticAI |
| **Optimization** | Optuna, scikit-learn, scipy, Bayesian optimization |
| **Data & Analytics** | NumPy, pandas, matplotlib, plotly |
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Cloud** | boto3, httpx, aiohttp |

---

## Troubleshooting

### `Python 3.11+ required`
The zip was built for a specific Python version (check the filename). Ensure the matching Python version is installed:
```bash
python3 --version
```

### `No module named 'traigent'`
The virtual environment is not activated:
```bash
source .venv/bin/activate
```

### `externally-managed-environment` error
You're trying to install system-wide. The installer creates a virtual environment to avoid this — just use `bash install.sh`.

### Wheel compatibility errors
The zip contains Linux x86_64 wheels. If you're on a different architecture (ARM, etc.), contact us for a compatible build.

---

## Support

| | |
|---|---|
| **Portal** | https://portal.traigent.ai |
| **Documentation** | https://docs.traigent.ai |
| **Email** | support@traigent.ai |
