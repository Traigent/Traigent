## HotpotQA Walkthrough — What This Demo Shows

This walkthrough lets you experience how Traigent optimizes a multi‑hop question‑answering agent on the HotpotQA distractor setting. You will see Traigent explore different agent configurations (models, prompting, retrieval depth, etc.), automatically score answers, and surface the best trade‑offs between quality, cost, and latency.

### What you’ll see

- Multi‑hop QA behavior on a small HotpotQA slice (two‑step reasoning over multiple paragraphs).
- Side‑by‑side runs of different configurations: model choice, temperature, max tokens, and retrieval depth (k).
- Automatic scoring with Answer Exact‑Match (EM) and F1, plus cost and latency accounting per run.
- A concise summary of best‑performing configurations and the Pareto frontier (quality vs. cost/latency).
- Optional single‑question traces to inspect how retrieval depth and prompt style affect answers.

### Why HotpotQA?

HotpotQA requires combining evidence from multiple passages to answer a single question. That makes it perfect for demonstrating how Traigent tunes retrieval and reasoning “knobs” and measures the impact on outcomes.

## How the demo works

We use the HotpotQA “distractor” format (each question ships with 10 paragraphs: 2 relevant + 8 distractors). Traigent tests variations such as:

- Retrieval depth `k` (how many paragraphs to include).
- Prompt style (direct answer vs. step‑by‑step reasoning).
- Model family and size (e.g., gpt‑4o‑mini, gpt‑4o, gpt‑3.5‑turbo).
- Temperature and max tokens.
- Optional reranking on/off.

Objectives and reports include:

- Quality: Answer EM and F1.
- Cost: per‑call USD estimates from token usage.
- Latency: p95 or average end‑to‑end time.
- Optional safety/response‑length checks when enabled.

## Run the demo

You can explore HotpotQA optimization in two ways.

### Quick setup script

Run `./install.sh` to provision a dedicated virtual environment, install Traigent with all walkthrough dependencies, and generate a sample HotpotQA dataset if needed. Re-run the script any time; it skips completed steps (use `FORCE_REINSTALL=1` to force dependency reinstalls).

#### What the script does
- Prompts you to **use your currently activated virtual environment** (make sure it’s active before answering “yes”) or to let the script create a dedicated `.venv` inside this folder.
- Installs Traigent plus the integration requirements into the selected environment (skipped if already present unless `FORCE_REINSTALL=1`). If you’d rather install manually, run `pip install -r walkthrough/hotpotQA/requirements.txt` followed by `pip install -e .` inside the environment of your choice.
- Generates a lightweight HotpotQA sample dataset the first time it runs. We default to `validation[:50]`, i.e., the first 50 examples from the official validation split, because it is quick to download (via Hugging Face `datasets`) and still showcases multi-hop behavior. Adjust `HOTPOT_SAMPLE_SLICE` for larger samples or replace the file with the official distractor dev set when you need full fidelity.
- No large Wikipedia dump is required. The HotpotQA distractor format already includes the supporting paragraphs inside the dataset, so the walkthrough operates entirely on local files—models do not reach out to the live web unless you modify the pipeline to do so.

### Run the optimization demo

```bash
./run_demo.sh
```

The script wraps `run_demo.py`, which optimizes a compact mock-mode HotpotQA agent with high-impact variables (`model`, `temperature`, `retriever_k`, `prompt_style`, `retrieval_reranker`, `max_output_tokens`). Optuna-backed search is enabled by default, and the mock simulator makes it fast to inspect results without real API keys.

#### Run against live models

1. Export your API keys (only the providers you plan to target are required):
   ```bash
   export OPENAI_API_KEY="sk-..."      # required for gpt-4o / gpt-4o-mini
   export ANTHROPIC_API_KEY="sk-ant-..."  # required for haiku-3.5
   ```
2. Disable mock mode (`true` by default in the demo) and launch with the full command:
   ```bash
   TRAIGENT_MOCK_MODE=false ./run_demo.sh
   ```

The real-mode pipeline streams the retrieved context into OpenAI (`gpt-4o`, `gpt-4o-mini`) and Anthropic (`haiku-3.5`) models, so expect live API usage, latency, and billing. Leave `TRAIGENT_MOCK_MODE=true` (default) to keep runs local and deterministic.

The demo simultaneously tunes several high-impact variables:
- `model` — compare OpenAI `gpt-4o` / `gpt-4o-mini` vs. Anthropic `haiku-3.5`.
- `temperature` — adjust generation determinism (0.1, 0.3, 0.7).
- `retriever_k` — number of paragraphs pulled from the HotpotQA context (3, 5, 8).
- `prompt_style` — direct answer (`vanilla`) vs. chain-of-thought reasoning (`cot`).
- `retrieval_reranker` — optional reranking pass (`none`, `mono_t5`).
- `max_output_tokens` — cap on answer length (256 or 384 tokens).

### Option A — Interactive walkthrough (guided)

Use the interactive launcher and follow the prompts. The RAG chapter mirrors the HotpotQA flow and highlights the same optimization levers.

```bash
./launch_walkthrough.sh
```

### Option B — Case‑study CLI (scripted)

Run the dedicated HotpotQA scenario via the paper experiments CLI. Start in mock mode (no API keys required):

```bash
TRAIGENT_MOCK_MODE=true \
python paper_experiments/cli.py optimize \
  --scenario hotpotqa \
  --algorithm optuna_nsga2 \
  --trials 20 \
  --parallel-trials 4 \
  --export-trials trial_history.json \
  --export-figures
```

To use real models, set your API keys and turn off mock mode:

```bash
export OPENAI_API_KEY="sk-..."
# export ANTHROPIC_API_KEY="sk-ant-..."   # if you enable Anthropic models

python paper_experiments/cli.py optimize \
  --scenario hotpotqa \
  --algorithm optuna_nsga2 \
  --trials 60 \
  --parallel-trials 6 \
  --mock-mode off \
  --export-trials trial_history.json \
  --export-figures
```

## What to look for in the results

Artifacts are written under `paper_experiments/artifacts/hotpotqa/<run-id>/`:

- `baseline_manual.json` — metrics for a hand‑crafted baseline configuration.
- `tvo_summary.json` — optimizer output with best configurations and Pareto frontier.
- `trial_history.json` — per‑trial metrics (used for figure export).
- `figure_data/` — optional data for charts (hypervolume over time, feasible vs. pruned trials).

When reading the tables, compare how increasing `k` or enabling step‑by‑step prompting changes EM/F1, cost, and latency. Traigent’s optimizer will highlight configurations that improve quality without unnecessary cost or delay.

## Customizing the demo

- Dataset: A small demo slice ships at `paper_experiments/case_study_rag/datasets/hotpotqa_dev_subset.jsonl`. Use `data_ingest.py` to create a larger dev slice.
- Knobs: Modify the configuration space in `paper_experiments/case_study_rag/pipeline.py` to add/remove models, k‑values, or prompting styles.
- Constraints: Toggle structure‑aware pruning with `--constraints on|off` to see how it shapes the search.

---

This demonstration is meant to make the optimization effects tangible: tweak retrieval and reasoning levers, see how answers improve, and understand the cost/latency trade‑offs—without changing your agent code.
