# HotpotQA Multi-Hop QA Optimization Demo

This walkthrough demonstrates how Traigent optimizes a multi-hop question-answering agent using the HotpotQA benchmark. You'll see how different configurations (model, temperature, retrieval depth, prompt style) affect quality, cost, and latency.

## What You'll Learn

- **Multi-objective optimization**: Balance quality, cost, and latency simultaneously
- **RAG parameter tuning**: Optimize retrieval depth (`k`) and reranking strategies
- **Prompt style optimization**: Compare direct answers vs chain-of-thought reasoning
- **Mock mode**: Fast iteration without API costs

## Quick Start

### Mock Mode (No API Keys Required)

```bash
# From the repository root:
python walkthrough/hotpotQA/run_demo.py
```

Mock mode is the default—no flags needed.

### Real Mode (With LLM API Calls)

```bash
# For OpenAI models (gpt-4o, gpt-4o-mini):
export OPENAI_API_KEY="sk-..."

# For Anthropic models (haiku-3.5):
export ANTHROPIC_API_KEY="sk-ant-..."

python walkthrough/hotpotQA/run_demo.py --real
```

## What the Demo Optimizes

The demo explores these configuration parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `model` | gpt-4o, gpt-4o-mini, haiku-3.5 | LLM to use |
| `temperature` | 0.1, 0.3, 0.7 | Generation randomness |
| `retriever_k` | 3, 5, 8 | Number of context passages to retrieve |
| `prompt_style` | vanilla, cot | Direct answer vs chain-of-thought |
| `retrieval_reranker` | none, mono_t5 | Re-score retrieved passages for relevance |
| `max_output_tokens` | 256, 384 | Maximum response length |

**Objectives optimized:**
- `quality` - Answer correctness (Exact Match + F1 token overlap, weighted)
- `latency_p95_ms` - 95th percentile response time
- `cost_usd_per_1k` - Cost per 1000 tokens

## Sample Output (Mock Mode)

```
============================================================
  Traigent HotpotQA Multi-Hop QA Optimization Demo
============================================================
  Mode: MOCK
  (No API keys required - using simulated responses)

Optimization Configuration:
  Objectives: quality, latency_p95_ms, cost_usd_per_1k
  Total combinations: 216 (3 model x 3 temperature x 3 retriever_k x 2 prompt_style x 2 retrieval_reranker x 2 max_output_tokens)
  Max trials: 12
  Parallel trials: 4

Configuration Space:
  - model: ['gpt-4o', 'gpt-4o-mini', 'haiku-3.5']
  - temperature: [0.1, 0.3, 0.7]
  - retriever_k: [3, 5, 8]
  - prompt_style: ['vanilla', 'cot']
  - retrieval_reranker: ['none', 'mono_t5']
  - max_output_tokens: [256, 384]

Starting optimization...

================================================================================
  Trial Results (MOCK - 12 trials)
================================================================================
  # | Model        | Temp |  K | Style   |  Quality |  Latency |       Cost
--------------------------------------------------------------------------------
> 1 | gpt-4o-mini  |  0.1 |  5 | vanilla |   72.1%* |  2750ms  | $0.00023*
  2 | haiku-3.5    |  0.3 |  3 | vanilla |   62.3%  |  2250ms* | $0.00033
  3 | gpt-4o       |  0.1 |  5 | cot     |   47.1%  |  5050ms  | $0.00750
...
--------------------------------------------------------------------------------
Legend: * = Best in column, > = Overall best configuration

Best Configuration Found:
----------------------------------------
  model: gpt-4o-mini
  temperature: 0.1
  retriever_k: 5
  prompt_style: vanilla

Sample Question & Answer:
----------------------------------------
  Q: What is the capital of the country where the Eiffel Tower is located?

  Model Answer: Paris

  Expected: Paris

============================================================
  Demo Complete!
============================================================
```

## Why HotpotQA?

HotpotQA requires combining evidence from multiple passages to answer a single question (multi-hop reasoning). This makes it ideal for demonstrating:

- How retrieval depth (`k`) affects answer quality
- Trade-offs between model capability and cost
- Impact of chain-of-thought prompting on accuracy

## Dataset

This demo includes 8 sample examples for quick testing. For production evaluation, download the full HotpotQA benchmark.

### Included Sample Data (8 examples)

The demo ships with 8 custom multi-hop questions designed to demonstrate Traigent's optimization capabilities:

- **Bridge questions**: "What is the capital of the country where the Eiffel Tower is located?"
- **Bridge questions**: "Who founded the company that created the iPhone?"

### Using the Full HotpotQA Benchmark

For comprehensive evaluation with the full 7,405-example validation set:

```bash
# From the repository root:

# 1. Download the official HotpotQA distractor dev set (~47MB)
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# 2. Set the path and run (the file is now in your repo root)
export HOTPOTQA_DATASET_PATH="$PWD/hotpot_dev_distractor_v1.json"
python walkthrough/hotpotQA/run_demo.py --real
```

The `HOTPOTQA_DATASET_PATH` environment variable tells the demo where to find the benchmark file. Use an absolute path or `$PWD` to avoid issues.

**Important notes:**

- The full benchmark runs take longer and cost more in real mode
- Default loads 100 examples; to load more, edit `paper_experiments/case_study_rag/dataset.py` and change `max_examples or 100` to your desired number
- Unset the environment variable to return to bundled examples:

```bash
unset HOTPOTQA_DATASET_PATH
```

**License**: HotpotQA is released under [CC BY-SA 4.0](https://hotpotqa.github.io/).
See the [official HotpotQA website](https://hotpotqa.github.io/) for citation and terms.

### Example Data Structure

Each example includes:
- The question
- Supporting context passages (2 gold + distractors)
- The expected answer
- Metadata (question type, difficulty level)

## Why Results May Vary

Even in mock mode, results can vary slightly because:
- Optuna's sampling is stochastic (different trial orderings)
- Multi-objective optimization may find different Pareto-optimal points

In real mode, additional variation comes from:
- LLM non-determinism (even at low temperature)
- API latency fluctuations

### Understanding Quality Scores

The quality metric uses **Exact Match + F1 scoring** (standard for HotpotQA). The demo includes answer extraction to handle verbose model outputs:

- Expected: `"Paris"`
- Model answer: `"Based on my analysis, the answer is Paris."`
- Extracted answer: `"Paris"` → **100% match**

Real mode typically achieves **48-71% quality** depending on model:
- **gpt-4o-mini**: 67-71% (concise answers score best)
- **haiku-3.5**: 54-67% (competitive with extraction)
- **gpt-4o**: 48-50% (verbose outputs hurt precision)

### Cost and Token Tracking

The demo uses LangChain's integration with OpenAI and Anthropic. Traigent automatically captures token usage and cost from LangChain responses via its interceptor—no manual tracking needed. This allows parallel execution in both mock and real modes.

## File Structure

```
walkthrough/hotpotQA/
├── README.md           # This file
├── run_demo.py         # Main demo script
├── run_demo.sh         # Shell wrapper
├── install.sh          # Environment setup script
└── requirements.txt    # Python dependencies

paper_experiments/case_study_rag/
├── dataset.py          # HotpotQA dataset loader
├── metrics.py          # Quality/latency/cost metrics
├── pipeline.py         # Response formatting utilities
└── simulator.py        # Mock answer generator
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'traigent'"

Run from the repository root after installing:
```bash
pip install -e .
python walkthrough/hotpotQA/run_demo.py
```

### "OPENAI_API_KEY not set" (in real mode)

Export your API key before running:
```bash
export OPENAI_API_KEY="sk-..."
python walkthrough/hotpotQA/run_demo.py --real
```

## Next Steps

After running this demo, explore:

- **[Walkthrough Examples](../examples/)**: More Traigent patterns (RAG, multi-provider, privacy modes)
- **[Core Examples](../../examples/core/)**: Production-ready example patterns
- **[SDK Documentation](../../docs/)**: Full API reference

---

This demo shows the optimization effects without changing your agent code—just decorate with `@traigent.optimize()` and let Traigent find the best configuration.
