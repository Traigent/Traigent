# Traigent Walkthrough

Learn Traigent by doing. Simple, clean examples showing how to use Traigent for LLM optimization.

## What You'll Learn

- **Zero-code optimization** - Add `@traigent.optimize` and watch it find the best config
- **Injection modes** - Context, Parameter, and Seamless approaches
- **Multi-objective optimization** - Balance accuracy, cost, and latency
- **RAG tuning** - Optimize retrieval + generation together
- **Custom evaluators** - LLM-as-Judge for subjective tasks
- **Privacy modes** - Local-only execution for sensitive data

## Requirements

Install Traigent with integrations to run all examples:

```bash
pip install traigent[integrations]
```

This includes:

- `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-community` - LLM framework
- `faiss-cpu` - Vector search for RAG examples
- `openai`, `anthropic`, `google-generativeai` - LLM providers

## Quick Start

```bash
# Mock examples - no API keys needed
python walkthrough/mock/01_tuning_qa.py

# Real examples - use real LLM calls when a provider key is set;
# otherwise they warn and fall back to the matching mock walkthrough
export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
python walkthrough/real/01_tuning_qa.py

# Multi-provider example (any one key works; without keys it falls back to mock)
export OPENAI_API_KEY="your-key"  # Or ANTHROPIC_API_KEY / GOOGLE_API_KEY  <!-- pragma: allowlist secret -->
python walkthrough/real/07_multi_provider.py
```

Check your environment with `python walkthrough/utils/check_environment.py`.

## Run All Walkthrough Examples

```bash
# Run all mock examples (no API keys needed)
bash walkthrough/test_all_examples.sh --mock

# Run all real examples (uses provider keys when set, otherwise falls back to mock)
export OPENAI_API_KEY="your-key"  # pragma: allowlist secret
bash walkthrough/test_all_examples.sh --real

# Disable fallback and fail fast when keys are missing (useful for CI)
export TRAIGENT_REQUIRE_REAL=1
```

## Structure

```text
walkthrough/
├── README.md              # This file
├── demo/                  # Optional demo/video support scripts
├── mock/                  # No API keys needed, instant results
├── real/                  # Real LLM calls when keyed, mock fallback when keys are missing
├── datasets/              # Pre-built evaluation datasets (20 examples each)
├── utils/                 # Shared utilities (scoring, helpers, mock answers)
└── test_all_examples.sh   # Run all examples script
```

## Examples Overview

Eight hands-on examples that build on each other:

| #  | Example          | Description                                | Injection Mode | Dataset             | Evaluation Method          |
|----|------------------|--------------------------------------------|----------------|---------------------|----------------------------|
| 01 | Simple           | Basic model and temperature tuning         | Context        | simple_questions    | Exact Match                |
| 02 | Zero Code Change | Seamless mode intercepts hardcoded values  | Seamless       | simple_questions    | Exact Match                |
| 03 | Parameter Mode   | Explicit configuration control             | Parameter      | simple_questions    | Exact Match                |
| 04 | Multi-Objective  | Balance accuracy, cost, and latency        | Context        | classification      | Exact Match                |
| 05 | RAG              | Optimize retrieval + parallel eval         | Context        | rag_questions       | Semantic Similarity        |
| 06 | Custom Evaluator | LLM-as-Judge for code generation           | Context        | code_gen            | LLM-as-Judge (GPT-4o-mini) |
| 07 | Multi-Provider   | Use any LLM vendor (OpenAI, Claude, Gemini)| Context        | simple_questions_10 | Exact Match                |
| 08 | Privacy Modes    | Local-only privacy-first execution         | Context        | simple_questions    | Exact Match                |

Injection modes are explained in depth here: [Injection Modes Guide](../docs/user-guide/injection_modes.md).

Each example has **mock** (no API keys) and **real** (actual LLM calls when keys are present, otherwise automatic mock fallback) variants.

Note: Example 05 runs parallel evaluation by default. Pause-on-error prompts only
appear in sequential mode (set `TRAIGENT_PARALLEL=0`).

## Optional Extras

These are not part of the core 8-step walkthrough or `test_all_examples.sh`, but
they are useful companion material:

- `walkthrough/mock/09_rag_multi_objective.py`: mock-only RAG tradeoff example
- `walkthrough/demo/optimize_and_observe.py`: shows how to combine `@optimize` and `@observe` on the same method, with mock/real modes and optimization scale presets
- `walkthrough/demo/run_guided_optimize_and_observe_demo.sh`: guided FE demo that pauses between baseline observability, optimization, and post-best-config observability runs
- `walkthrough/demo/rag_agent.py`: standalone pre-Traigent RAG baseline (requires `OPENAI_API_KEY`)
- `walkthrough/demo/optimize_rag.py`: replay of a recorded multi-objective run

### Quick Notes for New Users

- **01 Simple**: Context mode (default) with basic controls (model + temperature).
- **02 Zero Code Change**: Shows seamless interception without changing your code.
- **03 Parameter Mode**: Explicit config parameters passed into your function.
- **04 Multi-Objective**: Trade off accuracy, cost, and latency.
- **05 RAG**: Retrieval + generation tuning. `k` is the number of documents to retrieve; `retrieval_method` is `similarity` (vector embeddings) or `keyword` (text matching). You implement the retrieval logic in your function; Traigent finds the optimal parameter combination. This example enables parallel eval by default; disable with `TRAIGENT_PARALLEL=0`.
- **06 Custom Evaluator**: LLM-as-judge scoring for code generation.
- **07 Multi-Provider**: Use any LLM vendor (OpenAI, Anthropic Claude, Google Gemini) in the same optimization. Set the relevant API keys and Traigent finds the best model across providers. Uses a 10-example dataset (~100 API calls); switch to `simple_questions.jsonl` for more thorough evaluation (~200 calls).
- **08 Privacy Modes**: Local-only privacy-first run for now (no cloud/hybrid required).

## Datasets

Each dataset contains **20 examples** with varying difficulty levels to ensure different model configurations show measurable differences.

### simple_questions.jsonl (and simple_questions_10.jsonl)

General knowledge Q&A with three difficulty tiers:

- **Easy (10)**: Factual questions with single-word answers (e.g., "What is 2+2?", "Capital of France?")
- **Medium (5)**: Questions requiring brief explanations (e.g., "What causes tides?")
- **Hard (5)**: Complex questions needing nuanced reasoning (e.g., "Explain opportunity cost with an example")

The `_10` variant contains 10 balanced examples (5 easy, 3 medium, 2 hard) for faster runs with fewer API calls.

**Evaluation**: Exact match against expected output (case-insensitive word containment; stopwords ignored; numeric tokens must match exactly; >=80% of expected tokens required; simple prefix matching for word variants)

### classification.jsonl

Sentiment analysis with ambiguous cases:

- **Clear (10)**: Obvious positive/negative sentiment (e.g., "Best purchase ever!", "Complete waste of money")
- **Ambiguous (10)**: Subtle or mixed sentiment requiring careful analysis (e.g., "Not bad, not great", "Frustrating setup but eventually worked")

**Evaluation**: Exact match (positive/negative/neutral)

### rag_questions.jsonl

Traigent documentation Q&A for RAG testing:

- **Simple (10)**: Direct questions with factual answers from docs
- **Complex (10)**: Questions requiring synthesis of multiple document sections

**Evaluation**: Semantic similarity - checks if key concepts from expected answer appear in response

### code_gen.jsonl

Python code generation tasks with difficulty progression:

- **Easy (5)**: Basic functions (add numbers, check even, reverse string)
- **Medium (6)**: Intermediate algorithms (palindrome check, binary search, word frequency)
- **Hard (9)**: Advanced algorithms (LRU cache, Dijkstra's algorithm, serialize binary tree)

**Note**: This dataset has no `expected_output` field. That is intentional: the custom evaluator (LLM-as-Judge or heuristic fallback) scores code directly against the task description, so reference answers are not required.

**Evaluation**: LLM-as-Judge using GPT-4o-mini with detailed rubric:

- Correctness (40%): Does the code solve the task correctly?
- Code Quality (30%): Is it well-structured and Pythonic?
- Documentation (30%): Does it have proper docstrings/comments?

## Evaluation Methods

### Exact Match (Default)

Implemented locally with a lightweight token match (case-insensitive, stopwords ignored, numeric tokens must match). By default it requires ~80% of expected tokens to appear. If you want stricter behavior, you can swap this for a simple exact string match. Good for:

- Factual Q&A
- Classification tasks
- Multiple choice

### Semantic Similarity

Checks if key concepts appear in the response. Good for:

- RAG applications
- Open-ended questions
- Paraphrased answers

### LLM-as-Judge (Example 06)

Uses a separate LLM to evaluate response quality with a detailed rubric.
In mock mode, Example 06 uses a lightweight heuristic scorer (function signature, error-free code, docs) to avoid LLM calls.

**Judge Prompt Features**:

- Clear evaluation criteria with scoring guidelines
- Structured JSON output for reliable parsing
- Weighted scoring across multiple dimensions
- Fallback to heuristics if the judge call fails (real mode)

## Mock vs Real

| Mock (`mock/`)        | Real (`real/`)           |
|-----------------------|--------------------------|
| No API keys           | Uses provider keys when present |
| Instant results       | Real LLM calls or mock fallback |
| Simulated accuracy    | Actual model performance |
| Simulated best config | Real best config per run |
| Great for learning    | Production-ready         |
| ~50 lines each        | ~60-150 lines each       |

## Execution Modes & Privacy

These walkthrough examples use **local execution mode** (`edge_analytics`), which keeps LLM calls and optimization on your machine. This is the recommended starting point for learning Traigent.

- **Local / Edge Analytics**: LLM calls and optimization run locally. Results stay on your machine. Optionally send anonymized analytics without sharing prompts/responses.

Example 08 demonstrates a fully local, privacy-first path with `TRAIGENT_OFFLINE_MODE=true`, local result storage, and no backend communication.

### Local Results Folder (Example 08)

By default, walkthrough examples store results under:

```text
walkthrough/.traigent_local/
```

Inside you'll see:

- `experiments/`: per-function runs and trial results
  - Example: `experiments/answer_question/runs/<run_id>/`
- `sessions/`: lightweight session metadata (one file per optimization session)
- `cache/` and `joblib/`: local caches used by the optimizer

If you want a different location, set `TRAIGENT_RESULTS_FOLDER` before running.

## Benchmarks

For advanced benchmarks like HotpotQA multi-hop QA optimization, see the
[TraigentDemo benchmarks](https://github.com/Traigent/TraigentDemo/tree/main/benchmarks).

## Further Reading

- Mock mode: See [examples/README.md](../examples/README.md) ("Run any example in mock mode" section)
- Example guide: [Examples Guide](../docs/examples/README.md)
- Getting started: [Start Here](../docs/examples/START_HERE.md)

## Next Steps

After completing the walkthrough:

- Explore [examples/](../examples/) for more advanced patterns
- Read the [SDK Documentation](../docs/README.md)
- Try the Playground for interactive optimization (see repo root `playground/`)
