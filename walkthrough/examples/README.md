# Traigent Walkthrough Examples

Simple, clean examples showing how to use Traigent for LLM optimization.

## Requirements

Install Traigent with integrations to run all examples:

```bash
pip install traigent[integrations]
```

This includes:

- `langchain`, `langchain-openai`, `langchain-community` - LLM framework
- `faiss-cpu` - Vector search for RAG examples
- `openai`, `anthropic` - LLM providers

## Quick Start

```bash
# Mock examples - no API keys needed
python walkthrough/examples/mock/01_tuning_qa.py

# Real examples - requires OpenAI API key
export OPENAI_API_KEY="your-key"
python walkthrough/examples/real/01_tuning_qa.py
```

Check your environment with `python walkthrough/examples/utils/check_environment.py`.

## Run All Walkthrough Examples

```bash
# Run all mock examples (no API keys needed)
bash walkthrough/examples/test_all_examples.sh --mock

# Run all real examples (requires OpenAI API key)
export OPENAI_API_KEY="your-key"
bash walkthrough/examples/test_all_examples.sh --real
```

## Further Reading

- Mock mode: See `../../examples/README.md` ("Run any example in mock mode" section)
- Example guide: `../../examples/docs/EXAMPLES_GUIDE.md`
- Getting started: `../../examples/docs/START_HERE.md`

## Structure

```text
examples/
├── README.md              # This file
├── mock/                  # No API keys needed, instant results
├── real/                  # Requires API keys, real LLM calls
├── datasets/              # Pre-built evaluation datasets (20 examples each)
├── utils/                 # Shared utilities (scoring, helpers, mock answers)
└── test_all_examples.sh   # Run all examples script
```

## Examples Overview

| #  | Example          | Description                                | Injection Mode | Dataset          | Evaluation Method          |
|----|------------------|--------------------------------------------|----------------|------------------|----------------------------|
| 01 | Simple           | Basic model and temperature tuning         | Context        | simple_questions | Exact Match                |
| 02 | Zero Code Change | Seamless mode intercepts hardcoded values  | Seamless       | simple_questions | Exact Match                |
| 03 | Parameter Mode   | Explicit configuration control             | Parameter      | simple_questions | Exact Match                |
| 04 | Multi-Objective  | Balance accuracy, cost, and latency        | Context        | classification   | Exact Match                |
| 05 | RAG              | Optimize retrieval + parallel eval         | Context        | rag_questions    | Semantic Similarity        |
| 06 | Custom Evaluator | LLM-as-Judge for code generation           | Context        | code_gen         | LLM-as-Judge (GPT-4o-mini) |
| 07 | Privacy Modes    | Local-only privacy-first execution         | Context        | simple_questions | Exact Match                |

Injection modes are explained in depth here: [Injection Modes Guide](../../docs/user-guide/injection_modes.md).

Quick notes for new users:

- **01 Simple**: Context mode (default) with basic controls (model + temperature).
- **02 Zero Code Change**: Shows seamless interception without changing your code.
- **03 Parameter Mode**: Explicit config parameters passed into your function.
- **04 Multi-Objective**: Trade off accuracy, cost, and latency.
- **05 RAG**: Retrieval + generation tuning. `k` is the number of documents to retrieve; `retrieval_method` is `similarity` (vector embeddings) or `keyword` (text matching). You implement the retrieval logic in your function; Traigent finds the optimal parameter combination. This example enables parallel eval by default; disable with `TRAIGENT_PARALLEL=0`.
- **06 Custom Evaluator**: LLM-as-judge scoring for code generation.
- **07 Privacy Modes**: Local-only privacy-first run for now (no cloud/hybrid required).

## Datasets

Each dataset contains **20 examples** with varying difficulty levels to ensure different model configurations show measurable differences.

### simple_questions.jsonl

General knowledge Q&A with three difficulty tiers:

- **Easy (10)**: Factual questions with single-word answers (e.g., "What is 2+2?", "Capital of France?")
- **Medium (5)**: Questions requiring brief explanations (e.g., "What causes tides?")
- **Hard (5)**: Complex questions needing nuanced reasoning (e.g., "Explain opportunity cost with an example")

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

## Mock Output Controls

Optional environment variables for adjusting mock output verbosity:

- `TRAIGENT_SHOW_DETAIL_LOGS=1` re-enables per-call logs in mock examples (used in 03/05).
- `TRAIGENT_GEN_LOG_EVERY=10` logs every 10th generation in mock Example 6.

Set these in your terminal before running an example:

```bash
export TRAIGENT_SHOW_DETAIL_LOGS=1
export TRAIGENT_GEN_LOG_EVERY=10
python walkthrough/examples/mock/06_custom_evaluator.py
```

Or set them in code before imports:

```python
import os
os.environ["TRAIGENT_SHOW_DETAIL_LOGS"] = "1"
```

## Progress Output (Optional)

Long-running walkthrough examples pass `show_progress=True` into `.optimize(...)` to show
lightweight progress updates (trial number, best score, cost) on stderr. This is off by
default in the SDK and opt-in per run.

Enable globally (overrides code):

```bash
export TRAIGENT_SHOW_PROGRESS=1
```

Disable even if an example enables it:

```bash
export TRAIGENT_SHOW_PROGRESS=0
```

Or enable per run in code:

```python
results = await optimized_fn.optimize(
    algorithm="random",
    max_trials=10,
    show_progress=True,
)
```

### LLM-as-Judge (Example 06)

Uses a separate LLM to evaluate response quality with a detailed rubric.
In mock mode, Example 06 uses a lightweight heuristic scorer (function signature, error-free code, docs) to avoid LLM calls.

**Judge Prompt Features**:

- Clear evaluation criteria with scoring guidelines
- Structured JSON output for reliable parsing
- Weighted scoring across multiple dimensions
- Fallback to heuristics if the judge call fails (real mode)

```python
# Example from 06_custom_evaluator.py
def llm_code_evaluator(output: str, expected: str, **kwargs) -> float:
    """Uses GPT-4o-mini to evaluate code quality."""
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # ... detailed scoring rubric prompt ...
    result = judge.invoke(prompt)
    return weighted_score
```

## Mock vs Real

| Mock (`mock/`)     | Real (`real/`)           |
|--------------------|--------------------------|
| No API keys        | Requires OPENAI_API_KEY  |
| Instant results    | Real LLM calls           |
| Simulated accuracy | Actual model performance |
| Simulated best config | Real best config per run |
| Great for learning | Production-ready         |
| ~50 lines each     | ~60-150 lines each       |

## Execution Modes & Privacy

These walkthrough examples use **local execution mode** (`edge_analytics`), which keeps all data on your machine. This is the recommended starting point for learning Traigent.

- **Local / Edge Analytics**: LLM calls and optimization run locally. Results stay on your machine. Optionally send anonymized analytics without sharing prompts/responses.

Example 07 demonstrates privacy-first local execution with local result storage.

### Local Results Folder (Example 07)

By default, walkthrough examples store results under:

```
walkthrough/examples/.traigent_local/
```

Inside you'll see:

- `experiments/`: per-function runs and trial results
  - Example: `experiments/answer_question/runs/<run_id>/`
- `sessions/`: lightweight session metadata (one file per optimization session)
- `cache/` and `joblib/`: local caches used by the optimizer

If you want a different location, set `TRAIGENT_RESULTS_FOLDER` before running.

## Why Varying Difficulty?

The datasets include varying difficulty levels so that:

1. **Different models can show different results** - in real runs, stronger models often do better on harder questions (mock results are simulated and won't necessarily reflect this)
2. **Temperature effects are visible** - lower temperature tends to help on factual tasks; higher temperature may help on open-ended generation (limited in this walkthrough)
3. **Optimization is meaningful** - There's room for improvement across configurations
4. **Edge cases are covered** - Ambiguous examples test model robustness

### Why results can change between runs

Even with the same code, results can vary slightly because:

- LLMs are nondeterministic (even at low temperature)
- The dataset is small (20 items), so a few misses can swing accuracy by ~5-10%
- Cost/accuracy trade-offs can shift the best config
- Model behavior can drift over time

If you want more stability, increase dataset size, increase trials, or optimize for accuracy only.
