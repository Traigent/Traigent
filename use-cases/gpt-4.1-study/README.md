# GPT-4.1 Replication Study

A comprehensive study to replicate and validate key experiments from [OpenAI's GPT-4.1 announcement](https://openai.com/index/gpt-4-1/) using the Traigent SDK optimization framework.

## Overview

On April 14, 2025, OpenAI announced GPT-4.1, GPT-4.1 mini, and GPT-4.1 nano - models claiming significant improvements in coding, instruction following, and long context comprehension. This study aims to independently validate these claims through controlled experiments.

### Key Claims from OpenAI

| Capability | GPT-4.1 | GPT-4o | Improvement |
|------------|---------|--------|-------------|
| SWE-bench Verified | 54.6% | 33.2% | +21.4% |
| Aider Polyglot (diff) | 53% | 18% | +35% |
| Instruction Following (hard) | 49% | 29% | +20% |
| MultiChallenge | 38.3% | 27.8% | +10.5% |
| OpenAI-MRCR (2 needle, 128k) | 57.2% | 31.9% | +25.3% |
| ComplexFuncBench | 65.5% | 66.5% | -1% |

## Models Compared

| Model | Context Window | Cost (per 1M tokens) |
|-------|----------------|---------------------|
| gpt-4.1 | 1M tokens | $2.00 input / $8.00 output |
| gpt-4.1-mini | 1M tokens | $0.40 input / $1.60 output |
| gpt-4.1-nano | 1M tokens | $0.10 input / $0.40 output |
| gpt-4o | 128K tokens | $2.50 input / $10.00 output |
| gpt-4o-mini | 128K tokens | $0.15 input / $0.60 output |

## Experiments

### 1. Coding Performance

**Objective:** Validate GPT-4.1's claimed superiority in code generation and diff format reliability.

**Test Categories:**
- Code generation from specifications (10 tasks)
- Code modification with diff output format (10 tasks)
- Bug fixing and debugging (5 tasks)

**Metrics:**
- `task_completion`: Whether the generated code solves the task
- `diff_compliance`: Adherence to search/replace diff format
- `extraneous_edit_rate`: Frequency of unnecessary code changes
- `cost`: Token cost per task

**Blog Reference:** GPT-4.1 scores 54.6% on SWE-bench Verified vs 33.2% for GPT-4o. Extraneous edits dropped from 9% to 2%.

### 2. Instruction Following

**Objective:** Test model ability to follow complex, multi-faceted instructions reliably.

**Test Categories (based on OpenAI's internal IF eval):**
- Format following (JSON, XML, YAML, Markdown) - 8 tasks
- Negative instructions ("don't mention X") - 6 tasks
- Ordered instructions ("first do A, then B") - 6 tasks
- Content requirements ("must include X") - 5 tasks
- Ranking/ordering output - 5 tasks

**Metrics:**
- `format_compliance`: Output matches required format
- `instruction_adherence`: All specified constraints followed
- `negative_instruction_compliance`: Avoids prohibited content
- `cost`: Token cost per task

**Blog Reference:** GPT-4.1 scores 49.1% on hard instruction following vs 29.2% for GPT-4o.

### 3. Long Context Comprehension

**Objective:** Evaluate retrieval accuracy and multi-hop reasoning across large contexts.

**Test Categories:**
- Single needle retrieval (5 tasks) - find specific info in large context
- Multi-needle disambiguation (8 tasks) - like OpenAI-MRCR benchmark
- Multi-hop reasoning (7 tasks) - like Graphwalks, requires reasoning across positions

**Metrics:**
- `retrieval_accuracy`: Correct information retrieved from context
- `multi_hop_accuracy`: Correct multi-step reasoning
- `cost`: Token cost per task

**Blog Reference:** GPT-4.1 achieves 57.2% on 2-needle MRCR at 128k vs 31.9% for GPT-4o.

### 4. Function Calling Reliability

**Objective:** Test tool use accuracy and parameter correctness.

**Test Categories:**
- Single tool selection (8 tasks) - choose correct tool for task
- Multi-tool orchestration (8 tasks) - coordinate multiple tools
- Complex parameter schemas (9 tasks) - handle nested/complex params

**Metrics:**
- `tool_selection_accuracy`: Correct tool chosen
- `parameter_accuracy`: Parameters correctly specified
- `cost`: Token cost per task

**Blog Reference:** GPT-4.1 scores 65.5% on ComplexFuncBench vs 66.5% for GPT-4o (similar performance).

## Quick Start

### Prerequisites

```bash
# Install Traigent with dev dependencies
make install-dev

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

### Run in Mock Mode (Testing)

```bash
# Test without API calls
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest use-cases/gpt-4.1-study/
```

### Run Individual Experiments

```bash
# Coding experiment
python -m use_cases.gpt_4_1_study.agent.coding_agent

# Instruction following experiment
python -m use_cases.gpt_4_1_study.agent.instruction_following_agent

# Long context experiment
python -m use_cases.gpt_4_1_study.agent.long_context_agent

# Function calling experiment
python -m use_cases.gpt_4_1_study.agent.function_calling_agent
```

### Run Full Study

```bash
python -m use_cases.gpt_4_1_study.run_study
```

## Configuration Space

All agents use the following base configuration:

```python
configuration_space = {
    "model": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "temperature": [0.0, 0.3],
}
```

Additional experiment-specific parameters:
- **Coding:** `output_format` (whole_file, diff)
- **Long Context:** `context_strategy` (full_context, chunked)
- **Function Calling:** `parallel_tool_calls` (True, False)

## Data Sources

> **Important:** This study uses a mix of real benchmark data and synthetic examples.

### Real Benchmarks (Available)

| Benchmark | Source | Integration |
|-----------|--------|-------------|
| **SWE-bench Verified** | [HuggingFace](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | `datasets/swebench/` |
| **Aider Polyglot** | [GitHub](https://github.com/Aider-AI/polyglot-benchmark) | `datasets/aider/` |
| **IFEval** | [Google Research](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | `datasets/ifeval/` |

### Synthetic/Inspired Datasets

| Dataset | Status | Notes |
|---------|--------|-------|
| `coding_dataset.jsonl` | Synthetic | Example tasks following SWE-bench/Aider methodology |
| `instruction_following_dataset.jsonl` | Synthetic | Tasks inspired by OpenAI's internal IF eval categories |
| `long_context_dataset.jsonl` | Synthetic | MRCR-inspired multi-needle tasks (original not yet released) |
| `function_calling_dataset.jsonl` | Synthetic | Tasks inspired by ComplexFuncBench/Taubench |

### Benchmarks NOT Yet Public (with Open-Source Alternatives)

The following benchmarks mentioned in the blog are **not publicly available**, but we've identified open-source alternatives:

| Missing Benchmark | Open-Source Alternative | Source |
|-------------------|------------------------|--------|
| **OpenAI-MRCR** | [BABILong](https://github.com/booydar/babilong) | NeurIPS 2024 - Multi-needle long context |
| **OpenAI-MRCR** | [LongBench v2](https://github.com/THUDM/LongBench) | ACL 2025 - Up to 2M token context |
| **Graphwalks** | [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) | COLM 2024 - Multi-hop reasoning |
| **MultiChallenge** | [Multi-IF](https://github.com/facebookresearch/Multi-IF) | Facebook - Multi-turn IF |
| **ComplexFuncBench** | [ToolBench](https://github.com/OpenBMB/ToolBench) | ICLR 2024 - 16K+ real APIs |
| **ComplexFuncBench** | [AgentBench](https://github.com/THUDM/AgentBench) | ICLR 2024 - LLM-as-agent |

### Using Real Benchmarks

```bash
# List available benchmarks
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --list

# === Original Benchmarks ===

# Download SWE-bench Verified (50 samples)
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --swebench --limit 50

# Download IFEval
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --ifeval --limit 50

# Clone Aider polyglot benchmark
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --aider

# === Open-Source Alternatives ===

# Download BABILong (MRCR alternative - long context)
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --babilong --limit 50

# Download LongBench v2 (MRCR alternative - up to 2M tokens)
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --longbench --limit 50

# Clone MultiHop-RAG (Graphwalks alternative - multi-hop reasoning)
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --multihop-rag

# Download Multi-IF (MultiChallenge alternative - instruction following)
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --multi-if --limit 50

# Clone ToolBench (ComplexFuncBench alternative - 16K+ APIs)
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --toolbench

# Download all available benchmarks
python use-cases/gpt-4.1-study/datasets/benchmark_loader.py --all
```

## Dataset Format

All datasets use JSONL format:

```json
{"input": {"task": "...", "context": "..."}, "output": {"expected": "..."}}
```

### Dataset Sizes (Synthetic)

| Experiment | Tasks | Categories |
|------------|-------|------------|
| Coding | 25 | Generation, Diff, Bug Fixing |
| Instruction Following | 30 | Format, Negative, Ordered, Content, Ranking |
| Long Context | 20 | Single Needle, Multi-Needle, Multi-Hop |
| Function Calling | 25 | Single Tool, Multi-Tool, Complex Params |

## Expected Results

Based on OpenAI's blog claims, we expect to observe:

1. **Coding:** GPT-4.1 should significantly outperform GPT-4o on task completion and diff format compliance
2. **Instruction Following:** GPT-4.1 should show ~20% improvement on hard instruction following tasks
3. **Long Context:** GPT-4.1 should excel at multi-needle retrieval tasks
4. **Function Calling:** Performance should be similar between GPT-4.1 and GPT-4o

## Directory Structure

```
gpt-4.1-study/
├── README.md                           # This file
├── __init__.py
├── agent/
│   ├── __init__.py
│   ├── coding_agent.py                 # Code generation/review tests
│   ├── instruction_following_agent.py  # IF compliance tests
│   ├── long_context_agent.py           # Retrieval/reasoning tests
│   └── function_calling_agent.py       # Tool use tests
├── datasets/
│   ├── __init__.py
│   ├── coding_dataset.jsonl
│   ├── instruction_following_dataset.jsonl
│   ├── long_context_dataset.jsonl
│   └── function_calling_dataset.jsonl
├── eval/
│   ├── __init__.py
│   ├── coding_evaluator.py
│   ├── instruction_evaluator.py
│   ├── long_context_evaluator.py
│   └── function_calling_evaluator.py
└── results/
    └── .gitkeep
```

## References

- [OpenAI GPT-4.1 Announcement](https://openai.com/index/gpt-4-1/)
- [SWE-bench Verified](https://www.swebench.com/)
- [Aider Polyglot Benchmark](https://aider.chat/docs/leaderboards/)
- [Scale MultiChallenge](https://scale.com/leaderboard)
- [OpenAI-MRCR Dataset](https://github.com/openai/mrcr) (when released)
- [Graphwalks Dataset](https://github.com/openai/graphwalks) (when released)

## Contributing

When adding new test cases:

1. Follow the existing JSONL format in datasets/
2. Ensure mock mode compatibility
3. Update the evaluator if new metrics are needed
4. Run `make format && make lint` before committing

## License

This study is part of the Traigent SDK examples and follows the project's license.
