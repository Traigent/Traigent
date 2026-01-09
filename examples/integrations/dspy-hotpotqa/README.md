# DSPy + Traigent: HotPotQA Multi-Hop QA Optimization

This example demonstrates the recommended workflow for combining **DSPy's prompt optimization** with **Traigent's hyperparameter optimization** using the official HotPotQA dataset.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Three-Stage Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Stage 1: DSPy Prompt Generation                                │
│   ┌─────────────┐                                                │
│   │  trainset   │──▶ DSPy BootstrapFewShot ──▶ 3 Prompt Variants │
│   │ (200 examples)                                               │
│   └─────────────┘                                                │
│                                                                   │
│   Stage 2: Traigent Hyperparameter Optimization                  │
│   ┌─────────────┐                                                │
│   │   devset    │──▶ Traigent @optimize ──▶ Best Config          │
│   │ (100 examples)   (prompts + model + temp + tokens)           │
│   └─────────────┘                                                │
│                                                                   │
│   Stage 3: Final Validation                                       │
│   ┌─────────────┐                                                │
│   │   testset   │──▶ Evaluate Best Config ──▶ Test Accuracy      │
│   │ (50 examples)    (held-out, never seen)                      │
│   └─────────────┘                                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Approach?

| Tool | Strength | Used For |
|------|----------|----------|
| **DSPy** | Prompt optimization, few-shot selection | Generating effective prompt variants |
| **Traigent** | Hyperparameter optimization, multi-objective | Finding best prompt + model + settings |

### Dataset Separation (Critical!)

Following [DSPy's official recommendations](https://dspy.ai/learn/evaluation/overview/):

> "Ideally **200 examples or more to prevent overfitting** when running longer optimization runs."

| Dataset | Examples | Stage | Purpose |
|---------|----------|-------|---------|
| `trainset` | 200 | Stage 1 | DSPy prompt generation |
| `devset` | 100 | Stage 2 | Traigent hyperparameter optimization |
| `testset` | 50 | Stage 3 | Final evaluation (never seen) |

## Quick Start

### 1. Install Dependencies

```bash
pip install traigent[dspy]
```

### 2. Download Dataset

```bash
cd examples/integrations/dspy-hotpotqa
python download_data.py
```

This downloads the HotPotQA dataset from HuggingFace and saves it locally.

### 3. Run in Mock Mode (No API Calls)

```bash
TRAIGENT_MOCK_LLM=true python run.py
```

### 4. Run with Real LLMs

```bash
export OPENAI_API_KEY=your-key-here
python run.py
```

## What Happens

### Stage 1: DSPy Prompt Generation

Uses DSPy's `BootstrapFewShot` optimizer to generate diverse prompt strategies:

1. **Direct prompt**: Simple question-answer format
2. **Chain-of-thought**: Step-by-step reasoning
3. **Multi-hop reasoning**: Explicit multi-source synthesis

```python
from traigent.integrations import DSPyPromptOptimizer

optimizer = DSPyPromptOptimizer(method="bootstrap")
result = optimizer.optimize_prompt(
    module=CoTQA(),
    trainset=dspy_trainset[:50],
    metric=exact_match,
    max_bootstrapped_demos=2,
)
```

### Stage 2: Traigent Optimization

Searches over the configuration space:

```python
@traigent.optimize(
    configuration_space={
        "prompt_template": Choices(prompt_variants),  # From Stage 1
        "model": Choices(["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]),
        "temperature": Choices([0.0, 0.3, 0.5, 0.7]),
        "max_tokens": Choices([100, 200, 300]),
    },
    objectives=["accuracy", "cost", "latency"],
    eval_dataset="devset.jsonl",
)
def hotpotqa_agent(question: str, **config) -> str:
    ...
```

### Stage 3: Validation

Evaluates the best configuration on the held-out test set:

```python
test_accuracy = validate_on_test_set(agent, best_config, testset)
```

## Expected Output

```
================================================================
DSPy + Traigent Integration: HotPotQA Optimization
================================================================

Loading HotPotQA dataset...
  Train: 200 examples (for DSPy prompt optimization)
  Dev: 100 examples (for Traigent hyperparameter optimization)
  Test: 50 examples (for final validation)

================================================================
Stage 1: Generating Prompt Variants with DSPy
================================================================
  Using 50 examples for prompt generation
  Generating CoT variant with DSPy BootstrapFewShot...
    Generated with 2 demos, score: 0.85

  Generated 3 prompt variants:
    1. Answer the following question directly and concis...
    2. Think step by step to answer this question....
    3. This is a multi-hop question that may require co...

================================================================
Stage 2: Traigent Hyperparameter Optimization
================================================================
  Created agent with 3 prompt variants
  Configuration space size: 108 configs
  Max trials: 30

  Running Traigent optimization...
  Optimization complete!
  Best validation score: 0.78

================================================================
Stage 3: Validation on Held-Out Test Set
================================================================
  Best config: {
      "prompt_template": "Think step by step...",
      "model": "gpt-4o-mini",
      "temperature": 0.0,
      "max_tokens": 200
  }
  Evaluating on 50 test examples...

  Test Set Accuracy: 76.0% (38/50)

================================================================
Summary
================================================================
  Prompt variants generated: 3
  Validation score: 0.78
  Test accuracy: 76.0%
  Best model: gpt-4o-mini
  Best temperature: 0.0
```

## Files

```
dspy-hotpotqa/
├── README.md              # This file
├── download_data.py       # Dataset downloader
├── run.py                 # Main example
├── data/                  # Downloaded datasets (gitignored)
│   ├── hotpotqa_train.jsonl
│   ├── hotpotqa_dev.jsonl
│   └── hotpotqa_test.jsonl
└── results/               # Optimization results
    └── optimization_results.json
```

## Dataset: HotPotQA

**HotPotQA** is a multi-hop question answering dataset featuring natural questions that require reasoning over multiple Wikipedia articles.

- **Source**: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)
- **License**: CC BY-SA 4.0
- **Citation**:
  ```
  @inproceedings{yang2018hotpotqa,
    title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
    author={Yang, Zhiping and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William and Salakhutdinov, Ruslan and Manning, Christopher D.},
    booktitle={EMNLP},
    year={2018}
  }
  ```

## Key Learnings

1. **Separate your data**: Use different splits for DSPy training and Traigent evaluation
2. **DSPy generates diversity**: Use DSPy to create prompt variants, not just one optimized prompt
3. **Traigent finds the best combo**: Let Traigent search over prompts + hyperparameters together
4. **Validate on held-out data**: Always keep a test set that's never seen during optimization

## Related Documentation

- [DSPy Integration Guide](../../docs/DSPY_INTEGRATION.md)
- [DSPy Official Docs](https://dspy.ai/)
- [Traigent Quickstart](../../quickstart/README.md)
