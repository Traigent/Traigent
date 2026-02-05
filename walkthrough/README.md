# Traigent Walkthrough

Learn Traigent by doing. This walkthrough takes you from zero to optimizing real AI applications.

## What You'll Learn

- **Zero-code optimization** - Add `@traigent.optimize` and watch it find the best config
- **Injection modes** - Context, Parameter, and Seamless approaches
- **Multi-objective optimization** - Balance accuracy, cost, and latency
- **RAG tuning** - Optimize retrieval + generation together
- **Custom evaluators** - LLM-as-Judge for subjective tasks
- **Privacy modes** - Local-only execution for sensitive data

## Choose Your Path

### [examples/](examples/) - Progressive Examples (Start Here)

Seven hands-on examples that build on each other:

| Example | What You Learn |
|---------|----------------|
| 01 Simple | Basic model + temperature tuning |
| 02 Zero Code Change | Seamless interception |
| 03 Parameter Mode | Explicit config control |
| 04 Multi-Objective | Trade-off optimization |
| 05 RAG | Retrieval + parallel eval |
| 06 Custom Evaluator | LLM-as-Judge scoring |
| 07 Privacy Modes | Local-only execution |

Each example has **mock** (no API keys) and **real** (actual LLM calls) variants.

Note: Example 05 runs parallel evaluation by default. Pause-on-error prompts only
appear in sequential mode (set `TRAIGENT_PARALLEL=0`).

```bash
# Quick start - no API keys needed
python walkthrough/examples/mock/01_tuning_qa.py

# Run all mock examples
bash walkthrough/examples/test_all_examples.sh --mock
```

### [hotpotQA/](hotpotQA/) - Advanced Multi-Hop QA Demo

A complete case study optimizing a multi-hop question-answering agent on the HotpotQA benchmark. Demonstrates:

- Multi-step reasoning over multiple paragraphs
- Retrieval depth tuning (`k` parameter)
- Prompt style optimization (direct vs chain-of-thought)
- Cross-provider comparison (OpenAI, Anthropic)
- Pareto frontier analysis (quality vs cost vs latency)

```bash
cd walkthrough/hotpotQA
./install.sh    # Setup environment
./run_demo.sh   # Run optimization (mock mode by default)
```

## Next Steps

After completing the walkthrough:

- Explore [examples/](../examples/) for more advanced patterns
- Read the [SDK Documentation](../docs/README.md)
- Try the [Playground](../playground/) for interactive optimization
