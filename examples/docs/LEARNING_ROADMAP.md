# Traigent Learning Roadmap

A concise path from first run to production-ready evaluations.

## Tracks
- Beginner (60-90 min): mock-mode runs, core knobs.
- Builder (2-3 hours): balance cost/latency/accuracy, add safety, ship to CI.
- Researcher (3-4 hours): advanced playbooks, metrics, RAG evaluation.

## Week 1: Foundations (beginner)
1. Sanity check: `TRAIGENT_MOCK_LLM=true python examples/core/simple-prompt/run.py`
2. RAG toggle: `TRAIGENT_MOCK_LLM=true python examples/core/hello-world/run.py`
3. Few-shot prompts: `TRAIGENT_MOCK_LLM=true python examples/core/few-shot-classification/run.py`
4. Trade-offs: `TRAIGENT_MOCK_LLM=true python examples/core/multi-objective-tradeoff/run_anthropic.py`

Outcome: you can run examples, read metrics, and tweak configuration spaces.

## Week 2: Applied patterns (builder)
1. Control spend/latency: `examples/core/token-budget-summarization/run.py`
2. Quality and tone: `examples/core/prompt-style-optimization/run.py`
3. Safety: `examples/core/safety-guardrails/run.py` -> `examples/advanced/ai-engineering-tasks/p1_safety_guardrails/`
4. Structured outputs: `examples/core/structured-output-json/run.py`
5. Tool calling: `examples/core/tool-use-calculator/run.py`

Outcome: you can constrain costs, add guardrails, and tune prompts.

## Week 3: Advanced exploration (researcher)
1. Execution modes and rollouts: `examples/advanced/execution-modes/` (local patterns + roadmap stubs)
2. Context engineering & RAG eval: `examples/advanced/ai-engineering-tasks/p0_context_engineering/` and `examples/advanced/ragas/`
3. Metrics and analysis: `examples/advanced/results-analysis/` and `examples/advanced/metric-registry/`
4. CI/CD integration: `examples/integrations/ci-cd/`

Outcome: you can experiment with new metrics, evaluate RAG systems, and integrate Traigent into pipelines.

## Running and debugging
```bash
pip install -e ".[examples]"
export TRAIGENT_MOCK_LLM=true
ls examples/datasets/<example-name>
rg "optimize(" examples
```
If something fails: see `TROUBLESHOOTING.md` or rerun with `TRAIGENT_LOG_LEVEL=DEBUG`.
