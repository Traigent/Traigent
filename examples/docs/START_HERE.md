# START HERE - Traigent Examples Navigator

Pick an example in under a minute.

## Quick picks
- 2 min: `TRAIGENT_MOCK_LLM=true python examples/core/simple-prompt/run.py`
- 5 min: `TRAIGENT_MOCK_LLM=true python examples/core/rag-optimization/run.py`
- 8 min: `TRAIGENT_MOCK_LLM=true python examples/core/few-shot-classification/run.py`
- 10 min: `TRAIGENT_MOCK_LLM=true python examples/core/multi-objective-tradeoff/run_anthropic.py`
- 30+ min: browse `examples/advanced/` playbooks

## Choose by goal
- Lower cost/latency: multi-objective tradeoff, token-budget summarization, execution-modes (advanced).
- Better accuracy: few-shot classification, prompt-style optimization, AI engineering tasks (few-shot selection).
- Safety: safety-guardrails, AI engineering tasks (safety).
- Structured JSON: structured-output-json, AI engineering tasks (structured output).
- Tools: tool-use-calculator.
- Long docs: chunking-long-context, AI engineering tasks (context engineering).
- Prompt tests: prompt-ab-test.
- CI/CD: `examples/integrations/ci-cd/`.

## Directory map (current)
```text
examples/
|- core/
|  |- simple-prompt/
|  |- rag-optimization/
|  |- few-shot-classification/
|  |- multi-objective-tradeoff/
|  |- token-budget-summarization/
|  |- structured-output-json/
|  |- tool-use-calculator/
|  |- prompt-style-optimization/
|  |- chunking-long-context/
|  |- safety-guardrails/
|  `- prompt-ab-test/
|
|- advanced/
|  |- execution-modes/
|  |- results-analysis/
|  |- ai-engineering-tasks/
|  |- ragas/
|  `- metric-registry/
|
|- integrations/
|  |- ci-cd/
|  `- bedrock/
|
|- datasets/
|- docs/
|- templates/
|- utils/
`- tvl/
```

## Recommended path (beginner)
1. `core/simple-prompt/` - sanity check and mock mode setup
2. `core/rag-optimization/` - RAG toggle and telemetry
3. `core/few-shot-classification/` - few-shot concepts
4. `core/multi-objective-tradeoff/` - balance accuracy, latency, cost
5. `advanced/ai-engineering-tasks/` - specialized playbooks

## Run instructions
```bash
cd /path/to/Traigent
pip install -e ".[examples]"
export TRAIGENT_MOCK_LLM=true
python examples/core/simple-prompt/run.py
```

With real APIs (optional):
```bash
export ANTHROPIC_API_KEY="your-key" # pragma: allowlist secret
export OPENAI_API_KEY="your-key" # pragma: allowlist secret
python examples/core/multi-objective-tradeoff/run_anthropic.py
```

## Pro tips
- Stay in repo root so relative paths resolve.
- Use mock mode first to avoid spend.
- Every example notes its dataset under `examples/datasets/<example>/`.
- Copy an example folder into your project to bootstrap a workflow.

Need more? See `EXAMPLES_GUIDE.md` or `TROUBLESHOOTING.md`.
