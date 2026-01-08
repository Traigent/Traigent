# Traigent

**If you can evaluate it, optimize it. Specify evaluation set and eval, and start optimizing today.**

[![Build](https://github.com/Traigent/Traigent/actions/workflows/ci.yml/badge.svg)](https://github.com/Traigent/Traigent/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Traigent adds zero-code-change optimization to existing agents, RAG pipelines, and LangGraph flows so you can improve accuracy and reduce cost without refactoring or extra dev time. Works with LangChain, LlamaIndex, and direct API calls (OpenAI, Anthropic, etc.). CrewAI and AutoGen support is in testing.

---

## Golden Path: Router Agent + LiteLLM

Decorate your existing router, pass an evaluation set, and Traigent will tune across OpenAI, Anthropic, and Google models via LiteLLM.

```python
import traigent
from litellm import completion

EVAL_DATASET = "eval/support_tickets.jsonl"  # Required

PROVIDER_MODELS = [
    "gpt-4o-mini",             # OpenAI
    "claude-3-haiku-20240307", # Anthropic
    "gemini/gemini-pro",       # Google
]

def llm_judge(output: str, expected: str, **kwargs) -> float:
    """LLM-as-a-judge accuracy scorer (0.0-1.0)."""
    judge = completion(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Score 0-1. Expected: {expected}\nActual: {output}\nScore:",
        }],
    )
    return float(str(judge.choices[0].message.content).strip())

@traigent.optimize(
    configuration_space={
        "router_model": PROVIDER_MODELS,
        "agent_model": PROVIDER_MODELS,
        "temperature": [0.1, 0.4],
    },
    eval_dataset=EVAL_DATASET,
    objectives=["accuracy", "cost"],  # Aim: maximize accuracy, minimize cost
    scoring_function=llm_judge,       # Custom accuracy metric
    cost_limit=0.01,                  # Optional safety budget (USD)
)
def support_router(ticket: str) -> str:
    cfg = traigent.get_config()
    route = str(completion(
        model=cfg["router_model"],
        messages=[{"role": "user", "content": f"Route: {ticket}\nAnswer: billing|tech"}],
    ).choices[0].message.content).lower()

    system = "You are a billing specialist." if "billing" in route else "You are a tech specialist."
    response = completion(
        model=cfg["agent_model"],
        temperature=cfg["temperature"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": ticket}],
    )
    return str(response.choices[0].message.content)

results = await support_router.optimize(algorithm="bayesian", max_trials=20)
print(results.best_config, results.best_score)
```

Optimization only runs with an evaluation set. Evaluation datasets are JSONL with input/output fields:

```jsonl
{"input": {"ticket": "Refund request for order #123"}, "output": "billing"}
```

Traigent tracks cost and latency automatically; accuracy comes from your evaluator or dataset.
LLM-as-a-judge is a common way to score accuracy when exact matching is not enough.
Install LiteLLM with `pip install litellm` and set provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
See the [Evaluation Guide](docs/guides/evaluation.md) and [Complete Function Specification](docs/api-reference/complete-function-specification.md) for metrics and optimization parameters.

---

## Installation

### Add to Your Existing Project (Recommended)

```bash
# From your project virtualenv
pip install -e /path/to/Traigent[integrations]

# Or with uv (faster)
uv pip install -e /path/to/Traigent[integrations]
```

```python
# Then in your code:
import traigent

@traigent.optimize(...)
def your_agent(...):
    ...  # Your unchanged code
```

Requirements: Python 3.11+ on Linux, macOS, or Windows.

### Run Bundled Examples / Dev Setup

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,integrations]"

# Test without API keys
export TRAIGENT_MOCK_LLM=true
python examples/quickstart/01_simple_qa.py
```

---

## How It Works (Router Agents + LangGraph)

Traigent optimizes existing functions in place. For router agents, it tunes the handoff to the right specialist and the configs used by each step in a sequential chain.

Flow example: `User -> Router Agent -> Specialized Agent -> Response`

| Architecture           | Support | Notes                                           |
| ---------------------- | ------- | ----------------------------------------------- |
| Single Agent           | Full    | Any LLM-powered function                        |
| RAG                    | Full    | Optimize retriever depth, chunking, and models  |
| Router Agents          | Full    | 1-to-1 handoff to a specialized agent           |
| LangGraph              | Full    | Sequential agent chains                         |
| Multi-Agent (parallel) | Roadmap | Multiple tunables simultaneously                |

---

## Traigent Cloud: Deep Insights and AI Planning

The SDK runs locally. A Traigent Cloud API key unlocks advanced capabilities:

- AI Planner: natural language to agent, benchmark, and measure generation; template configs and draft iteration.
- Advanced Insights: trade-off analysis (up to 4 measures), correlations, trend analysis, robustness scoring, Pareto frontier, log-scale normalization.
- Cost Tracking: per-request cost calculation, budget-aware metrics, cost vs quality recommendations.
- Security and Audit: API key lifecycle (create, rotate, revoke), audit logs, compliance reports, 2FA.
- Rate Limiting and Usage: real-time monitoring, alerts before limits, history, admin overrides.
- Analytics Dashboard: system health monitoring, performance trend sparklines, model performance explorer.

Run locally today with `execution_mode="edge_analytics"`.
Get a Cloud API key at <https://traigent.ai/>.

---

## Quick Reference

```python
@traigent.optimize(
    configuration_space={"model": [...], "temperature": [...]},
    eval_dataset="path/to/data.jsonl",
    objectives=["accuracy", "cost"],
    scoring_function=my_evaluator,  # Optional custom accuracy metric
    execution_mode="edge_analytics",
    cost_limit=2.0,
)
```

```python
results = await agent.optimize(algorithm="bayesian", max_trials=50)
agent.apply_best_config(results)
answer = agent("Your query here")
```

```bash
traigent optimize module.py -a bayesian -n 20
traigent validate path/to/data.jsonl
traigent results
```

---

## Cost Warning

Traigent runs multiple trials = multiple API calls = costs add up.

| Safety Setting           | Command                              |
| ------------------------ | ------------------------------------ |
| Mock Mode (no API calls) | `export TRAIGENT_MOCK_LLM=true`      |
| Cost Limit               | `export TRAIGENT_RUN_COST_LIMIT=2.0` |

See [DISCLAIMER.md](DISCLAIMER.md) for details.

---

## Resources

| Resource                                  | Description                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| [SDK Documentation](docs/README.md)       | Full API reference and guides                                                              |
| [Examples Navigator](examples/index.html) | Serve locally: `python -m http.server -d examples 8000`                                    |
| [Walkthrough](walkthrough/README.md)      | Simple end-to-end tutorial                                                                 |
| [Playground](playground/)                 | Create agents and run optimization (`streamlit run playground/traigent_control_center.py`) |
| [Use Cases](use-cases/)                   | Richer datasets and real-world scenarios                                                   |

---

## License

Apache 2.0 - See [LICENSE](LICENSE)

---

**[Get Started](docs/getting-started/GETTING_STARTED.md)** | **[Examples](examples/)** | **[GitHub Issues](https://github.com/Traigent/Traigent/issues)** | **[Discord](https://discord.gg/traigent)**
