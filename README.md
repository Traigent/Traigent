# Traigent

**If you can evaluate it, optimize it. Specify evaluation set and eval, and start optimizing today.**

[![Build](https://github.com/Traigent/Traigent/actions/workflows/ci.yml/badge.svg)](https://github.com/Traigent/Traigent/actions)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Traigent adds zero-code-change optimization to existing agents, RAG pipelines, and sequential chains so you can improve accuracy and reduce cost without refactoring or extra dev time. Works with LangChain, LlamaIndex, CrewAI, AutoGen, and direct API calls (OpenAI, Anthropic, etc.).

---

## 🚀 Golden Path: Router Agent + LiteLLM

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

## 📦 Installation

### Add to Your Existing Project (Recommended)

```bash
# From your project virtualenv
pip install -e "/path/to/Traigent[integrations]"

# Or with uv (faster)
uv pip install -e "/path/to/Traigent[integrations]"
```

```python
# Then in your code:
import traigent

@traigent.optimize(...)
def your_agent(...):
    ...  # Your unchanged code
```

Requirements: Python 3.11–3.13 on Linux, macOS, or Windows.

#### Feature Sets

| Feature Set | Description | Use Case |
|-------------|-------------|----------|
| `[core]` | Basic functionality (default) | Minimal install |
| `[analytics]` | Analytics and visualization | View optimization results |
| `[bayesian]` | Bayesian optimization | Advanced optimization algorithms |
| `[integrations]` | Framework integrations | LangChain, OpenAI, Anthropic |
| `[playground]` | Interactive UI | Streamlit control center |
| `[examples]` | Example dependencies | Run all demo scripts |
| `[dev]` | Development tools | pytest, black, ruff, mypy |
| `[all]` | Complete installation | Everything above |

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

## 🏗️ How It Works

Traigent optimizes existing functions in place. For router agents, it tunes the handoff to the right specialist and the configs used by each step in a sequential chain.

Flow example: `User -> Router Agent -> Specialized Agent -> Response`

| Architecture           | Support     | Notes                                          |
| ---------------------- | ----------- | ---------------------------------------------- |
| Single Agent           | Full        | Any LLM-powered function                       |
| RAG                    | Full        | Optimize retriever depth, chunking, and models |
| Router Agents          | Full        | 1-to-1 handoff to a specialized agent          |
| Sequential Chains      | Full        | Multi-step agent pipelines                     |
| Multi-Agent (parallel) | Coming Soon | Multiple tunables simultaneously               |

---

## ☁️ Traigent Cloud (Coming Soon)

The SDK executes your code locally. By default Traigent runs in `execution_mode="edge_analytics"` (local).

`execution_mode="cloud"` and `execution_mode="hybrid"` are reserved for Traigent Cloud integration, but are not yet supported in this build and will raise `NotYetSupported` when optimization runs.

To run fully local (no Traigent backend communication), set `TRAIGENT_OFFLINE_MODE=true`.

The Traigent Cloud roadmap includes:

- AI Planner: natural language to agent, benchmark, and measure generation; template configs and draft iteration.
- Advanced Insights: trade-off analysis (up to 4 measures), correlations, trend analysis, robustness scoring, Pareto frontier, log-scale normalization.
- Cost Tracking: per-request cost calculation, budget-aware metrics, cost vs quality recommendations.
- Security and Audit: API key lifecycle (create, rotate, revoke), audit logs, compliance reports, 2FA.
- Rate Limiting and Usage: real-time monitoring, alerts before limits, history, admin overrides.
- Analytics Dashboard: system health monitoring, performance trend sparklines, model performance explorer.

Get updates (and a Cloud API key when available) at <https://traigent.ai/>.

---

## ⚡ Quick Reference

```python
@traigent.optimize(
    configuration_space={"model": [...], "temperature": [...]},
    eval_dataset="path/to/data.jsonl",
    objectives=["accuracy", "cost"],
    scoring_function=my_evaluator,  # Optional custom accuracy metric
    # execution_mode defaults to "edge_analytics" (local)
    # execution_mode="cloud",  # Not yet supported (raises NotYetSupported)
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

## ⚠️ Cost Warning

Traigent runs multiple trials = multiple API calls = costs add up.

| Safety Setting           | Command                              |
| ------------------------ | ------------------------------------ |
| Mock Mode (no API calls) | `export TRAIGENT_MOCK_LLM=true`      |
| Cost Limit               | `export TRAIGENT_RUN_COST_LIMIT=2.0` |

See [DISCLAIMER.md](DISCLAIMER.md) for details.

---

## 📚 Resources

| Resource                                  | Description                                                                                |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| [SDK Documentation](docs/README.md)       | Full API reference and guides                                                              |
| [Examples Navigator](examples/index.html) | Serve locally: `python -m http.server -d examples 8000`                                    |
| [Walkthrough](walkthrough/README.md)      | Simple end-to-end tutorial                                                                 |
| [Playground](playground/)                 | Create agents and run optimization (`streamlit run playground/traigent_control_center.py`) |
| [Use Cases](use-cases/)                   | Richer datasets and real-world scenarios                                                   |

---

## 🌐 Execution Modes

Traigent executes your code locally in this build; `execution_mode="cloud"` and `execution_mode="hybrid"` are reserved for Traigent Cloud and are not yet supported.

| Mode | Notes |
| ---- | ----- |
| **Local** (`edge_analytics`, default) | Runs locally; set `TRAIGENT_OFFLINE_MODE=true` to disable backend communication |
| **Hybrid** (`hybrid`) | Not yet supported (raises `NotYetSupported`) |
| **Cloud** (`cloud`) | Not yet supported (raises `NotYetSupported`) |

---

## 🎯 Quick Capability Matrix

| Feature | What It Does | Why It's Amazing |
| ------- | ------------ | ---------------- |
| Seamless Injection | Override params without code changes | Zero migration effort |
| Budget Rails | Real-time cost enforcement with approval | Never exceed $2 by accident |
| Multi-Agent Tuning | Tune entire pipelines together | 1 run vs N sequential runs |
| TVL Specs | Declarative optimization intent | Version control your optimization strategy |
| RAGAS Integration | RAG-specific metrics built-in | Faithfulness, relevance, precision |
| Parallel Batching | Cost-aware concurrent execution | Smart budget distribution |

---

## ✨ Unique Capabilities

Beyond the basics, Traigent includes powerful features such as:

| Capability | Description |
| ---------- | ----------- |
| **Injection Modes (default: context)** | Context (default), Parameter, Attribute, Seamless for different trade-offs. See [docs/user-guide/injection_modes.md](docs/user-guide/injection_modes.md). |
| **3-Tier Evaluation** | Exact match (default) → Custom scorers → LLM-as-Judge/RAGAS |
| **Budget Rails** | `TRAIGENT_RUN_COST_LIMIT` + handshake approval + EMA estimation |
| **2-Level Parallelism** | `example_concurrency` (per trial) + `trial_concurrency` (simultaneous configs) |
| **Multi-Objective Aggregation** | WEIGHTED_SUM, HARMONIC, CHEBYSHEV + BANDED objectives with TOST testing |
| **TVL (Traigent Validation Language)** | YAML specs for objectives, constraints, budgets - optimizer-agnostic |
| **Constraint DSL** | Functional, operator, or fluent syntax for config constraints |
| **Smart Pruning** | Median/Percentile/Threshold/Timeout pruners + adaptive early stopping |
| **Sample Budget Leasing** | `max_total_examples` caps evaluations = 60-80% cost reduction |
| **100+ LLM Providers** | Via LiteLLM - optimize across providers in a single run |

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)

---

**[Get Started](docs/getting-started/GETTING_STARTED.md)** | **[Examples](examples/)** | **[GitHub Issues](https://github.com/Traigent/Traigent/issues)** | **[Discord](https://discord.gg/traigent)**

---

**Current Version**: 0.9.0 (Beta)
