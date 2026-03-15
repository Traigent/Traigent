<h4 align="center">Optimize any LLM agent with one decorator</h4>

<p align="center">
  <a href="https://github.com/Traigent/Traigent/actions/workflows/tests.yml"><img src="https://github.com/Traigent/Traigent/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://docs.traigent.ai"><img src="https://img.shields.io/badge/docs-traigent.ai-brightgreen.svg" alt="Docs"></a>
</p>

Traigent finds the best LLM parameters for your specific task — model, temperature, prompts, RAG settings, and more — by running controlled experiments. Add one decorator to your existing code, and Traigent handles the rest.

> **Runs multiple LLM trials** — use `TRAIGENT_MOCK_LLM=true` to test without spending money, or set `TRAIGENT_RUN_COST_LIMIT=2.0` to cap spend. See [Cost Management](#cost-management).

**Quick Install:**

```bash
git clone https://github.com/Traigent/Traigent.git && cd Traigent
pip install -e ".[recommended]"
```

**Try it now:**

```bash
export TRAIGENT_MOCK_LLM=true
python walkthrough/mock/01_tuning_qa.py
```

<p align="center">
  <a href="https://docs.traigent.ai">Documentation</a> &middot;
  <a href="https://portal.traigent.ai">Portal</a> &middot;
  <a href="docs/getting-started/GETTING_STARTED.md">Quickstart</a> &middot;
  <a href="examples/">Examples</a>
</p>

---

## Choose Your Path

| Goal | Resource | Time |
|------|----------|------|
| **Get started quickly** | [Quick Start Guide](docs/getting-started/GETTING_STARTED.md) | 5 min |
| **Try examples locally** | [Mock walkthrough](walkthrough/mock/) (8 progressive steps) | 15 min |
| **Understand the architecture** | [Architecture Overview](#-architecture-overview) | 5 min |
| **Connect to Traigent Cloud** | [Cloud Setup](#-traigent-cloud) | 5 min |
| **Read the full API reference** | [Decorator Reference →](docs/api-reference/decorator-reference.md) | — |

<details>
<summary>Full documentation index</summary>

| | |
| --- | --- |
| **Get started** | [Installation](docs/getting-started/installation.md) · [5-minute tutorial](docs/getting-started/GETTING_STARTED.md) |
| **User guides** | [Injection Modes](docs/user-guide/injection_modes.md) · [Configuration Spaces](docs/user-guide/configuration-spaces.md) · [Tuned Variables](docs/user-guide/tuned_variables.md) · [Evaluation](docs/user-guide/evaluation_guide.md) |
| **Advanced** | [Agent Optimization](docs/user-guide/agent_optimization.md) · [Optuna Integration](docs/user-guide/optuna_integration.md) · [JS Bridge](docs/guides/js-bridge.md) |
| **API reference** | [Decorator Reference](docs/api-reference/decorator-reference.md) · [Constraint DSL](docs/features/constraint-dsl.md) |

</details>

---

## 🎬 See Traigent in Action

> Click any demo to play the animated version.

| Demo | |
|------|-|
| **LLM Agent Optimization** | [![Optimization Demo](docs/demos/output/optimize-still.svg)](docs/demos/output/optimize.svg) |
| **Optimization Callbacks** | [![Callbacks Demo](docs/demos/output/hooks-still.svg)](docs/demos/output/hooks.svg) |
| **Agent Configuration Hooks** | [![Agent Hooks Demo](docs/demos/output/github-hooks-still.svg)](docs/demos/output/github-hooks.svg) |

---

## 🏗️ Architecture Overview

**How it works:**

1. **Suggest** — the optimizer proposes a configuration to test
2. **Inject** — Traigent overrides your function's parameters with the proposed config
3. **Evaluate** — your function runs against the dataset, scored by the evaluator
4. **Record** — results update the optimizer's model
5. **Repeat** — loop continues until budget/trials exhausted, then outputs results

![Architecture Overview](docs/demos/output/architecture.svg)

**[Read the full architecture guide →](docs/architecture/ARCHITECTURE.md)**

---

## 🚀 Walkthrough — 8 runnable examples

All examples run with `TRAIGENT_MOCK_LLM=true` — no API keys needed.

| # | Run | What you'll learn |
|---|-----|-------------------|
| 1 | `python walkthrough/mock/01_tuning_qa.py` | Basic model + temperature optimization |
| 2 | `python walkthrough/mock/02_zero_code_change.py` | Seamless mode — zero code changes to existing code |
| 3 | `python walkthrough/mock/03_parameter_mode.py` | Explicit config access via `traigent.get_config()` |
| 4 | `python walkthrough/mock/04_multi_objective.py` | Balance accuracy, cost, and latency |
| 5 | `python walkthrough/mock/05_rag_parallel.py` | RAG optimization with parallel evaluation |
| 6 | `python walkthrough/mock/06_custom_evaluator.py` | Define your own success metrics |
| 7 | `python walkthrough/mock/07_multi_provider.py` | Compare OpenAI, Anthropic, Google in one run |
| 8 | `python walkthrough/mock/08_privacy_modes.py` | Local-only privacy-first execution |

**[Browse reference examples →](examples/) · [Injection modes →](docs/user-guide/injection_modes.md)**

---

## 📦 Installation

| Requirement | Supported |
|-------------|-----------|
| **Python** | 3.11, 3.12, 3.13, 3.14 |
| **Platform** | Linux (tested on Ubuntu), macOS, Windows |

```bash
git clone https://github.com/Traigent/Traigent.git && cd Traigent
pip install -e ".[recommended]"
```

> Not on PyPI yet — install from source. Use `uv pip install` for faster installs.

| Feature Set | Description |
|-------------|-------------|
| `[recommended]` | All user-facing features (default) |
| `[integrations]` | LangChain, OpenAI, Anthropic adapters |
| `[analytics]` | Visualization and analytics |
| `[bayesian]` | Bayesian optimization (TPE, NSGA-II) |
| `[all]` | Everything |

**[Full installation guide →](docs/getting-started/installation.md)**

### Cost Management

| Setting | How |
|---------|-----|
| Testing (no API calls) | `TRAIGENT_MOCK_LLM=true` |
| Cost Limit | `TRAIGENT_RUN_COST_LIMIT=2.0` (default: $2/run) |

Cost estimates are approximations. Actual billing is determined by your LLM provider. See [DISCLAIMER.md](DISCLAIMER.md) for details.

### Working with Results

```python
result = await my_agent.optimize(algorithm="random", max_trials=10)
print(result.best_config)  # {'model': 'gpt-4o-mini', 'temperature': 0.1}
print(result.best_score)   # 0.94

my_agent.apply_best_config(result)  # Apply for future calls
```

Results are stored in `.traigent_local/`. Use `traigent results` to list past runs, `traigent plot <name>` to visualize.

---

### ☁️ Traigent Cloud

Connect your SDK to [Traigent Portal](https://portal.traigent.ai) to view optimization results, compare trials, and collaborate with your team.

**1. Create an account**

Sign up at [portal.traigent.ai](https://portal.traigent.ai) — enter your email, name, organization, and a password. Verify your email to activate.

**2. Create an API key**

Once logged in, click your name (top-right) → **API Keys** → **+ Create API Key**. Copy the key — it is shown only once.

**3. Connect the SDK**

Option A — CLI login (recommended for local development):
```bash
traigent auth login
```

Option B — environment variable (recommended for CI/automation):
```bash
export TRAIGENT_API_KEY="sk_..."
```

**4. Run — results appear in the portal automatically**

```bash
python your_optimization.py
```

**Credential priority order:**

| Credential  | 1st (highest)                  | 2nd                    | 3rd (default)        |
|-------------|--------------------------------|------------------------|----------------------|
| API Key     | `TRAIGENT_API_KEY` env var     | Stored CLI credentials | None (local only)    |
| Backend URL | `TRAIGENT_BACKEND_URL` env var | Stored CLI credentials | `portal.traigent.ai` |

> **Tip:** No environment variables needed after `traigent auth login` — the SDK picks up stored credentials automatically.

### Multi-provider optimization

Use [LiteLLM](https://github.com/BerriAI/litellm) to compare models across OpenAI, Anthropic, Google, Mistral, and 100+ providers with a single interface:

```python
@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "claude-3-haiku-20240307", "gemini/gemini-pro"],
        "temperature": [0.1, 0.5, 0.9],
    },
    objectives=["accuracy", "cost"],
    eval_dataset="data/qa_samples.jsonl",
)
def multi_provider_agent(question: str) -> str:
    config = traigent.get_config()
    response = litellm.completion(
        model=config.get("model"),
        temperature=config.get("temperature"),
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content
```

---

## 📏 Evaluation

Provide a JSONL dataset — Traigent scores outputs using semantic similarity by default:

```jsonl
{"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"}
{"input": {"question": "Explain ML"}, "output": "Machine learning uses data and algorithms"}
```

- `input` (required): your function's parameter names as keys
- `output` (optional): expected output for accuracy scoring

**[Evaluation guide →](docs/guides/evaluation.md)** — custom evaluators, dataset formats, troubleshooting

## 🎯 Execution Modes

| Mode | Status | Privacy | Algorithm | Best For |
|------|--------|---------|-----------|----------|
| **Local** (`edge_analytics`) | ✅ Available | ✅ Complete | All (Random/Grid/Bayesian/Optuna) | All use cases |
| **Hybrid** | ✅ Available | ✅ Execution local | All (Random/Grid/Bayesian/Optuna) | Balanced approach |
| **Cloud** | 🚧 Coming Soon | ⚠️ Metadata | Random/Grid/Bayesian | Production, teams |

**[Execution modes guide →](docs/guides/execution-modes.md)** — mode comparisons, privacy details, migration path

### Quick Reference

| Parameter | Where | Description |
|-----------|-------|-------------|
| `configuration_space` | `@traigent.optimize()` | Parameters to test (required) |
| `objectives` | `@traigent.optimize()` | Metrics to optimize for |
| `eval_dataset` | `@traigent.optimize()` | Dataset for evaluation |
| `algorithm` | `.optimize()` call | `"random"`, `"grid"`, `"bayesian"` |
| `max_trials` | `.optimize()` call | Number of configurations to test |

---

## 🎯 Injection Modes

| Mode | Best for | How |
|------|----------|-----|
| **Seamless** (default) | Existing codebases | Traigent intercepts `ChatOpenAI`, `as_retriever`, etc. — zero code changes |
| **Parameter** | New development | Receives `TraigentConfig` object with explicit `config.get("key")` access |

**[Injection modes guide →](docs/user-guide/injection_modes.md)**

---

## 💻 CLI

```bash
traigent optimize module.py -a grid -n 10   # Run optimization
traigent validate data.jsonl -o accuracy     # Validate dataset
traigent results                             # List past runs
traigent plot <name> -p progress             # Visualize results
traigent auth login                          # Authenticate with portal
traigent --help                              # Full command reference
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Zero-code integration** | Add `@traigent.optimize()` to existing code — no refactoring |
| **Multi-algorithm** | Random, Grid, Bayesian (TPE, NSGA-II, CMA-ES) via Optuna |
| **Multi-objective** | Optimize accuracy, latency, cost, and custom metrics simultaneously |
| **Framework support** | LangChain, OpenAI SDK, Anthropic, LiteLLM, and any LLM provider |
| **Cost tracking** | Integrated tokencost library with 500+ model pricing |
| **Parallel execution** | Concurrent trials and example-level parallelism |
| **Privacy-first** | Local execution mode keeps all data on your machine |

**[TraigentDemo →](https://github.com/Traigent/TraigentDemo)** — Streamlit playground, use cases, and research benchmarks

---

## 🛠️ Development

```bash
pip install -e ".[all,dev]"              # Install with dev dependencies
TRAIGENT_MOCK_LLM=true pytest            # Run tests
make format && make lint                 # Format and lint
```

**[Architecture guide →](docs/architecture/ARCHITECTURE.md) · [Project structure →](docs/architecture/project-structure.md)**

## 🤝 Contributing

We welcome bug reports and feature requests via [GitHub Issues](https://github.com/Traigent/Traigent/issues). For security vulnerabilities, please email security@traigent.ai.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | `pip install -e ".[recommended]"` or check venv is activated |
| 0.0% accuracy | Set `TRAIGENT_MOCK_LLM=true`, or check dataset format |
| Missing API keys | Copy `.env.example` to `.env`; or use mock mode |
| Permission errors | Create a fresh venv |

---

## 🌟 Community

- **Discord**: Join our community (coming soon)
- **[GitHub Issues](https://github.com/Traigent/Traigent/issues)**: Report bugs or request features
- **[GitHub Discussions](https://github.com/Traigent/Traigent/discussions)**: Ask questions and share ideas

---

**[Get Started →](docs/getting-started/GETTING_STARTED.md)** | **[Examples →](examples/)** | **[Portal →](https://portal.traigent.ai)**
