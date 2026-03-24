# Traigent

**Traigent is an AI Agent infrastructure that allows companies to take AI agents out of the lab and deploy them at high scale with high confidence.**

**Our mission:** Anything you can measure, we can improve. Whether it's accuracy, speed of response, cost, or any other business metric — we bring strong results that deliver real business value.



<p align="center">
  <a href="https://github.com/Traigent/Traigent/actions/workflows/tests.yml"><img src="https://github.com/Traigent/Traigent/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL-3.0"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://docs.traigent.ai"><img src="https://img.shields.io/badge/docs-traigent.ai-brightgreen.svg" alt="Docs"></a>
</p>

**Traigent is an AI Agent infrastructure that allows companies to take AI agents out of the lab and deploy them at high scale with high confidence.**

Our mission: **Anything you can measure, we can improve.** Whether it's accuracy, speed of response, cost, or any other business metric — we bring strong results that deliver real business value.

> **Runs multiple LLM trials** — use `TRAIGENT_MOCK_LLM=true` to test without spending money, or set `TRAIGENT_RUN_COST_LIMIT=2.0` to cap spend. See [Cost Management](#cost-management).

**Quick Install (`uv` recommended):**

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Installs uv if it is not already available in your shell.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv
source .venv/bin/activate
uv pip install -e ".[recommended]"
```

Prefer `uv` for new environments. For Windows PowerShell steps and a `pip`
fallback, see [Installation details](#installation).

**Try it now — no API keys needed:**

```bash
python hello_world.py
```

**Here's what `hello_world.py` does — one decorator, automatic optimization:**

```python
import traigent
from langchain_openai import ChatOpenAI

DATASET = "examples/datasets/quickstart/qa_samples.jsonl"

@traigent.optimize(
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.0, 0.7, 1.0],
    },
    objectives=["accuracy"],
    eval_dataset=DATASET,
)
def answer(question: str) -> str:
    cfg = traigent.get_config()
    llm = ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    return llm.invoke(question).content
```

---

## Using it in your own code

<p align="center">
  <a href="https://portal.traigent.ai">Portal</a> &middot;
  <a href="docs/getting-started/GETTING_STARTED.md">Quickstart</a> &middot;
  <a href="examples/">Examples</a> &middot;
  <a href="docs/agent-skill.md">Skill</a> &middot;
  <a href="docs/walkthrough.md">Walkthrough</a>
</p>

---

## Choose Your Path

| Goal | Resource | Time |
|------|----------|------|
| **Get started quickly** | [Quick Start Guide](docs/getting-started/GETTING_STARTED.md) | 5 min |
| **Understand the architecture** | [Architecture Overview](#-architecture-overview) | 5 min |
| **Connect to Traigent Cloud** | [Cloud Setup](#-traigent-cloud) | 5 min |
| **Try examples locally, see them on the cloud** | [Mock walkthrough](walkthrough/mock/) (8 steps) → [Portal](https://portal.traigent.ai) | 15 min |
| **Read the full API reference** | [Decorator Reference →](docs/api-reference/decorator-reference.md) | — |

<details>
<summary>Full documentation index</summary>

| | |
| --- | --- |
| **Get started** | [Installation](docs/getting-started/installation.md) · [5-minute tutorial](docs/getting-started/GETTING_STARTED.md) |
| **User guides** | [Injection Modes](docs/user-guide/injection_modes.md) · [Configuration Spaces](docs/user-guide/configuration-spaces.md) · [Evaluation](docs/user-guide/evaluation_guide.md) |
| **Tunable Variable Language** | [TVL Guide](docs/user-guide/tuned_variables.md) |
| **Advanced** | [Agent Optimization](docs/user-guide/agent_optimization.md) · [Optuna Integration](docs/user-guide/optuna_integration.md) · [JS Bridge](docs/guides/js-bridge.md) |
| **API reference** | [Decorator Reference](docs/api-reference/decorator-reference.md) · [Constraint DSL](docs/features/constraint-dsl.md) |

</details>

---

<details>
<summary>🎬 See Traigent in Action — click to play demos</summary>

| Demo | |
|------|-|
| **LLM Agent Optimization** | [![Optimization Demo](docs/demos/output/optimize-still.svg)](docs/demos/output/optimize.svg) |
| **Optimization Callbacks** | [![Callbacks Demo](docs/demos/output/hooks-still.svg)](docs/demos/output/hooks.svg) |
| **Agent Configuration Hooks** | [![Agent Hooks Demo](docs/demos/output/github-hooks-still.svg)](docs/demos/output/github-hooks.svg) |

</details>

<details>
<summary>🏗️ Architecture Overview — how it works</summary>

1. **Suggest** — the optimizer proposes a configuration to test
2. **Inject** — Traigent overrides your function's parameters with the proposed config
3. **Evaluate** — your function runs against the dataset, scored by the evaluator
4. **Record** — results update the optimizer's model
5. **Repeat** — loop continues until budget/trials exhausted, then outputs results

![Architecture Overview](docs/demos/output/architecture.svg)

**[Read the full architecture guide →](docs/architecture/ARCHITECTURE.md)**

</details>

---

## 🚀 Walkthrough — 8 runnable examples

All examples run with `TRAIGENT_MOCK_LLM=true` — no API keys needed.

<details>
<summary>Show all 8 walkthrough steps</summary>

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

</details>

**[Browse reference examples →](examples/) · [Injection modes →](docs/user-guide/injection_modes.md)**

---

### ☁️ Traigent Cloud

Connect to [Traigent Portal](https://portal.traigent.ai) to view results, compare trials, and collaborate.

1. **Sign up** at [portal.traigent.ai](https://portal.traigent.ai) — verify your email to activate
2. **Create an API key** — click your name (top-right) → **API Keys** → **+ Create API Key**
3. **Connect** — run `traigent auth login` or set `export TRAIGENT_API_KEY="sk_..."`  <!-- pragma: allowlist secret -->
4. **Run** — results appear in the portal automatically

<details>
<summary>Credential priority and multi-provider setup</summary>

| Credential  | 1st (highest)                  | 2nd                    | 3rd (default)        |
|-------------|--------------------------------|------------------------|----------------------|
| API Key     | `TRAIGENT_API_KEY` env var     | Stored CLI credentials | None (local only)    |
| Backend URL | `TRAIGENT_BACKEND_URL` env var | Stored CLI credentials | `portal.traigent.ai` |

> **Tip:** No env vars needed after `traigent auth login` — the SDK picks up stored credentials automatically.

**Multi-provider optimization** — use [LiteLLM](https://github.com/BerriAI/litellm) to compare OpenAI, Anthropic, Google, Mistral, and 100+ providers:

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

</details>

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
| **Error resilience** | Interactive pause on rate limits and budget caps — resume or stop gracefully |
| **Live progress** | Auto-enabled progress bar in interactive terminals (`progress_bar=False` to disable) |
| **Privacy-first** | Local execution mode keeps all data on your machine |

**[TraigentDemo →](https://github.com/Traigent/TraigentDemo)** — Streamlit playground, use cases, and research benchmarks

---

<details>
<summary>📦 Installation details, execution modes, CLI, and more</summary>

### Installation

Python 3.11+ on Linux, macOS, or Windows. For coordinated release validation, install from this repository source tree.

| Feature Set | Description |
|-------------|-------------|
| `[recommended]` | All user-facing features (default) |
| `[integrations]` | LangChain, OpenAI, Anthropic adapters |
| `[analytics]` | Visualization and analytics |
| `[bayesian]` | Bayesian optimization (TPE, NSGA-II) |
| `[all]` | Everything |

**[Full installation guide →](docs/getting-started/installation.md)**

Preferred source install with `uv`:

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Installs uv if it is not already available in your shell.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[recommended]"
```

Windows PowerShell with `uv`:

```powershell
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Installs uv if it is not already available in this shell.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  irm https://astral.sh/uv/install.ps1 | iex
}

uv venv .venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[recommended]"
```

Fallback source install with `pip`:

```bash
git clone https://github.com/Traigent/Traigent.git
cd Traigent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[recommended]"
```

### Cost Management

| Setting | How |
|---------|-----|
| Testing (no API calls) | `TRAIGENT_MOCK_LLM=true` |
| Cost Limit | `TRAIGENT_RUN_COST_LIMIT=2.0` (default: $2/run) |

Cost estimates are approximations. See [DISCLAIMER.md](DISCLAIMER.md) for details.

### Evaluation

Provide a JSONL dataset — Traigent scores outputs using semantic similarity by default:

```jsonl
{"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"}
{"input": {"question": "Explain ML"}, "output": "Machine learning uses data and algorithms"}
```

- `input` (required): your function's parameter names as keys
- `output` (optional): expected output for accuracy scoring

**[Evaluation guide →](docs/guides/evaluation.md)** — custom evaluators, dataset formats, troubleshooting

### Execution Modes

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
| `progress_bar` | `.optimize()` call | `True` / `False` / `None` (auto) — live progress bar |

### Injection Modes

| Mode | Best for | How |
|------|----------|-----|
| **Seamless** (default) | Existing codebases | Traigent intercepts `ChatOpenAI`, `as_retriever`, etc. — zero code changes |
| **Parameter** | New development | Receives `TraigentConfig` object with explicit `config.get("key")` access |

**[Injection modes guide →](docs/user-guide/injection_modes.md)**

### CLI

```bash
traigent optimize module.py -a grid -n 10   # Run optimization
traigent validate data.jsonl -o accuracy     # Validate dataset
traigent results                             # List past runs
traigent plot <name> -p progress             # Visualize results
traigent auth login                          # Authenticate with portal
traigent --help                              # Full command reference
```

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | `uv pip install -e ".[recommended]"` (or `pip install -e ".[recommended]"` in a pip-managed venv) |
| 0.0% accuracy | Set `TRAIGENT_MOCK_LLM=true`, or check dataset format |
| Missing API keys | Copy `.env.example` to `.env`; or use mock mode |
| Permission errors | Create a fresh `uv` venv and reinstall dependencies |

</details>

---

## 🛠️ Development

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"           # Install with dev dependencies
TRAIGENT_MOCK_LLM=true pytest            # Run tests
make format && make lint                 # Format and lint
```

**[Architecture guide →](docs/architecture/ARCHITECTURE.md) · [Project structure →](docs/architecture/project-structure.md)**

## 🤝 Contributing

We welcome bug reports and feature requests via [GitHub Issues](https://github.com/Traigent/Traigent/issues). For security vulnerabilities, please email security@traigent.ai.

## 📄 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0-only) - see the [LICENSE](LICENSE) file for details.

---

**[Get Started →](docs/getting-started/GETTING_STARTED.md)** | **[Examples →](examples/)** | **[Portal →](https://portal.traigent.ai)** | **[Skill →](docs/agent-skill.md)** | **[Walkthrough →](docs/walkthrough.md)** | **[GitHub Issues](https://github.com/Traigent/Traigent/issues)** | **[Discussions](https://github.com/Traigent/Traigent/discussions)**
