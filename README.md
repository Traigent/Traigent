# ✨ TraiGent: Find the Perfect AI Parameters for Your Task - Zero Code Changes Required!

**Current Version**: 0.8.0 (Beta)

---

## Cost Warning

TraiGent optimizes LLM applications by running multiple trials across configurations.
**This can result in significant API costs.**

| Recommendation      | How                                                         |
| ------------------- | ----------------------------------------------------------- |
| Development/Testing | Use `TRAIGENT_MOCK_MODE=true`                               |
| Control Spending    | Set `TRAIGENT_RUN_COST_LIMIT=2.0` (default: $2 USD per run) |
| Before Production   | Review the [DISCLAIMER.md](DISCLAIMER.md)                   |

**Important**: Cost estimates are approximations. Actual billing is determined by your LLM provider.

---

Start with the curated experiments in `examples/`—each scenario ships with a README plus ready-to-run commands (including the required `export` statements) so you can iterate locally without guessing the setup.

> 💡 **Local Playground**: Run the interactive Streamlit control center locally with `streamlit run playground/traigent_control_center.py`. Also explore the `examples/` directory for end-to-end flows.

> **Note**: Research papers and experimental code have been moved to a separate repository:
> [Traigent-Experiments](https://github.com/Traigent/Traigent-Experiments)

## 🎬 See Traigent in Action

### LLM Agent Optimization

![Traigent Optimization Demo](docs/demos/output/optimize.svg)

### Optimization Callbacks

![Traigent Callbacks Demo](docs/demos/output/hooks.svg)

### Agent Configuration Hooks

![Traigent Agent Hooks Demo](docs/demos/output/github-hooks.svg)

## 🚀 Quick Example: See Tuned Variables in Action

> **Want to run this now?** First [install TraiGent](#-quick-installation), then use the ready-to-run quickstart examples (no API keys needed):
>
> ```bash
> export TRAIGENT_MOCK_MODE=true
> python examples/quickstart/01_simple_qa.py
> ```
>
> The `examples/quickstart/` directory contains runnable versions that work without API keys.

```python
import traigent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from traigent.api.decorators import EvaluationOptions, ExecutionOptions

# Load environment variables (API keys, etc.)
load_dotenv()

@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],  # 🎯 Adaptive Variable #1
        "temperature": [0.1, 0.5, 0.9]                         # 🎯 Adaptive Variable #2
    },
    objectives=["accuracy", "cost"],    # What to optimize for
    # Dataset file path (relative to examples/datasets/quickstart/)
    evaluation=EvaluationOptions(eval_dataset="examples/datasets/quickstart/qa_samples.jsonl"),
    execution=ExecutionOptions(execution_mode="edge_analytics"),
)
def simple_qa_agent(question: str) -> str:
    """Simple Q&A agent with Tuned Variables"""

    # These values will be automatically optimized by TraiGent!
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",     # 🎯 TraiGent tests: gpt-3.5-turbo, gpt-4o-mini, gpt-4o
        temperature=0.7            # 🎯 TraiGent tests: 0.1, 0.5, 0.9
    )

    # Normal LLM invocation - TraiGent intercepts and optimizes
    response = llm.invoke(f"Question: {question}\nAnswer:")
    print(f"🔧 Using: model={llm.model_name}, temp={llm.temperature}")
    return response.content

# That's it! TraiGent will find the best model & temperature for YOUR specific use case
```

## 📊 Full Customer Support Example with RAG

```python
import asyncio
import traigent
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
from traigent.api.decorators import EvaluationOptions

# Load environment variables (API keys, etc.)
load_dotenv()

# Define your knowledge base (used during optimization and inference)
KNOWLEDGE_BASE = [
    "Returns accepted within 30 days with original receipt",
    "Free shipping on orders over $50",
    "Contact support@example.com for order issues",
]

@traigent.optimize(
    configuration_space={
        "model": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "k": [3, 5, 10]  # RAG retrieval depth
    },
    evaluation=EvaluationOptions(eval_dataset="rag_feedback.jsonl")  # Provide your dataset
)
def customer_support_agent(query: str, knowledge_base: list = KNOWLEDGE_BASE) -> str:
    """Answer customer questions using RAG"""

    # Your existing code - TraiGent optimizes these automatically!
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",     # Current: gpt-3.5-turbo
        temperature=0.7            # Current: 0.7
    )

    vectorstore = Chroma.from_texts(knowledge_base)
    docs = vectorstore.similarity_search(query, k=5)  # Current: k=5
    context = "\n".join([doc.page_content for doc in docs])

    response = llm.invoke(f"Context: {context}\nQuestion: {query}\nAnswer:")
    print(f"🔧 Using: {llm.model_name}, temp={llm.temperature}, k={len(docs)}")
    return response.content

# Step 1: Find optimal configuration
# Note: algorithm and max_trials are passed to .optimize(), not the decorator
results = asyncio.run(customer_support_agent.optimize(
    algorithm="random",  # Options: "random", "grid", "bayesian"
    max_trials=20        # Number of configurations to test
))

# Step 2: Apply best configuration
customer_support_agent.apply_best_config(results)

# Step 3: Use optimized agent (uses default KNOWLEDGE_BASE, or pass custom)
answer = customer_support_agent("What's your return policy?")
# 🔧 Using: gpt-4o-mini, temp=0.1, k=3  # ← Shows optimized parameters!

# Step 4: View optimization results
print(f"Best config: {results.best_config}")
print(f"Best score: {results.best_score}")
# Output:
# Best config: {'model': 'gpt-4o-mini', 'temperature': 0.1, 'k': 3}
# Best score: 0.94
```

### Need custom weights or minimize a different metric?

Lists like `["accuracy", "cost"]` are fine for most runs—TraiGent automatically infers sensible orientations and equal weights. When you want explicit control, provide an `ObjectiveSchema`:

```python
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

custom_objectives = ObjectiveSchema.from_objectives(
    [
        ObjectiveDefinition("accuracy", orientation="maximize", weight=0.7),
        ObjectiveDefinition("cost", orientation="minimize", weight=0.3),
    ]
)

@traigent.optimize(
    objectives=custom_objectives,
    configuration_space={
        # Use tuples for continuous ranges, lists for categorical values
        "temperature": (0.0, 1.0),  # Continuous range
        "top_p": (0.1, 1.0),        # Continuous range
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"],  # Categorical
    },
    eval_dataset="qa_samples.jsonl",
)
def weighted_agent(question: str) -> str:
    ...
```

> **Tip**: See `examples/quickstart/03_custom_objectives.py` for a complete working example.

### Injection modes & default values

TraiGent can inject parameters in two ways:

- **Seamless (default)**: your original literals remain in place until a trial overrides them. Provide `default_config={"temperature": 0.3}` if you want a different starting point for the first trial or a new value for `reset()`.
- **Parameter mode** (`injection_mode="parameter"`): TraiGent passes a `TraigentConfig` (built from your `default_config`) into the parameter you nominate (e.g. `config`). Access values with `config.get("foo", fallback)` so missing keys fall back cleanly when the default config is empty or partial.

**↗️ Try TraiGent now - see the results above in under 5 minutes!**

### TVL Specs: The Foundation Layer

TVL (TraiGent Validation Language) defines the _what_—constraints, objectives, and boundaries—while leaving the _how_ to any compatible optimizer. The power is in the specification, not the implementation.

```python
@traigent.optimize(tvl_spec="docs/tvl/tvl-website/client/public/examples/ch1_motivation_experiment.tvl.yml")
def rag_agent(query: str) -> str:
    ...
```

TVL sections control the configuration space, objectives, constraints, and budgets—no
extra arguments required. The CLI also accepts `traigent optimize ... --tvl-spec path`
and an optional `--tvl-environment staging` flag.

> 💡 **Why specifications matter**: A TVL spec can be validated by any conformant tool—TraiGent today, your internal optimizer tomorrow. The foundation is the contract, not the implementation.

## 📦 Quick Installation

**Requirements**: Python 3.11 or higher

Get started with TraiGent in under 2 minutes:

### Method 1: From Source (Recommended for Examples)

#### Using pip (Traditional)

```bash
# Clone the repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment (requires Python 3.11+)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install TraiGent in development mode with all dependencies
pip install -e ".[dev,integrations,analytics]"

# Or install everything
pip install -e ".[all]"

# Verify installation
python -c "import traigent; print('✅ TraiGent installed successfully')"
```

#### Using uv (Faster - Recommended)

> **First time?** Install uv: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
# Clone the repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install TraiGent in development mode (10-100x faster!)
uv pip install -e ".[dev,integrations,analytics]"

# Or install everything
uv pip install -e ".[all]"

# Verify installation
python -c "import traigent; print('✅ TraiGent installed successfully')"
```

**Why use uv?**

- ⚡ 10-100x faster dependency resolution
- 🎯 Drop-in replacement for pip
- 📦 Works with existing `pyproject.toml`
- 🔒 More reliable dependency resolution

### Method 2: From PyPI (Coming Soon)

> ⚠️ **Note**: TraiGent is not yet published to PyPI. Please use **Method 1 (From Source)** above for now.

<!--
Once available on PyPI, you can install with:
```bash
pip install traigent
# or
uv pip install traigent
```
-->

### Environment Configuration

#### Mock Mode

For local development and testing without API keys:

```bash
export TRAIGENT_MOCK_MODE=true
```

**What mock mode does:**

- Returns simulated responses from your decorated functions (no real LLM calls)
- Skips TraiGent backend/cloud connections
- Generates realistic mock metrics (accuracy, cost, latency) for testing

**What mock mode does NOT do:**

- It does not disable the optimization loop itself—trials still run, configs are still tested
- It does not affect compute resources or parallelism

#### Restricted Environments

If running in containers, CI, or environments with limited permissions, you may see errors or warnings. Set these variables as needed:

```bash
export TRAIGENT_RESULTS_FOLDER=./results   # If home directory isn't writable
export LOKY_MAX_CPU_COUNT=1                # If you see joblib/semaphore permission errors
```

#### Local Storage

Optimization results are stored in `.traigent_local/` in your working directory (or customize with `local_storage_path` parameter). Logs go to `TRAIGENT_RESULTS_FOLDER` (defaults to `~/.traigent`).

<!-- Backend configuration (for TraiGent Cloud users - coming soon)
export TRAIGENT_API_URL=http://localhost:5000/api/v1
export TRAIGENT_BACKEND_URL=http://localhost:5000
export TRAIGENT_API_KEY=<api key issued in the TraiGent app>
-->

### Available Feature Sets

When installing TraiGent, you can choose specific feature sets:

| Feature Set      | Description                   | Use Case                         |
| ---------------- | ----------------------------- | -------------------------------- |
| `[core]`         | Basic functionality (default) | Minimal install                  |
| `[analytics]`    | Analytics and visualization   | View optimization results        |
| `[bayesian]`     | Bayesian optimization         | Advanced optimization algorithms |
| `[integrations]` | Framework integrations        | LangChain, OpenAI, Anthropic     |
| `[playground]`   | Interactive UI                | Streamlit control center         |
| `[examples]`     | Example dependencies          | Run all demo scripts             |
| `[dev]`          | Development tools             | pytest, black, ruff, mypy        |
| `[all]`          | Complete installation         | Everything above                 |

**Recommended installs:**

```bash
# For running examples and development
pip install -e ".[dev,integrations,analytics]"

# For the Streamlit playground UI
pip install -e ".[playground]"

# For everything (largest install)
pip install -e ".[all]"
```

### Next Steps

1. **Try the quickstart examples** (recommended first):

   ```bash
   export TRAIGENT_MOCK_MODE=true
   python examples/quickstart/01_simple_qa.py
   python examples/quickstart/02_customer_support_rag.py
   python examples/quickstart/03_custom_objectives.py
   ```

   > 💡 If you see `joblib will operate in serial mode` warnings, that's harmless—see [Restricted Environments](#restricted-environments) to suppress them.

2. **Run the curated walkthroughs**: Explore `examples/core/simple-prompt/run.py` and other examples (each README shows the `export` commands to copy)

3. **Set up API keys** (optional): Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

4. **Deep dive**: Start with `examples/README.md` and `examples/docs/EXAMPLES_GUIDE.md` for experiment-specific instructions

> **Note**: `TRAIGENT_MOCK_MODE=true` runs examples without real API calls. The quickstart commands above include this export.

---

## 🎮 Interactive UI - Get Started in Minutes

> 💡 **Local Playground**: Launch the interactive Streamlit control center locally:
>
> ```bash
> # Install playground dependencies first (if not already installed)
> pip install -e ".[playground]"
>
> # Then launch the control center
> streamlit run playground/traigent_control_center.py
> ```

The TraiGent Control Center provides a user-friendly interface to:

- Define problems using natural language
- Test and compare different AI agents
- Visualize performance metrics
- Export optimal configurations

## 📏 Evaluation

TraiGent evaluates your AI agent's performance by comparing outputs to expected results using semantic similarity, custom evaluators, or mock mode for testing.

**Quick Start:**

```python
# Simple evaluation with default semantic similarity
@traigent.optimize(
    configuration_space={
        "temperature": [0.1, 0.5, 0.9],
        "model": ["gpt-3.5-turbo", "gpt-4o-mini"]
    },
    eval_dataset="qa_samples.jsonl",  # JSONL format
    objectives=["accuracy", "cost"]
)
def my_agent(question: str) -> str:
    return process_question(question)
```

**Dataset Format (JSONL):** Each line must be valid JSON with these fields:

```jsonl
{"input": {"question": "What is AI?"}, "output": "Artificial Intelligence"}
{"input": {"question": "Explain ML"}, "output": "Machine learning uses data and algorithms"}
```

- `input` (required): Dictionary with your function's parameter names as keys
- `output` (optional): Expected output for accuracy evaluation

**Learn More:** See the [Evaluation Guide](docs/guides/evaluation.md) for:

- Dataset formats and creation
- Custom evaluator patterns (RAG, classification, code generation)
- Troubleshooting low accuracy
- Mock mode testing
- Best practices

## 🎯 Execution Modes

TraiGent supports local execution with cloud modes planned:

| Mode                         | Status         | Privacy            | Algorithm            | Best For          |
| ---------------------------- | -------------- | ------------------ | -------------------- | ----------------- |
| **Local** (`edge_analytics`) | ✅ Available   | ✅ Complete        | Random/Grid/Bayesian | All use cases     |
| **Cloud**                    | 🚧 Coming Soon | ⚠️ Metadata        | Bayesian             | Production, teams |
| **Hybrid**                   | 🚧 Coming Soon | ✅ Execution local | Bayesian             | Balanced approach |

**Quick Start - Local Mode (Recommended):**

```python
@traigent.optimize(
    execution_mode="edge_analytics",  # Full privacy, works today
    local_storage_path="./my_optimizations"
)
def my_agent(query: str) -> str:
    return process_query(query)
```

> **Local Storage**: When using `edge_analytics` mode, TraiGent creates a `.traigent_local/` directory in your project root to store optimization state, trial results, and configuration data. This directory is automatically created on first run and can be safely deleted to reset optimization state. You can customize the location using the `local_storage_path` parameter.

**Learn More:** See the [Execution Modes Guide](docs/guides/execution-modes.md) for:

- Detailed mode comparisons and feature matrices
- Privacy-safe analytics and what data is tracked
- Intelligent upgrade recommendations
- Migration path from local to cloud
- Security best practices

### ✨ Zero-Code Integration - Keep Your Code Unchanged!

TraiGent works with your existing code through a simple decorator. Here's how the example above works step by step:

**🎯 The Magic: Parameter Interception**

- TraiGent automatically detects `ChatOpenAI()` and `similarity_search()` calls
- During optimization, it overrides your hardcoded values with test configurations
- Your original code stays exactly the same - no refactoring needed!

**📊 Optimization Results You'll See:**

```bash
🔧 Trial 1/20: gpt-3.5-turbo, temp=0.7, k=5 → 81% accuracy, $0.15/1K
🔧 Trial 5/20: gpt-4o-mini, temp=0.1, k=3 → 94% accuracy, $0.12/1K
🔧 Trial 12/20: gpt-4o, temp=0.1, k=3 → 97% accuracy, $0.48/1K
💡 Best configuration found: gpt-4o-mini, temp=0.1, k=3
```

**🚀 Business Impact:**

- **15% accuracy improvement** (81% → 94%)
- **20% cost reduction** ($0.15 → $0.12 per 1K queries)
- **Zero development time** - just add a decorator

**⚙️ Optimization Parameters:**

| Parameter | Where | Description |
|-----------|-------|-------------|
| `configuration_space` | `@traigent.optimize()` decorator | Define what parameters to test |
| `objectives` | `@traigent.optimize()` decorator | Metrics to optimize for |
| `eval_dataset` | `@traigent.optimize()` decorator | Dataset for evaluation |
| `algorithm` | `.optimize()` method call | Search algorithm: `"random"`, `"grid"`, `"bayesian"` |
| `max_trials` | `.optimize()` method call | Number of configurations to test |

```python
# Decorator defines WHAT to optimize
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.1, 0.9]},
    objectives=["accuracy", "cost"],
    eval_dataset="data.jsonl"
)
def my_agent(query: str) -> str:
    ...

# Method call defines HOW to optimize
results = await my_agent.optimize(
    algorithm="random",  # Search strategy
    max_trials=20        # Number of trials
)
```

### 🧠 Tuned Variables: The Core Concept

**Tuned Variables** are configuration parameters whose optimal values dynamically change based on:

1. **Objective shifts**: Changes in optimization priorities (e.g., prioritizing cost reduction vs. accuracy)
2. **Environmental changes**: New model availability, data distribution shifts, or context evolution

Unlike static configuration parameters (like API endpoints or credentials), **Tuned Variables** directly influence agent behavior and require continuous optimization:

#### Examples of Tuned Variables:

- **Model Selection**: `gpt-3.5-turbo` → `gpt-4o` based on accuracy/cost trade-offs
- **Temperature**: `0.1` (factual) → `0.9` (creative) based on task requirements
- **Retrieval Depth**: `k=3` (fast) → `k=10` (comprehensive) based on precision needs
- **Output Format**: `json` vs `text` based on downstream processing requirements

#### Static vs Tuned Variables:

```python
# ❌ Static variables (don't optimize these)
database_url = "postgresql://..."
api_key = "sk-..."

# ✅ Tuned Variables (optimize these for agent performance)
model = "gpt-4o-mini"        # Cost vs accuracy trade-off
temperature = 0.3            # Creativity vs consistency
k = 5                        # Retrieval depth vs speed
format = "json"              # Structured vs natural output
```

### 🎯 TraiGent's Two-Mode Strategy: The Best of Both Worlds

TraiGent offers **two powerful modes** designed specifically for software engineers working with AI agents. Unlike general optimization libraries, TraiGent understands agent patterns and can automatically optimize LLM calls, retrieval parameters, and agent logic.

## 🎯 Configuration Injection Modes

TraiGent supports two ways to inject optimized parameters into your code:

### Seamless Mode (Default) - Zero Code Changes

Perfect for optimizing existing agents without refactoring:

```python
@traigent.optimize(
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.1, 0.9]}
)
def my_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Auto-optimized!
    return llm.invoke(query).content
```

TraiGent automatically intercepts framework calls (`ChatOpenAI`, `as_retriever`, etc.) and injects optimized values.

### Parameter Mode - Explicit Control

For new development with full type safety:

```python
from traigent import TraigentConfig

@traigent.optimize(
    injection_mode="parameter",
    configuration_space={"model": ["gpt-4o-mini"], "k": [3, 5, 10]}
)
def my_agent(query: str, config: TraigentConfig) -> str:
    llm = ChatOpenAI(model=config.get("model"))  # Explicit access
    return llm.invoke(query).content
```

**Which to use?**

- **Seamless**: Existing codebases, rapid adoption, zero migration
- **Parameter**: New development, type safety, complex logic

## 🌟 Natural Language Problem Definition

### Define Problems with AI Understanding

```python
# Use the UI or programmatically define problems
from playground.problem_management import SmartProblemAnalyzer

analyzer = SmartProblemAnalyzer()
result = await analyzer.analyze_and_generate(
    description="I need to classify customer emails by department",
    count=50  # AI generates 50 test examples
)

# AI automatically:
# - Detects this is a classification problem
# - Generates relevant test examples
# - Suggests appropriate metrics
# - Creates optimal prompt templates
```

### 🔍 Smart Agent Discovery Process

1. **Describe Your Problem**: "I need to analyze customer feedback sentiment"
2. **AI Generates Test Cases**: Automatically creates diverse examples
3. **Test Multiple Agents**: Compare GPT-3.5, GPT-4, Claude, etc.
4. **See Clear Results**: Visual comparison of accuracy, cost, and speed
5. **Export Best Config**: Use the optimal settings in production

## 💻 CLI Commands

The CLI provides local optimization, validation, results management, and template generation:

```bash
# Help and version info
traigent --help
traigent --version   # Quick version check
traigent info        # Detailed version, features, and integrations

# Quiet mode (suppress logs) or verbose mode
traigent --quiet info    # Errors only
traigent --verbose info  # Full logging

# Algorithms
traigent algorithms

# Optimize decorated functions in a Python file
# Note: The module must have @traigent.optimize decorated functions with
# configuration_space and eval_dataset defined
traigent optimize path/to/module.py -a grid -n 10

# Validate dataset and configuration files
traigent validate path/to/dataset.jsonl -o accuracy -o cost
traigent validate_config config.json

# Manage and visualize results
traigent results
traigent plot <result_name> -p progress

# Generate example templates
traigent generate -t basic -o traigent_example.py

# Verify optimization improves over defaults
traigent check path/to/module.py --threshold=10
```

## 📊 Real Results from Real Users

```python
# Before TraiGent: Guessing at configurations
llm = ChatOpenAI(model="gpt-4", temperature=0.7)  # Expensive and maybe not optimal

# After TraiGent: Data-driven decisions
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  # 95% accuracy at 10x less cost!
```

**Typical improvements:**

- 💰 **Cost Reduction**: 60-90% lower costs
- 🎯 **Accuracy Gains**: 5-15% better performance
- ⏱️ **Speed Boost**: 2-5x faster responses
- 🤖 **Model Discovery**: Find hidden gems like Claude Haiku
- 📈 **Usage Insights**: Understand your optimization patterns with privacy-safe analytics

## ✨ Key Features

### 🚀 **Zero-Code-Change Integration**

- **🎯 Works with Your Code**: Automatically optimizes LangChain, OpenAI SDK, and more
- **🔄 No Modifications Needed**: Your existing code stays exactly the same
- **🧠 Smart Testing**: Intelligently explores different models and parameters
- **⚡ Simple Decorator**: Just add `@traigent.optimize()` - that's it!

### 🎮 **User-Friendly Interface**

- **📝 Natural Language**: Describe problems in plain English
- **🤖 AI Understanding**: Automatic problem classification and example generation
- **📊 Visual Results**: Clear charts comparing agent performance
- **📤 Easy Export**: One-click configuration export

### 📦 **Production Ready**

- **🎨 Multiple Optimization Algorithms**: Grid search, Random search, and Bayesian optimization
- **📊 Multi-Objective Optimization**: Optimize for accuracy, latency, cost, and custom metrics
- **💰 Accurate Cost Tracking**: Integrated tokencost library supports 500+ models with real-time pricing
- **🔗 Framework Support**: LangChain, OpenAI SDK, Anthropic, and more
- **🤝 Dual Execution Models**: Privacy-first local or cloud-powered testing
- **🎮 Interactive UI**: User-friendly Streamlit Playground
- **⚙️ Smart Testing**: Automatic parameter exploration and comparison
- **⚡ Real-Time Progress**: Watch as different agents are tested
- **🔌 Extensible Design**: Add custom models and evaluation metrics
- **📝 Problem Templates**: 9 standardized AI problem types
- **🧪 Production-Ready**: Battle-tested with comprehensive logging

### 🛠️ **Intelligent Features**

- **🧠 Natural Language Understanding**: Describe problems in plain English
- **🎭 Claude SDK Integration**: Smart problem analysis and classification
- **🔄 Cost Optimization**: Smart sampling reduces costs by 60-80%
- **🌐 Comprehensive Testing**: Compare accuracy, cost, and speed
- **🏗️ Platform Agnostic**: Works with any LLM provider
- **💰 Transparent Pricing**: See exactly what each agent costs
- **🚀 Parallel Testing**: Test multiple configurations simultaneously
  - Example-level concurrency: set `parallel_config={"example_concurrency": 8}` in `@traigent.optimize(...)` or override per run
  - Trial-level concurrency: set `parallel_config={"trial_concurrency": 4}` (e.g., `.optimize(parallel_config={...})`) to evaluate multiple configs at once

#### Parallel Execution Examples

```python
import traigent
from traigent.evaluators.base import Dataset, EvaluationExample

# Small dataset
ds = Dataset(examples=[
    EvaluationExample(input_data={"x": i}, expected_output=f"val-{i}") for i in range(8)
])

@traigent.optimize(
    eval_dataset=ds,
    objectives=["accuracy"],
    configuration_space={"p": [1, 2, 3, 4]},
    execution_mode="edge_analytics",
    parallel_config={"example_concurrency": 4, "trial_concurrency": 2},
)
def fn(x: int) -> str:
    import time; time.sleep(0.1)  # Simulate work
    return f"val-{x}"

# Override at call-site if needed
# results = await fn.optimize(
#     parallel_config={"example_concurrency": 8, "trial_concurrency": 4}
# )
```

#### Hybrid + Privacy

```python
@traigent.optimize(
    eval_dataset=ds,
    objectives=["accuracy"],
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"]},
    execution_mode="hybrid",
    privacy_enabled=True   # Never transmit input/output/prompts
)
def agent(x: int) -> str:
    return f"val-{x}"
```

- **🔐 Privacy Options**: Keep sensitive data on your servers

### 💰 **Cost Tracking & Optimization**

TraiGent includes professional-grade cost tracking powered by the **tokencost** library:

- **500+ Models Supported**: OpenAI, Anthropic, Google, Cohere, Mistral, and more
- **Real-Time Pricing**: Always up-to-date pricing information
- **Automatic Updates**: No manual pricing table maintenance needed
- **Detailed Breakdown**: Track input tokens, output tokens, and total costs
- **Multi-Provider**: Compare costs across different LLM providers
- **Cost Objectives**: Optimize for cost alongside accuracy and performance

```python
# Cost information is automatically tracked during optimization
results = await my_agent.optimize()
print(f"Total optimization cost: ${results.total_cost:.4f}")
print(f"Best configuration cost per call: ${results.best_config_cost:.6f}")
```

- **📊 Privacy-Safe Analytics**: Track optimization patterns with zero sensitive data
- **🎯 Smart Insights**: Get personalized upgrade recommendations based on usage
- **🏃‍♂️ Gradual Migration**: Start local, upgrade selectively based on real value

## 🎓 Quick Examples

### 🎮 Using the Interactive UI

1. **Launch the Control Center:**

   ```bash
   streamlit run playground/traigent_control_center.py
   ```

2. **Define Your Problem:**

   - Click "Problem Manager" → "Define Your Problem"
   - Type: "I need to summarize long documents"
   - AI generates test cases automatically

3. **Find the Best Agent:**
   - Click "Explore Agents"
   - Select models to compare (GPT-3.5, GPT-4, etc.)
   - Click "Find Best Agent"
   - See results in real-time!

### 💻 Programmatic Usage

```python
# Your proprietary function stays local
def analyze_customer_data(customer_id: str, query: str) -> str:
    # Sensitive business logic here
    customer = load_customer_data(customer_id)  # Private data
    return generate_response(customer, query)

# Optimize with cloud guidance, execute locally
@traigent.optimize(
    eval_dataset="customer_queries.jsonl",
    objectives=["accuracy", "response_time"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": (0.0, 0.5),
        "response_style": ["concise", "detailed", "friendly"]
    },
    execution_mode="interactive"  # Model 1
)
def optimized_analyzer(customer_id: str, query: str) -> str:
    config = traigent.get_trial_config()  # Only valid during optimization
    # Your logic with optimized parameters
    return analyze_customer_data(customer_id, query)

# Run optimization and reuse the best settings later
# result = await optimized_analyzer.optimize()
# print(result.best_config)                 # Best trial config
# print(optimized_analyzer.current_config)  # Applied to future calls
```

**Config access: during vs. after**

| When you're running             | Use this                      | Notes                                               |
| ------------------------------- | ----------------------------- | --------------------------------------------------- |
| During optimization             | `traigent.get_trial_config()` | Raises `OptimizationStateError` if no active trial. |
| After optimization completes    | `result.best_config`          | Returned by `func.optimize()`.                      |
| When calling the function later | `func.current_config`         | Automatically set to the best config.               |

### ☁️ Model 2: Cloud-Based Agent Optimization (Coming Soon)

> **Note**: Cloud optimization is under development. The API below shows the planned interface.

```python
from traigent.cloud.models import AgentSpecification

# Define agent for cloud optimization
support_agent = AgentSpecification(
    id="support-bot-v2",
    name="Customer Support Bot",
    agent_type="conversational",
    agent_platform="openai",
    prompt_template="""You are an expert support agent.

    Customer: {customer_query}
    History: {conversation_history}

    Provide a helpful, empathetic response.""",
    model_parameters={
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 200
    },
    guidelines=[
        "Be empathetic and understanding",
        "Provide actionable solutions",
        "Escalate complex issues"
    ]
)

# Optimize in the cloud
async def optimize_support_agent():
    async with TraiGentCloudClient(api_key="your-key") as client:
        response = await client.optimize_agent(
            agent_spec=support_agent,
            dataset=support_conversations,
            configuration_space={
                "model": ["gpt-4o-mini", "gpt-4o"],
                "temperature": (0.3, 0.9),
                "max_tokens": [150, 250, 350]
            },
            objectives=["customer_satisfaction", "resolution_rate", "cost"]
        )

        # Cloud handles everything
        print(f"Optimization started: {response.optimization_id}")
```

### 🎯 Real-World: LangChain + OpenAI Optimization

```python
from langchain_openai import OpenAI
from langchain import LLMChain, PromptTemplate
import traigent

# Your existing LangChain code - unchanged!
def analyze_sentiment(text: str) -> str:
    llm = OpenAI(model="gpt-4o-mini", temperature=0.5)
    prompt = PromptTemplate(
        template="Analyze sentiment of: {text}\nSentiment:",
        input_variables=["text"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text)

# Optimize it with zero code changes!
@traigent.optimize(
    eval_dataset="sentiment_test_set.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "davinci-002"],
        "temperature": [0.0, 0.3, 0.7, 1.0]
    },
    # Seamless optimization is enabled by default!
)
def analyze_sentiment_optimized(text: str) -> str:
    # EXACT SAME CODE - just copy-pasted!
    llm = OpenAI(model="gpt-4o-mini", temperature=0.5)
    prompt = PromptTemplate(
        template="Analyze sentiment of: {text}\nSentiment:",
        input_variables=["text"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text)
```

### 🔥 Multi-Framework Optimization

```python
# Works with any framework - OpenAI SDK example
import openai

@traigent.optimize(
    eval_dataset="translations.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o"],
        "temperature": [0.1, 0.5, 0.9],
        "max_tokens": [100, 500, 1000]
    }
    # Seamless framework override is automatic!
def translate_text(text: str, target_language: str) -> str:
    # Your existing OpenAI code - no changes needed!
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Will be overridden during optimization
        temperature=0.3,         # Will be overridden during optimization
        max_tokens=500,          # Will be overridden during optimization
        messages=[
            {"role": "system", "content": f"Translate to {target_language}"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
```

### 📊 Advanced: Multi-Objective with Cost Optimization

```python
@traigent.optimize(
    eval_dataset="complex_tasks.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={
        "model": ["gpt-4o-mini", "gpt-4o", "claude-2"],
        "temperature": [0.0, 0.5, 1.0],
        "max_tokens": [100, 500, 2000]
    },
    optimization_strategy={
        "max_cost_budget": 100.0,  # Stop when $100 spent
        "exploration_ratio": 0.3,   # 30% exploration, 70% exploitation
        "adaptive_sample_size": True  # Smart dataset subset selection
    }
)
def complex_reasoning_task(query: str) -> str:
    # Your production code stays exactly the same!
    llm = OpenAI(model="gpt-4o", temperature=0.7, max_tokens=1000)
    # ... rest of your complex logic ...
    return result
```

## 📚 Pre-built Examples

TraiGent comes with ready-to-use examples in `examples/`:

### Core Examples

Located in `examples/core/`:

- **simple-prompt**: Basic prompt optimization
- **hello-world**: Q&A with RAG optimization
- **few-shot-classification**: Few-shot learning patterns
- **multi-objective-tradeoff**: Balance accuracy vs cost with weighted objectives
- **token-budget-summarization**: Optimize within token limits
- **structured-output-json**: JSON schema validation
- **tool-use-calculator**: Function calling optimization
- **prompt-style-optimization**: Tune prompt style and tone
- **safety-guardrails**: Content moderation patterns

Each example includes:

- Complete `run.py` with production-quality code
- Evaluation datasets in `examples/datasets/`
- Mock mode support (no API keys needed!)
- Inline documentation

### Running Examples

```bash
# First, ensure you have the dependencies installed
pip install -e ".[integrations]"

# Run any example in mock mode
export TRAIGENT_MOCK_MODE=true
python examples/core/simple-prompt/run.py
```

## 🤔 Why TraiGent?

### The Problem with Traditional Optimization

❌ **Privacy Concerns**: Sending proprietary data to external services
❌ **Limited Control**: Black-box optimization without transparency
❌ **High Costs**: Testing every configuration on full datasets
❌ **Integration Pain**: Rewriting code for optimization tools
❌ **Vendor Lock-in**: Tied to specific optimization platforms

### TraiGent's Dual-Model Solution

✅ **Privacy First**: Choose local execution with cloud guidance
✅ **Full Transparency**: See exactly what's being optimized and why
✅ **60-80% Cost Reduction**: Smart dataset subset selection
✅ **Zero Code Changes**: Works with your existing functions
✅ **Platform Agnostic**: Works with any LLM provider
✅ **Smart Analytics**: Privacy-safe insights guide your optimization journey
✅ **Gradual Adoption**: Start local, see value, upgrade selectively

### Choose Your Approach

🏠 **Model 1**: Keep data local, get cloud intelligence
☁️ **Model 2**: Leverage full cloud power for agents
🎭 **Hybrid**: Start local, refine in cloud

**Result**: Complete control over your optimization strategy

## 📚 Documentation

- **[Quick Start Guide](docs/getting-started/GETTING_STARTED.md)**: Get started in 5 minutes
- **[Playground UI Guide](playground/README.md)**: Interactive Playground
- **[Architecture Guide](docs/architecture/ARCHITECTURE.md)**: Technical design details
- **[Secrets Management](docs/guides/secrets_management.md)**: Secure AWS-backed workflow
- **[Examples](examples/)**: Working code examples
- **[Evaluation Guide](docs/guides/evaluation.md)**: Dataset formats and custom evaluators
- **[Execution Modes](docs/guides/execution-modes.md)**: Local, cloud, and hybrid modes

## 🛠️ Development

### Project Structure

```
Traigent/
├── traigent/          # Main SDK package
│   ├── api/           # Public API and decorators
│   ├── core/          # Core orchestration logic
│   ├── optimizers/    # Optimization algorithms
│   ├── cloud/         # Cloud integration
│   └── integrations/  # Framework integrations
├── playground/        # Interactive UI (Streamlit control center)
├── examples/          # Example scripts and demos
├── tests/             # Test suite and configurations
├── docs/              # All documentation
├── scripts/           # Development and automation tools
└── requirements/      # Dependency specifications
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
# Or use our comprehensive test runner
python scripts/test/run_tests.py

# Run linting (scripts organized in scripts/linting/)
./scripts/linting/run_linters.sh
# Or individually:
ruff check traigent/
ruff format traigent/

# Install pre-commit hooks
pre-commit install
```

### 📂 Clean Project Organization

The project maintains a clean, professional structure:

- **Core directories** with clear, single purposes
- **All scripts centralized** in `scripts/` with subdirectory organization
- **All documentation unified** in `docs/` with logical grouping
- **Clean root directory** with only essential files

## 🤝 Contributing

We welcome contributions! See the guidelines below for how to get started.

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=traigent --cov-report=html

# Run specific test module
pytest tests/unit/test_api.py
```

### Areas for Contribution

- **🏗️ New AI Models**: Add support for Claude, Cohere, Edge Analytics models
- **🧠 Better Testing**: Improve subset selection algorithms
- **🎮 UI Features**: Enhance the Streamlit Playground
- **📊 Metrics**: Add domain-specific evaluation metrics
- **📖 Documentation**: Share your success stories
- **🐛 Bug Fixes**: Help improve reliability

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting

### Verify Installation

Run our verification script to check all dependencies:

```bash
python scripts/validation/verify_installation.py
```

### Installation Issues

If you encounter installation errors:

1. **Missing Dependencies** (e.g., `ModuleNotFoundError: No module named 'langchain_openai'`):

   ```bash
   pip install -r requirements/requirements-integrations.txt
   pip install -e .
   ```

2. **Upgrade pip**: `pip install --upgrade pip`

3. **Clear pip cache**: `pip cache purge`

4. **Install in stages**:
   ```bash
   pip install -e .              # First, basic installation
   pip install -r requirements/requirements-integrations.txt  # Then integrations
   pip install -e ".[dev]"       # Finally dev dependencies if needed
   ```

### Common Gotchas for New Users

#### README Code Examples vs Quickstart Files

The code examples in this README demonstrate patterns and concepts but may require API keys. The **actually runnable** examples are in `examples/quickstart/`:

```bash
# These work immediately with mock mode - no API keys needed!
export TRAIGENT_MOCK_MODE=true
python examples/quickstart/01_simple_qa.py   # Simple Q&A
python examples/quickstart/02_customer_support_rag.py  # RAG example
python examples/quickstart/03_custom_objectives.py     # Custom weights
```

#### Required Parameters

The `@traigent.optimize` decorator **requires** a `configuration_space` parameter:

```python
# ❌ This will fail - missing configuration_space
@traigent.optimize(eval_dataset="data.jsonl", objectives=["accuracy"])

# ✅ This works - configuration_space is required
@traigent.optimize(
    configuration_space={"temperature": [0.1, 0.5, 0.9]},
    eval_dataset="data.jsonl",
    objectives=["accuracy"]
)
```

#### Dataset Paths

Some dataset files have symlinks in the repo root for convenience:

- `qa_samples.jsonl` → `examples/datasets/quickstart/qa_samples.jsonl`
- `rag_feedback.jsonl` → `examples/quickstart/rag_feedback.jsonl`

For other datasets, use the full path or create your own symlink.

#### Mock Mode and API Keys

`TRAIGENT_MOCK_MODE=true` prevents TraiGent from making real API calls during optimization, but if your agent code directly instantiates `ChatOpenAI()` or similar, you still need the API key set (even if unused). The quickstart examples avoid this by using mock responses internally.

### Common Issues

#### 0.0% Accuracy in Results

If you see 0.0% accuracy:

- **Enable Mock Mode**: Set `TRAIGENT_MOCK_MODE=true` for realistic demo values
- **Check Dataset Format**: Ensure your dataset follows the correct format (see Evaluation section)
- **Use Custom Evaluator**: For non-exact matches, provide a custom evaluator function
- **Verify API Keys**: Ensure OPENAI_API_KEY is set for embedding-based evaluation

#### Missing Environment Variables

```bash
# Create a .env file with your API keys
echo "OPENAI_API_KEY=your-key-here" >> .env
echo "TRAIGENT_API_KEY=your-key-here" >> .env

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

#### Import Errors

- **ModuleNotFoundError**: Ensure you're in the correct directory and virtual environment
- **langchain_openai not found**: Install with `pip install langchain-openai`
- **langchain_chroma not found**: Install with `pip install langchain-chroma`
- **dotenv not found**: Install with `pip install python-dotenv`

#### Other Issues

- **Permission errors**: Use `pip install --user` or ensure venv is activated
- **Dependency conflicts**: Try creating a fresh virtual environment
- **Memory issues**: Use smaller datasets or reduce batch sizes

## 🌟 Community

- **[Discord](https://discord.gg/traigent)**: Join our community
- **[GitHub Issues](https://github.com/Traigent/Traigent/issues)**: Report bugs or request features
- **[GitHub Discussions](https://github.com/Traigent/Traigent/discussions)**: Ask questions and share ideas

## 🙏 Acknowledgments

- Built with ❤️ by the TraiGent team and community
- Inspired by the needs of LLM developers worldwide
- Thanks to all our [contributors](https://github.com/Traigent/Traigent/graphs/contributors)

---

**[Get Started](docs/getting-started/GETTING_STARTED.md)** | **[Examples](examples/)** | **[Evaluation Guide](docs/guides/evaluation.md)**
