# Traigent SDK Documentation

**The first optimization platform that enhances your LLM applications without changing your code.**

![Traigent Logo](assets/traigent.png)

## 🚀 Quick Start

```python
import traigent
from langchain_openai import ChatOpenAI

@traigent.optimize(
    eval_dataset="examples/datasets/hello-world/evaluation_set.jsonl",
    objectives=["accuracy", "cost"],
    configuration_space={"model": ["gpt-4o-mini", "gpt-4o"], "temperature": [0.0, 0.7]},
)
def answer_question(question: str) -> str:
    cfg = traigent.get_config()  # Active trial/applied config
    llm = ChatOpenAI(model=cfg.get("model"), temperature=cfg.get("temperature"))
    return llm.invoke(question).content

# Async-safe in Traigent
# results = await answer_question.optimize(max_trials=5)
```

## 📚 Documentation

### 🎯 [Getting Started](getting-started/)
- **[Installation](getting-started/installation.md)** - Install Traigent SDK
- **[Getting Started](getting-started/GETTING_STARTED.md)** - Your first optimization in 5 minutes

### 📖 [User Guide](user-guide/)
- **[Agent Optimization](user-guide/agent_optimization.md)** - Optimize AI agents and workflows
- **[Choosing Models](user-guide/choosing_optimization_model.md)** - Select the right optimization model
- **[Injection Modes](user-guide/injection_modes.md)** - Different ways to integrate Traigent
- **[Interactive Optimization](user-guide/interactive_optimization.md)** - Real-time optimization workflows
- **[Evaluation Guide](user-guide/evaluation_guide.md)** - Measure and improve performance
- **[Optuna Integration](user-guide/optuna_integration.md)** - Optuna-backed optimizers, coordinator, and adapter

### 🔧 [API Reference](api-reference/)
- **[Complete Function Specification](api-reference/complete-function-specification.md)** - Full API documentation

### 🏗️ [Architecture](architecture/)
- **[System Architecture](architecture/ARCHITECTURE.md)** - How Traigent works internally
- **[Project Structure](architecture/project-structure.md)** - Codebase organization

### 🤝 [Contributing](contributing/)
- **[Contributing Guide](contributing/CONTRIBUTING.md)** - How to contribute to Traigent
- **[Code of Conduct](contributing/CODE_OF_CONDUCT.md)** - Community guidelines
- **[Security Policy](contributing/SECURITY.md)** - Security reporting and practices

## 🎯 What is Traigent?

Traigent is a **zero-code optimization platform** that automatically finds the best configurations for your LLM applications. Simply add a decorator to your existing functions, and Traigent will:

- 🎯 **Optimize multiple objectives** (accuracy, cost, speed, etc.)
- 🔄 **Test different configurations** automatically
- 📊 **Provide detailed analytics** on performance
- 🚀 **Improve your applications** without code changes

## 💡 Key Features

- **Zero Code Changes** - Works with existing functions
- **Multi-Objective Optimization** - Balance accuracy, cost, and speed
- **Framework Agnostic** - Works with OpenAI, Anthropic, LangChain, and more
- **Production Ready** - Built for enterprise deployment
- **Real-time Analytics** - Monitor optimization progress live

## 🌐 Examples Navigator

- Open: `examples/index.html` (repo copy)
- Serve locally (avoids file:// fetch restrictions):
  - `python -m http.server -d examples 8000`
  - Visit `http://localhost:8000/`
- Run icon: Each code block has a play button that copies a ready-to-run mock-mode command.

## 🚀 Next Steps

1. **[Install Traigent](getting-started/installation.md)** - Get set up in minutes
2. **[Follow the Getting Started Guide](getting-started/GETTING_STARTED.md)** - Your first optimization
3. **Open the Examples** – `examples/index.html` and try the sections
4. **[Explore User Guides](user-guide/)** - Learn advanced features
5. **[Check API Reference](api-reference/)** - Detailed technical documentation

## 🧪 Mock Mode (No API Keys)

- Enable: `TRAIGENT_MOCK_MODE=true` (no API keys needed)
- Benefits: realistic scores, zero cost, quick iteration
- Example:
  - `TRAIGENT_MOCK_MODE=true python examples/core/hello-world/run.py`
  - Or copy a command from the Examples Navigator "Run" icon

---

**Need help?** Check our guides above or reach out to the community!
