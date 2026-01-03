# Traigent Examples

This directory contains professional, production-ready examples demonstrating the Traigent SDK across various use cases and optimization techniques.

## 🌐 Interactive Examples Gallery

Browse all examples visually with the web-based gallery:

```bash
python -m http.server -d examples 8000
# Open http://localhost:8000/index.html
```

The gallery provides:

- Visual cards for each example with difficulty levels and estimated runtime
- Direct links to source code and documentation
- Copy-paste run commands for each example
- Organized by category: Core, Advanced, Integrations, TVL

## 📂 Directory Overview

| Folder | Description | Count |
| ------ | ----------- | ----- |
| **`quickstart/`** | **Start here!** Ready-to-run README examples | 3 examples |
| **`core/`** | Essential examples introducing core Traigent concepts | 10 examples |
| **`advanced/`** | Advanced optimization patterns and specialized workflows | 5 categories |
| **`integrations/`** | Framework and platform integrations (CI/CD, Bedrock, etc.) | 2+ integrations |
| **`datasets/`** | Shared evaluation datasets and test data | 9+ datasets |
| **`docs/`** | Comprehensive documentation, guides, and tutorials | Multiple guides |
| **`templates/`** | Boilerplate templates for creating new examples | 2 templates |
| **`utils/`** | Reusable utility modules shared across examples | Shared utilities |
| **`tvl/`** | TVL (Traigent Variable Language) specification examples | TVL specs |
| **`archive/`** | Historical artifacts and legacy examples for reference | Archive |

## ⚡ Quickstart Examples (Start Here!)

These are the exact examples from the main README, ready to run immediately:

```bash
pip install -e ".[examples]"          # Ensure example deps
export TRAIGENT_MOCK_LLM=true        # No API keys needed
python examples/quickstart/01_simple_qa.py           # Basic Q&A optimization
python examples/quickstart/02_customer_support_rag.py # RAG with customer support
python examples/quickstart/03_custom_objectives.py    # Custom weighted objectives
```

Each quickstart example demonstrates a key concept from the README with working code.

## 🚀 Core Examples

Perfect for learning Traigent fundamentals. Each example is self-contained with clear documentation.

| Example | Description | Key Concepts | Runtime |
| ------- | ----------- | ------------ | ------- |
| **`simple-prompt`** | Basic prompt optimization | Decorator usage, config space, seamless injection | ~1 min |
| **`hello-world`** | Simple Q&A optimization with RAG | Configuration spaces, objectives, parameter injection | ~2 min |
| **`few-shot-classification`** | Few-shot learning for text classification | Few-shot prompting, example selection | ~3 min |
| **`multi-objective-tradeoff`** | Balance accuracy, cost, and latency (use `run_anthropic.py` or `run_openai.py`) | Multi-objective optimization, Pareto fronts | ~5 min |
| **`token-budget-summarization`** | Optimize within strict token budgets | Token constraints, budget management | ~3 min |
| **`structured-output-json`** | Generate and validate JSON outputs | Schema validation, structured outputs | ~3 min |
| **`tool-use-calculator`** | Function calling and tool orchestration | Tool usage, deterministic operations | ~2 min |
| **`prompt-style-optimization`** | Tune prompt style and tone | Prompt engineering, style optimization | ~4 min |
| **`chunking-long-context`** | Handle documents exceeding context limits | Chunking strategies, long-context handling | ~4 min |
| **`safety-guardrails`** | Apply content moderation and safety | Safety filters, guardrails, moderation | ~3 min |
| **`prompt-ab-test`** | A/B test different prompt variants | Prompt comparison, statistical testing | ~3 min |

**Each example includes:**

- ✅ Complete `run.py` with production-quality code (except `multi-objective-tradeoff` which has provider-specific files)
- ✅ Evaluation datasets in `datasets/<example-name>/`
- ✅ Mock mode support (no API keys needed!)
- ✅ Inline documentation and comments

## 🎓 Advanced Examples

For users comfortable with Traigent basics who want to explore advanced patterns:

| Category | Examples | Focus Area |
| -------- | -------- | ---------- |
| **`execution-modes/`** | 6 examples | Local patterns plus roadmap-only cloud/hybrid stubs |
| **`results-analysis/`** | 2 examples | Analyzing and visualizing optimization results |
| **`ai-engineering-tasks/`** | 5 examples | Common AI engineering challenges (context, few-shot, structured output, safety, token budgets) |
| **`ragas/`** | RAG evaluation | Specialized RAG metrics and evaluation |
| **`metric-registry/`** | Custom metrics | Building and registering custom evaluation metrics |

## 🔌 Integration Examples

Production-ready integrations with popular platforms and workflows:

- **`ci-cd/`** - Continuous integration examples (Math Q&A pipeline)
- **`bedrock/`** - AWS Bedrock integration patterns
- **More coming soon:** LangChain, OpenAI SDK, Azure OpenAI, Google Vertex AI

## 📊 Working With Datasets

All shared datasets are centralized in `examples/datasets/` for easy access:

```python
from pathlib import Path

# Reference datasets from any example
DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
eval_data = DATASETS_ROOT / "hello-world" / "evaluation_set.jsonl"
```

### Dataset Path Convention

All examples reference datasets using a relative path from the example file:
```python
# Standard pattern used in all examples:
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "<example-name>"
```

When creating custom examples, maintain this directory structure:
```
examples/
├── core/
│   └── your-example/
│       └── run.py          # Uses parents[2] to reach datasets/
└── datasets/
    └── your-example/
        └── evaluation_set.jsonl
```

**Available datasets:**

- `data/qa_samples.jsonl` - Q&A samples for README examples
- `simple-prompt/` - Basic summarization examples
- `hello-world/` - Simple Q&A examples
- `few-shot-classification/` - Classification with examples
- `multi-objective-tradeoff/` - Multi-objective test cases
- `token-budget-summarization/` - Summarization tasks
- `structured-output-json/` - JSON schema validation
- `tool-use-calculator/` - Calculator function calls
- `prompt-style-optimization/` - Style variations
- `chunking-long-context/` - Long documents
- `safety-guardrails/` - Safety test cases
- `prompt-ab-test/` - A/B test variants

**Matrix files:** Test specifications and CTD matrices are in `datasets/matrices/`.

## 🏃 Running Examples

### Quick Start (No API Keys!)

```bash
# Clone the repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Install Traigent
pip install -e .

# Run any example in mock mode (no API keys)
export TRAIGENT_MOCK_LLM=true
python examples/core/simple-prompt/run.py
```

### With Real LLM APIs

```bash
# Set your API keys
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Run examples with real LLM calls
python examples/core/multi-objective-tradeoff/run_anthropic.py
```

### Recommended Learning Path

1. **Quickstart:** `quickstart/01_simple_qa.py` (1 min) - README examples that just work
2. **Basics:** `core/simple-prompt/run.py` (1 min) - The absolute basics
3. **Learn:** `core/hello-world/run.py` (2 min) - RAG & parameter injection
4. **Explore:** `core/few-shot-classification/run.py` (3 min) - Few-shot optimization
5. **Deep Dive:** `core/multi-objective-tradeoff/run_anthropic.py` (5 min) - Real optimization
6. **Advance:** `advanced/ai-engineering-tasks/` - Specialized patterns
7. **Integrate:** `integrations/ci-cd/` - Production workflows

## 🛠️ Creating New Examples

Follow these best practices when contributing new examples:

1. **Choose the right category:**
   - `core/` - Fundamental concepts for beginners
   - `advanced/` - Specialized patterns for experienced users
   - `integrations/` - Platform or framework integrations

2. **Use the template:**

   ```bash
   cp templates/run_template.py examples/core/<YOUR_EXAMPLE>/run.py
   ```

3. **Include all essentials:**
   - ✅ Self-contained `run.py` with clear documentation
   - ✅ Evaluation dataset in `datasets/<YOUR_EXAMPLE>/`
   - ✅ Mock mode support for testing without API keys
   - ✅ Clear comments explaining key concepts
   - ✅ Expected output examples

4. **Test thoroughly:**

   ```bash
   # Test in mock mode
   TRAIGENT_MOCK_LLM=true python examples/core/<YOUR_EXAMPLE>/run.py

   # Test with real APIs
   python examples/core/<YOUR_EXAMPLE>/run.py
   ```

5. **Update documentation:**
   - Add your example to this README
   - Update relevant guides in `docs/`

## 📚 Documentation

- **[START_HERE.md](docs/START_HERE.md)** - Quick navigation guide
- **[EXAMPLES_GUIDE.md](docs/EXAMPLES_GUIDE.md)** - Comprehensive walkthrough
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command reference
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API_PATTERNS.md](docs/API_PATTERNS.md)** - API usage patterns

## 💡 Tips for Success

**For Beginners:**

- Start with `core/simple-prompt/` to verify your setup
- Use `TRAIGENT_MOCK_LLM=true` to learn without API costs
- Read inline comments in examples - they explain key concepts
- Follow the recommended learning path above

**For Advanced Users:**

- Explore `advanced/` for specialized patterns
- Check `integrations/` for production deployment examples
- Review `datasets/matrices/` for comprehensive test coverage
- Contribute your own examples and patterns!

**Performance Tips:**

- Start with small datasets and few trials
- Use mock mode for development and testing
- Enable caching to avoid redundant API calls
- Monitor costs during optimization

## 🐛 Troubleshooting

**Common issues:**

| Problem | Solution |
| ------- | -------- |
| `ModuleNotFoundError: No module named 'traigent'` | Run `pip install -e .` from repository root |
| `API key not found` | Export your API key or use `TRAIGENT_MOCK_LLM=true` |
| `Example doesn't run` | Check example's inline comments for prerequisites |
| `0.0% accuracy` | Use mock mode or provide valid API keys |

For more help, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) or open an issue.

## 🤝 Contributing

We welcome contributions! To add new examples:

1. Follow the structure and patterns of existing examples
2. Include mock mode support for testing
3. Provide clear documentation and comments
4. Test thoroughly before submitting
5. Update this README and relevant docs

See our [Contributing Guide](../docs/guides/CONTRIBUTING.md) for more details.

## 📝 Additional Resources

- **Main README:** [../README.md](../README.md)
- **Documentation:** [../docs/](../docs/)
- **Website:** <https://traigent.ai> (coming soon)
- **Discord:** Join our community for support
- **GitHub Issues:** Report bugs and request features

---

**Ready to get started?** Run your first example:

```bash
export TRAIGENT_MOCK_LLM=true
python examples/core/simple-prompt/run.py
```

Questions? Check [docs/START_HERE.md](docs/START_HERE.md) for a quick navigation guide!
