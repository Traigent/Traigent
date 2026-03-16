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
| **`core/`** | Essential examples introducing core Traigent concepts | 12 examples |
| **`advanced/`** | Advanced optimization patterns and specialized workflows | 5 categories |
| **`integrations/`** | Framework and platform integrations (CI/CD, Bedrock, etc.) | 2+ integrations |
| **`datasets/`** | Shared evaluation datasets and test data | 9+ datasets |
| **`docs/`** | Comprehensive documentation, guides, and tutorials | Multiple guides |
| **`templates/`** | Boilerplate templates for creating new examples | 2 templates |
| **`utils/`** | Reusable utility modules shared across examples | Shared utilities |
| **`tvl/`** | TVL (Traigent Variable Language) specification examples | TVL specs |
| **`archive/`** | Historical artifacts and legacy examples for reference | Archive |

## ⚡ Quickstart Examples (Start Here!)

Run these quickstart examples to get started:

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
| **`rag-optimization`** | Simple Q&A optimization with RAG | Configuration spaces, objectives, parameter injection | ~2 min |
| **`few-shot-classification`** | Few-shot learning for text classification | Few-shot prompting, example selection | ~3 min |
| **`multi-objective-tradeoff`** | Balance accuracy, cost, and latency (use `run_anthropic.py` or `run_openai.py`) | Multi-objective optimization, Pareto fronts | ~5 min |
| **`token-budget-summarization`** | Optimize within strict token budgets | Token constraints, budget management | ~3 min |
| **`structured-output-json`** | Generate and validate JSON outputs | Schema validation, structured outputs | ~3 min |
| **`tool-use-calculator`** | Function calling and tool orchestration | Tool usage, deterministic operations | ~2 min |
| **`prompt-style-optimization`** | Tune prompt style and tone | Prompt engineering, style optimization | ~4 min |
| **`chunking-long-context`** | Handle documents exceeding context limits | Chunking strategies, long-context handling | ~4 min |
| **`safety-guardrails`** | Apply content moderation and safety | Safety filters, guardrails, moderation | ~3 min |
| **`prompt-ab-test`** | A/B test different prompt variants | Prompt comparison, statistical testing | ~3 min |
| **`production-deployment`** | Load and apply optimized configs in production | Config persistence, hot-swap, deployment patterns | ~2 min |
| **`error-handling`** | Graceful failure modes and fallback patterns | Error recovery, budget limits, timeouts, validation | ~2 min |

### What You'll Learn

- **simple-prompt**: The minimal Traigent setup. Shows how to add the `@traigent.optimize` decorator to an existing function and define a configuration space (model, temperature). Great first example.

- **rag-optimization**: Introduces RAG patterns with parameter injection. Learn how Traigent passes optimized parameters (like `k` for retrieval depth) directly into your function.

- **few-shot-classification**: Optimize few-shot example selection. Traigent finds which examples improve classification accuracy - useful for prompt engineering without manual trial-and-error.

- **multi-objective-tradeoff**: Balance competing objectives (accuracy vs. cost vs. latency). Learn Pareto optimization to find configurations that don't sacrifice one metric for another.

- **token-budget-summarization**: Work within token limits. Shows how to optimize summarization quality while staying under a strict token budget - critical for cost control.

- **structured-output-json**: Generate reliable JSON outputs. Learn schema validation and how Traigent optimizes for both correctness and schema compliance.

- **tool-use-calculator**: Function calling patterns. Traigent optimizes when and how to use tools, finding the right balance between tool calls and direct responses.

- **prompt-style-optimization**: Tune tone and style. Optimize prompt templates for specific audiences or use cases without manually testing variations.

- **chunking-long-context**: Handle documents that exceed context windows. Learn chunking strategies and how Traigent finds optimal chunk sizes and overlap.

- **safety-guardrails**: Content moderation integration. Shows how to optimize for accuracy while maintaining safety constraints - essential for production deployments.

- **prompt-ab-test**: Statistical A/B testing for prompts. Compare prompt variants with statistical significance, not just single-run comparisons.

- **production-deployment**: Production deployment workflow. Optimize, save the best config to JSON, load it in production, and run with a frozen configuration.

- **error-handling**: Graceful error handling patterns. Demonstrates invalid config detection, budget limits, timeout handling, preflight validation, and fallback to default configs.

**Each example includes:**

- ✅ Complete `run.py` with production-quality code (except `multi-objective-tradeoff` which has provider-specific files)
- ✅ Evaluation datasets in `datasets/<example-name>/`
- ✅ Mock mode support (no API keys needed!)
- ✅ Inline documentation and comments

## 🎓 Advanced Examples

For users comfortable with Traigent basics who want to explore advanced patterns:

| Category | Examples | Focus Area |
| -------- | -------- | ---------- |
| **`execution-modes/`** | 2 examples | Local execution patterns with edge_analytics mode |
| **`results-analysis/`** | 2 examples | Analyzing and visualizing optimization results |
| **`ai-engineering-tasks/`** | 5 examples | Common AI engineering challenges (context, few-shot, structured output, safety, token budgets) |
| **`ragas/`** | 3 examples | Specialized RAG metrics and evaluation |
| **`metric-registry/`** | Custom metrics | Building and registering custom evaluation metrics |

### Advanced Category Details

- **execution-modes/**: Learn execution patterns. `edge_analytics` keeps data local while sending anonymized metrics. Cloud and hybrid modes coming in a future release.

- **results-analysis/**: Post-optimization analysis. Visualize trial results, compare configurations, and extract insights from completed optimization runs.

- **ai-engineering-tasks/**: Real-world AI engineering patterns. Each example tackles a specific challenge: context window management, few-shot learning, structured outputs, safety, and token budgets.

- **ragas/**: RAG-specific evaluation using the RAGAS framework. Learn to measure faithfulness, answer relevancy, and context precision for retrieval-augmented generation.
  - `basics/` - Standard RAGAS metrics integration
  - `column_map/` - Custom column mapping for non-standard datasets
  - `with_llm/` - LLM-powered evaluation (requires API key in real mode)

- **metric-registry/**: Build custom evaluation metrics. Register domain-specific scorers that integrate seamlessly with Traigent's optimization loop.

## 🔌 Integration Examples

Production-ready integrations with popular platforms and workflows:

- **`deepeval/`** - DeepEval evaluation metrics as optimization objectives. Use research-backed metrics (relevancy, faithfulness, hallucination) to drive Traigent optimization.
- **`ci-cd/`** - Continuous integration examples (Math Q&A pipeline). Run Traigent optimizations as part of your CI/CD workflow.
- **`bedrock/`** - AWS Bedrock integration patterns. Use Traigent with Amazon's managed LLM service.

## 📊 Working With Datasets

All shared datasets are centralized in `examples/datasets/` for easy access:

```python
from pathlib import Path

# Reference datasets from any example
DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
eval_data = DATASETS_ROOT / "rag-optimization" / "evaluation_set.jsonl"
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

| Dataset | Size | Difficulty Tiers | Evaluation |
| ------- | ---- | ---------------- | ---------- |
| `simple-prompt/` | 10 | Mixed | Exact match |
| `rag-optimization/` | 15 | Easy/Medium/Hard | Exact match |
| `few-shot-classification/` | 20 | Clear/Ambiguous | Exact match |
| `multi-objective-tradeoff/` | 15 | Mixed | Multi-metric |
| `token-budget-summarization/` | 10 | Short/Long docs | Token + quality |
| `structured-output-json/` | 12 | Simple/Complex schemas | Schema validation |
| `tool-use-calculator/` | 15 | Basic/Multi-step | Exact match |
| `prompt-style-optimization/` | 10 | Formal/Casual | Style scoring |
| `chunking-long-context/` | 8 | Medium/Long docs | Semantic similarity |
| `safety-guardrails/` | 20 | Safe/Harmful inputs | Safety + accuracy |
| `prompt-ab-test/` | 15 | Mixed | Statistical comparison |

### Why Difficulty Tiers Matter

Datasets include varying difficulty so that:

- Different model configurations show measurable differences
- Optimization has room for meaningful improvement
- Edge cases test robustness (e.g., ambiguous classifications, boundary inputs)

### Evaluation Methods

- **Exact match**: Case-insensitive token comparison (80% threshold)
- **Semantic similarity**: Key concept presence checking
- **Schema validation**: JSON structure and type compliance
- **Multi-metric**: Weighted combination of accuracy, cost, latency

**Matrix files:** Test specifications and CTD matrices are in `datasets/matrices/`.

## 🏃 Running Examples

### Test Script (Run Examples by Category)

Use `test_all_examples.sh` to run examples by category:

```bash
cd examples

# Run examples by category (mock mode, no API keys)
./test_all_examples.sh core               # Run 12 core examples
./test_all_examples.sh quickstart         # Run 3 quickstart examples
./test_all_examples.sh tvl                # Run 5 TVL tutorial examples
./test_all_examples.sh multi-objective    # Run 5 multi-objective variants
./test_all_examples.sh walkthrough        # Run 8 walkthrough examples
./test_all_examples.sh advanced-walkthrough # Run 5 advanced walkthrough examples
./test_all_examples.sh manifest           # Run all 37 manifest examples (strict)
./test_all_examples.sh all                # Run all categories

# Run with real API keys
./test_all_examples.sh --real core
./test_all_examples.sh --real all

# Show help
./test_all_examples.sh --help
```

**Categories:**

| Category | Examples | Description |
| -------- | -------- | ----------- |
| `core` | 12 | Main Traigent feature demonstrations |
| `quickstart` | 3 | Ready-to-run README examples |
| `tvl` | 5 | TVL specification tutorials |
| `multi-objective` | 5 | Multi-objective optimization variants |
| `walkthrough` | 8 | Tutorial walkthrough examples |
| `advanced-walkthrough` | 5 | Advanced walkthrough mock examples |
| `ragas` | 3 | RAGAS evaluation integration |
| `docs` | 2 | Documentation inline examples |
| `manifest` | 37 | All publication-ready examples (strict) |
| `all` | — | Run all categories |

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
3. **Learn:** `core/rag-optimization/run.py` (2 min) - RAG & parameter injection
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

See our [Contributing Guide](../CONTRIBUTING.md) for more details.

## 📝 Additional Resources

- **Main README:** [../README.md](../README.md)
- **Documentation:** [../docs/](../docs/)
- **Website:** <https://traigent.ai>
- **GitHub Issues:** Report bugs and request features

---

**Ready to get started?** Run your first example:

```bash
export TRAIGENT_MOCK_LLM=true
python examples/core/simple-prompt/run.py
```

Questions? Check [docs/START_HERE.md](docs/START_HERE.md) for a quick navigation guide!
