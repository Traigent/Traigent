# Traigent Tutorials

This directory contains in-depth tutorials that demonstrate complete workflows combining multiple Traigent features.

## Available Tutorials

### [DSPy + Tuned Variables Combined](./dspy_tuned_variables_combined.py)

A comprehensive tutorial showing how to combine:
- **Callable Auto-Discovery**: Automatically discover retriever functions from source code
- **TunedCallable Pattern**: Manage function-valued variables with per-callable parameters
- **Domain Presets**: Use LLMPresets and RAGPresets for domain-aware optimization
- **DSPy Integration**: Create optimizable prompt choices
- **Variable Analysis**: Post-optimization analysis with VariableAnalyzer

**Run in mock mode (no API key needed):**
```bash
TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true python dspy_tuned_variables_combined.py
```

**Run with real LLM:**
```bash
OPENAI_API_KEY=your-key python dspy_tuned_variables_combined.py
```

## Prerequisites

```bash
# Install required packages
pip install traigent[dspy] traigent-tuned-variables
```

## Related Documentation

- [Tuned Variables Guide](../../docs/user-guide/tuned_variables.md) - Complete reference
- [DSPy Integration Guide](../docs/DSPY_INTEGRATION.md) - DSPy-specific documentation
- [Examples README](../README.md) - Overview of all examples
