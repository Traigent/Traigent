# Traigent Playground

This directory contains the Traigent Playground - a comprehensive Streamlit application for managing LLM optimization problems and running Traigent optimizations.

## 🚀 Quick Start

```bash
# From the TraigentSDK root directory
python -m venv .venv
source .venv/bin/activate
pip install -e ".[playground]"  # or: pip install -r playground/requirements_streamlit.txt
.venv/bin/streamlit run playground/traigent_control_center.py
```

## 📁 Directory Structure

```
playground/
├── traigent_control_center.py    # Main Streamlit application
├── problem_management/            # Problem creation and management utilities
├── problem_generation/            # Intelligent problem generation system
├── streamlit_utils/              # UI utilities for Streamlit
├── langchain_problems/           # Generated and stored problems
├── optimization_results/         # Optimization run results
├── optimization_logs/            # Detailed optimization logs
├── optimization_storage.py       # Storage handler for results
├── optimization_callbacks.py     # Callbacks for optimization tracking
├── problem_manager.py           # CLI tool for problem management
└── generate_problem_suite.py    # Batch problem generation tool
```

## 🎯 Features

### Problem Management

- Create new optimization problems from natural language descriptions
- Intelligent example generation using Claude Code SDK
- Problem validation and quality analysis
- Import/export capabilities

### Optimization Execution

- Run optimizations with various strategies (grid, random, adaptive) in local (`edge_analytics`) mode
- Real-time progress tracking and visualization
- Cost tracking and budget management
- Multi-objective optimization support

### Results Analysis

- Interactive results visualization
- Performance metrics comparison
- Export results in various formats
- Historical run comparisons

## 🛠️ CLI Tools

### Problem Manager

```bash
# Create a new problem
python playground/problem_manager.py create \
  --description "Customer support ticket classification" \
  --examples 50

# Add examples to existing problem
python playground/problem_manager.py add-examples customer_support --count 20

# Analyze problem quality
python playground/problem_manager.py analyze customer_support
```

### Batch Problem Generator

```bash
# Generate multiple problems at scale
python playground/generate_problem_suite.py \
  --problems 10 \
  --examples 100 \
  --parallel
```

## 📊 Data Storage

- **langchain_problems/**: Problem definitions and examples
- **optimization_results/**: JSON files with optimization results
- **optimization_logs/**: Detailed logs including trial data and LLM usage

## 🔧 Configuration

See `STREAMLIT_README.md` for detailed configuration options and advanced usage.
