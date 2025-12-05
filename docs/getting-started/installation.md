# TraiGent SDK Installation Guide

This guide covers all installation methods for TraiGent SDK, from basic usage to enterprise deployment.

## 🚀 Quick Start

### Method 1: Direct Installation from GitHub (Recommended)

Install directly from GitHub without cloning:

#### Using pip (Traditional)

```bash
# Basic installation
pip install git+https://github.com/Traigent/Traigent.git

# With specific feature sets
pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[integrations]"  # LangChain, OpenAI, etc.
pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[examples]"      # All example dependencies
pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[all]"           # Everything

# Install from specific branch
pip install "git+https://github.com/Traigent/Traigent.git@main#egg=traigent[all]"
```

#### Using uv (Faster - 10-100x speed improvement)

```bash
# Basic installation (10-100x faster!)
uv pip install git+https://github.com/Traigent/Traigent.git

# With specific feature sets
uv pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[integrations]"
uv pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[examples]"
uv pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[all]"

# Install from specific branch
uv pip install "git+https://github.com/Traigent/Traigent.git@main#egg=traigent[all]"
```

**Why use uv?**
- ⚡ 10-100x faster dependency resolution and installation
- 🎯 Drop-in replacement for pip (same commands)
- 📦 Works with existing `pyproject.toml` and requirements files
- 🔒 More reliable dependency resolution
- 💾 Efficient caching system

**Installing uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Method 2: Clone and Install (For Development)

#### Using pip (Traditional)

```bash
# Clone repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all features
pip install -e ".[dev,integrations,analytics]"

# Or install everything
pip install -e ".[all]"

# Verify installation
python -c "import traigent; print('✅ TraiGent installed successfully')"
```

#### Using uv (Faster - Recommended for Development)

```bash
# Clone repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment with uv (faster)
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode (10-100x faster!)
uv pip install -e ".[dev,integrations,analytics]"

# Or install everything
uv pip install -e ".[all]"

# Verify installation
python -c "import traigent; print('✅ TraiGent installed successfully')"

# Open the Examples Navigator (serve locally to avoid file:// CORS)
python -m http.server -d examples 8000  # then visit http://localhost:8000/
```

## 📦 Available Feature Sets

| Feature Set | Command | Description | Use Case |
|------------|---------|-------------|----------|
| **Core** | `pip install traigent` | Minimal dependencies | Basic SDK functionality |
| **Analytics** | `traigent[analytics]` | numpy, pandas, matplotlib | Data analysis & visualization |
| **Bayesian** | `traigent[bayesian]` | scikit-learn, scipy | Bayesian optimization |
| **Integrations** | `traigent[integrations]` | LangChain, OpenAI, Anthropic, MLflow, WandB | Framework integrations |
| **Security** | `traigent[security]` | JWT, cryptography, FastAPI | Enterprise security features |
| **Visualization** | `traigent[visualization]` | matplotlib, plotly | Advanced charts & plots |
| **Playground** | `traigent[playground]` | streamlit, plotly | Interactive UI |
| **Examples** | `traigent[examples]` | All demo dependencies | Run documentation examples |
| **Test** | `traigent[test]` | pytest, coverage, ragas | Testing dependencies only |
| **Dev** | `traigent[dev]` | pytest, black, ruff, mypy | Development & testing tools |
| **All** | `traigent[all]` | Everything above | Complete installation |
| **Enterprise** | `traigent[enterprise]` | Same as `[all]` | Enterprise deployments |

### Future: PyPI Installation (Coming Soon)

Once published to PyPI, you'll be able to install with:

```bash
# Basic installation (minimal dependencies)
pip install traigent

# With specific features
pip install "traigent[analytics]"
pip install "traigent[integrations]"

# With enterprise security
pip install "traigent[security]"

# All features (complete platform)
pip install "traigent[all]"

# Enterprise bundle (production deployment)
pip install "traigent[enterprise]"
```

## 📦 Installation Methods

### Method 1: Development Installation from Source

```bash
# Clone repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements/requirements.txt

# Install package
pip install -e .

# For demo development
cd demos
pip install -r requirements.txt
```

### Method 2: Requirements Files

For more control over dependencies:

```bash
# Basic installation
pip install -r requirements/requirements.txt

# Development environment
pip install -r requirements/requirements-dev.txt

# With integrations
pip install -r requirements/requirements-integrations.txt

# Complete installation
pip install -r requirements/requirements-all.txt
```

### Method 3: Poetry (If Available)

If the project includes a pyproject.toml:

```bash
# Clone repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Install with Poetry
poetry install

# Install with specific groups
poetry install --with dev

# Install all optional dependencies
poetry install --all-extras
```

### Method 4: Virtual Environment Setup

Recommended for isolated development:

```bash
# Create virtual environment
python -m venv traigent-env

# Activate (Linux/Mac)
source traigent-env/bin/activate

# Activate (Windows)
traigent-env\Scripts\activate

# Clone and install
git clone https://github.com/Traigent/Traigent.git
cd Traigent
pip install -r requirements/requirements.txt
pip install -e .

# Verify installation
python -c "import traigent; print('✅ TraiGent SDK installed successfully')"
```

## 🎯 Feature-Based Installation

### For Basic Optimization
```bash
# From source
pip install -r requirements/requirements.txt
pip install -e .
# Includes: Core optimization, Grid/Random search, Basic evaluation
```

### For Framework Integration
```bash
pip install -r requirements/requirements-integrations.txt
# Adds: LangChain, OpenAI SDK, Anthropic, and more
```

### For Development
```bash
pip install -r requirements/requirements-dev.txt
# Adds: Testing tools, linters, formatters
```

### For Complete Platform
```bash
pip install -r requirements/requirements-all.txt
# Includes: All features and dependencies
```

## 🔧 Development Installation

### For Contributors

```bash
# Clone and setup development environment
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode with dev dependencies
pip install -r requirements/requirements-dev.txt
pip install -e .

# Run tests to verify setup
python -m pytest tests/unit/ -v
```

### For Extension Development

```bash
# Install with all dependencies
pip install -r requirements/requirements-all.txt
pip install -e .

# Additional development tools
pip install pytest pytest-asyncio black isort mypy ruff
```

## 🏢 Enterprise Installation

### Production Deployment

```bash
# Full installation with all features
pip install -r requirements/requirements-all.txt
pip install -e .

# Set environment variables
export TRAIGENT_EXECUTION_MODE=cloud  # or local, hybrid
export TRAIGENT_API_KEY=your-api-key  # if using cloud mode
```

### Docker Installation

```dockerfile
FROM python:3.11-slim

# Copy source code
COPY . /app
WORKDIR /app

# Install TraiGent SDK
RUN pip install -r requirements/requirements.txt
RUN pip install -e .

# Start application
CMD ["python", "your_app.py"]
```

### Environment Variables

```bash
# Create .env file from template
cp .env.example .env

# Edit .env with your configuration
# Required for demos and cloud features:
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
TRAIGENT_API_KEY=your-traigent-key  # For cloud mode
TRAIGENT_EXECUTION_MODE=local  # local, cloud, or hybrid
TRAIGENT_RESULTS_FOLDER=~/.traigent/results  # For local storage
```

## ✅ Verification

### Basic Installation Check

```bash
# Verify core installation
python -c "import traigent; print(f'TraiGent SDK installed successfully')"

# Check available optimizers
python -c "
from traigent.optimizers.registry import OptimizerRegistry
registry = OptimizerRegistry()
print('Available optimizers:', registry.list_optimizers())
"

# Test basic functionality
python -c "
import traigent
@traigent.optimize(
    configuration_space={'x': [1,2,3]},
    objectives=['accuracy'],
    eval_dataset='test.jsonl'
)
def test(): pass
print('✅ Basic functionality working')
"
```

### Feature Verification

```bash
# Check if integrations are available
python -c "
try:
    from traigent.integrations import framework_override
    print('✅ Framework integrations available')
except ImportError:
    print('❌ Integrations not installed')
"

# Check if cloud features are available
python -c "
try:
    from traigent.cloud import backend_client
    print('✅ Cloud features available')
except ImportError:
    print('❌ Cloud features not available')
"
```

### Run Demo

```bash
# Launch the interactive UI
python scripts/launch_control_center.py

# Run a simple optimization demo
cd demos/01-fundamentals/quickstart
python examples/core/hello-world/run.py

# Test temperature optimization
cd demos/01-fundamentals/06-temperature-optimization
python temperature_optimizer.py
```

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'traigent'"
```bash
# Make sure you're in the correct directory
cd Traigent
# Install in development mode
pip install -e .
```

#### "ModuleNotFoundError: No module named 'langchain'"
```bash
# Install integration dependencies
pip install -r requirements/requirements-integrations.txt
```

#### Missing API Keys
```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Virtual Environment Issues

```bash
# Clear pip cache
pip cache purge

# Upgrade pip
pip install --upgrade pip

# Fresh virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt
pip install -e .
```

## 📋 Dependency Overview

### Core Dependencies (Always Installed)
- `click>=8.0.0` - CLI framework
- `rich>=10.0.0` - Rich terminal output
- `pydantic>=2.0.0` - Data validation
- `aiohttp>=3.8.0` - Async HTTP client
- `numpy>=1.20.0` - Numerical computing

### Optional Dependencies by Feature

| Feature Group | Key Dependencies | Requirements File |
|---------------|------------------|------------------|
| Core | numpy, pydantic, aiohttp | requirements.txt |
| Integrations | langchain, openai, anthropic | requirements-integrations.txt |
| Development | pytest, black, ruff, mypy | requirements-dev.txt |
| All Features | All of the above | requirements-all.txt |

## 🎯 Recommended Installations

### For New Users
```bash
# Basic installation from source
git clone https://github.com/Traigent/Traigent.git
cd Traigent
pip install -r requirements/requirements.txt
pip install -e .
```

### For Development
```bash
pip install -r requirements/requirements-dev.txt
pip install -e .
```

### For Production
```bash
pip install -r requirements/requirements-all.txt
pip install -e .
```

---

## 🚀 UV Package Manager (Recommended)

### What is uv?

`uv` is an extremely fast Python package installer and resolver written in Rust. It's designed as a drop-in replacement for `pip` with 10-100x performance improvements.

### Why Choose uv?

| Feature | pip | uv |
|---------|-----|-----|
| **Speed** | Standard | 10-100x faster |
| **Dependency Resolution** | Slow on large projects | Near-instant |
| **Caching** | Basic | Advanced, efficient |
| **Compatibility** | Standard | Drop-in replacement |
| **Installation Size** | Small | ~10MB (single binary) |

### Installing uv

```bash
# macOS/Linux (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip (if you prefer)
pip install uv

# Using Homebrew (macOS)
brew install uv

# Using cargo (Rust)
cargo install --git https://github.com/astral-sh/uv uv
```

### Using uv with TraiGent

#### Development Workflow

```bash
# 1. Clone repository
git clone https://github.com/Traigent/Traigent.git
cd Traigent

# 2. Create virtual environment (much faster than python -m venv)
uv venv

# 3. Activate environment
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# 4. Install TraiGent in development mode
uv pip install -e ".[dev,integrations,analytics]"

# That's it! Same commands as pip, but 10-100x faster
```

#### Production Deployment

```bash
# Install from GitHub (production)
uv pip install "git+https://github.com/Traigent/Traigent.git#egg=traigent[all]"

# Or from PyPI when available
uv pip install "traigent[all]"
```

### Common uv Commands

```bash
# Create virtual environment
uv venv                          # Creates .venv/
uv venv custom-name              # Custom name

# Install packages (drop-in replacement for pip)
uv pip install package-name      # Install package
uv pip install -e .              # Editable install
uv pip install -e ".[dev]"       # With extras
uv pip install -r requirements.txt  # From requirements file

# Uninstall packages
uv pip uninstall package-name

# List installed packages
uv pip list

# Show package info
uv pip show traigent

# Freeze dependencies
uv pip freeze > requirements.txt
```

### Key Differences from pip

#### ✅ What's the Same
- Command syntax (`uv pip install` vs `pip install`)
- Works with `pyproject.toml`, `requirements.txt`, and setup.py
- Same package index (PyPI)
- Same virtual environment structure

#### ⚡ What's Better
- **10-100x faster** dependency resolution
- **Parallel downloads** and installations
- **Better caching** - faster subsequent installs
- **More reliable** dependency resolution
- **Single binary** - easy to install and update

#### ⚠️ Important Notes
- Use `uv pip` not just `uv` for pip-compatible commands
- Virtual environments created with `uv venv` are standard Python venvs
- Can mix uv and pip commands in the same environment (though uv is faster)

### Performance Comparison

Real-world TraiGent installation times:

```bash
# Traditional pip installation
time pip install -e ".[all]"
# Result: ~180 seconds (3 minutes)

# Using uv
time uv pip install -e ".[all]"
# Result: ~8 seconds (22x faster!)
```

### Migration from pip to uv

No migration needed! Just replace `pip` with `uv pip`:

```bash
# Before (pip)
pip install -e ".[dev]"
pip install langchain-openai
pip list

# After (uv) - same commands!
uv pip install -e ".[dev]"
uv pip install langchain-openai
uv pip list
```

### Troubleshooting uv

#### Command not found
```bash
# Make sure uv is in your PATH
echo $PATH  # Check if ~/.cargo/bin is included

# Add to PATH (Linux/Mac)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Permission issues
```bash
# Use --user flag if needed (though usually not required)
uv pip install --user traigent
```

#### Virtual environment issues
```bash
# uv venv creates standard Python virtual environments
# You can activate them the same way as venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### Additional Resources

- **Official Documentation**: https://github.com/astral-sh/uv
- **Installation Guide**: https://astral.sh/uv
- **GitHub Repository**: https://github.com/astral-sh/uv

---

## 📞 Support

- **Repository**: https://github.com/Traigent/Traigent
- **Issues**: https://github.com/Traigent/Traigent/issues
- **Documentation**: See README.md and docs/ folder
