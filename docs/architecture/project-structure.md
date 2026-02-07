# Traigent SDK Project Structure

This document outlines the current structure of the Traigent SDK project.

## Root Directory

```
Traigent/
├── README.md                  # Main project documentation
├── LICENSE                    # Apache-2.0 license
├── pyproject.toml             # Build, packaging, and tool config
├── uv.lock                    # Lockfile (uv)
├── traigent/                  # Core SDK source code
├── tests/                     # Test suite
├── examples/                  # Usage examples and demo runners
├── walkthrough/               # Interactive tutorials with datasets
├── docs/                      # Documentation (this folder)
├── tools/                     # Architecture analysis and dev utilities
├── scripts/                   # Utility and automation scripts
├── plugins/                   # Optional plugin packages
├── requirements/              # Dependency sets and guides
└── configs/                   # Configuration, baselines, and runtime
    ├── auto_tune_config.yaml
    ├── env-templates/
    ├── baselines/             # Performance baselines
    └── runtime/               # Runtime configs
```

> **Note**: Use-cases, demos, playground, paper experiments, and non-essential tools
> have been moved to the [TraigentDemo](https://github.com/Traigent/TraigentDemo) repository.

## Key Directories Explained

### `/traigent/`
Core SDK source code, including:
- `api/` public decorators and functions
- `config/` configuration and injection providers
- `core/` orchestration and optimized function
- `optimizers/` algorithms (grid/random/optuna/interactive)
- `evaluators/` dataset and evaluation
- `integrations/` framework adapters (LangChain/OpenAI/Anthropic)
- `cloud/` cloud client models (experimental)
- `storage/`, `utils/`, `visualization/`, `tvl/`, etc.

### `/tests/`
Unit, integration, e2e, security, and performance suites.

### `/docs/`
Documentation content:
- `getting-started/` quickstart and install
- `user-guide/` how-to guides
- `api-reference/` API docs
- `architecture/` design docs
- `marketing/` marketing materials and content calendars
- `feature_matrices/`, `traceability/`, `planned_features/`, `plans/`

### `/examples/`
Usage examples and the Examples Navigator (`examples/index.html`).

### `/walkthrough/`
Interactive step-by-step tutorials with datasets.

### `/tools/`
Architecture analysis (`tools/architecture/`) and environment utilities.
CI workflow dependency via `.github/workflows/architecture-analysis.yml`.

### `/scripts/`
Helper scripts (e.g., launchers, verification, utilities).

### `/plugins/`
Optional plugin packages: analytics, tracing, tuned-variables, UI.

### `/requirements/`
Dependency sets by feature. See `requirements/README.md` for install guidance.

## File Organization Principles

1. **Clean Root**: Only essential project directories at root level
2. **Grouped Dependencies**: All requirements files in dedicated directory
3. **Logical Grouping**: Related functionality grouped together
4. **Clear Naming**: Descriptive directory and file names
5. **Separation of Concerns**: Demos and use-cases live in TraigentDemo repo

## Notes

This structure provides a clean, scalable layout that makes the project easy to navigate for both users and contributors.
