# Traigent SDK Project Structure

This document outlines the current structure of the Traigent SDK project.

## Root Directory (current)

```
Traigent/
├── README.md                  # Main project documentation
├── LICENSE                    # MIT license
├── pyproject.toml             # Build and packaging config
├── pytest.ini                 # Test configuration
├── requirements/              # Dependency sets and guides
├── traigent/                  # Core SDK source code
├── tests/                     # Test suite
├── examples/                  # Examples and demo runners
├── scripts/                   # Utility scripts
├── docs/                      # Documentation (this folder)
├── reports/                   # Project reports and audits
├── baselines/                 # Baseline artifacts
├── data/, results*/           # Local datasets and run outputs
├── docs/tvl/                  # TVL website source and assets
└── integrations_* / tools/…   # Integration plans, tooling, inventories
```

## Key Directories Explained

### `/requirements/`
Dependency sets by feature. See `requirements/README.md` for install guidance.

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
- `feature_matrices/`, `traceability/`, `planned_features/`, `plans/`

### `/examples/`
Usage examples and the Examples Navigator (`examples/index.html`).

### `/reports/`
Project audits, quality reviews, and status reports.

### `/scripts/`
Helper scripts (e.g., launchers, verification, utilities).

## File Organization Principles

1. **Clean Root**: Only essential project files in root directory
2. **Grouped Dependencies**: All requirements files in dedicated directory
3. **Archived Reports**: Temporary analysis files moved to docs/archive
4. **Logical Grouping**: Related functionality grouped together
5. **Clear Naming**: Descriptive directory and file names

## Notes

This structure provides a clean, scalable layout that makes the project easy to navigate for both users and contributors.
