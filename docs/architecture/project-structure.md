# Traigent SDK Project Structure

This document outlines the current structure of the Traigent SDK project.

## Root Directory (current)

```
Traigent/
├── README.md                  # Main project documentation
├── LICENSE                    # MIT license
├── pyproject.toml             # Build and packaging config
├── uv.lock                    # Lockfile (uv)
├── pytest.ini                 # Test configuration
├── requirements/              # Dependency sets and guides
├── traigent/                  # Core SDK source code
├── tests/                     # Test suite
├── examples/                  # Examples and demo runners
├── use-cases/                 # End-to-end agent use cases
├── demos/                     # TVL demo scripts
├── scripts/                   # Utility scripts
├── docs/                      # Documentation (this folder)
├── baselines/                 # Baseline artifacts
├── data/, local_results/, results-ci/ # Local datasets and run outputs
├── paper_experiments/         # Research case study pipeline (historical)
├── tools/                     # Automation and traceability utilities
└── marketing/, runtime/, stress_tests/ # Support tooling and experiments
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

### `/demos/`
Terminal demo scripts (TVL) for marketing and docs.

### `/tools/`
Automation and traceability utilities.

### `/scripts/`
Helper scripts (e.g., launchers, verification, utilities).

## File Organization Principles

1. **Clean Root**: Only essential project files in root directory
2. **Grouped Dependencies**: All requirements files in dedicated directory
3. **Analysis Outputs**: Generated artifacts live in `local_results/`, `results-ci/`, `mlruns/`, or `htmlcov/`
4. **Logical Grouping**: Related functionality grouped together
5. **Clear Naming**: Descriptive directory and file names

## Notes

This structure provides a clean, scalable layout that makes the project easy to navigate for both users and contributors.
