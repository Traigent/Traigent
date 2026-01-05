# Traigent UI Plugin

Streamlit-based UI and playground for Traigent SDK.

## Features

- **Interactive Playground**: Experiment with optimization configurations
- **Result Visualization**: View and analyze optimization results
- **Problem Generation**: Generate test problems for optimization
- **Quick Start Wizard**: Guided setup for new optimizations

## Installation

```bash
pip install traigent-ui
```

Or with the UI bundle:

```bash
pip install traigent[ui]
```

## Usage

### Launch the UI

```bash
# Using the CLI command
traigent-ui

# Or using streamlit directly
streamlit run -m traigent_ui.app
```

### Programmatic Access

```python
from traigent_ui import launch_playground

# Launch the Streamlit app
launch_playground()
```

## Components

### Streamlit Core (`streamlit_core/`)
- `quick_start.py`: Quick start wizard
- `optimization.py`: Optimization configuration UI
- `browse_results.py`: Result visualization
- `custom_problems.py`: Custom problem creation
- `navigation.py`: App navigation
- `components.py`: Reusable UI components
- `state.py`: Session state management

### Streamlit Utils (`streamlit_utils/`)
- `example_validation.py`: Example validation utilities
- `progress_tracking.py`: Progress tracking components

### Problem Management (`problem_management/`)
- Problem templates and management
- LangChain problem examples

### Problem Generation (`problem_generation/`)
- Automatic problem generation utilities

## Requirements

- Python 3.10+
- traigent >= 0.9.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- pandas >= 1.5.0

## License

MIT License - see LICENSE file for details.
