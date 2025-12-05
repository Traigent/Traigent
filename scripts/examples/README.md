# 🚀 Examples & Demo Scripts

Scripts for launching demonstrations, running examples, and showcasing functionality.

## Scripts

- **`launch_control_center.py`** - Launch TraiGent Playground (Streamlit UI)
- **`launch_control_center.sh`** - Shell script launcher for control center
- **`run_examples.py`** - Run all example scripts with comprehensive testing
- **`build_gallery.py`** - Regenerate `examples/docs/index.html` from `catalog.yaml`

## Usage

```bash
# Launch interactive playground
python scripts/examples/launch_control_center.py
# OR
./scripts/examples/launch_control_center.sh

# Run all core examples
python scripts/examples/run_examples.py --base examples/core --pattern run.py

# Run advanced gallery (structure only)
python scripts/examples/run_examples.py --base examples/advanced --pattern run.py --no-structure-validation

# Rebuild the examples landing page after updating catalog.yaml
python scripts/examples/build_gallery.py
```

## Features

### TraiGent Control Center
- Interactive Streamlit dashboard
- Real-time optimization monitoring
- Configuration management
- Results visualization

### Example Runner
- Automated example execution
- Comprehensive test coverage
- Performance benchmarking
- Error reporting and debugging
