# 🔧 Scripts Directory

This directory contains all automation, development, and utility scripts for the Traigent SDK project.

## 📁 Directory Structure

```
scripts/
├── README.md                    # This file - comprehensive script documentation
├── maintenance/                 # 🛠️ Active maintenance tools (RECOMMENDED)
├── analysis/                    # 🔍 Code analysis and pattern detection
├── documentation/               # 📚 Documentation management and dashboards
├── examples/                    # 🚀 Demo scripts and interactive tools
├── integration/                 # 🔗 Integration testing and storage inspection
├── setup/                       # ⚙️ Installation and environment setup
├── test/                        # 🧪 Test automation scripts
├── validation/                  # ✅ Code and documentation validation
├── auto_tune/                   # 🎯 Auto-tuning and optimization scripts
├── config/                      # ⚙️ Configuration files
├── development/                 # 🔨 Development utilities
├── hooks/                       # 🪝 Git hooks and automation
├── linting/                     # 🧹 Code quality and linting
├── quality/                     # 📈 Code quality analysis tools
├── smoke/                       # 💨 Smoke testing scripts
└── utilities/                   # 🛠️ General utilities
```

## ⚠️ **IMPORTANT - New Maintenance Structure**

**For all ongoing maintenance, use these tools in `scripts/maintenance/`:**
- **`traigent_quality_manager.py`** - Unified code quality management
- **`safe_cleanup_manager.py`** - Comprehensive project cleanup
- **`performance_monitor.py`** - Performance monitoring and profiling

## Script Categories by Directory

### 🛠️ **Maintenance (`maintenance/`)** - **USE THESE TOOLS**
**Active maintenance tools** for ongoing project maintenance:

#### **`traigent_quality_manager.py`** - Comprehensive Code Quality Tool
- Unified reporting (flake8, ruff, mypy, custom checks)
- Safe automated fixing with dry-run and rollback
- Automatic backups before making changes
- Interactive and automated modes

```bash
# Generate quality report
python scripts/maintenance/traigent_quality_manager.py --report

# Preview fixes
python scripts/maintenance/traigent_quality_manager.py --fix --dry-run

# Apply fixes interactively
python scripts/maintenance/traigent_quality_manager.py --fix

# Apply fixes automatically
python scripts/maintenance/traigent_quality_manager.py --fix --auto-yes
```

#### **`safe_cleanup_manager.py`** - Comprehensive Project Cleanup Tool
- Safe removal of temporary files, caches, and old reports
- Automatic backup before cleanup
- Interactive and automated modes
- Undo capability

```bash
# Analyze cleanup opportunities
python scripts/maintenance/safe_cleanup_manager.py --analyze

# Preview cleanup
python scripts/maintenance/safe_cleanup_manager.py --cleanup --dry-run

# Interactive cleanup
python scripts/maintenance/safe_cleanup_manager.py --cleanup --interactive
```

#### **`performance_monitor.py`** - System Performance Monitoring
- Monitor resource usage during development
- Track performance metrics over time
- Generate performance reports

### 🔍 **Analysis (`analysis/`)**
**Code analysis and pattern detection scripts:**
- **`find_deprecated_patterns.py`** - Find deprecated patterns after refactoring
- **`find_old_files.py`** - Identify old or unused files
- **`duplicate_detector.py`** - Code duplication detection utility
- **`generate_file_inventory.py`** - Generate comprehensive file inventory

### 📚 **Documentation (`documentation/`)**
**Documentation management and dashboards:**
- **`doc_dashboard.py`** - Interactive documentation health dashboard
- **`sync_docs.py`** - Documentation synchronization and management

### 🚀 **Examples (`examples/`)**
**Demo scripts and interactive tools:**
- **`launch_control_center.py`** - Launch Traigent Playground (Streamlit UI)
- **`run_examples.py`** - Run all example scripts with comprehensive testing

### 🔗 **Integration (`integration/`)**
**Integration testing and storage inspection:**
- **`test-local-integration.py`** - Local integration testing and validation
- **`view_traigent_storage.py`** - Traigent storage inspection and debugging

### ⚙️ **Setup (`setup/`)**
**Installation and environment setup scripts:**
- **`setup_test_environment.py`** - Test environment setup
- **`setup_auto_tuning.sh`** - Auto-tuning environment setup
- **`quickstart.py`** - Quick start for new users
- **`install-dev.sh`** - Development environment installation

### ✅ **Validation (`validation/`)**
**Code and installation validation:**
- **`verify_installation.py`** - Installation verification and diagnostics
- **`verify_tokencost_integration.py`** - TokenCost integration verification

### ⚙️ Configuration (`config/`)
Configuration files moved from root for better organization:
- **`mkdocs.yml`** - MkDocs documentation site configuration
- **`tox.ini`** - Tox testing automation configuration
- **`.pre-commit-config.yaml`** - Pre-commit hooks configuration

### 🛠️ Development (`development/`)
Development and testing utilities:
- **`test_imports.py`** - Security and module import verification script

### 🔍 Linting (`linting/`)
Code quality and linting automation:
- **`install_linters.sh`** - Installs all required linters and formatters
- **`run_linters.sh`** - Runs comprehensive linting checks across the project

## Core Scripts Overview

### Development Setup

#### `setup_test_environment.py`
Sets up the test environment for Traigent development. Run this after cloning the repository for the first time.

**What it does:**
- Creates necessary test fixture directories
- Generates mock JWT keys for testing
- Creates `.env.test` file with test credentials
- Sets up MCP validation fixtures
- Checks for optional test dependencies

**Usage:**
```bash
python scripts/setup/setup_test_environment.py
```

### Testing

#### `test/run_tests.py`
Comprehensive test runner with coverage reporting.

**Usage:**
```bash
python scripts/test/run_tests.py
```

#### `test/run_new_functionality_tests.py`
Simplified test runner for new functionality (apply_best_config and get_optimization_insights).
Runs without requiring full pytest setup and external dependencies.

**Usage:**
```bash
python scripts/test/run_new_functionality_tests.py
```

### User Interface

#### `launch_control_center.py`
Launches the interactive Traigent Playground UI.

**Usage:**
```bash
python scripts/examples/launch_control_center.py
# Or use the shell script:
./scripts/examples/launch_control_center.sh
```

### Utilities

#### `test_setup.py`
Additional test setup utilities.

#### `test-local-integration.py`
Tests local (offline) mode integration functionality.

## Quick Usage Examples

### Development Setup - **NEW ORGANIZATION**
```bash
# Complete development environment setup
./scripts/setup/install-dev.sh

# Quick start for new users
python scripts/setup/quickstart.py

# Set up test environment
python scripts/setup/setup_test_environment.py

# Install all linters and formatters
./scripts/linting/install_linters.sh
```

### Code Quality Management - **NEW TOOLS**
```bash
# 🛠️ RECOMMENDED: Use the new unified quality manager
python scripts/maintenance/traigent_quality_manager.py --report

# Generate and apply fixes safely
python scripts/maintenance/traigent_quality_manager.py --fix --dry-run
python scripts/maintenance/traigent_quality_manager.py --fix

# Performance profiling
python scripts/maintenance/performance_monitor.py

# Legacy linting (still available)
./scripts/linting/run_linters.sh
```

### Project Cleanup Management - **NEW TOOL**
```bash
# 🛠️ RECOMMENDED: Use the new unified cleanup manager
python scripts/maintenance/safe_cleanup_manager.py --analyze

# Safe cleanup with preview
python scripts/maintenance/safe_cleanup_manager.py --cleanup --dry-run
python scripts/maintenance/safe_cleanup_manager.py --cleanup --interactive

# Automated cleanup (with backup)
python scripts/maintenance/safe_cleanup_manager.py --cleanup --auto
```

### Testing and Verification - **NEW ORGANIZATION**
```bash
# Run comprehensive tests
./scripts/test/run_tests.sh

# Verify security imports
python scripts/development/test_imports.py

# Local integration testing
python scripts/integration/test-local-integration.py

# Verify installation
python scripts/validation/verify_installation.py
```

### Analysis and Exploration - **NEW CATEGORY**
```bash
# Find deprecated patterns
python scripts/analysis/find_deprecated_patterns.py

# Detect code duplication
python scripts/analysis/duplicate_detector.py

# Generate file inventory
python scripts/analysis/generate_file_inventory.py

# Documentation dashboard
python scripts/documentation/doc_dashboard.py
```

### Examples and Demos - **NEW CATEGORY**
```bash
# Launch Traigent Playground
python scripts/examples/launch_control_center.py

# Run all examples
python scripts/examples/run_examples.py

# View Traigent storage
python scripts/integration/view_traigent_storage.py
```

## Path Management

All moved scripts now use relative paths and auto-navigate to project root:
```bash
# Standard pattern in moved scripts
cd "$(dirname "$0")/../.." # Navigate to project root from subdirectory
```

Scripts can be run from any location and will automatically find the correct project structure.

## Virtual Environment

Scripts have been updated to use the unified `venv` virtual environment:
- ✅ `venv/` - Primary virtual environment (updated)
- ❌ `traigent_test_env/` - Deprecated, references removed

## Migration Notes

### Moved from Root
These files were organized from project root:
- Configuration files → `scripts/config/`
- Development utilities → `scripts/development/`
- Linting automation → `scripts/linting/`

### Updated References
All scripts updated with:
- Correct relative paths
- Unified virtual environment references
- Auto-navigation to project root
- Improved error handling

## 🚨 **Transition to New Maintenance Approach**

**IMPORTANT CHANGE**: The scripts directory has been reorganized for safety and maintainability.

### ✅ **What to Use Now**
- **`scripts/maintenance/traigent_quality_manager.py`** - For all code quality needs
- **`scripts/maintenance/safe_cleanup_manager.py`** - For all cleanup needs
- **`scripts/maintenance/performance_monitor.py`** - For performance monitoring

### 🔒 **Safety Features**
All new maintenance tools include:
- **Dry-run mode** - Preview changes without applying them
- **Automatic backups** - Create backups before making changes
- **Rollback capability** - Undo changes if needed
- **Interactive confirmation** - Ask before destructive operations
- **Comprehensive logging** - Track all operations

## Best Practices

1. **Always activate virtual environment**: `source venv/bin/activate`
2. **Install development dependencies**: `pip install -e ".[dev]"`
3. **Run setup after fresh clone**: `python scripts/setup/setup_test_environment.py`
4. **🛠️ Use new maintenance tools**: Always use tools in `scripts/maintenance/` for ongoing work
5. **Always run with --dry-run first**: Preview changes before applying them
6. **Keep backups**: Maintenance tools create automatic backups
7. **Check logs**: Review logs in `scripts/maintenance/logs/` if issues occur
8. **Scripts auto-navigate**: Can be run from any directory

---
*Last updated: September 11, 2025 - Major maintenance reorganization*
