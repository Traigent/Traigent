# Getting Started with Code Review System

## 🎯 Overview

This guide will help you set up and run the Code Review System for your Python project.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Node.js 18+** (for the web viewer)
- **API Access** to an LLM provider (Anthropic Claude, OpenAI, etc.)
- **Git** (optional, for version control)

## 🚀 Installation

### Step 1: Set Up Python Environment

```bash
# From the repository root, move into the tool directory
cd tools/code_review/

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all,dev]"
```

### Step 2: Set Up API Credentials

```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-api-key-here"
# OR
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
# .env
ANTHROPIC_API_KEY=your-api-key-here
OPENAI_API_KEY=your-api-key-here
```

### Step 3: Set Up Web Viewer (Optional)

```bash
# Navigate to viewer
cd viewer/

# Install dependencies
npm install

# Build for production (optional)
npm run build

# Or run in development mode
npm run dev
```

## 🔧 Configuration

### Basic Configuration

The system works out of the box with sensible defaults. Review tracks are defined in:

```
automation/
├── code_quality/instructions.md
├── performance/instructions.md
├── security/instructions.md
└── soundness_correctness/instructions.md
```

### Custom Configuration

To customize checks, edit the `required_checks.json` file in each track:

```json
{
  "scope": ["function", "class", "module"],
  "min_issues": 1,
  "required_categories": ["complexity", "maintainability"]
}
```

## 📝 Running Your First Review

### Option 1: Review Entire Project

```bash
# From the repository root
python tools/code_review/automation/run_all_validations.py --folder traigent/
```

### Option 2: Review Specific Module

```bash
# Review single file from the repo root
python tools/code_review/automation/run_all_validations.py --module traigent/core/orchestrator.py
```

### Option 3: Using Individual Track Scripts

```bash
# Run specific track review from the repo root
python tools/code_review/automation/code_quality/validate_module.py \
  --module traigent/core/orchestrator.py \
  --report reports/1_quality/automated_reviews/code_quality/traigent/core/orchestrator.py.review.json
```

## 📊 Viewing Results

### Web Viewer (Recommended)

```bash
cd viewer/
npm run dev

# Open browser to http://localhost:5173
```

Features:
- Interactive browsing
- Filtering by track, severity, module
- Full-text search
- Export to JSON/CSV
- Dark mode

### Command Line

```bash
# View review file
cat reports/1_quality/automated_reviews/code_quality/module.py.review.json | jq

# Count total issues
cat reports/1_quality/automated_reviews/code_quality/module.py.review.json | jq '.summary.total_issues'

# List high severity issues
cat reports/1_quality/automated_reviews/code_quality/module.py.review.json | jq '.issues[] | select(.severity == "high")'
```

### Python Script

```python
import json
from pathlib import Path

def load_review(module_path, track="code_quality"):
    reviews_root = Path("../../reports/1_quality/automated_reviews")
    review_file = reviews_root / track / f"{module_path}.review.json"
    with open(review_file) as f:
        return json.load(f)

# Load and analyze
review = load_review("traigent/core/orchestrator.py")
print(f"Total Issues: {review['summary']['total_issues']}")

for issue in review['issues']:
    if issue['severity'] == 'high':
        print(f"HIGH: {issue['description']}")
```

## 🎨 Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up environment
cd tools/code_review/
source .venv/bin/activate
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Run reviews on your project
python automation/run_all_validations.py --folder ../traigent/

# 3. Start the viewer
cd viewer/
npm run dev

# 4. Open browser to http://localhost:5173
# Browse results, filter, search, export

# 5. Fix issues in your code
# vim /path/to/your/project/module.py

# 6. Re-run reviews to verify fixes
cd ..
python automation/run_all_validations.py --module ../traigent/core/orchestrator.py
```

## 🔍 Understanding Review Tracks

### Code Quality
**Purpose**: Assess maintainability and best practices
**Focus Areas**:
- Code complexity
- Naming conventions
- Documentation quality
- Design patterns

### Performance
**Purpose**: Identify efficiency improvements
**Focus Areas**:
- Algorithm complexity
- Data structure choices
- Caching opportunities
- Resource usage

### Security
**Purpose**: Find security vulnerabilities
**Focus Areas**:
- Input validation
- Authentication/authorization
- Data encryption
- Injection vulnerabilities

### Soundness & Correctness
**Purpose**: Verify logic correctness
**Focus Areas**:
- Error handling
- Edge cases
- Boundary conditions
- Type safety

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'anthropic'`
```bash
# Solution: Install dependencies
pip install -e ".[all]"
```

**Issue**: `API key not found`
```bash
# Solution: Set environment variable
export ANTHROPIC_API_KEY="your-key"
# Or check .env file
```

**Issue**: `Permission denied: output/code_quality/`
```bash
# Solution: Create output directories
mkdir -p output/{code_quality,performance,security,soundness_correctness}
```

**Issue**: Viewer won't start
```bash
# Solution: Reinstall dependencies
cd viewer/
rm -rf node_modules/
npm install
npm run dev
```

### Getting Help

1. Check logs in `automation/*.log`
2. Verify API keys are set
3. Ensure output directories exist
4. Check Python/Node versions match requirements

## 📚 Next Steps

- **Add Custom Tracks**: See [adding-tracks.md](adding-tracks.md)
- **CI/CD Integration**: See [integration-guide.md](integration-guide.md)
- **Customize Instructions**: Edit `automation/<track>/instructions.md`
- **Advanced Configuration**: See main [README.md](../README.md)

## 🤝 Contributing

Found a bug? Want to add a feature? See the main README for contribution guidelines.

---

**Happy Reviewing!** 🎉
