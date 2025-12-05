# Code Review System

A comprehensive, reusable code review automation system for Python projects with visual review interface.

## 🎯 Overview

This standalone package (located in `tools/code_review/` in this repo) provides:
- **Automated Code Reviews**: Multi-track review system (code quality, performance, security, correctness)
- **Interactive Viewer**: React-based web interface to browse and analyze review results
- **Standardized Output**: JSON-based review reports with validation
- **Extensible Framework**: Easy to add custom review tracks and checks

## 📁 Structure

```
tools/code_review/
├── automation/          Python automation scripts
│   ├── code_quality/   Code quality review prompts and checks
│   ├── performance/    Performance review prompts and checks
│   ├── security/       Security review prompts and checks
│   ├── soundness_correctness/  Logic and correctness checks
│   ├── _shared/        Shared utilities and validators
│   ├── run_all_validations.py  Main validation orchestrator
│   ├── INSTRUCTIONS.md  Review protocol documentation
│   └── README.md       Automation documentation
│
├── viewer/             React + TypeScript web interface
│   ├── src/           Source code
│   ├── public/        Static assets
│   ├── package.json   Dependencies
│   └── README.md      Viewer documentation
│
├── docs/              Additional documentation
│   ├── getting-started.md
│   ├── adding-tracks.md
│   └── integration-guide.md
│
├── .gitignore        Git ignore patterns
├── setup.py          Package installation
└── README.md         This file
```

> Report outputs are stored in `reports/1_quality/automated_reviews/<track>/` at the repository root.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+ (for viewer)
- LLM API access (Anthropic Claude, OpenAI, etc.)

### Installation

```bash
# From the repository root
cd tools/code_review

# Install Python dependencies
pip install -e .

# Install viewer dependencies
cd viewer/
npm install
```

### Running Reviews

```bash
# From the repository root, run all validation checks
python tools/code_review/automation/run_all_validations.py --folder traigent/

# Run for specific module
python tools/code_review/automation/run_all_validations.py --module traigent/core/cache_policy.py
```

### Viewing Results

```bash
# Start the viewer
cd viewer/
npm run dev

# Open browser to http://localhost:5173
# Browse review results interactively
```

## 📊 Review Tracks

### 1. Code Quality
- **Focus**: Maintainability, readability, best practices
- **Checks**: Complexity, naming, documentation, patterns
- **Output**: `reports/1_quality/automated_reviews/code_quality/`

### 2. Performance
- **Focus**: Efficiency, optimization opportunities
- **Checks**: Algorithms, data structures, bottlenecks
- **Output**: `reports/1_quality/automated_reviews/performance/`

### 3. Security
- **Focus**: Vulnerabilities, security best practices
- **Checks**: Input validation, authentication, encryption
- **Output**: `reports/1_quality/automated_reviews/security/`

### 4. Soundness & Correctness
- **Focus**: Logic correctness, edge cases
- **Checks**: Error handling, boundary conditions, invariants
- **Output**: `reports/1_quality/automated_reviews/soundness_correctness/`

## 🔧 Configuration

### Adding Custom Tracks

1. Create new track directory: `automation/your_track/`
2. Add prompts: `automation/your_track/instructions.md`
3. Add checks: `automation/your_track/required_checks.json`
4. Update `run_all_validations.py` TRACKS dictionary

### Customizing Checks

Edit `automation/<track>/required_checks.json`:

```json
{
  "scope": ["function", "class", "module"],
  "min_issues": 1,
  "required_categories": ["category1", "category2"]
}
```

## 📝 Output Format

Reviews are saved as JSON files:

```json
{
  "module_path": "path/to/module.py",
  "review_track": "code_quality",
  "issues": [
    {
      "severity": "medium",
      "category": "complexity",
      "scope": "function",
      "location": "function_name:42",
      "description": "Issue description",
      "recommendation": "Suggested fix"
    }
  ],
  "summary": {
    "total_issues": 5,
    "by_severity": {"high": 1, "medium": 3, "low": 1}
  },
  "metadata": {
    "reviewed_at": "2024-10-14T12:00:00Z",
    "reviewer": "claude-sonnet-4.5",
    "automated": true
  }
}
```

## 🌐 Web Viewer Features

- **Interactive Browse**: Navigate through all review results
- **Filtering**: Filter by track, severity, module
- **Search**: Full-text search across issues
- **Drill-Down**: Click to see full issue details
- **Export**: Export filtered results to JSON/CSV
- **Dark Mode**: Toggle dark/light theme

## 🔄 Integration

### CI/CD Integration

```yaml
# .github/workflows/code-review.yml
- name: Run Code Reviews
  run: |
    python tools/code_review/automation/run_all_validations.py --folder traigent
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
python tools/code_review/automation/run_all_validations.py --folder traigent || exit 1
```

## 📦 Standalone Git Project Setup

To convert this into a standalone git repository:

```bash
# Initialize git
cd tools/code_review/
git init
git add .
git commit -m "Initial commit: Code Review System v1.0"

# Add remote (optional)
git remote add origin <your-repo-url>
git push -u origin main
```

## 🤝 Contributing

This is designed to be a reusable asset. To contribute:

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

[Specify your license here]

## 🔗 Related Projects

- **TraiGent SDK**: AI optimization framework (parent project)
- **OptiGen**: LLM experimentation platform

## 📞 Support

For issues, questions, or contributions, please contact [your contact info].

---

**Version**: 1.0.0
**Last Updated**: October 14, 2024
**Extracted From**: TraiGent SDK cleanup effort
