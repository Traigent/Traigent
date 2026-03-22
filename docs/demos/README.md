# Traigent SDK Demo Videos

This directory contains animated terminal demos showcasing Traigent's LLM optimization capabilities.

## Demo Videos

| Demo | Description | Key Concepts |
|------|-------------|--------------|
| **optimize** | Core LLM Agent Optimization | `@traigent.optimize()`, tuned variables, objectives, evaluation datasets |
| **hooks** | Optimization Callbacks | `OptimizationCallback`, lifecycle hooks, progress tracking, parameter importance |
| **github-hooks** | Agent Configuration Hooks | traigent-validate, cost/performance constraints, push rejection on violations |

## Quick Start

```bash
cd docs/demos

# Make scripts executable
chmod +x record-demos.sh scripts/*.sh

# Generate demos
./record-demos.sh
```

## Output Files

After running `record-demos.sh`:

- `output/optimize.cast` - asciinema recording (Video 1)
- `output/hooks.cast` - asciinema recording (Video 2)
- `output/github-hooks.cast` - asciinema recording (Video 3)
- `output/optimize.svg` - animated SVG (if svg-term available)
- `output/hooks.svg` - animated SVG (if svg-term available)
- `output/github-hooks.svg` - animated SVG (if svg-term available)

## Prerequisites

### Required
- Python 3.x
- Bash shell

### Optional (for SVG generation)
```bash
# Install svg-term-cli for animated SVG output
npm install -g svg-term-cli
```

## Directory Structure

```
docs/demos/
├── README.md               # This file
├── record-demos.sh         # Master generation script
├── scripts/
│   ├── generate-cast.py    # Python cast file generator
│   ├── demo-optimize.sh    # Video 1: Core optimization
│   ├── demo-hooks.sh       # Video 2: Optimization callbacks
│   └── demo-github-hooks.sh # Video 3: GitHub hooks and CI/CD
├── output/
│   ├── *.cast              # asciinema format
│   └── *.svg               # Animated SVGs for GitHub
└── test_agents/            # Sample agents + traigent.yml for hooks demo
```

## Embedding in README

```markdown
## See Traigent in Action

### LLM Agent Optimization
![Traigent Optimization](output/optimize.svg)

### Optimization Callbacks
![Traigent Callbacks](output/hooks.svg)

### GitHub Hooks & CI/CD
![Traigent GitHub Hooks](output/github-hooks.svg)
```

## Key Messages

### Video 1: Core Optimization
- **Zero code changes** - just add the decorator
- **Tuned variables** = parameters that affect agent behavior (model, temperature, etc.)
- **Objectives** = what you want to optimize (accuracy, cost, latency)
- **Evaluation-driven** = uses test datasets to measure improvement
- **Results** = find the best configuration automatically

### Video 2: Optimization Callbacks

- **Lifecycle hooks** for monitoring optimization progress
- **Built-in callbacks** for common use cases (progress bar, logging, statistics)
- **Custom callbacks** for notifications (Slack, email, dashboards)
- **Statistics** for parameter importance analysis

### Video 3: Agent Configuration Hooks

- **traigent-validate** blocks pushes with invalid agent configurations
- **Cost constraints** enforce budget limits (max cost per query, monthly budget)
- **Performance constraints** ensure accuracy/latency meet baseline
- **Model constraints** allow/block specific models for production
- **Push rejection** prevents expensive mistakes before they reach production

## Manual Playback

You can preview cast files with asciinema:

```bash
# Install asciinema
pip install asciinema

# Play a recording
asciinema play docs/demos/output/optimize.cast
```

## Reference

See `DEMO_VIDEO_GUIDE.md` for implementation details and customization options.
