# User Guide

Comprehensive guides for using Traigent SDK features and capabilities.

## Contents

- **[Agent Optimization](agent_optimization.md)** - Optimize AI agents and multi-step workflows
- **[Choosing Models](choosing_optimization_model.md)** - Select the right `algorithm` / `offline` routing model for your use case
- **[Configuration Spaces](configuration-spaces.md)** - Define tunable variables, parameter ranges, and presets
- **[Tuned Variables](tuned_variables.md)** - Auto-detect tunable parameters and use the callable discovery API
- **[Injection Modes](injection_modes.md)** - Different ways to integrate Traigent with your code
- **[Interactive Optimization](interactive_optimization.md)** - Real-time optimization workflows and monitoring
- **[Evaluation Guide](evaluation_guide.md)** - Measure and improve model performance
- **[Smart Optimization](smart_optimization.md)** - Use backend-routed smart optimizer strategies

## Learning Path

**New Users**: Start with [Injection Modes](injection_modes.md) to understand integration options

**Basic Optimization**: Learn [Choosing Models](choosing_optimization_model.md) for your use case

**Tuned Variables**: Use [Tuned Variables](tuned_variables.md) to auto-detect parameters and discover callables

**Advanced Users**: Explore [Agent Optimization](agent_optimization.md), [Interactive Optimization](interactive_optimization.md), and [Smart Optimization](smart_optimization.md)

**Evaluation**: Use [Evaluation Guide](evaluation_guide.md) to measure success

## CLI & Observability Tools

- Onboarding: `traigent onboard`, browser `traigent auth device-login`, `traigent quickstart`, and `traigent first-prompt --agent claude|cursor|codex`.
- Local MCP: `traigent mcp serve` provides the stdio MCP server for coding agents.
- Observability: `add_agent_span()` adds user-defined agent workflow spans; see [telemetry](../api-reference/telemetry.md).

See [CHANGELOG.md](../../CHANGELOG.md) for what's new in each release.
