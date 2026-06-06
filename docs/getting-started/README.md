# Getting Started

Quick start guides to get you up and running with Traigent SDK.

## Contents

- **[Installation](installation.md)** - Install Traigent SDK and dependencies
- **[Getting Started](GETTING_STARTED.md)** - Your first optimization in 5 minutes
- **[Testing](testing.md)** - Run tests safely in mock mode
- **[Minimum Integration](minimal-integration.md)** - Fastest API/SDK path to first successful run

## Quick Path

1. [Install Traigent](installation.md)
2. [Follow the getting started tutorial](GETTING_STARTED.md)
3. [Use minimum integration checklist](minimal-integration.md) to validate readiness
4. [Explore user guides](../user-guide/) for advanced features

## New in 0.12.0

- Onboarding CLI: `traigent onboard`, `traigent auth device-login`, `traigent quickstart`, and `traigent first-prompt --agent claude|cursor|codex`.
- Local agent tooling: `traigent mcp serve` exposes the stdio MCP server for coding agents.
- Configuration recommendations: `traigent recommend`, `recommend_configuration_space()`, and `list_recommendation_agent_types()`; see [configuration spaces](../user-guide/configuration-spaces.md).
- Advisory strategy presets: `max_accuracy_then_cheapest_within_epsilon`, `quality_floor_min_cost`, and `pareto_frontier`; see [decorator reference](../api-reference/decorator-reference.md).
- Workflow observability: `add_agent_span()` records user-defined agent spans; see [telemetry](../api-reference/telemetry.md).
