# Changelog

All notable changes to Traigent SDK are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Multi-agent parameter and measure mapping system
- `AgentConfiguration` types for explicit agent groupings
- `agent` parameter on `Range`, `Choices`, `IntRange` classes
- `agent_prefixes` for prefix-based agent inference
- TVL support for `agent` field on tvars

### Changed
- Backend session metadata now includes `agent_configuration`
- Improved type safety across API types

## [0.9.0] - 2025-01-09

### Added
- Core optimization decorator (`@traigent.optimize`)
- Grid, Random, and Bayesian optimization strategies
- TVL (Traigent Variable Language) specification support
- LangChain and DSPy integration adapters
- Edge analytics execution mode
- Cloud and hybrid execution modes
- Privacy-preserving optimization mode
- Parallel trial execution support
- Stop conditions (plateau detection, max trials)
- Plugin architecture for extensibility

### Changed
- Refactored plugin architecture for modularity

### Security
- JWT-based authentication for cloud operations
- Input validation and sanitization
