# Traigent SDK Launch Messaging (Working Doc)

## One-liner

Traigent helps teams **treat AI agents like production software**: define behavior under specification, tune measurable variables, and ship behind CI/CD quality gates.

## Launch state (keep it explicit)

- **OSS today**: local execution (`edge_analytics`), mock mode, CI example patterns.
- **Roadmap unless provisioned**: managed cloud/hybrid backend, hosted eval infrastructure, production SLAs.

## Core narrative (repeat weekly)

1. **AI agents are software**: they need specs, tests, gates, and ops—not “prompt tweaking”.
2. **Tuned Variables** are the new configuration layer: dynamic, measurable, continuously optimized.
3. **Agents Under Specification**: a spec becomes the “contract” for what matters (objectives/constraints/budgets/promotion rules).
4. **Agent CI/CD + AgentsOps**: regression detection, promotion gates, observability, drift → re-tune → safe deploy.

## Terminology (pick one for public consistency)

**Decision:** TVL = **Tuned Variable Language**.

Use “TVL spec” when you want brevity; expand to “Tuned Variable Language (TVL)” the first time in long-form content.

## Positioning pillars (what we want people to remember)

- **Specification-first**: declare objectives, constraints, budgets, and promotion rules.
- **Low-friction adoption (for common stacks)**: decorator + injection + framework overrides for supported patterns; custom stacks may need adapter code.
- **Production discipline**: run evaluations, compare, and gate changes (like unit tests + CI for agents).
- **Privacy + control**: local execution mode available; privacy flags; telemetry can be disabled.

## Voice & tone (default)

- Technical peer, not salesy: concrete examples, honest limitations, clear CTAs.
- Avoid hype words and absolute claims (“revolutionary”, “always”, “zero effort”).
- When mentioning statistics, prefer plain English first; keep formal terms for ML appendix.

## Why now (market timing angles)

Pick 1–2 per post/article and keep it concrete:
- **Agents are shipping into production** → regressions and drift become operational problems.
- **Cost blowouts** → teams need budgets, tradeoffs, and repeatable tuning.
- **Governance pressure** (security, compliance, reliability) → “spec + gates” becomes a forcing function.

## Competitive framing (one-liner)

- Observability tools help you **see** agent behavior; Traigent helps you **systematically improve and gate** it against a spec. More in `marketing/AUDIENCE_CHANNEL_STRATEGY.md`.

## Claims to avoid (until explicitly true)

- “Cloud/hybrid is available now” (OSS currently focuses on local `edge_analytics`; cloud/hybrid is roadmap-compatible).
- “Always cheaper / always better” (use probabilistic language; show measurement + gates instead).
- “No telemetry ever” (telemetry exists; it’s opt-out and can be minimized with privacy settings).

## Calls-to-action (rotate; keep them concrete)

- **Try it locally (no API spend)**: run quickstart with mock mode:
  - `export TRAIGENT_MOCK_MODE=true`
  - `python examples/quickstart/01_simple_qa.py`
- **CI gates example**: point to `examples/integrations/ci-cd/README.md`
- **Minimal agent spec template**: `marketing/templates/AGENT_SPEC_TEMPLATE.md`
- **Integration expectations**: `marketing/INTEGRATION_COMPATIBILITY_MATRIX.md`
- **Read the API**: `docs/api-reference/decorator-reference.md`
- **Privacy/telemetry**: `docs/api-reference/telemetry.md`
- **Architecture (for platform teams)**: `docs/architecture/ARCHITECTURE.md`

## Visual assets already in-repo

- Optimization demo: `docs/demos/output/optimize.svg`
- Callbacks demo: `docs/demos/output/hooks.svg`
- GitHub hooks demo: `docs/demos/output/github-hooks.svg`

## Anchor narrative asset (development workflow)

- Agent development playbook draft: `docs/agent_development_script.md`
- Series plan for repurposing by audience/channel: `marketing/AGENT_DEV_PLAYBOOK_SERIES.md`
- Audience/channel mapping + jargon ladder: `marketing/AUDIENCE_CHANNEL_STRATEGY.md`

## Hashtag set (mix 3–6 per post; avoid stuffing)

- #AIAgents
- #LLM
- #SoftwareEngineering
- #MLOps
- #DevOps
- #CICD
- #AgentOps
