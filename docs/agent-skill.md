# AI Agent Skills

Traigent publishes [Agent Skills](https://agentskills.io/) that teach AI coding
agents how to adopt, validate, and run Traigent optimizations.

When you ask Claude Code, Codex, Cursor, or another coding assistant to
"optimize this code with Traigent," the skills should use a **dry-run-first
workflow**: scan and prepare changes without spending tokens, validate in mock
mode, then switch to real execution only when you explicitly approve it.

## Compatible Agents

Works with Claude Code, Cursor, GitHub Copilot, OpenAI Codex, Gemini CLI, Windsurf, and [30+ more](https://agentskills.io/).

## Install

**Recommended — canonical skill bundle:**
```bash
npx skills add Traigent/agents-skills --skill traigent --skill traigent-optimizer
```

> **Note:** The older `Traigent/agent-skills` repo (without the trailing `s`)
> is deprecated and pinned to older skill content.
> Use `Traigent/agents-skills` instead.

If you cloned the SDK repo, fallback skills are available at
[`.agents/skills/traigent/SKILL.md`](../.agents/skills/traigent/SKILL.md) and
[`.agents/skills/traigent-optimizer/SKILL.md`](../.agents/skills/traigent-optimizer/SKILL.md).
The external `Traigent/agents-skills` bundle remains the source of truth.

## What The Skills Do

| Skill | Use it for |
| --- | --- |
| `traigent-optimizer` | Existing-code adoption: run `traigent optimizer scan`, present top candidates, produce a `decorate` dry-run plan, and wait for objective/dataset confirmation. |
| `traigent` | Dry-run validation and real-run handoff after a function is decorated. |
| `traigent-quickstart` | First install, mock mode, and first example optimization. |
| `traigent-configuration-space` | Search-space design, constraints, and typed parameters. |
| `traigent-run-optimization` | Trial budget, algorithms, cost controls, and execution. |

## Helper Boundaries

The skills orchestrate SDK commands; they are not a second implementation of
Traigent. `traigent optimizer scan/decorate` is the adoption assistant. MCP and
hybrid are backend/portal integration layers, not first-run onboarding tools.
Governed autosearch is advanced TVL search after a TVL program already exists.
