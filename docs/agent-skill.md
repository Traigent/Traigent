# AI Agent Skill

Traigent ships a built-in [Agent Skill](https://agentskills.io/) — a set of instructions that teach your AI coding agent how to set up and run Traigent optimizations.

When you ask your agent to "optimize this function with Traigent," the skill guides it through a **dry-run-first workflow**: validate everything at zero cost in mock mode, then switch to real execution only when you say so.

## Compatible Agents

Works with Claude Code, Cursor, GitHub Copilot, OpenAI Codex, Gemini CLI, Windsurf, and [30+ more](https://agentskills.io/).

## Install

**Option A — npx (all 7 SDK skills):**
```bash
npx skills add Traigent/agents-skills
```

**Option B — already included:**

If you cloned the Traigent repo, the skill is already at [`.agents/skills/traigent/SKILL.md`](../.agents/skills/traigent/SKILL.md). Agents that support the `.agents/` convention will discover it automatically.

## What the Skill Does

1. Sets up the `@traigent.optimize()` decorator
2. Validates your dataset and config space
3. Runs a mock optimization (zero cost) to verify the full pipeline
4. Reports results and estimates real costs
5. Waits for your go-ahead before spending real tokens

## View the Skill

[Read the full skill instructions →](../.agents/skills/traigent/SKILL.md)
