# AI Agent Skill

Traigent publishes [Agent Skills](https://agentskills.io/) — sets of instructions that teach your AI coding agent how to set up and run Traigent optimizations.

When you ask your agent to "optimize this function with Traigent," the skill guides it through a **dry-run-first workflow**: validate the documented Traigent path at zero provider cost in mock mode, then switch to real execution only when you say so. If your code makes provider calls outside Traigent's optimized/evaluated path, stub those calls explicitly before treating the rehearsal as guaranteed $0.

## Compatible Agents

Works with Claude Code, Cursor, GitHub Copilot, OpenAI Codex, Gemini CLI, Windsurf, and [30+ more](https://agentskills.io/).

## Install

The canonical SDK skill source is [`Traigent/traigent-skills`](https://github.com/Traigent/traigent-skills). SDK checkouts do not vendor `.agents/` or `.claude/skills/` copies; install from the remote repo so your local agent cache can be updated independently.

**Install all SDK skills:**
```bash
npx skills add Traigent/traigent-skills --skill '*'
```

**Install only the Python entry points:**
```bash
npx skills add Traigent/traigent-skills --skill traigent --skill traigent-quickstart
```

## What the Skill Does

1. Sets up the `@traigent.optimize()` decorator
2. Validates your dataset and config space
3. Runs a mock optimization (zero cost) to verify the full pipeline
4. Reports results and estimates real costs
5. Waits for your go-ahead before spending real tokens

## View the Skills

[Read the canonical skill instructions →](https://github.com/Traigent/traigent-skills/tree/main/skills/traigent)
