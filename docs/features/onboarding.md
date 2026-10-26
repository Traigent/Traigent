# Onboarding & CLI

Traigent provides guided CLI surfaces for getting a Python project from
install to a safe first run.

## Commands

```bash
traigent onboard
traigent auth device-login
traigent first-prompt --agent codex
traigent quickstart
```

`traigent onboard` detects common Python project markers, supported coding
agents, current authentication state, MCP availability, and then prints or runs
the next setup steps.

In a terminal, onboarding can offer to:

- add `traigent[integrations]` to the project
- start device login unless `--no-login` is set
- install Traigent agent skills — `--profile beginner` (default) via
  `npx skills add Traigent/traigent-first-run`, or `--profile advanced` via
  `npx skills add Traigent/traigent-skills`
- register MCP for detected agents when `traigent mcp --help` succeeds
- run the packaged mock quickstart
- print the first prompt for the detected coding agent

In non-interactive mode, onboarding prints a plan and JSON block instead of
starting login by default. Use `--login` to run device login after the plan.

## Exit codes

These exit codes cover these four commands only. They are pre-1.0 and may
change in a minor release. They are pinned by
`tests/unit/cli/test_documented_exit_codes.py`.

| Command | Exit | Meaning |
|---|---|---|
| `traigent quickstart` | 0 | Mock demo finished; plumbing check passed |
| `traigent quickstart` | 1 | No trial succeeded, so no best configuration was found |
| `traigent validate` | 0 | Dataset passed validation |
| `traigent validate` | 1 | Dataset path or content failed validation |
| `traigent validate` | 2 | Usage error, such as a dataset path that does not exist |
| `traigent check` | 0 | All checked functions improve on defaults, `--dry-run` completed, or nothing was found with `--allow-empty` |
| `traigent check` | 1 | No optimizable function found, the module path does not exist (reported as 1, not 2), or a function did not improve on its defaults |
| `traigent check` | 2 | Usage error, such as a missing `MODULE_PATH` argument |
| `traigent optimize` | 0 | Every optimizable function in the file was optimized |
| `traigent optimize` | 1 | File failed to load, no decorated function was found, or an optimization failed |
| `traigent optimize` | 2 | Usage error, such as a file path that does not exist |

In general, any other failure exits 1; usage errors exit 2 (click's default).

## Device Login

```bash
traigent auth device-login
traigent auth device-login --write-env
traigent auth device-login --backend-url https://portal.traigent.ai
```

The command starts an RFC 8628-style browser device-code flow against the
resolved API base, which defaults to the cloud portal at
`https://portal.traigent.ai`. It prints a verification URL and one-time code,
then polls until the backend returns a project-scoped API key or the request
expires.

By default, credentials are stored in the secure local credential store. With
`--write-env`, the command also writes project credentials to a local `.env`
file in the current directory.

## First Prompt

```bash
traigent first-prompt --agent claude
traigent first-prompt --agent cursor
traigent first-prompt --agent codex
```

The first prompt is a paste block for coding agents. It tells the agent to read
the Traigent agent guide, inspect the project, run dry-run/mock mode first, avoid
real provider spend without approval, and finish with evidence and changed
files.

## Quickstart

```bash
traigent quickstart
```

`quickstart` runs the bundled mock-mode demo as a plumbing check: it verifies
the install and the optimize loop using a mock LLM and simulated scores. It is
not a model comparison and does not report a winner; for a real optimization,
set a provider API key (see `walkthrough/real/01_tuning_qa.py`). It does not
require provider API keys and does not make LLM provider calls. LiteLLM may still try an import-time
model-cost-map fetch and fall back to bundled pricing data when offline; the
guarantee is no provider spend, not zero outbound packets.

## Copy-Paste Example

```bash
python3 -m pip install "traigent[integrations]"
traigent auth device-login --write-env
traigent quickstart
traigent first-prompt --agent codex
```

Honesty note: `traigent onboard` is a guided setup helper, not an installer that
silently changes every project. Interactive dependency, skill, MCP, and `.env`
steps ask for consent; non-interactive mode prints a plan for a caller to run.
