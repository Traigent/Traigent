# Specialized Prompt: Docs & Examples Review

Append this after `00_META_PROMPT.md`.

```text
Set `review_type` to `docs_examples`.

Focus priorities (highest first):
1) Factual correctness: docs match actual behavior in changed code.
2) Executability: examples are runnable/copy-paste-safe with explicit prerequisites.
3) Compatibility: versions, flags, env vars, and API signatures are aligned.
4) Safety: examples do not normalize insecure patterns (secret leakage, unsafe defaults).
5) User outcome clarity: expected outputs and failure guidance are accurate.

Review method:
- Cross-check changed public API/function signatures against docs/examples.
- Validate import paths, argument names/defaults, and command syntax.
- Verify examples include required setup (auth/env/dependencies) and realistic output.
- Check migration notes if behavior changed.
- Check for stale or contradictory statements across touched docs.

Must-fix triggers (usually P0/P1):
- Example or command that fails for default user setup due to incorrect instructions.
- Docs that describe nonexistent or materially incorrect API behavior.
- Missing warning on changed/breaking behavior that will cause user-facing failures.
- Example that instructs insecure handling of credentials or sensitive data.

Medium-risk examples (usually P2):
- Outdated option names or parameters with easy workaround.
- Missing edge-case caveat that causes confusion for common workflows.
- Incomplete migration guidance for changed behavior.

De-prioritize (usually P3 unless risk-elevated):
- Grammar/style improvements without ambiguity reduction.
- Structural docs preferences outside PR scope.
```
