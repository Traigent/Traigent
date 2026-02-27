# Claude Opus Consultation Notes

Date: 2026-02-26

Command used:

```bash
claude --model claude-opus-4-6 --effort high -p "<meta-prompt design request>"
```

Key recommendations incorporated:

1. Use one shared base skeleton across all review types.
2. Enforce depth tiers (`quick|standard|deep`) to control review scope.
3. Require evidence per finding (file/lines + scenario + impact + fix).
4. Add strict anti-hallucination rules and explicit uncertainty labeling.
5. Use a machine-parseable JSON schema for downstream automation.
6. Separate specialized prompts by domain instead of one combined mega-prompt.
7. Tie findings severity directly to merge gates (`APPROVE|REQUEST_CHANGES|BLOCK`).

Implementation mapping:

- Base skeleton: `00_META_PROMPT.md`
- Domain prompts: `01` through `05`
- Evidence and gate rules: centralized in `00_META_PROMPT.md`
