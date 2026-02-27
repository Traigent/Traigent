# Pre-PR Prompt Pack

This folder contains reusable prompts for high-signal pre-PR reviews.

## Files

- `00_META_PROMPT.md`: Shared review protocol (tiers, severity, gates, evidence, JSON schema).
- `01_CODE_REVIEW_PROMPT.md`: Correctness/reliability/API compatibility review.
- `02_TEST_REVIEW_PROMPT.md`: Test quality, coverage, determinism, and signal review.
- `03_SECURITY_REVIEW_PROMPT.md`: App security and supply-chain focused review.
- `04_DOCS_EXAMPLES_REVIEW_PROMPT.md`: Documentation and examples accuracy review.
- `05_PRD_SPEC_REVIEW_PROMPT.md`: PRD/spec quality and implementability review.
- `SOURCES.md`: External references used to design this pack.

## How To Use

1. Copy `00_META_PROMPT.md`.
2. Append one specialized prompt file.
3. Fill in placeholders (`{repo}`, `{base_branch}`, `{diff}`, `{pr_description}`, `{depth}`).
4. Run one prompt per review dimension (do not merge all dimensions into one mega-prompt).
5. Merge JSON outputs across dimensions and apply gate rules.

## Suggested Sequence Per PR

1. `01_CODE_REVIEW_PROMPT.md`
2. `02_TEST_REVIEW_PROMPT.md`
3. `03_SECURITY_REVIEW_PROMPT.md`
4. `04_DOCS_EXAMPLES_REVIEW_PROMPT.md` (if docs/examples changed)
5. `05_PRD_SPEC_REVIEW_PROMPT.md` (for design/spec PRs)

## Gate Policy

Use the gate policy from `00_META_PROMPT.md`:
- `BLOCK` for any critical issue.
- `REQUEST_CHANGES` for unresolved high-impact issues.
- `APPROVE` only when blockers are clear and risk is acceptable.
