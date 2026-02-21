# Archive Branches Summary (2026-02-18)

## Snapshot
- Date: 2026-02-18
- Current remote branches: `5`
- Current remote branches list:
  - `main`
  - `develop`
  - `copilot/add-llm-support-to-issues` (open PR #113)
  - `feat/test-improvements-hybrid-validation` (open PR #117)
  - `feature/reasoning-tuned-variables` (active local worktree branch)
- Canonical archive tags kept: `15`
- Redundant archive tags removed: `16` (including `archive/2026-02-18/feature__text2sql-eval-generation-poc`)
- Containment status among remaining archive tags: `no overlaps` (none is fully contained in another remaining archive tag)

## How To Read This
- `Unique vs main` / `Unique vs develop` = number of patch-level commits (`git cherry`, `+`) present in the archive tag and not present in that base branch.
- PR linkage is based on `gh pr list --state all --head <original-branch>`.

## Archive Inventory (Canonical Set)

| Archive Tag | Original Branch | Head | Last Commit | PR Linkage | Unique vs main | Unique vs develop | Last Commit Subject |
|---|---|---|---|---|---:|---:|---|
| `archive/2026-02-18/api-tests-Mastra-JS` | `api-tests-Mastra-JS` | `f813b14` | 2026-02-10, Israel | `#105 MERGED->develop` | 25 | 20 | docs(hybrid): remove audit file, moved to GitHub issue #107 |
| `archive/2026-02-18/code_fixes` | `code_fixes` | `0379e35` | 2026-02-08, Israel | `-` | 64 | 64 | docs: Add what-can-you-optimize guide and improve user-guide navigation |
| `archive/2026-02-18/dev__real-llm-examples_exp` | `dev/real-llm-examples_exp` | `1c03b4e` | 2026-01-06, EladTraignet | `-` | 8 | 8 | feat: Add retrieval logging to RAG optimization |
| `archive/2026-02-18/feat__openapi-enhanacment-from-develop` | `feat/openapi-enhanacment-from-develop` | `fdc9df2` | 2026-02-10, nimrodbusany | `#104 MERGED->develop` | 20 | 15 | docs(hybrid-api): fix 8 issues from doc audit (Issue #107) |
| `archive/2026-02-18/feature__haystack-integration` | `feature/haystack-integration` | `68f7fcc` | 2025-12-23, nimrod | `-` | 4 | 4 | docs: Fix README code examples and add advanced guides |
| `archive/2026-02-18/feature__injection-mode-consolidation` | `feature/injection-mode-consolidation` | `4a3dc69` | 2026-01-26, nimrodbusany | `#47 MERGED->develop` | 1 | 1 | test: Remove ATTRIBUTE injection mode references from tests |
| `archive/2026-02-18/feature__quality-improvement` | `feature/quality-improvement` | `a1748de` | 2026-02-07, nimrodbusany | `#96 MERGED->main` | 4 | 4 | fix: Add _estimate_optimization_cost delegation stub to orchestrator |
| `archive/2026-02-18/feature__quickstart-detailed-csv-output` | `feature/quickstart-detailed-csv-output` | `4271417` | 2025-12-17, EladTraignet | `-` | 6 | 6 | feat(examples): Add summary row with pass rates to main CSV |
| `archive/2026-02-18/feature__real-llm-examples` | `feature/real-llm-examples` | `dde2d5d` | 2025-12-31, EladTraignet | `-` | 80 | 80 | feat: Add real LLM examples and working files |
| `archive/2026-02-18/feature__results-table-ansi-colors` | `feature/results-table-ansi-colors` | `95f1601` | 2026-02-12, Elad@Traigent.ai | `-` | 38 | 31 | refactor: Rename Value→Grade column and VAL→GRD badge throughout results table |
| `archive/2026-02-18/fix__codex-audit-remediation` | `fix/codex-audit-remediation` | `4f38d9b` | 2026-02-08, nimrodbusany | `#103 MERGED->develop; #102 MERGED->develop` | 10 | 6 | Align hybrid API contract and Traigent integration flow |
| `archive/2026-02-18/fix__cost-tracking-prompt-tokens` | `fix/cost-tracking-prompt-tokens` | `acb04ef` | 2026-01-26, EladTraignet | `-` | 25 | 25 | fix: Add SDK docstrings to prevent config injection misuse |
| `archive/2026-02-18/rag-up-wip` | `rag-up-wip` | `099d251` | 2025-12-31, EladTraignet | `-` | 90 | 90 | feat: Add knowledge base documents for RAG example |
| `archive/2026-02-18/research__ITR_epsilon_delta` | `research/ITR_epsilon_delta` | `4cac744` | 2026-02-16, Elad@Traigent.ai | `-` | 9 | 9 | feat: Fix and rerun all 6 non-LLM paper experiments (A, E, F, G, H, I) |
| `archive/2026-02-18/revert-release-review-v0.8.0` | `revert-release-review-v0.8.0` | `328e4d0` | 2025-12-13, nimrodbusany | `#14 CLOSED->main` | 1 | 1 | Revert "chore(release): Add release review automation and post-release tracking" |

## Suggested Handling (High-Level)
- Keep as primary long-term archives (large unique work streams):
  - `archive/2026-02-18/feature__real-llm-examples`
  - `archive/2026-02-18/rag-up-wip`
  - `archive/2026-02-18/code_fixes`
  - `archive/2026-02-18/feature__results-table-ansi-colors`
  - `archive/2026-02-18/research__ITR_epsilon_delta`
- Keep medium-term (targeted but unique branches):
  - `archive/2026-02-18/fix__cost-tracking-prompt-tokens`
  - `archive/2026-02-18/api-tests-Mastra-JS`
  - `archive/2026-02-18/feat__openapi-enhanacment-from-develop`
  - `archive/2026-02-18/fix__codex-audit-remediation`
- Lower-risk candidates for eventual archive pruning (already PR-linked and/or low uniqueness):
  - `archive/2026-02-18/feature__injection-mode-consolidation`
  - `archive/2026-02-18/feature__quality-improvement`
  - `archive/2026-02-18/revert-release-review-v0.8.0`
  - `archive/2026-02-18/feature__haystack-integration`
  - `archive/2026-02-18/feature__quickstart-detailed-csv-output`
  - `archive/2026-02-18/dev__real-llm-examples_exp`

## Restore Commands
- Restore a branch from archive:
  - `git push origin refs/tags/archive/2026-02-18/<tag-name>:refs/heads/<branch-name>`
- Example:
  - `git push origin refs/tags/archive/2026-02-18/feature__real-llm-examples:refs/heads/feature/real-llm-examples`
