# SDK Production Readiness Issue Tracker

Last updated: 2026-03-13

This file tracks GitHub issue triage decisions made during the production-readiness pass for the Python SDK repo. It is intentionally repo-focused: issues owned by cloud infra, portal, or broader product work are recorded here but kept separate from SDK defects.

Current open issue count in `Traigent/Traigent`: 29

## Active SDK Issues

No currently verified SDK release-blocking issues remain after the local fixes for `#246`, `#325`, and `#277`. The remaining open public issues are enhancement/backlog work that should be prioritized by release value rather than issue number.

| Issue | Title | Classification | Current status | Next action |
| --- | --- | --- | --- | --- |

## Fixed Locally Pending Push

| Issue | Title | Classification | Current status | Verification |
| --- | --- | --- | --- | --- |
| #246 | Cost estimator overestimates by ~1000x in hybrid_api mode | SDK defect | Implemented locally. The estimator now uses the most expensive resolvable model candidate from optimizer config space and honors optional `estimated_tokens_per_example` metadata from hybrid config-space discovery instead of always assuming `2000/500` tokens. Matching schema changes were also made on `traigent-api` branch `fix/hybrid-config-space-token-estimates`, and first-party adopter rollout is tracked in `docs/reviews/hybrid_config_space_token_estimate_rollout.md`. Keep the GitHub issue open until changes are pushed/merged. | `.venv/bin/pytest -q tests/unit/hybrid/test_protocol.py` -> `42 passed`; `.venv/bin/pytest -q tests/unit/hybrid/test_discovery.py` -> `30 passed`; `.venv/bin/pytest -q tests/unit/core/test_cost_estimator.py` -> `23 passed`; `.venv/bin/pytest -q tests/unit/core/test_orchestrator.py -k 'cost_estimate or optimizer_model_candidates or hybrid_token_estimate'` -> `4 passed` |
| #325 | Hybrid API: pass config to Evaluate endpoint for evaluation-time tuning | SDK defect | Implemented locally. `HybridAPIEvaluator` now threads the same trial config used for `/execute` into `HybridEvaluateRequest.config` for two-phase hybrid evaluation, including preserving `{}` instead of omitting config, so evaluation-time tunables such as judge-model selection can reach `/traigent/v1/evaluate` consistently. Keep the GitHub issue open until changes are pushed/merged. | `.venv/bin/python -m py_compile traigent/evaluators/hybrid_api.py tests/unit/evaluators/test_hybrid_api_evaluator.py` succeeded; direct async sanity checks confirmed the evaluate request carries `{'model': 'gpt-4', 'judge_model': 'gpt-4.1-mini'}`, preserves `timeout_ms=12500`, and keeps empty config as `{}` |
| #277 | when no api exist, show user link to - https://portal.traigent.ai/login to get api key | SDK UX cleanup | Implemented locally for the user-facing CLI paths called out by the issue. Auth failure messaging now points users to `https://portal.traigent.ai/login`, and the local sync upsell points to the portal login and portal home instead of the old signup/app URLs. Keep the GitHub issue open until changes are pushed/merged. | `.venv/bin/python -m py_compile traigent/cli/auth_commands.py traigent/cli/local_commands.py` succeeded; literal repo search confirms the old CLI `https://traigent.ai/signup` link is gone |
| #238 | sdk - enforce (pre run) up to 100% objective weights, objective as percentage | SDK enhancement | Mostly already superseded by `ObjectiveSchema`, which automatically normalizes positive weights. Implemented locally: documented that percentage-style weights are valid and now reject zero-weight objectives so dead objectives fail fast instead of being silently normalized to `0.0`. Keep the GitHub issue open until changes are pushed/merged, then likely close or re-scope the remaining non-code product questions. | `.venv/bin/python -m py_compile traigent/core/objectives.py tests/unit/core/test_objectives.py` succeeded; direct sanity check confirmed `70/30 -> {'accuracy': 0.7, 'cost': 0.3}` and zero-weight definitions now raise `Weight must be positive` |

## Public Repo Recommendation

For a public Python SDK repo, keep only issues that are safe and useful for external contributors or SDK users. Move cloud posture, private roadmap, backend/portal ownership, internal process, and research initiative issues out of this repo before making it public.

### Keep Public In This Repo

- #246
- #325
- #277
- #267
- #238
- #237
- #235
- #234
- #169
- #157
- #86
- #85
- #84
- #83
- #81
- #80
- #79
- #78
- #77
- #76
- #75
- #74
- #72
- #71
- #70
- #68
- #63
- #62
- #29

The SDK repo has been trimmed to this public-safe set.

## Re-Scoped Or External To This Repo

| Issue | Title | Classification | Decision |
| --- | --- | --- | --- |
| #300 | AWS Cloud: Users with administrator access should have MFA enabled | Cloud / infra-security | Keep off the SDK release path. Tracked as AWS posture, not branch-local SDK code. |
| #243 | AWS Cloud: EC2 IAM roles should require IMDSv2 | Cloud / infra-security | Keep off the SDK release path. Tracked as AWS posture, not branch-local SDK code. |
| #328 | Portal registration flow should work for all users, not just whitelisted ones | Cloud / portal | Not implemented in this repo. Treat as portal/backend issue. |
| #323 | Cloud walkthrough from a laptop is not yet supported in the Python SDK examples | Product / roadmap | Current walkthroughs are intentionally local-first; cloud mode is not yet supported here. |
| #336 | create backend portal - to admin users | Cloud / portal | Admin-console request, not SDK repo work. |
| #326 | JWT-only auth: backend does not map write/admin scopes to experiment.write | Backend permission model with SDK mitigation | CLI flow is mitigated in SDK; remaining gap is backend JWT scope mapping / JWT-only usage. |
| #337 | Evaluate AWS Strands support for Python SDK / cloud integration | Product / integration enhancement | Optional future integration backlog, not part of the current production-readiness gate. |

## Transferred Out Of This Repo

| Original issue | Destination | Reason |
| --- | --- | --- |
| #315 | `Traigent/TraigentBackend#106` | Production cloud log health and monitoring is backend/frontend ops work. |
| #324 | `Traigent/TraigentFrontend#132` | Dependabot PR handling belongs in the frontend repo. |
| #301 | `Traigent/TraigentBackend#107` | Loki log-format work is backend/cloud ops. |
| #319 | `Traigent/TraigentBackend#108` | Portal user provisioning is backend/cloud-ops work. |
| #306 | `Traigent/TraigentBackend#109` | Backend/frontend tracing rollout is cloud/backend work. |
| #271 | `Traigent/TraigentBackend#110` | Statistical-significance badge exposure is backend/API work. |
| #336 | `Traigent/TraigentBackend#111` | Admin portal work is backend/cloud portal ownership. |
| #328 | `Traigent/TraigentBackend#112` | Registration/verification flow is backend/cloud portal ownership. |
| #326 | `Traigent/TraigentBackend#113` | JWT scope mapping is backend auth/permission ownership. |
| #290 | `Traigent/TraigentFrontend#133` | Registration-page password UX belongs in the frontend repo. |

## Transferred To Management Repo

| Original issue | Destination | Reason |
| --- | --- | --- |
| #320 | `Traigent/traigent-pricing-simulator#1` | Fundraising/business task moved out of SDK repo. |
| #318 | `Traigent/traigent-pricing-simulator#2` | Fundraising/business task moved out of SDK repo. |
| #317 | `Traigent/traigent-pricing-simulator#3` | Hiring/founding-team task moved out of SDK repo. |
| #275 | `Traigent/traigent-pricing-simulator#4` | Legal/ops task moved out of SDK repo. |
| #274 | `Traigent/traigent-pricing-simulator#5` | GTM/design-partner follow-up moved out of SDK repo. |
| #242 | `Traigent/traigent-pricing-simulator#6` | Marketing/personal-branding task moved out of SDK repo. |
| #241 | `Traigent/traigent-pricing-simulator#7` | Marketing/personal-branding task moved out of SDK repo. |
| #338 | `Traigent/traigent-pricing-simulator#8` | Podcast/content task moved out of SDK repo. |
| #329 | `Traigent/traigent-pricing-simulator#9` | Legal/IP task moved out of SDK repo. |
| #296 | `Traigent/traigent-pricing-simulator#10` | Publication/content task moved out of SDK repo. |
| #295 | `Traigent/traigent-pricing-simulator#11` | Publication/content task moved out of SDK repo. |
| #258 | `Traigent/traigent-pricing-simulator#12` | Planning-board upkeep moved out of SDK repo. |
| #244 | `Traigent/traigent-pricing-simulator#13` | n8n demo/product exploration task moved out of SDK repo. |
| #155 | `Traigent/traigent-pricing-simulator#14` | Video/demo-content task moved out of SDK repo. |
| #291 | `Traigent/traigent-pricing-simulator#15` | Cross-repo release-management task moved out of SDK repo. |
| #321 | `Traigent/traigent-pricing-simulator#16` | Finance/cloud-credits task moved out of SDK repo. |
| #337 | `Traigent/traigent-pricing-simulator#17` | Strategic/private integration roadmap moved out of public SDK repo. |
| #330 | `Traigent/traigent-pricing-simulator#18` | Internal process/tooling task moved out of public SDK repo. |
| #323 | `Traigent/traigent-pricing-simulator#19` | Private cross-repo roadmap item moved out of public SDK repo. |
| #302 | `Traigent/traigent-pricing-simulator#20` | Internal PR review/process task moved out of public SDK repo. |
| #300 | `Traigent/traigent-pricing-simulator#21` | Private infra-security posture issue moved out of public SDK repo. |
| #243 | `Traigent/traigent-pricing-simulator#22` | Private infra-security posture issue moved out of public SDK repo. |
| #211 | `Traigent/traigent-pricing-simulator#23` | Research validation work moved out of public SDK repo. |
| #210 | `Traigent/traigent-pricing-simulator#24` | Research algorithm work moved out of public SDK repo. |
| #207 | `Traigent/traigent-pricing-simulator#25` | Research algorithm work moved out of public SDK repo. |
| #204 | `Traigent/traigent-pricing-simulator#26` | Research gap analysis moved out of public SDK repo. |
| #198 | `Traigent/traigent-pricing-simulator#27` | Research redesign work moved out of public SDK repo. |
| #195 | `Traigent/traigent-pricing-simulator#28` | Research categorization work moved out of public SDK repo. |
| #194 | `Traigent/traigent-pricing-simulator#29` | Research visualization work moved out of public SDK repo. |
| #193 | `Traigent/traigent-pricing-simulator#30` | Research categorization work moved out of public SDK repo. |
| #192 | `Traigent/traigent-pricing-simulator#31` | Research categorization work moved out of public SDK repo. |

## Non-Blocking SDK Enhancement Backlog

| Issue | Title | Classification | Decision |
| --- | --- | --- | --- |
| #238 | sdk - enforce (pre run) up to 100% objective weights, objective as percentage | SDK enhancement | Keep open as the single umbrella for stricter objective-weight validation / consistency. Not a production-readiness blocker. |
| #267 | toxicity etc checks for sdk | SDK enhancement | Keep open as safety/product enhancement backlog, not a production-readiness blocker for current core SDK release. |
| #234 | SDK: align error codes, internal docs, and maxItems with traigent-api spec (post PR #231) | SDK enhancement | Keep open as a concrete spec-alignment follow-up. Real SDK work, but not as urgent as `#246` / `#325`. |
| #235 | SDK - clean and more organize (E2E tests to their repo (e2e-tests), benchmarks to TraigentDemo) | SDK cleanup backlog | Broad cleanup/repo-organization umbrella; split into scoped tasks if acted on. |
| #237 | SDK enhancement features | SDK cleanup backlog | Broad post-merge cleanup umbrella; split into scoped tasks if acted on. |

## Closed During Triage

| Issue | Title | Reason closed |
| --- | --- | --- |
| #170 | fix(security): path injection vulnerability in config_generator/apply.py (S2083) | Verified fixed in current `develop`; sanitization and regression tests already exist. |
| #43 | Cost tracking does not capture prompt template tokens | Verified fixed in current `develop`; prompt-template-aware mock token estimation is covered by tests. |
| #334 | where see a DB of our users - who signed in ? when are they last active? | Closed as duplicate of `#336`. |
| #331 | add observabilty and evaluation framework to traigent | Closed as stale umbrella issue; foundations already exist, remaining work should be narrower productization issues. |
| #315 | make sure cloud logs do not have any errors on the backend and frontend | Closed in SDK repo as external cloud/backend/frontend ops work. |
| #303 | make sure in the UI is refernincg to weight - when giving a score to the results ( if i wan taccuracy - it should have the best score ) | Closed as resolved/stale; the issue body said the frontend PR was already merged. |
| #324 | on the FE repo - there are many PR from dependabot - what are they ? how should they be merged? | Closed as frontend-repo maintenance work, not a Python SDK issue. |
| #299 | enforce - objective weights up to 1 on sdk | Closed as duplicate of `#238`. |
| #307 | Review Nimrod's LangGraph demos (Nexxen) | Closed as an ad hoc review request rather than SDK engineering work. |
| #327 | fix errors or warning over cloud usage (backend/frontend) (Elad duplicate) | Closed as duplicate cloud/backend/frontend ops work. |
| #304 | Move text2sql to traigent demo `|` Do a follow up with @EladTraignet on that | Closed as demo/repo-organization follow-up, not a current SDK production-readiness issue. |

## Next Review Queue

| Issue | Title | Why it is next |
| --- | --- | --- |
| next TBD | Remaining unlabeled or mixed-scope issues | Continue separating true SDK defects from cloud/portal/product backlog items. |

## Notes

- This tracker is for repo triage state, not a full implementation plan.
- Only issues in `Active SDK Issues` should be treated as current Python SDK engineering work.
- `Fixed Locally Pending Push` means the code change exists in the local `develop` worktree and has targeted verification, but the GitHub issue has not yet been closed remotely.
- Cloud, portal, and infrastructure items should be kept on separate boards or clearly labeled to avoid mixing release blockers with external dependencies.
- Management, roadmap, research, and private security items are now tracked in `Traigent/traigent-pricing-simulator` instead of the public-facing SDK repo.
