# main → develop ancestry-sync log

After each promotion, `main` must be merged back into `develop` with a **real merge
commit** so `main` stays an ancestor of `develop` — otherwise the next promotion
re-enters the manual "preserve main-only fixes" reconciliation (see the
2026-08-09 round's runbook, `tools/ops-runbook/finish-develop-main-sync.sh` in the
workspace shell repo).

Two operational gotchas, learned the hard way:

- **Never squash a sync PR.** Squashing lands main's *content* but flattens the
  merge commit, so main does **not** become an ancestor and the sync is silently
  defeated. This repo's merge queue uses the `merge` method, which is safe.
- **A content-empty sync PR breaks `required-pr-gate`.** When develop already
  contains all of main's content, the sync diff is zero files; the gate's
  classifier fails closed on `changed_file_count=0` ("broken diff", deliberately
  indistinguishable from a misfired path filter). That is why sync PRs carry a
  one-line entry in this log: it keeps the diff non-empty so the gate can run and
  the queue can accept the PR, and it documents the round.

| Date | PR | main commits linked | Notes |
|------|----|---------------------|-------|
| 2026-08-09 | #2148 | 22 | First scripted round; merged via queue (`merge`). |
| 2026-08-11 | #2162 | — | Ancestry-only (content already on develop); this file added to un-break the empty-diff gate. |
