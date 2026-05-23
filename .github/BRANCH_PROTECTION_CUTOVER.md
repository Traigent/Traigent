# Branch protection cutover

The Python SDK currently relies on release PR discipline more than required status checks.
If branch protection later requires a fast gate, introduce a single stable
`required-pr-gate` job first, let it complete successfully once on the target branch, and
only then add it to the ruleset.

Required workflows must always create the required check name. Use in-workflow change
detection and an aggregator that accepts intentionally skipped jobs.
