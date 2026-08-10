# Vendored fp2 conformance corpus

`agent_lifecycle_cases.json` in this directory is vendored **byte-for-byte**
(not hand-edited) from TraigentSchema:

- Source path: `tests/data/fp2/agent_lifecycle_cases.json`
- Source commit: `fbf8d734f865073f9ce52744377516c7a94a8295` (`origin/develop`, 2026-08-10)
- SHA-256 of the vendored file: `971c3fc71117a02b318148d8ca33c720f8f05b937f790f98809a24f31beff36c`

Vendored, not imported via a git dependency: `pyproject.toml` forbids a
direct git dependency on TraigentSchema even inside an extra (PyPI rejects
it). The corresponding algorithm port lives at
`traigent/utils/fp2.py` (see its module docstring for the same provenance
note).

To detect drift, diff this file against
`git -C <TraigentSchema checkout> show origin/develop:tests/data/fp2/agent_lifecycle_cases.json`
and re-vendor (updating both this file and the commit/SHA above) whenever
TraigentSchema's corpus changes. Nothing re-checks this automatically.
