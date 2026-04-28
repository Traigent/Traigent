# Aikido Triage Notes

This file records repository-local triage decisions for Aikido findings that
cannot be safely fixed by code changes alone.

## 2026-04-25 SDK License Finding

- Aikido issue: `235535549`
- Aikido issue group: `27772214`
- Type: `license`
- Severity: `critical`
- Affected package: `traigent@0.11.4`
- Affected file: `uv.lock`
- Related license: `AGPL-3.0-only`
- Disposition: intentional project license, requires Aikido issue ignore

The finding is for the SDK root package's declared license, not for a
third-party dependency license drift. The repository consistently declares
`AGPL-3.0-only` in `pyproject.toml`, includes the AGPL license text in
`LICENSE`, and documents separate commercial licensing in
`COMMERCIAL-LICENSING.md` and `CONTRIBUTOR-LICENSING.md`.

Do not resolve this by excluding `uv.lock` from Aikido scanning. That would
also suppress dependency, malware, and future license findings from the lockfile.

Do not change the SDK license as a scanner remediation without explicit legal
and business approval. If Aikido comments on a PR for this finding, reply:

```text
@AikidoSec ignore: Intentional SDK project license. Traigent open-source releases are AGPL-3.0-only and separate commercial licenses are documented in COMMERCIAL-LICENSING.md and CONTRIBUTOR-LICENSING.md. This is the root package license, not a third-party dependency license drift.
```
