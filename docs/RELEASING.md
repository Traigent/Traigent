# Releasing

## Versioning

This repository uses Changesets for manual versioning and changelog generation.

- Create a changeset for any public API, export, or user-visible behavior change.
- Manual publish remains the default release policy.
- Changelog generation is automated, but publishing is not.

## Commands

```bash
npm run changeset
npm run changeset:status
npm run changeset:version
```

Temporary CI bypass for internal-only or docs-only pull requests:

```bash
ALLOW_MISSING_CHANGESET=1 node scripts/ci/check-changeset.mjs
```

## Semver Guidance

- `patch`: bug fixes, hardening, internal compatibility fixes without intended API expansion
- `minor`: additive exports, new supported workflows, new stable helper APIs
- `major`: breaking public API, export, packaging, or behavioral contract changes

## Release Flow

1. Ensure CI is green.
2. Add or confirm the required changesets.
3. Run `npm run changeset:version`.
4. Review generated version and changelog changes.
5. Build and run package smoke tests.
6. Publish manually:

```bash
npm publish --access public
git tag vX.Y.Z
git push origin vX.Y.Z
```

## Changelog

Changesets generates changelog entries as part of the versioning flow. Review generated notes before publishing.
