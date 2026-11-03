#!/usr/bin/env bash
# Refuse to continue when a release tag for the package version already exists
# and does not resolve to the commit being published. A missing tag is fine
# (the release job creates it later); any lookup error fails closed.
#
# Usage: assert_version_tag_matches_sha.sh <sha> <package-version>
set -euo pipefail

SHA="${1:?usage: assert_version_tag_matches_sha.sh <sha> <package-version>}"
VERSION="${2:?usage: assert_version_tag_matches_sha.sh <sha> <package-version>}"
TAG_NAME="v${VERSION}"
TAG_REF="refs/tags/${TAG_NAME}"

# show-ref: 0 = exists, 1 = absent, anything else = lookup error (fail closed).
rc=0
git show-ref --verify --quiet "$TAG_REF" || rc=$?
if [ "$rc" -eq 1 ]; then
  echo "Tag $TAG_NAME does not exist yet; nothing to compare."
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "::error::Refusing to publish: could not look up tag $TAG_NAME (exit $rc)."
  exit 1
fi

if ! TAG_COMMIT="$(git rev-parse --verify "${TAG_REF}^{commit}")"; then
  echo "::error::Refusing to publish: could not resolve tag $TAG_NAME to a commit."
  exit 1
fi
if [ "$TAG_COMMIT" != "$SHA" ]; then
  echo "::error::Refusing to publish: tag $TAG_NAME points at $TAG_COMMIT, not $SHA."
  exit 1
fi
echo "Tag $TAG_NAME points at $SHA."
