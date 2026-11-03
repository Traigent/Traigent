#!/usr/bin/env bash
# Refuse to continue unless the commit being published is reachable from
# origin/main and, when a tag name is given, the tag resolves to that commit.
#
# Usage: assert_sha_on_main.sh <sha> [tag-name]
#
# Env: MAIN_REMOTE (default "origin"), MAIN_BRANCH (default "main"),
#      SKIP_FETCH=1 skips `git fetch` (tests only).
set -euo pipefail

SHA="${1:?usage: assert_sha_on_main.sh <sha> [tag-name]}"
TAG_NAME="${2:-}"
REMOTE="${MAIN_REMOTE:-origin}"
BRANCH="${MAIN_BRANCH:-main}"

if [ "${SKIP_FETCH:-0}" != "1" ]; then
  git fetch --no-tags "$REMOTE" "$BRANCH"
fi

if ! git merge-base --is-ancestor "$SHA" "$REMOTE/$BRANCH"; then
  echo "::error::Refusing to publish: $SHA is not reachable from $REMOTE/$BRANCH."
  exit 1
fi

if [ -n "$TAG_NAME" ]; then
  TAG_COMMIT="$(git rev-parse --verify "refs/tags/$TAG_NAME^{commit}")"
  if [ "$TAG_COMMIT" != "$SHA" ]; then
    echo "::error::Refusing to publish: tag $TAG_NAME points at $TAG_COMMIT, not $SHA."
    exit 1
  fi
fi

echo "$SHA is on $REMOTE/$BRANCH${TAG_NAME:+ and tag $TAG_NAME points at it}."
