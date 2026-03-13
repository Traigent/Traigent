#!/usr/bin/env node

import { execFileSync } from 'node:child_process';

const baseRef = process.env['GITHUB_BASE_REF'] || 'main';
const allowMissingChangeset = process.env['ALLOW_MISSING_CHANGESET'] === '1';

const relevantMatchers = [
  /^src\//,
  /^package\.json$/,
  /^README\.md$/,
  /^docs\/api-reference\//,
  /^docs\/getting-started\//,
  /^docs\/CLIENT_CODE_GUIDE\.md$/,
  /^docs\/NATIVE_JS_PARITY_MATRIX\.md$/,
  /^docs\/HYBRID_JS_PARITY_MATRIX\.md$/,
];

function readChangedFiles() {
  try {
    const diff = execFileSync('git', ['diff', '--name-only', `origin/${baseRef}...HEAD`], {
      encoding: 'utf8',
    }).trim();
    return diff === '' ? [] : diff.split('\n').filter(Boolean);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(
      `[changeset] Failed to compare against origin/${baseRef}. Ensure the base ref is fetched in CI or locally before running the changeset gate.`
    );
    console.error(`[changeset] Underlying git error: ${message}`);
    process.exit(1);
  }
}

function main() {
  const changedFiles = readChangedFiles();
  const relevantFiles = changedFiles.filter((file) =>
    relevantMatchers.some((matcher) => matcher.test(file))
  );
  const hasChangeset = changedFiles.some(
    (file) =>
      file.startsWith('.changeset/') && file.endsWith('.md') && file !== '.changeset/README.md'
  );

  if (allowMissingChangeset) {
    console.log('[changeset] ALLOW_MISSING_CHANGESET=1, skipping enforcement.');
    return;
  }

  if (relevantFiles.length === 0) {
    console.log('[changeset] No release-relevant files changed.');
    return;
  }

  if (hasChangeset) {
    console.log('[changeset] Changeset found.');
    return;
  }

  console.error('[changeset] Missing changeset for release-relevant changes:');
  for (const file of relevantFiles) {
    console.error(` - ${file}`);
  }
  console.error('\nRun `npm run changeset` and commit the generated file.');
  process.exit(1);
}

main();
