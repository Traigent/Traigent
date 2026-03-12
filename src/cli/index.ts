#!/usr/bin/env node

import { parseArgs } from 'node:util';

import { detectTunedVariablesInFiles } from './detect.js';
import { runSeamlessMigration } from './migrate.js';

interface CLIValues {
  function?: string;
  help?: boolean;
  write?: boolean;
  'include-low-confidence'?: boolean;
}

function printHelp(): void {
  console.error(`
Traigent CLI

Usage:
  traigent migrate seamless [paths...] [--write]
  traigent detect tuned-variables [paths...] [--function name] [--include-low-confidence]

Options:
  --write   Apply rewrites in place
  --function  Limit tuned-variable detection to a named function
  --include-low-confidence  Include low-confidence discovery candidates
  --help    Show this help message
`);
}

function formatDiagnostic(diagnostic: {
  kind: 'rewritten' | 'rejected';
  message: string;
  line?: number;
  column?: number;
}): string {
  const location =
    diagnostic.line !== undefined
      ? `:${diagnostic.line}:${(diagnostic.column ?? 0) + 1}`
      : '';
  return `  [${diagnostic.kind}]${location} ${diagnostic.message}`;
}

async function main(): Promise<void> {
  const positionals = process.argv.slice(2);
  if (positionals.length === 0 || positionals[0] === '--help') {
    printHelp();
    return;
  }

  const [command, subcommand, ...rest] = positionals;

  const { values, positionals: targetPaths } = parseArgs({
    args: rest,
    options: {
      function: { type: 'string' },
      help: { type: 'boolean' },
      'include-low-confidence': { type: 'boolean' },
      write: { type: 'boolean' },
    },
    allowPositionals: true,
  });

  const cliValues = values as CLIValues;
  if (cliValues.help) {
    printHelp();
    return;
  }

  if (command === 'detect' && subcommand === 'tuned-variables') {
    const results = detectTunedVariablesInFiles(targetPaths, {
      functionName: cliValues.function,
      includeLowConfidence: cliValues['include-low-confidence'] ?? false,
    });
    console.log(JSON.stringify(results, null, 2));
    return;
  }

  if (command !== 'migrate' || subcommand !== 'seamless') {
    printHelp();
    process.exitCode = 1;
    return;
  }

  const result = await runSeamlessMigration(targetPaths, {
    write: cliValues.write ?? false,
  });

  if (result.files.length === 0) {
    console.error('[traigent] No seamless rewrite candidates found.');
    return;
  }

  for (const file of result.files) {
    console.error(
      `[traigent] ${file.blocked ? 'blocked' : file.changed ? 'updated' : 'inspected'} ${file.file} (${file.rewrittenCount} rewrites, ${file.rejectedCount} rejected)`,
    );
    for (const diagnostic of file.diagnostics) {
      console.error(formatDiagnostic(diagnostic));
    }
    if (file.blocked) {
      console.error(
        '  [blocked] File was not modified because rejected seamless patterns were found. Fix them first or use explicit context/parameter injection.',
      );
    }
  }

  console.error(
    `[traigent] processed=${result.totalFiles} changed=${result.changedFiles} blocked=${result.blockedFiles} rewritten=${result.rewrittenCount} rejected=${result.rejectedCount}`,
  );

  if (result.rejectedCount > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[traigent] Migration failed:', error);
  process.exitCode = 1;
});
