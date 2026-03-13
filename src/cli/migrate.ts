import { promises as fs } from 'node:fs';
import path from 'node:path';

import { transformSeamlessSource, type SeamlessDiagnostic } from '../seamless/transform.js';

const SUPPORTED_EXTENSIONS = new Set([
  '.js',
  '.mjs',
  '.cjs',
  '.ts',
  '.mts',
  '.cts',
  '.jsx',
  '.tsx',
]);

export interface SeamlessMigrationOptions {
  write: boolean;
  cwd?: string;
}

export interface SeamlessMigrationFileResult {
  file: string;
  changed: boolean;
  blocked: boolean;
  rewrittenCount: number;
  rejectedCount: number;
  diagnostics: SeamlessDiagnostic[];
}

export interface SeamlessMigrationResult {
  files: SeamlessMigrationFileResult[];
  totalFiles: number;
  changedFiles: number;
  blockedFiles: number;
  rewrittenCount: number;
  rejectedCount: number;
}

async function collectFiles(inputPath: string): Promise<string[]> {
  const stat = await fs.stat(inputPath);
  if (stat.isFile()) {
    return SUPPORTED_EXTENSIONS.has(path.extname(inputPath)) ? [inputPath] : [];
  }

  if (!stat.isDirectory()) {
    return [];
  }

  const entries = await fs.readdir(inputPath, { withFileTypes: true });
  const nestedFiles = await Promise.all(
    entries
      .filter((entry) => entry.name !== 'node_modules' && entry.name !== 'dist')
      .map((entry) => collectFiles(path.join(inputPath, entry.name)))
  );

  return nestedFiles.flat();
}

export async function runSeamlessMigration(
  paths: readonly string[],
  options: SeamlessMigrationOptions
): Promise<SeamlessMigrationResult> {
  const cwd = options.cwd ?? process.cwd();
  const resolvedPaths = paths.length > 0 ? paths : ['.'];
  const files = (
    await Promise.all(
      resolvedPaths.map((targetPath) => collectFiles(path.resolve(cwd, targetPath)))
    )
  ).flat();

  const results: SeamlessMigrationFileResult[] = [];

  for (const file of files) {
    const source = await fs.readFile(file, 'utf8');
    const transformed = transformSeamlessSource(source, { filename: file });
    const rejectedCount = transformed.diagnostics.filter(
      (diagnostic) => diagnostic.kind === 'rejected'
    ).length;
    const blocked = rejectedCount > 0;
    const changed = transformed.changed && !blocked;

    if (options.write && changed) {
      await fs.writeFile(file, `${transformed.code}\n`, 'utf8');
    }

    if (!transformed.changed && transformed.diagnostics.length === 0) {
      continue;
    }

    results.push({
      file,
      changed,
      blocked,
      rewrittenCount: transformed.rewrittenCount,
      rejectedCount,
      diagnostics: transformed.diagnostics,
    });
  }

  return {
    files: results,
    totalFiles: files.length,
    changedFiles: results.filter((result) => result.changed).length,
    blockedFiles: results.filter((result) => result.blocked).length,
    rewrittenCount: results.reduce((total, result) => total + result.rewrittenCount, 0),
    rejectedCount: results.reduce(
      (total, result) =>
        total + result.diagnostics.filter((diagnostic) => diagnostic.kind === 'rejected').length,
      0
    ),
  };
}
