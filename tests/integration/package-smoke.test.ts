/**
 * Package consumer smoke tests.
 *
 * These tests `npm pack` the built SDK, install the tarball into a
 * temporary project, and verify that every declared subpath export
 * resolves correctly under both ESM (`import`) and CJS (`require`).
 *
 * This catches real publish-time breakage that unit tests miss:
 *   - missing files in the tarball
 *   - broken `exports` map entries
 *   - ESM/CJS dual-format issues
 *
 * Peer dependencies (ai, openai, @langchain/core) are NOT installed
 * in the consumer project. CJS eagerly requires them at load time,
 * so CJS tests tolerate MODULE_NOT_FOUND for known peer deps while
 * still failing if the SDK package itself cannot be resolved.
 */

import { execSync, ExecSyncOptionsWithStringEncoding } from 'node:child_process';
import { existsSync, mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, it, expect, beforeAll, afterAll } from 'vitest';

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

const ROOT = resolve(import.meta.dirname, '..', '..');

const KNOWN_PEER_DEPS = ['ai', 'openai', '@langchain/core', '@babel/core'];

function run(cmd: string, cwd: string): string {
  return execSync(cmd, { cwd, encoding: 'utf-8', timeout: 60_000 }).trim();
}

function parsePackedTarballFilename(output: string): string {
  const trimmed = output.trim();

  try {
    const jsonStart = trimmed.lastIndexOf('\n[');
    const jsonText = jsonStart >= 0 ? trimmed.slice(jsonStart + 1) : trimmed;
    const parsed = JSON.parse(jsonText) as Array<{ filename?: string }>;
    const filename = parsed.at(-1)?.filename;
    if (typeof filename === 'string' && filename.endsWith('.tgz')) {
      return filename;
    }
  } catch {
    // Fall through to line-based parsing for npm variants that mix extra output in stdout.
  }

  for (const line of trimmed.split(/\r?\n/).reverse()) {
    const candidate = line.trim();
    if (candidate.endsWith('.tgz')) {
      return candidate;
    }
  }

  throw new Error(`Unable to determine packed tarball filename from output:\n${output}`);
}

/**
 * Run a node script and return { code, stderr }.
 * Does not throw on non-zero exit — lets the caller inspect the result.
 */
function runScript(file: string, cwd: string): { code: number; stderr: string } {
  const opts: ExecSyncOptionsWithStringEncoding = { cwd, encoding: 'utf-8', timeout: 60_000 };
  try {
    execSync(`node "${file}"`, { ...opts, stdio: ['pipe', 'pipe', 'pipe'] });
    return { code: 0, stderr: '' };
  } catch (err: unknown) {
    const e = err as { status?: number; stderr?: string };
    return { code: e.status ?? 1, stderr: e.stderr ?? '' };
  }
}

/**
 * Returns true if the stderr indicates a missing peer dependency
 * (expected when peer deps are not installed in the consumer project).
 */
function isPeerDepError(stderr: string): boolean {
  if (!stderr.includes('MODULE_NOT_FOUND') && !stderr.includes('ERR_MODULE_NOT_FOUND')) {
    return false;
  }
  return KNOWN_PEER_DEPS.some((dep) => stderr.includes(`'${dep}'`) || stderr.includes(`"${dep}"`));
}

/* ------------------------------------------------------------------ */
/*  Fixture: pack once, reuse across all tests                        */
/* ------------------------------------------------------------------ */

let tarballPath: string;
let tmpDir: string;

beforeAll(() => {
  const cliRunnerPath = join(ROOT, 'dist', 'cli', 'runner.js');
  if (!existsSync(cliRunnerPath)) {
    run('npm run build', ROOT);
  }

  // Pack the already-built SDK
  const tarball = parsePackedTarballFilename(run('npm pack --json --pack-destination /tmp', ROOT));
  tarballPath = join('/tmp', tarball);

  // Create a temp consumer project
  tmpDir = mkdtempSync(join(tmpdir(), 'traigent-smoke-'));

  // Minimal package.json — "type":"module" for ESM tests
  writeFileSync(
    join(tmpDir, 'package.json'),
    JSON.stringify({ name: 'smoke-consumer', version: '0.0.0', type: 'module', private: true })
  );

  // Install the tarball (ignore peer deps — we only test the export map)
  run(`npm install --no-save --legacy-peer-deps "${tarballPath}"`, tmpDir);
}, /* 2 min timeout */ 120_000);

afterAll(() => {
  try {
    rmSync(tmpDir, { recursive: true, force: true });
  } catch {
    /* best-effort */
  }
  try {
    rmSync(tarballPath, { force: true });
  } catch {
    /* best-effort */
  }
});

/* ------------------------------------------------------------------ */
/*  ESM import tests                                                  */
/* ------------------------------------------------------------------ */

describe('ESM imports', () => {
  /**
   * Verify an ESM import resolves. If it fails only because a peer dep
   * is absent, that still counts as a pass (the SDK module was found).
   */
  function testEsmImport(specifier: string, namedExports?: string[]) {
    const exportCheck = namedExports
      ? `const ok = [${namedExports.map((n) => `typeof ${n}`).join(', ')}].every(t => t !== 'undefined');
         if (!ok) process.exit(1);`
      : '';
    const importClause = namedExports ? `{ ${namedExports.join(', ')} }` : `* as _mod`;
    const script = `
      import ${importClause} from '${specifier}';
      ${exportCheck}
      process.exit(0);
    `;
    const slug = specifier.replace(/[/@]/g, '_');
    const file = join(tmpDir, `esm-test-${slug}.mjs`);
    writeFileSync(file, script);

    const { code, stderr } = runScript(file, tmpDir);
    if (code !== 0) {
      if (isPeerDepError(stderr)) {
        // Module resolved, peer dep not installed — acceptable
        return;
      }
      throw new Error(`ESM import of '${specifier}' failed:\n${stderr}`);
    }
  }

  it('import @traigent/sdk (main entry)', () => {
    testEsmImport('@traigent/sdk', [
      'TrialContext',
      'getTrialConfig',
      'getTrialParam',
      'isInTrial',
    ]);
  });

  it('import @traigent/sdk/langchain', () => {
    testEsmImport('@traigent/sdk/langchain', ['TraigentHandler']);
  });

  it('import @traigent/sdk/openai', () => {
    testEsmImport('@traigent/sdk/openai');
  });

  it('import @traigent/sdk/vercel-ai', () => {
    testEsmImport('@traigent/sdk/vercel-ai');
  });

  it('import @traigent/sdk/babel-plugin-seamless', () => {
    testEsmImport('@traigent/sdk/babel-plugin-seamless');
  });
});

/* ------------------------------------------------------------------ */
/*  CJS require tests                                                 */
/* ------------------------------------------------------------------ */

describe('CJS require', () => {
  /**
   * Verify a CJS require resolves. The CJS bundle eagerly loads peer
   * deps, so MODULE_NOT_FOUND for a known peer dep is tolerated.
   */
  function testCjsRequire(specifier: string, namedExports?: string[]) {
    const exportCheck = namedExports
      ? `const ok = [${namedExports.map((n) => `typeof mod.${n}`).join(', ')}].every(t => t !== 'undefined');
         if (!ok) process.exit(1);`
      : '';
    const script = `
      try {
        const mod = require('${specifier}');
        ${exportCheck}
        process.exit(0);
      } catch (e) {
        // Write error details to stderr for the parent to inspect
        process.stderr.write(e.code + ':' + (e.requireStack ? e.requireStack[0] : '') + ':' + e.message);
        process.exit(2);
      }
    `;
    const slug = specifier.replace(/[/@]/g, '_');
    const file = join(tmpDir, `cjs-test-${slug}.cjs`);
    writeFileSync(file, script);

    const { code, stderr } = runScript(file, tmpDir);
    if (code !== 0) {
      if (isPeerDepError(stderr)) {
        return;
      }
      throw new Error(`CJS require of '${specifier}' failed:\n${stderr}`);
    }
  }

  it('require @traigent/sdk (main entry)', () => {
    testCjsRequire('@traigent/sdk', [
      'TrialContext',
      'getTrialConfig',
      'getTrialParam',
      'isInTrial',
    ]);
  });

  it('require @traigent/sdk/langchain', () => {
    testCjsRequire('@traigent/sdk/langchain', ['TraigentHandler']);
  });

  it('require @traigent/sdk/openai', () => {
    testCjsRequire('@traigent/sdk/openai');
  });

  it('require @traigent/sdk/vercel-ai', () => {
    testCjsRequire('@traigent/sdk/vercel-ai');
  });

  it('require @traigent/sdk/babel-plugin-seamless', () => {
    testCjsRequire('@traigent/sdk/babel-plugin-seamless');
  });
});

/* ------------------------------------------------------------------ */
/*  Tarball content checks                                            */
/* ------------------------------------------------------------------ */

describe('tarball contents', () => {
  let tarballFiles: string[];

  beforeAll(() => {
    tarballFiles = run(`tar tf "${tarballPath}"`, tmpDir).split('\n');
  });

  it('includes ESM entry', () => {
    expect(tarballFiles.some((f) => f.includes('dist/index.js'))).toBe(true);
  });

  it('includes CJS entry', () => {
    expect(tarballFiles.some((f) => f.includes('dist/index.cjs'))).toBe(true);
  });

  it('includes type declarations', () => {
    expect(tarballFiles.some((f) => f.includes('dist/index.d.ts'))).toBe(true);
  });

  it('includes langchain integration', () => {
    expect(tarballFiles.some((f) => f.includes('dist/integrations/langchain/index.js'))).toBe(true);
    expect(tarballFiles.some((f) => f.includes('dist/integrations/langchain/index.cjs'))).toBe(
      true
    );
  });

  it('includes CLI runner', () => {
    expect(tarballFiles.some((f) => f.includes('dist/cli/runner.js'))).toBe(true);
  });

  it('does not include node_modules', () => {
    expect(tarballFiles.some((f) => f.includes('node_modules/'))).toBe(false);
  });

  it('does not include tests', () => {
    expect(tarballFiles.some((f) => f.includes('tests/'))).toBe(false);
  });

  it('does not include source .ts files', () => {
    expect(tarballFiles.some((f) => f.includes('/src/'))).toBe(false);
  });
});
