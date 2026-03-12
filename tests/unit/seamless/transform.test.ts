import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { transformSync } from '@babel/core';
import { parse } from '@babel/parser';
import traverse, { type NodePath } from '@babel/traverse';
import * as t from '@babel/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { runSeamlessMigration } from '../../../src/cli/migrate.js';
import { getTrialParam } from '../../../src/core/context.js';
import {
  clearRegisteredFrameworkTargets,
  registerFrameworkTarget,
} from '../../../src/integrations/registry.js';
import traigentSeamlessBabelPlugin, {
  formatSeamlessDiagnosticPreview,
} from '../../../src/seamless/babel-plugin.js';
import {
  transformSeamlessFunctionPath,
  transformSeamlessSource,
} from '../../../src/seamless/transform.js';
import { resolveSeamlessFunction } from '../../../src/seamless/runtime.js';
import { optimize, param } from '../../../src/optimization/spec.js';
import type { SeamlessDiagnostic } from '../../../src/seamless/transform.js';

const tempDirs: string[] = [];
const traverseAst = traverse.default ?? traverse;

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })),
  );
  vi.unstubAllEnvs();
});

beforeEach(() => {
  vi.unstubAllEnvs();
});

function transformFunctionSnippet(
  source: string,
  configKeys: string[],
): { code: string; diagnostics: string[]; rewrittenCount: number } {
  const ast = parse(`const fn = ${source};`, {
    sourceType: 'module',
    plugins: ['typescript', 'jsx'],
  });

  let code = '';
  let rewrittenCount = 0;
  const diagnostics: Array<{ message: string }> = [];

  traverseAst(ast, {
    VariableDeclarator(path: NodePath<t.VariableDeclarator>) {
      const init = path.get('init');
      if (
        init.isFunctionExpression() ||
        init.isFunctionDeclaration() ||
        init.isArrowFunctionExpression()
      ) {
        rewrittenCount = transformSeamlessFunctionPath(
          init,
          new Set(configKeys),
          'getTrialParam',
          '<test>',
          diagnostics as never[],
        );
        code = path.toString();
        path.stop();
      }
    },
  });

  return {
    code,
    rewrittenCount,
    diagnostics: diagnostics.map((entry) => entry.message),
  };
}

describe('seamless transform tooling', () => {
  it('rewrites matching local tuned variables and injects getTrialParam imports', () => {
    const source = `
      import { optimize, param } from '@traigent/sdk';

      async function answerQuestion(question) {
        const model = 'cheap';
        let temperature = 0.2;
        temperature = 0.4;
        return \`\${model}:\${temperature}:\${question}\`;
      }

      export const optimized = optimize({
        configurationSpace: {
          model: param.enum(['cheap', 'best']),
          temperature: param.float({ min: 0, max: 1, step: 0.2 }),
        },
        objectives: ['accuracy'],
      })(answerQuestion);
    `;

    const result = transformSeamlessSource(source, {
      filename: 'example.mjs',
    });

    expect(result.changed).toBe(true);
    expect(result.rewrittenCount).toBe(3);
    expect(result.code).toContain('getTrialParam');
    expect(result.code).toContain(`const model = getTrialParam("model", 'cheap')`);
    expect(result.code).toContain(
      `let temperature = getTrialParam("temperature", 0.2)`,
    );
    expect(result.code).toContain(
      `temperature = getTrialParam("temperature", 0.4)`,
    );
  });

  it('reports rejected seamless patterns without partially rewriting them', () => {
    const source = `
      import { optimize, param } from '@traigent/sdk';

      const defaults = { temperature: 0.2 };
      async function answerQuestion(question) {
        const { temperature } = defaults;
        return \`\${temperature}:\${question}\`;
      }

      export const optimized = optimize({
        configurationSpace: {
          temperature: param.float({ min: 0, max: 1, step: 0.2 }),
        },
        objectives: ['accuracy'],
      })(answerQuestion);
    `;

    const result = transformSeamlessSource(source, {
      filename: 'example.mjs',
    });

    expect(result.changed).toBe(false);
    expect(result.diagnostics.some((entry) => entry.kind === 'rejected')).toBe(
      true,
    );
  });

  it('skips files without optimize imports or static configuration space', () => {
    const noOptimize = transformSeamlessSource(
      `export async function answer() { const model = 'cheap'; return model; }`,
      { filename: 'no-optimize.mjs' },
    );
    expect(noOptimize.changed).toBe(false);
    expect(noOptimize.diagnostics).toHaveLength(0);

    const dynamicSpec = transformSeamlessSource(
      `
        import { optimize } from '@traigent/sdk';
        const spec = getSpec();
        async function answer() {
          const model = 'cheap';
          return model;
        }
        export const optimized = optimize(spec)(answer);
      `,
      { filename: 'dynamic-spec.mjs' },
    );
    expect(dynamicSpec.changed).toBe(false);
    expect(
      dynamicSpec.diagnostics.some((entry) =>
        entry.message.includes('static object literal'),
      ),
    ).toBe(true);

    const quotedKeys = transformSeamlessSource(
      `
        import { optimize as tune, param } from '@traigent/sdk';
        async function answer() {
          const model = 'cheap';
          return model;
        }
        export const optimized = tune({
          "configurationSpace": {
            "model": param.enum(['cheap', 'best']),
          },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'quoted-keys.mjs' },
    );
    expect(quotedKeys.changed).toBe(true);
    expect(quotedKeys.code).toContain(`const model = getTrialParam("model", 'cheap')`);

    const spreadConfig = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        const space = { model: param.enum(['cheap', 'best']) };
        async function answer() {
          const model = 'cheap';
          return model;
        }
        export const optimized = optimize({
          configurationSpace: { ...space },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'spread-config.mjs' },
    );
    expect(spreadConfig.changed).toBe(false);
    expect(
      spreadConfig.diagnostics.some((entry) =>
        entry.message.includes('static object literal'),
      ),
    ).toBe(true);
  });

  it('rejects unresolved optimize targets and update expressions', () => {
    const unresolved = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        export const optimized = optimize({
          configurationSpace: { model: param.enum(['cheap', 'best']) },
          objectives: ['accuracy'],
        })(factory());
      `,
      { filename: 'unresolved.mjs' },
    );
    expect(unresolved.changed).toBe(false);
    expect(
      unresolved.diagnostics.some((entry) =>
        entry.message.includes('directly resolvable local function'),
      ),
    ).toBe(true);

    const updateExpression = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        async function answer() {
          let retries = 1;
          retries++;
          return retries;
        }
        export const optimized = optimize({
          configurationSpace: { retries: param.int({ min: 0, max: 3 }) },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'update-expression.mjs' },
    );
    expect(updateExpression.changed).toBe(false);
    expect(updateExpression.code).not.toContain('getTrialParam');
    expect(
      updateExpression.diagnostics.some((entry) =>
        entry.message.includes('Increment/decrement'),
      ),
    ).toBe(true);
    expect(
      updateExpression.diagnostics.some((entry) =>
        entry.message.includes('No partial rewrite was applied'),
      ),
    ).toBe(true);
  });

  it('preserves existing getTrialParam imports and supports object/array/template defaults', () => {
    const result = transformSeamlessSource(
      `
        import { optimize, param, getTrialParam as readTvar } from '@traigent/sdk';
        async function answer(question) {
          const model = 'cheap';
          const prompt = \`helpful\`;
          const weights = [1, 2];
          const config = { model: 'cheap', enabled: true };
          return [readTvar('model', model), prompt, weights, config, question];
        }
        export const optimized = optimize({
          configurationSpace: {
            model: param.enum(['cheap', 'best']),
            prompt: param.enum(['helpful', 'direct']),
            weights: param.enum([[1, 2], [2, 3]]),
            config: param.enum([{ model: 'cheap', enabled: true }]),
          },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'existing-import.mjs' },
    );

    expect(result.changed).toBe(true);
    expect(result.code.match(/readTvar\(/g)?.length).toBeGreaterThanOrEqual(4);
    expect(result.code).not.toContain('import { getTrialParam');
  });

  it('supports inline optimize targets, merged config keys, and unary defaults', () => {
    const result = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';

        async function answer(question) {
          const retries = -1;
          const model = 'cheap';
          return \`\${model}:\${retries}:\${question}\`;
        }

        export const optimizedA = optimize({
          configurationSpace: {
            model: param.enum(['cheap', 'best']),
          },
          objectives: ['accuracy'],
        })(answer);

        export const optimizedB = optimize({
          configurationSpace: {
            retries: param.int({ min: -1, max: 3 }),
          },
          objectives: ['accuracy'],
        })(answer);

        export const optimizedInline = optimize({
          configurationSpace: {
            tone: param.enum(['calm', 'urgent']),
          },
          objectives: ['accuracy'],
        })(async function inlineAnswer() {
          const tone = 'calm';
          return tone;
        });
      `,
      { filename: 'merged-keys.mjs' },
    );

    expect(result.changed).toBe(true);
    expect(result.rewrittenCount).toBe(3);
    expect(result.code).toContain(`const retries = getTrialParam("retries", -1)`);
    expect(result.code).toContain(`const model = getTrialParam("model", 'cheap')`);
    expect(result.code).toContain(`const tone = getTrialParam("tone", 'calm')`);
  });

  it('covers direct function rewrites for assignment operators and pre-transformed values', () => {
    const existingGetter = transformFunctionSnippet(
      `async function answer() { const model = getTrialParam('model', 'cheap'); return model; }`,
      ['model'],
    );
    expect(existingGetter.rewrittenCount).toBe(0);

    const compoundAssignment = transformFunctionSnippet(
      `async function answer() { let retries = 1; retries += 1; return retries; }`,
      ['retries'],
    );
    expect(compoundAssignment.diagnostics.some((message) => message.includes('Only direct "="'))).toBe(true);

    const unsupportedReassignment = transformFunctionSnippet(
      `async function answer() { let settings = { enabled: true }; settings = otherSettings; return settings; }`,
      ['settings'],
    );
    expect(
      unsupportedReassignment.diagnostics.some((message) =>
        message.includes('supported literal values'),
      ),
    ).toBe(true);
  });

  it('covers array and object pattern helper branches directly', () => {
    const destructuredArray = transformFunctionSnippet(
      `async function answer() { const [temperature] = [0.2]; return temperature; }`,
      ['temperature'],
    );
    expect(
      destructuredArray.diagnostics.some((message) =>
        message.includes('Destructuring tuned variables'),
      ),
    ).toBe(true);

    const destructuredObject = transformFunctionSnippet(
      `async function answer() { const { temperature: configured = 0.2 } = defaults; return configured; }`,
      ['configured'],
    );
    expect(
      destructuredObject.diagnostics.some((message) =>
        message.includes('Destructuring tuned variables'),
      ),
    ).toBe(true);
  });

  it('rejects unsupported literals and module-style constants cleanly', () => {
    const result = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        const BASE_MODEL = 'cheap';

        async function answer() {
          const model = BASE_MODEL;
          const settings = { ['dynamic']: true };
          return [model, settings];
        }

        export const optimized = optimize({
          configurationSpace: {
            model: param.enum(['cheap', 'best']),
            settings: param.enum([{ dynamic: true }]),
          },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'unsupported-literals.mjs' },
    );

    expect(result.changed).toBe(false);
    expect(
      result.diagnostics.filter((entry) => entry.kind === 'rejected').length,
    ).toBeGreaterThanOrEqual(2);

    const numericKey = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        async function answer() {
          const settings = { 1: 'on' };
          return settings;
        }
        export const optimized = optimize({
          configurationSpace: {
            settings: param.enum([{ 1: 'on' }]),
          },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'numeric-key.mjs' },
    );
    expect(numericKey.changed).toBe(true);
    expect(numericKey.code).toContain('getTrialParam("settings"');
  });

  it('rejects non-function bindings, array spreads/holes, and rest destructuring patterns', () => {
    const nonFunctionBinding = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        const answer = factory();
        export const optimized = optimize({
          configurationSpace: { model: param.enum(['cheap', 'best']) },
          objectives: ['accuracy'],
        })(answer);
      `,
      { filename: 'non-function-binding.mjs' },
    );
    expect(nonFunctionBinding.changed).toBe(false);
    expect(
      nonFunctionBinding.diagnostics.some((entry) =>
        entry.message.includes('directly resolvable local function'),
      ),
    ).toBe(true);

    const restPattern = transformFunctionSnippet(
      `async function answer() { const { temperature, ...rest } = defaults; return [temperature, rest]; }`,
      ['rest'],
    );
    expect(
      restPattern.diagnostics.some((message) =>
        message.includes('Destructuring tuned variables'),
      ),
    ).toBe(true);

    const arrayHole = transformFunctionSnippet(
      `async function answer() { const retries = [1,,2]; return retries; }`,
      ['retries'],
    );
    expect(
      arrayHole.diagnostics.some((message) =>
        message.includes('literal, array literal, object literal, or template literal'),
      ),
    ).toBe(true);

    const arraySpread = transformFunctionSnippet(
      `async function answer() { const retries = [1, ...extra]; return retries; }`,
      ['retries'],
    );
    expect(
      arraySpread.diagnostics.some((message) =>
        message.includes('literal, array literal, object literal, or template literal'),
      ),
    ).toBe(true);

    const computedObject = transformFunctionSnippet(
      `async function answer() { const settings = { [dynamicKey]: true }; return settings; }`,
      ['settings'],
    );
    expect(
      computedObject.diagnostics.some((message) =>
        message.includes('literal, array literal, object literal, or template literal'),
      ),
    ).toBe(true);
  });

  it('covers missing bindings, imported bindings, and additional unsupported literal forms', () => {
    const missingBinding = transformSeamlessSource(
      `
        import sdk, { optimize as tune, param } from '@traigent/sdk';
        export const optimized = tune({
          "configurationSpace": { model: param.enum(['cheap', 'best']) },
          objectives: ['accuracy'],
        })(missingAnswer);
      `,
      { filename: 'missing-binding.mjs' },
    );
    expect(missingBinding.changed).toBe(false);
    expect(
      missingBinding.diagnostics.some((entry) =>
        entry.message.includes('directly resolvable local function'),
      ),
    ).toBe(true);

    const importedBinding = transformSeamlessSource(
      `
        import { optimize, param } from '@traigent/sdk';
        import { externalAnswer } from './external.js';
        export const optimized = optimize({
          configurationSpace: { model: param.enum(['cheap', 'best']) },
          objectives: ['accuracy'],
        })(externalAnswer);
      `,
      { filename: 'imported-binding.mjs' },
    );
    expect(importedBinding.changed).toBe(false);
    expect(
      importedBinding.diagnostics.some((entry) =>
        entry.message.includes('directly resolvable local function'),
      ),
    ).toBe(true);

    const uninitialized = transformFunctionSnippet(
      `async function answer() { let temperature; return temperature; }`,
      ['temperature'],
    );
    expect(
      uninitialized.diagnostics.some((message) =>
        message.includes('literal, array literal, object literal, or template literal'),
      ),
    ).toBe(true);

    const unsupportedUnary = transformFunctionSnippet(
      `async function answer() { const retries = ~1; return retries; }`,
      ['retries'],
    );
    expect(
      unsupportedUnary.diagnostics.some((message) =>
        message.includes('literal, array literal, object literal, or template literal'),
      ),
    ).toBe(true);

    const objectMethod = transformFunctionSnippet(
      `async function answer() { const settings = { enable() { return true; } }; return settings; }`,
      ['settings'],
    );
    expect(
      objectMethod.diagnostics.some((message) =>
        message.includes('literal, array literal, object literal, or template literal'),
      ),
    ).toBe(true);

    const restArrayPattern = transformFunctionSnippet(
      `async function answer() { const [...retries] = [1, 2]; return retries; }`,
      ['retries'],
    );
    expect(
      restArrayPattern.diagnostics.some((message) =>
        message.includes('Destructuring tuned variables'),
      ),
    ).toBe(true);

    const sparseArrayPattern = transformFunctionSnippet(
      `async function answer() { const [, retries] = [0, 1]; return retries; }`,
      ['retries'],
    );
    expect(
      sparseArrayPattern.diagnostics.some((message) =>
        message.includes('Destructuring tuned variables'),
      ),
    ).toBe(true);
  });

  it('exposes the same transform through the Babel plugin', () => {
    const source = `
      import { optimize, param } from '@traigent/sdk';
      const answer = async (question) => {
        const model = 'cheap';
        return \`\${model}:\${question}\`;
      };
      export const optimized = optimize({
        configurationSpace: {
          model: param.enum(['cheap', 'best']),
        },
        objectives: ['accuracy'],
      })(answer);
    `;

    const transformed = transformSync(source, {
      filename: 'plugin-example.mjs',
      plugins: [traigentSeamlessBabelPlugin],
      configFile: false,
      babelrc: false,
    });

    expect(transformed?.code).toContain('getTrialParam');
    expect(transformed?.code).toContain(`const model = getTrialParam("model", 'cheap')`);
  });

  it('makes the Babel plugin fail closed on rejected seamless patterns', () => {
    const source = `
      import { optimize, param } from '@traigent/sdk';
      const answer = async () => {
        let retries = 1;
        retries++;
        return retries;
      };
      export const optimized = optimize({
        configurationSpace: {
          retries: param.int({ min: 0, max: 3 }),
        },
        objectives: ['accuracy'],
      })(answer);
    `;

    expect(() =>
      transformSync(source, {
        filename: 'plugin-rejected.mjs',
        plugins: [traigentSeamlessBabelPlugin],
        configFile: false,
        babelrc: false,
      }),
    ).toThrow(/Refusing to emit a partial seamless transform/i);
  });

  it('formats multi-diagnostic Babel failures without a filename', () => {
    const source = `
      import { optimize, param } from '@traigent/sdk';
      async function answer() {
        const { temperature } = defaults;
        let retries = 1;
        retries++;
        const settings = { [dynamicKey]: true };
        const route = [1, ...extra];
        return [temperature, retries, settings, route];
      }
      export const optimized = optimize({
        configurationSpace: {
          temperature: param.float({ min: 0, max: 1, step: 0.1 }),
          retries: param.int({ min: 0, max: 3 }),
          settings: param.enum([{ enabled: true }]),
          route: param.enum([[1, 2]]),
        },
        objectives: ['accuracy'],
      })(answer);
    `;

    expect(() =>
      transformSync(source, {
        plugins: [traigentSeamlessBabelPlugin],
        configFile: false,
        babelrc: false,
      }),
    ).toThrow(/<unknown>|and 1 more|Refusing to emit a partial seamless transform/i);
  });

  it('formats seamless diagnostic previews with and without source locations', () => {
    const withLocation: SeamlessDiagnostic = {
      kind: 'rejected',
      message: 'bad pattern',
      filename: 'demo.mjs',
      line: 4,
      column: 2,
    };
    expect(formatSeamlessDiagnosticPreview(withLocation)).toBe(
      '- demo.mjs:4:3 bad pattern',
    );

    const withoutLocation: SeamlessDiagnostic = {
      kind: 'rejected',
      message: 'missing location',
    };
    expect(
      formatSeamlessDiagnosticPreview(withoutLocation, 'fallback.mjs'),
    ).toBe('- fallback.mjs missing location');
  });

  it('runs the codemod CLI helper in dry-run and write modes', async () => {
    const tempDir = await mkdtemp(path.join(os.tmpdir(), 'traigent-seamless-'));
    tempDirs.push(tempDir);
    const file = path.join(tempDir, 'agent.mjs');
    await writeFile(
      file,
      `
        import { optimize, param } from '@traigent/sdk';
        async function answerQuestion(question) {
          const model = 'cheap';
          return \`\${model}:\${question}\`;
        }
        export const optimized = optimize({
          configurationSpace: { model: param.enum(['cheap', 'best']) },
          objectives: ['accuracy'],
        })(answerQuestion);
      `,
      'utf8',
    );

    const dryRun = await runSeamlessMigration([file], {
      write: false,
    });
    expect(dryRun.changedFiles).toBe(1);
    expect((await readFile(file, 'utf8')).includes('getTrialParam')).toBe(false);

    const writeRun = await runSeamlessMigration([file], {
      write: true,
    });
    expect(writeRun.changedFiles).toBe(1);
    expect((await readFile(file, 'utf8')).includes('getTrialParam')).toBe(true);
  });

  it('blocks codemod writes when a file mixes rewritable and rejected seamless patterns', async () => {
    const tempDir = await mkdtemp(path.join(os.tmpdir(), 'traigent-seamless-'));
    tempDirs.push(tempDir);
    const file = path.join(tempDir, 'blocked-agent.mjs');
    await writeFile(
      file,
      `
        import { optimize, param } from '@traigent/sdk';
        async function answerQuestion(question) {
          const model = 'cheap';
          let retries = 1;
          retries++;
          return \`\${model}:\${retries}:\${question}\`;
        }
        export const optimized = optimize({
          configurationSpace: {
            model: param.enum(['cheap', 'best']),
            retries: param.int({ min: 0, max: 3 }),
          },
          objectives: ['accuracy'],
        })(answerQuestion);
      `,
      'utf8',
    );

    const dryRun = await runSeamlessMigration([file], {
      write: false,
    });
    expect(dryRun.changedFiles).toBe(0);
    expect(dryRun.blockedFiles).toBe(1);
    expect(dryRun.files[0]?.blocked).toBe(true);

    const writeRun = await runSeamlessMigration([file], {
      write: true,
    });
    expect(writeRun.changedFiles).toBe(0);
    expect(writeRun.blockedFiles).toBe(1);
    expect((await readFile(file, 'utf8')).includes('getTrialParam')).toBe(false);
  });

  it('recurses through directories and ignores unsupported extensions', async () => {
    const tempDir = await mkdtemp(path.join(os.tmpdir(), 'traigent-seamless-'));
    tempDirs.push(tempDir);
    await writeFile(
      path.join(tempDir, 'agent.mjs'),
      `
        import { optimize, param } from '@traigent/sdk';
        async function answer() {
          const model = 'cheap';
          return model;
        }
        export const optimized = optimize({
          configurationSpace: { model: param.enum(['cheap', 'best']) },
          objectives: ['accuracy'],
        })(answer);
      `,
      'utf8',
    );
    await writeFile(path.join(tempDir, 'notes.txt'), 'ignore me', 'utf8');

    const result = await runSeamlessMigration([tempDir], { write: false });
    expect(result.totalFiles).toBe(1);
    expect(result.changedFiles).toBe(1);
  });

  it('returns an empty migration result when there are no supported files', async () => {
    const tempDir = await mkdtemp(path.join(os.tmpdir(), 'traigent-seamless-'));
    tempDirs.push(tempDir);
    await writeFile(path.join(tempDir, 'notes.txt'), 'ignore me', 'utf8');

    const result = await runSeamlessMigration([tempDir], { write: false });
    expect(result.totalFiles).toBe(0);
    expect(result.files).toEqual([]);
  });

  it('uses experimental runtime seamless rewriting for self-contained functions', async () => {
    vi.stubEnv('TRAIGENT_ENABLE_EXPERIMENTAL_RUNTIME_SEAMLESS', '1');

    const wrapped = optimize({
      configurationSpace: {
        temperature: param.enum([0.2, 0.8]),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
      },
      evaluation: {
        data: [{ input: 'x', output: '0.8:x' }],
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
    })(async function answer(question: string) {
      const temperature = 0.2;
      return `${temperature}:${question}`;
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ temperature: 0.8 });
    expect(wrapped.seamlessResolution()).toMatchObject({
      path: 'runtime',
      experimental: true,
    });
  });

  it('uses framework registration and transformed functions as seamless-ready fast paths', () => {
    clearRegisteredFrameworkTargets();

    const registered = async function answer(question: string) {
      return question;
    };
    registerFrameworkTarget('openai');
    expect(
      resolveSeamlessFunction(registered, ['temperature'], ['openai'], true),
    ).toMatchObject({
      fn: registered,
      resolution: {
        path: 'framework',
        experimental: false,
      },
    });

    registerFrameworkTarget('langchain');
    expect(
      resolveSeamlessFunction(registered, ['temperature'], undefined, true),
    ).toMatchObject({
      resolution: {
        path: 'framework',
        targets: ['langchain', 'openai'],
      },
    });

    clearRegisteredFrameworkTargets();

    const alreadyTransformed = async function answer() {
      return getTrialParam('temperature', 0.2);
    };
    expect(
      resolveSeamlessFunction(alreadyTransformed, ['temperature'], undefined, true),
    ).toMatchObject({
      fn: alreadyTransformed,
      resolution: {
        path: 'pretransformed',
        experimental: false,
      },
    });
  });

  it('allows known globals during runtime seamless rewriting', async () => {
    vi.stubEnv('TRAIGENT_ENABLE_EXPERIMENTAL_RUNTIME_SEAMLESS', '1');

    const wrapped = optimize({
      configurationSpace: {
        temperature: param.enum([0.2, 0.8]),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
      },
      evaluation: {
        data: [{ input: 'x', output: '0.2:x' }],
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
    })(async function answer(question: string) {
      const temperature = 0.2;
      const rounded = Math.round(1);
      return `${temperature * rounded}:${question}`;
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.bestConfig).toEqual({ temperature: 0.2 });
    expect(wrapped.seamlessResolution()).toMatchObject({
      path: 'runtime',
    });
  });

  it('rejects unsupported runtime seamless cases before execution', () => {
    clearRegisteredFrameworkTargets();

    expect(() =>
      resolveSeamlessFunction(
        async (question: string) => {
          const answer = question;
          return answer;
        },
        ['temperature'],
        undefined,
        true,
      ),
    ).toThrow(/Run `traigent migrate seamless`|Babel plugin/i);

    expect(() =>
      resolveSeamlessFunction(
        (question: string) => {
          const temperature = 0.2;
          return `${temperature}:${question}`;
        },
        ['temperature'],
        undefined,
        true,
      ),
    ).toThrow(/arrow functions/i);

    expect(() =>
      resolveSeamlessFunction(
        async function answer(question: string) {
          return question;
        },
        ['temperature'],
        undefined,
        true,
      ),
    ).toThrow(/Run `traigent migrate seamless`|Babel plugin/i);
  });

  it('can disable automatic framework override resolution for seamless mode', () => {
    clearRegisteredFrameworkTargets();
    registerFrameworkTarget('openai');

    const pretransformed = async function answer() {
      return getTrialParam('temperature', 0.2);
    };

    expect(
      resolveSeamlessFunction(pretransformed, ['temperature'], undefined, false),
    ).toMatchObject({
      resolution: {
        path: 'pretransformed',
      },
    });
  });

  it('rejects experimental runtime seamless rewriting for non-self-contained functions', async () => {
    vi.stubEnv('TRAIGENT_ENABLE_EXPERIMENTAL_RUNTIME_SEAMLESS', '1');

    clearRegisteredFrameworkTargets();
    const defaults = { temperature: 0.2 };

    const wrapped = optimize({
      configurationSpace: {
        temperature: param.enum([0.2, 0.8]),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'seamless',
      },
      evaluation: {
        data: [{ input: 'x', output: '0.8:x' }],
        scoringFunction: (output, expectedOutput) =>
          output === expectedOutput ? 1 : 0,
      },
    })(async function answer(question: string) {
      const temperature = defaults.temperature;
      return `${temperature}:${question}`;
    });

    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 2,
      }),
    ).rejects.toThrow(/self-contained functions|Babel plugin/i);
  });

  it('rejects runtime seamless rewriting unless explicitly opted in', () => {
    clearRegisteredFrameworkTargets();

    expect(() =>
      resolveSeamlessFunction(
        async function answer(question: string) {
          const temperature = 0.2;
          return `${temperature}:${question}`;
        },
        ['temperature'],
        undefined,
        true,
      ),
    ).toThrow(/disabled by default|trusted local code/i);
  });
});
