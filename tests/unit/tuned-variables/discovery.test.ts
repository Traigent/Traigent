import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  discoverTunedVariables,
  discoverTunedVariablesFromFile,
  discoverTunedVariablesFromSource,
} from '../../../src/tuned-variables/index.js';

describe('discoverTunedVariablesFromSource', () => {
  it('discovers high-confidence tunables from framework-style request objects', () => {
    const source = `
      export async function answerQuestion(input) {
        const model = "gpt-4o-mini";
        const temperature = 0.7;
        const tone = "concise";
        return client.chat.completions.create({
          model,
          temperature,
          messages: [{ role: "user", content: input + " / tone=" + tone }],
        });
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'answerQuestion',
    });

    expect(result?.candidates.map((candidate) => candidate.name)).toEqual([
      'model',
      'temperature',
      'tone',
    ]);
    expect(result?.candidates[0]).toMatchObject({
      name: 'model',
      confidence: 'high',
      kind: 'string',
      supportedByConfigSpace: true,
      suggestedDefinition: {
        type: 'enum',
        values: ['gpt-4o-mini'],
      },
    });
    expect(result?.candidates[1]).toMatchObject({
      name: 'temperature',
      confidence: 'high',
      kind: 'float',
      supportedByConfigSpace: true,
      suggestedDefinition: {
        type: 'float',
      },
    });
  });

  it('skips reassigned variables and reports a warning', () => {
    const source = `
      export function answerQuestion(input) {
        let temperature = 0.7;
        if (input.length > 10) {
          temperature = 0.9;
        }
        return client.chat.completions.create({ temperature });
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'answerQuestion',
    });

    expect(result?.candidates).toEqual([]);
    expect(result?.warnings).toContain(
      'Skipped "temperature" because it is reassigned or mutated after declaration.'
    );
  });

  it('filters low-confidence generic literals by default', () => {
    const source = `
      export function helper() {
        const label = "hello";
        return label;
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'helper',
    });

    expect(result?.candidates).toEqual([]);
  });

  it('includes low-confidence candidates when requested', () => {
    const source = `
      export function helper() {
        const label = "hello";
        return label;
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'helper',
      includeLowConfidence: true,
    });

    expect(result?.candidates).toHaveLength(1);
    expect(result?.candidates[0]).toMatchObject({
      name: 'label',
      confidence: 'low',
    });
  });

  it('discovers multiple literal kinds and marks unsupported config-space values', () => {
    const source = `
      export const answerQuestion = async (input) => {
        const enabled = true;
        const retryCount = 3;
        const topP = 0.8;
        const toolConfig = { mode: "fast" };
        const labels = ["short", "safe"];
        const fallback = null;
        return client.chat.completions.create({
          enabled,
          retryCount,
          top_p: topP,
          toolConfig,
          labels,
          fallback,
          messages: [{ role: "user", content: input }],
        });
      };
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'answerQuestion',
      includeLowConfidence: true,
    });

    expect(result?.candidates).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'enabled',
          kind: 'boolean',
          supportedByConfigSpace: true,
          suggestedDefinition: {
            type: 'enum',
            values: [true, false],
          },
        }),
        expect.objectContaining({
          name: 'retryCount',
          kind: 'int',
          supportedByConfigSpace: true,
          suggestedDefinition: expect.objectContaining({
            type: 'int',
            step: 1,
          }),
        }),
        expect.objectContaining({
          name: 'topP',
          kind: 'float',
          supportedByConfigSpace: true,
          suggestedDefinition: expect.objectContaining({
            type: 'float',
          }),
        }),
        expect.objectContaining({
          name: 'toolConfig',
          kind: 'object',
          supportedByConfigSpace: false,
        }),
        expect.objectContaining({
          name: 'labels',
          kind: 'array',
          supportedByConfigSpace: false,
        }),
        expect.objectContaining({
          name: 'fallback',
          kind: 'null',
          supportedByConfigSpace: false,
        }),
      ])
    );
  });

  it('skips private bindings by default and can include them explicitly', () => {
    const source = `
      export function answerQuestion(input) {
        const _temperature = 0.2;
        return client.chat.completions.create({
          temperature: _temperature,
          messages: [{ role: "user", content: input }],
        });
      }
    `;

    const [defaultResult] = discoverTunedVariablesFromSource(source, {
      functionName: 'answerQuestion',
    });
    const [includedResult] = discoverTunedVariablesFromSource(source, {
      functionName: 'answerQuestion',
      includePrivate: true,
    });

    expect(defaultResult?.candidates).toEqual([]);
    expect(includedResult?.candidates[0]).toMatchObject({
      name: '_temperature',
      confidence: 'high',
    });
  });

  it('ignores unreferenced locals and supports exported arrow functions', () => {
    const source = `
      export const answerQuestion = async (input) => {
        const model = "gpt-4o-mini";
        const unusedThreshold = 0.3;
        return client.chat.completions.create({
          model,
          messages: [{ role: "user", content: input }],
        });
      };
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'answerQuestion',
      includeLowConfidence: true,
    });

    expect(result?.functionName).toBe('answerQuestion');
    expect(result?.candidates.map((candidate) => candidate.name)).toEqual(['model']);
  });

  it('treats common tunable names as medium confidence even without call sinks', () => {
    const source = `
      export function helper() {
        const temperature = 0.2;
        return temperature;
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'helper',
      includeLowConfidence: true,
    });

    expect(result?.candidates).toHaveLength(1);
    expect(result?.candidates[0]).toMatchObject({
      name: 'temperature',
      confidence: 'medium',
    });
  });

  it('skips unsupported literal forms such as computed object keys and spread arrays', () => {
    const source = `
      export function helper(input) {
        const labels = ["safe", ...fallbackLabels];
        const promptStyle = { [input]: "brief" };
        const dynamic = buildOptions();
        return { labels, promptStyle, dynamic };
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'helper',
      includeLowConfidence: true,
    });

    expect(result?.candidates).toEqual([]);
  });

  it('returns no results when the source contains no analyzable functions', () => {
    expect(discoverTunedVariablesFromSource('const value = 1;')).toEqual([]);
  });

  it('detects identifier-callee sinks and unary numeric literals', () => {
    const source = `
      export function helper() {
        const threshold = -2;
        const temperature = +0.3;
        complete(threshold);
        return temperature;
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'helper',
      includeLowConfidence: true,
    });

    expect(result?.candidates).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'threshold',
          kind: 'int',
          confidence: 'high',
        }),
        expect.objectContaining({
          name: 'temperature',
          kind: 'float',
        }),
      ])
    );
  });

  it('classifies non-common names as medium when they only flow into generic calls', () => {
    const source = `
      export function helper() {
        const label = "hello";
        return format(label);
      }
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      functionName: 'helper',
      includeLowConfidence: true,
    });

    expect(result?.candidates[0]).toMatchObject({
      name: 'label',
      confidence: 'medium',
      reason: 'Flows into a function call.',
    });
  });

  it('supports anonymous exported functions and skips destructured declarations', () => {
    const source = `
      export default async (input) => {
        const { model } = defaults;
        const promptStyle = \`brief\`;
        return client.chat.completions.create({
          style: promptStyle,
          messages: [{ role: "user", content: input }],
        });
      };
    `;

    const [result] = discoverTunedVariablesFromSource(source, {
      includeLowConfidence: true,
    });

    expect(result?.functionName).toBe('anonymous');
    expect(result?.candidates).toHaveLength(1);
    expect(result?.candidates[0]).toMatchObject({
      name: 'promptStyle',
      kind: 'string',
      confidence: 'high',
    });
  });
});

describe('discoverTunedVariables', () => {
  it('analyzes callable source directly', () => {
    async function answerQuestion(input: string) {
      const model = 'gpt-4o-mini';
      return `${model}:${input}`;
    }

    const result = discoverTunedVariables(answerQuestion, {
      includeLowConfidence: true,
    });

    expect(result.functionName).toBe('answerQuestion');
    expect(result.candidates.map((candidate) => candidate.name)).toContain('model');
  });

  it('supports anonymous functions and preserves anonymous naming', () => {
    const result = discoverTunedVariables(
      async function (input: string) {
        const temperature = 0.2;
        return `${input}:${temperature}`;
      },
      { includeLowConfidence: true }
    );

    expect(result.functionName).toBe('anonymous');
    expect(result.candidates[0]).toMatchObject({
      name: 'temperature',
      confidence: 'high',
    });
  });
});

describe('discoverTunedVariablesFromFile', () => {
  it('analyzes a file path', () => {
    const dir = mkdtempSync(join(tmpdir(), 'traigent-discovery-'));
    const file = join(dir, 'example.mjs');
    writeFileSync(
      file,
      `
        export function searchDocs(query) {
          const retrievalK = 5;
          return retriever.search({ query, k: retrievalK });
        }
      `,
      'utf8'
    );

    const [result] = discoverTunedVariablesFromFile(file, {
      functionName: 'searchDocs',
    });

    expect(result?.candidates).toHaveLength(1);
    expect(result?.candidates[0]).toMatchObject({
      name: 'retrievalK',
      confidence: 'high',
      supportedByConfigSpace: true,
    });
  });
});
