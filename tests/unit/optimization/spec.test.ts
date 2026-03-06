import { describe, expect, it } from 'vitest';

import {
  getOptimizationSpec,
  optimize,
  param,
  toHybridConfigSpace,
} from '../../../src/index.js';
import { ValidationError } from '../../../src/core/errors.js';

describe('optimization spec helpers', () => {
  it('attaches metadata without changing the wrapped function behavior', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini', 'gpt-4o']),
      },
      objectives: ['accuracy'],
    })(async (value: string) => value.toUpperCase());

    expect(await wrapped('ok')).toBe('OK');
    expect(getOptimizationSpec(wrapped)).toMatchObject({
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['gpt-4o-mini', 'gpt-4o'],
        },
      },
      objectives: [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
    });
    expect(typeof wrapped.optimize).toBe('function');
  });

  it('serializes config space to the hybrid tunables shape', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini', 'gpt-4o']),
        temperature: param.float({ min: 0, max: 1, scale: 'linear' }),
        max_retries: param.int({ min: 0, max: 3, scale: 'linear' }),
      },
      objectives: ['accuracy', 'cost'],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(toHybridConfigSpace(wrapped)).toEqual({
      tunables: [
        {
          name: 'model',
          type: 'enum',
          domain: { values: ['gpt-4o-mini', 'gpt-4o'] },
        },
        {
          name: 'temperature',
          type: 'float',
          domain: { range: [0, 1] },
          scale: 'linear',
        },
        {
          name: 'max_retries',
          type: 'int',
          domain: { range: [0, 3] },
          scale: 'linear',
        },
      ],
      constraints: {},
    });
  });

  it('rejects unknown string objectives', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['gpt-4o-mini']),
        },
        objectives: ['quality'],
      }),
    ).toThrow(ValidationError);
  });

  it('accepts explicit objective objects for custom metrics', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini']),
      },
      objectives: [{ metric: 'quality_score', direction: 'maximize' }],
    })(async () => ({ metrics: { quality_score: 0.9 } }));

    expect(getOptimizationSpec(wrapped)?.objectives).toEqual([
      { metric: 'quality_score', direction: 'maximize', weight: 1 },
    ]);
  });
});
