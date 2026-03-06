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
    expect(typeof wrapped.applyBestConfig).toBe('function');
    expect(typeof wrapped.currentConfig).toBe('function');
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

  it('stores wrapper-local applied config without creating runtime global state', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-4o-mini', 'gpt-4o']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'gpt-4o' ? 1 : 0.5,
      },
    }));

    expect(wrapped.currentConfig()).toBeUndefined();

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    const applied = wrapped.applyBestConfig(result);
    expect(applied).toEqual({ model: 'gpt-4o' });
    expect(wrapped.currentConfig()).toEqual({ model: 'gpt-4o' });
  });

  it('rejects log-scaled parameters with non-positive ranges or non-multiplicative steps', () => {
    expect(() =>
      param.float({ min: 0, max: 1, scale: 'log' }),
    ).toThrow(/require min\/max > 0/i);

    expect(() =>
      param.float({ min: 0.001, max: 1, scale: 'log', step: 1 }),
    ).toThrow(/require step to be greater than 1/i);
  });

  it('supports conditional parameters with equality conditions and default fallback', () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-3.5', 'gpt-4']),
        max_tokens: param.int({
          min: 256,
          max: 1024,
          step: 256,
          conditions: { model: 'gpt-4' },
          default: 512,
        }),
      },
      objectives: ['accuracy'],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(getOptimizationSpec(wrapped)?.configurationSpace.max_tokens).toEqual({
      type: 'int',
      min: 256,
      max: 1024,
      step: 256,
      scale: 'linear',
      conditions: { model: 'gpt-4' },
      default: 512,
    });
  });

  it('rejects conditional parameters without a valid default fallback', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['gpt-3.5', 'gpt-4']),
          max_tokens: param.int({
            min: 256,
            max: 1024,
            step: 256,
            conditions: { model: 'gpt-4' },
          }),
        },
        objectives: ['accuracy'],
      }),
    ).toThrow(/requires a default fallback value/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(['gpt-3.5', 'gpt-4']),
          max_tokens: param.int({
            min: 256,
            max: 1024,
            step: 256,
            conditions: { model: 'gpt-4' },
            default: 2048,
          }),
        },
        objectives: ['accuracy'],
      }),
    ).toThrow(/default must fall within min\/max/i);
  });

  it('rejects conditional dependency cycles and hybrid serialization for native-only conditionals', () => {
    expect(() =>
      optimize({
        configurationSpace: {
          alpha: param.enum(['x', 'y'], {
            conditions: { beta: 'x' },
            default: 'x',
          }),
          beta: param.enum(['x', 'y'], {
            conditions: { alpha: 'x' },
            default: 'x',
          }),
        },
        objectives: ['accuracy'],
      }),
    ).toThrow(/cannot form dependency cycles/i);

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-3.5', 'gpt-4']),
        max_tokens: param.int({
          min: 256,
          max: 1024,
          step: 256,
          conditions: { model: 'gpt-4' },
          default: 512,
        }),
      },
      objectives: ['accuracy'],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(() => toHybridConfigSpace(wrapped)).toThrow(
      /does not support conditional parameter "max_tokens" yet/i,
    );
  });
});
