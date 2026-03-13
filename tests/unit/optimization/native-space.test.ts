import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import { PythonRandom } from '../../../src/optimization/python-random.js';
import {
  applyDefaultConfig,
  buildDiscreteValues,
  buildFloatValues,
  buildIntValues,
  canonicalize,
  cartesianProduct,
  configKey,
  discreteCardinality,
  getOrderedParameterEntries,
  hashJson,
  sampleCandidateConfig,
  sampleParameter,
  stableJson,
} from '../../../src/optimization/native-space.js';
import type { NormalizedOptimizationSpec } from '../../../src/optimization/types.js';

function createNormalizedSpec(
  overrides: Partial<NormalizedOptimizationSpec> = {},
): NormalizedOptimizationSpec {
  return {
    configurationSpace: {
      temperature: {
        type: 'float',
        min: 0,
        max: 1,
        scale: 'linear',
        step: 0.5,
      },
      retries: {
        type: 'int',
        min: 1,
        max: 3,
        step: 1,
        scale: 'linear',
      },
      model: {
        type: 'enum',
        values: ['fast', 'best'],
      },
    },
    objectives: [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
    defaultConfig: {},
    constraints: [],
    safetyConstraints: [],
    injection: {
      mode: 'context',
    },
    execution: {
      mode: 'native',
      contract: 'trial',
      repsPerTrial: 1,
      repsAggregation: 'mean',
    },
    evaluation: {
      data: [{ id: 1 }],
    },
    ...overrides,
  };
}

describe('native-space helpers', () => {
  it('canonicalizes nested objects and arrays deterministically', () => {
    const value = {
      b: 2,
      a: {
        y: 2,
        x: 1,
      },
      c: [{ z: 2, a: 1 }],
    };

    expect(canonicalize(value)).toEqual({
      a: { x: 1, y: 2 },
      b: 2,
      c: [{ a: 1, z: 2 }],
    });
    expect(stableJson({ b: 2, a: 1 })).toBe('{"a":1,"b":2}');
    expect(configKey({ retries: 2, model: 'fast' })).toBe(
      stableJson({ retries: 2, model: 'fast' }),
    );
    expect(hashJson({ a: 1, b: 2 })).toBe(hashJson({ b: 2, a: 1 }));
  });

  it('orders parameters with model last and applies default config', () => {
    const spec = createNormalizedSpec({
      defaultConfig: {
        retries: 3,
      },
    });

    expect(getOrderedParameterEntries(spec.configurationSpace).map(([name]) => name)).toEqual(
      ['retries', 'temperature', 'model'],
    );
    expect(applyDefaultConfig(spec, { model: 'best' })).toEqual({
      retries: 3,
      model: 'best',
    });
  });

  it('builds linear and log int value grids and validates invalid log steps', () => {
    expect(
      buildIntValues('retries', {
        type: 'int',
        min: 1,
        max: 5,
        step: 2,
        scale: 'linear',
      }),
    ).toEqual([1, 3, 5]);

    expect(
      buildIntValues('retries', {
        type: 'int',
        min: 1,
        max: 20,
        step: 2,
        scale: 'log',
      }),
    ).toEqual([1, 2, 4, 8, 16, 20]);

    expect(() =>
      buildIntValues('retries', {
        type: 'int',
        min: 1,
        max: 20,
        step: 1,
        scale: 'log',
      }),
    ).toThrowError(ValidationError);
  });

  it('builds linear and log float value grids and validates missing steps', () => {
    expect(
      buildFloatValues('temperature', {
        type: 'float',
        min: 0,
        max: 1,
        step: 0.5,
        scale: 'linear',
      }),
    ).toEqual([0, 0.5, 1]);

    expect(
      buildFloatValues('learning_rate', {
        type: 'float',
        min: 0.001,
        max: 1,
        step: 10,
        scale: 'log',
      }),
    ).toEqual([0.001, 0.01, 0.1, 1]);

    expect(() =>
      buildFloatValues('temperature', {
        type: 'float',
        min: 0,
        max: 1,
        scale: 'linear',
      }),
    ).toThrowError(ValidationError);
  });

  it('builds discrete values and cardinality only for fully discrete spaces', () => {
    const spec = createNormalizedSpec();
    const entries = getOrderedParameterEntries(spec.configurationSpace);

    expect(buildDiscreteValues('model', spec.configurationSpace.model!)).toEqual([
      'fast',
      'best',
    ]);
    expect(discreteCardinality(entries)).toBe(18);
    expect(
      discreteCardinality([
        [
          'temperature',
          {
            type: 'float',
            min: 0,
            max: 1,
            scale: 'linear',
          },
        ],
      ]),
    ).toBeNull();
  });

  it('samples each supported parameter shape', () => {
    const random = new PythonRandom(42);

    const sampledEnum = sampleParameter(
      'model',
      { type: 'enum', values: ['fast', 'best'] },
      random,
    );
    const sampledInt = sampleParameter(
      'retries',
      { type: 'int', min: 1, max: 5, step: 2, scale: 'linear' },
      random,
    );
    const sampledLogInt = sampleParameter(
      'depth',
      { type: 'int', min: 1, max: 64, scale: 'log' },
      random,
    );
    const sampledFloat = sampleParameter(
      'temperature',
      { type: 'float', min: 0, max: 1, scale: 'linear' },
      random,
    );
    const sampledLogFloat = sampleParameter(
      'learning_rate',
      { type: 'float', min: 0.001, max: 1, scale: 'log', step: 10 },
      random,
    );

    expect(['fast', 'best']).toContain(sampledEnum);
    expect([1, 3, 5]).toContain(sampledInt);
    expect(typeof sampledLogInt).toBe('number');
    expect(sampledLogInt).toBeGreaterThanOrEqual(1);
    expect(sampledLogInt).toBeLessThanOrEqual(64);
    expect(typeof sampledFloat).toBe('number');
    expect(sampledFloat).toBeGreaterThanOrEqual(0);
    expect(sampledFloat).toBeLessThanOrEqual(1);
    expect([0.001, 0.01, 0.1, 1]).toContain(sampledLogFloat);
  });

  it('creates cartesian products and sampled candidate configs', () => {
    expect(
      cartesianProduct([
        ['model', ['fast', 'best']],
        ['retries', [1, 2]],
      ]),
    ).toEqual([
      { model: 'fast', retries: 1 },
      { model: 'fast', retries: 2 },
      { model: 'best', retries: 1 },
      { model: 'best', retries: 2 },
    ]);

    const random = new PythonRandom(7);
    const sampled = sampleCandidateConfig(
      getOrderedParameterEntries(createNormalizedSpec().configurationSpace),
      random,
    );

    expect(sampled.model).toMatch(/fast|best/);
    expect([1, 2, 3]).toContain(sampled.retries);
    expect([0, 0.5, 1]).toContain(sampled.temperature);
  });

  it('rejects unsupported discrete parameter types', () => {
    expect(() =>
      buildDiscreteValues(
        'unsupported',
        { type: 'custom' } as never,
      ),
    ).toThrowError(ValidationError);
  });
});
