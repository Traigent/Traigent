import { describe, expect, it } from 'vitest';

import { PythonRandom } from '../../../src/optimization/python-random.js';
import { suggestBayesianConfig } from '../../../src/optimization/native-bayesian.js';
import type {
  NormalizedOptimizationSpec,
  OptimizationTrialRecord,
} from '../../../src/optimization/types.js';

function createNormalizedSpec(
  overrides: Partial<NormalizedOptimizationSpec> = {},
): NormalizedOptimizationSpec {
  return {
    configurationSpace: {
      model: {
        type: 'enum',
        values: ['a', 'b', 'c', 'd'],
      },
      retries: {
        type: 'int',
        min: 0,
        max: 2,
        step: 1,
        scale: 'linear',
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

function trial(
  trialId: string,
  config: Record<string, unknown>,
  accuracy: number,
  status: OptimizationTrialRecord['status'] = 'completed',
): OptimizationTrialRecord {
  return {
    trialId,
    trialNumber: Number.parseInt(trialId, 10) || 0,
    config,
    metrics: { accuracy },
    status,
    metadata: {},
  };
}

describe('native-bayesian suggestion', () => {
  it('marks a discrete search space as exhaustive once all configs are seen', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['a', 'b'],
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [trial('1', { model: 'a' }, 0.1), trial('2', { model: 'b' }, 0.2)],
      new PythonRandom(1),
      10,
      () => true,
    );

    expect(result).toEqual({ config: null, exhaustive: true });
  });

  it('uses random seeding before enough trials have completed', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['a', 'b', 'c'],
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [],
      new PythonRandom(42),
      10,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect(['a', 'b', 'c']).toContain(result.config?.model);
  });

  it('ignores rejected trials when fitting the observed history', () => {
    const spec = createNormalizedSpec();
    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { model: 'a', retries: 0 }, 0.2),
        trial('2', { model: 'b', retries: 1 }, 0.8),
        trial('3', { model: 'c', retries: 2 }, 0.5, 'rejected'),
        trial('4', { model: 'c', retries: 1 }, 0.7),
        trial('5', { model: 'd', retries: 0 }, 0.6),
      ],
      new PythonRandom(7),
      20,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect(result.config).not.toEqual({ model: 'c', retries: 2 });
  });

  it('returns a non-exhaustive null result when constraints reject all continuous candidates', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        temperature: {
          type: 'float',
          min: 0,
          max: 1,
          scale: 'linear',
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [],
      new PythonRandom(3),
      10,
      () => false,
    );

    expect(result).toEqual({ config: null, exhaustive: false });
  });

  it('preserves default config while proposing unseen candidates', () => {
    const spec = createNormalizedSpec({
      defaultConfig: {
        systemPrompt: 'keep-it-short',
      },
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['a', 'b', 'c'],
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [trial('1', { model: 'a', systemPrompt: 'keep-it-short' }, 0.1)],
      new PythonRandom(9),
      10,
      () => true,
    );

    expect(result.config).not.toBeNull();
    expect(result.config?.systemPrompt).toBe('keep-it-short');
  });

  it('uses the final fallback path when the main candidate budget is exhausted', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        temperature: {
          type: 'float',
          min: 0,
          max: 1,
          scale: 'linear',
        },
      },
    });

    let attempts = 0;
    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { temperature: 0.1 }, 0.1),
        trial('2', { temperature: 0.2 }, 0.2),
        trial('3', { temperature: 0.3 }, 0.3),
        trial('4', { temperature: 0.4 }, 0.4),
        trial('5', { temperature: 0.5 }, 0.5),
      ],
      new PythonRandom(17),
      30,
      () => {
        attempts += 1;
        return attempts > 256;
      },
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect(attempts).toBeGreaterThan(256);
  });

  it('explores log-scaled int and float neighborhoods after the random warmup phase', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['a', 'b', 'c'],
        },
        depth: {
          type: 'int',
          min: 1,
          max: 16,
          step: 2,
          scale: 'log',
        },
        learning_rate: {
          type: 'float',
          min: 0.001,
          max: 1,
          step: 10,
          scale: 'log',
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { model: 'a', depth: 1, learning_rate: 0.001 }, 0.2),
        trial('2', { model: 'a', depth: 2, learning_rate: 0.01 }, 0.3),
        trial('3', { model: 'b', depth: 4, learning_rate: 0.1 }, 0.7),
        trial('4', { model: 'b', depth: 8, learning_rate: 1 }, 0.8),
        trial('5', { model: 'c', depth: 1, learning_rate: 0.1 }, 0.5),
        trial('6', { model: 'c', depth: 16, learning_rate: 0.01 }, 0.6),
      ],
      new PythonRandom(21),
      30,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect(['a', 'b', 'c']).toContain(result.config?.model);
    expect([1, 2, 4, 8, 16]).toContain(result.config?.depth);
    expect([0.001, 0.01, 0.1, 1]).toContain(result.config?.learning_rate);
  });

  it('explores linear neighborhoods for stepped ints and continuous floats', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        retries: {
          type: 'int',
          min: 0,
          max: 4,
          step: 1,
          scale: 'linear',
        },
        temperature: {
          type: 'float',
          min: 0,
          max: 1,
          scale: 'linear',
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { retries: 0, temperature: 0.1 }, 0.1),
        trial('2', { retries: 1, temperature: 0.2 }, 0.2),
        trial('3', { retries: 2, temperature: 0.4 }, 0.5),
        trial('4', { retries: 3, temperature: 0.6 }, 0.9),
        trial('5', { retries: 4, temperature: 0.9 }, 0.6),
      ],
      new PythonRandom(31),
      30,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect(typeof result.config?.temperature).toBe('number');
    expect((result.config?.temperature as number) >= 0).toBe(true);
    expect((result.config?.temperature as number) <= 1).toBe(true);
    expect([0, 1, 2, 3, 4]).toContain(result.config?.retries);
  });

  it('explores snapped linear float neighborhoods when steps are defined', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        retries: {
          type: 'int',
          min: 0,
          max: 4,
          step: 2,
          scale: 'linear',
        },
        temperature: {
          type: 'float',
          min: 0,
          max: 1,
          step: 0.25,
          scale: 'linear',
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { retries: 0, temperature: 0 }, 0.1),
        trial('2', { retries: 2, temperature: 0.25 }, 0.2),
        trial('3', { retries: 4, temperature: 0.5 }, 0.7),
        trial('4', { retries: 0, temperature: 0.75 }, 0.4),
        trial('5', { retries: 2, temperature: 1 }, 0.5),
      ],
      new PythonRandom(11),
      30,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect([0, 2, 4]).toContain(result.config?.retries);
    expect([0, 0.25, 0.5, 0.75, 1]).toContain(result.config?.temperature);
  });

  it('can recover when the warmup history contains only rejected trials', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        temperature: {
          type: 'float',
          min: 0,
          max: 1,
          scale: 'linear',
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { temperature: 0.1 }, 0.1, 'rejected'),
        trial('2', { temperature: 0.2 }, 0.2, 'rejected'),
        trial('3', { temperature: 0.3 }, 0.3, 'rejected'),
        trial('4', { temperature: 0.4 }, 0.4, 'rejected'),
        trial('5', { temperature: 0.5 }, 0.5, 'rejected'),
      ],
      new PythonRandom(27),
      30,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
  });

  it('handles fixed-range numeric dimensions during vectorization and local search', () => {
    const spec = createNormalizedSpec({
      configurationSpace: {
        model: {
          type: 'enum',
          values: ['a', 'b', 'c'],
        },
        retries: {
          type: 'int',
          min: 2,
          max: 2,
          step: 1,
          scale: 'linear',
        },
        temperature: {
          type: 'float',
          min: 0.5,
          max: 0.5,
          scale: 'linear',
        },
      },
    });

    const result = suggestBayesianConfig(
      spec,
      [
        trial('1', { model: 'a' }, 0.2),
        trial('2', { model: 'b', retries: 2 }, 0.6),
        trial('3', { model: 'c', temperature: 0.5 }, 0.4),
        trial('4', { model: 'a', retries: 2, temperature: 0.5 }, 0.3),
        trial('5', { model: 'b' }, 0.7),
        trial('6', { model: 'c', retries: 2, temperature: 0.5 }, 0.5),
      ],
      new PythonRandom(4),
      30,
      () => true,
    );

    expect(result.exhaustive).toBe(false);
    expect(result.config).not.toBeNull();
    expect(result.config?.retries).toBe(2);
    expect(result.config?.temperature).toBeCloseTo(0.5, 10);
  });
});
