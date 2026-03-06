import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import { optimize, param } from '../../../src/index.js';

describe('native optimize()', () => {
  it('runs grid optimization and selects the best config from built-in objectives', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy', 'cost'],
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async (trialConfig) => {
      const model = trialConfig.config.model;
      return {
        metrics: {
          accuracy: model === 'best' ? 0.95 : 0.6,
          cost: model === 'best' ? 0.8 : 0.1,
        },
        metadata: {
          total: trialConfig.dataset_subset.total,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ model: 'cheap' });
    expect(result.bestMetrics).toEqual({ accuracy: 0.6, cost: 0.1 });
    expect(result.trials).toHaveLength(2);
    expect(result.stopReason).toBe('completed');
    expect(result.totalCostUsd).toBeCloseTo(0.9, 10);
  });

  it('uses explicit objective objects for custom metrics', async () => {
    const wrapped = optimize({
      configurationSpace: {
        prompt: param.enum(['short', 'long']),
      },
      objectives: [{ metric: 'quality_score', direction: 'maximize' }],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        quality_score: trialConfig.config.prompt === 'long' ? 0.9 : 0.5,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ prompt: 'long' });
    expect(result.bestMetrics).toEqual({ quality_score: 0.9 });
  });

  it('produces deterministic random sampling when randomSeed is fixed', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
        temperature: param.float({ min: 0, max: 1, step: 0.5 }),
      },
      objectives: ['cost'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost:
          trialConfig.config.model === 'a'
            ? 0.1
            : trialConfig.config.model === 'b'
              ? 0.2
              : 0.3,
      },
    }));

    const first = await wrapped.optimize({
      algorithm: 'random',
      maxTrials: 4,
      randomSeed: 42,
    });
    const second = await wrapped.optimize({
      algorithm: 'random',
      maxTrials: 4,
      randomSeed: 42,
    });

    expect(first.trials.map((trial) => trial.config)).toEqual(
      second.trials.map((trial) => trial.config),
    );
  });

  it('throws when budget.maxCostUsd is used without numeric metrics.cost', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      budget: {
        maxCostUsd: 1,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      }),
    ).rejects.toThrow(ValidationError);
  });

  it('throws when optimize() is called on a wrapped function that does not return trial metrics', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => 'not-a-trial-result');

    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      }),
    ).rejects.toThrow(ValidationError);
  });

  it('stops early when the accumulated cost budget is reached', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
      },
      objectives: ['cost'],
      budget: {
        maxCostUsd: 0.25,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost:
          trialConfig.config.model === 'a'
            ? 0.1
            : trialConfig.config.model === 'b'
              ? 0.2
              : 0.3,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.stopReason).toBe('budget');
    expect(result.trials).toHaveLength(2);
    expect(result.totalCostUsd).toBeCloseTo(0.3, 10);
  });
});
