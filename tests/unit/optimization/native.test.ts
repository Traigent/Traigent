import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import { optimize, param } from '../../../src/index.js';

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function branin(x: number, y: number): number {
  const a = 1;
  const b = 5.1 / (4 * Math.PI ** 2);
  const c = 5 / Math.PI;
  const r = 6;
  const s = 10;
  const t = 1 / (8 * Math.PI);
  return a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * Math.cos(x) + s;
}

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
        retries: param.int({ min: 0, max: 2 }),
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

  it('supports log-scaled float grid search', async () => {
    const wrapped = optimize({
      configurationSpace: {
        learning_rate: param.float({
          min: 0.00001,
          max: 0.1,
          scale: 'log',
          step: 10,
        }),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: Number(trialConfig.config.learning_rate),
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.trials.map((trial) => trial.config.learning_rate)).toEqual([
      0.00001,
      0.0001,
      0.001,
      0.01,
      0.1,
    ]);
    expect(result.stopReason).toBe('completed');
  });

  it('requires step for float grid search', async () => {
    const wrapped = optimize({
      configurationSpace: {
        temperature: param.float({ min: 0, max: 1, scale: 'linear' }),
      },
      objectives: ['accuracy'],
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
        maxTrials: 2,
      }),
    ).rejects.toThrow(/float parameters to define step/i);
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
    ).rejects.toThrow(
      /budget\.maxCostUsd requires every trial to return numeric metrics\.cost/i,
    );
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

  it('returns a timeout stop reason when a trial exceeds timeoutMs', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['slow']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => {
      await delay(50);
      return {
        metrics: {
          accuracy: 1,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
      timeoutMs: 10,
    });

    expect(result.stopReason).toBe('timeout');
    expect(result.errorMessage).toMatch(/timeout/i);
    expect(result.trials).toHaveLength(0);
  });

  it('returns an error stop reason when the trial function throws', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => {
      throw new Error('boom');
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.stopReason).toBe('error');
    expect(result.errorMessage).toContain('boom');
    expect(result.trials).toHaveLength(0);
  });

  it('supports sequential bayesian optimization for smooth objectives', async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: -5, max: 10 }),
        y: param.float({ min: 0, max: 15 }),
      },
      objectives: [{ metric: 'score', direction: 'maximize' }],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      const x = Number(trialConfig.config.x);
      const y = Number(trialConfig.config.y);
      return {
        metrics: {
          score: -branin(x, y),
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 24,
      randomSeed: 7,
    });

    expect(result.trials).toHaveLength(24);
    expect(result.stopReason).toBe('maxTrials');
    expect(Number(result.bestConfig?.x)).toBeGreaterThanOrEqual(-5);
    expect(Number(result.bestConfig?.x)).toBeLessThanOrEqual(10);
    expect(Number(result.bestConfig?.y)).toBeGreaterThanOrEqual(0);
    expect(Number(result.bestConfig?.y)).toBeLessThanOrEqual(15);
    expect(Number(result.bestMetrics?.score)).toBeGreaterThan(-5);
  });
});
