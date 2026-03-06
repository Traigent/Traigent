import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import {
  TrialContext,
  getTrialConfig,
  getTrialParam,
  optimize,
  param,
} from '../../../src/index.js';

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
      maxTrials: 5,
      randomSeed: 42,
    });
    const second = await wrapped.optimize({
      algorithm: 'random',
      maxTrials: 5,
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

  it('applies conditional parameter defaults during grid search', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-3.5', 'gpt-4']),
        max_tokens: param.int({
          min: 256,
          max: 768,
          step: 256,
          conditions: { model: 'gpt-4' },
          default: 512,
        }),
        temperature: param.float({
          min: 0.1,
          max: 0.5,
          step: 0.2,
          conditions: { model: 'gpt-3.5' },
          default: 0.3,
        }),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          trialConfig.config.model === 'gpt-4'
            ? Number(trialConfig.config.max_tokens) / 1000
            : Number(trialConfig.config.temperature),
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.trials.map((trial) => trial.config)).toEqual([
      { model: 'gpt-3.5', max_tokens: 512, temperature: 0.1 },
      { model: 'gpt-3.5', max_tokens: 512, temperature: 0.3 },
      { model: 'gpt-3.5', max_tokens: 512, temperature: 0.5 },
      { model: 'gpt-4', max_tokens: 256, temperature: 0.3 },
      { model: 'gpt-4', max_tokens: 512, temperature: 0.3 },
      { model: 'gpt-4', max_tokens: 768, temperature: 0.3 },
    ]);
    expect(result.bestConfig).toEqual({
      model: 'gpt-4',
      max_tokens: 768,
      temperature: 0.3,
    });
  });

  it('preserves conditional activation rules in deterministic random search', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['gpt-3.5', 'gpt-4']),
        max_tokens: param.int({
          min: 256,
          max: 768,
          step: 256,
          conditions: { model: 'gpt-4' },
          default: 512,
        }),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          trialConfig.config.model === 'gpt-4'
            ? Number(trialConfig.config.max_tokens)
            : 1,
      },
    }));

    const first = await wrapped.optimize({
      algorithm: 'random',
      maxTrials: 6,
      randomSeed: 7,
    });
    const second = await wrapped.optimize({
      algorithm: 'random',
      maxTrials: 6,
      randomSeed: 7,
    });

    expect(first.trials.map((trial) => trial.config)).toEqual(
      second.trials.map((trial) => trial.config),
    );

    for (const trial of first.trials) {
      if (trial.config.model === 'gpt-4') {
        expect([256, 512, 768]).toContain(trial.config.max_tokens);
      } else {
        expect(trial.config.max_tokens).toBe(512);
      }
    }
  });

  it('keeps conditional parameters valid during bayesian optimization', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
        temperature: param.float({
          min: 0.1,
          max: 0.9,
          step: 0.2,
          conditions: { model: 'cheap' },
          default: 0.3,
        }),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          trialConfig.config.model === 'best'
            ? 1
            : 0.5 + Number(trialConfig.config.temperature) / 10,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 8,
      randomSeed: 5,
    });

    expect(result.stopReason).toBe('completed');
    for (const trial of result.trials) {
      if (trial.config.model === 'best') {
        expect(trial.config.temperature).toBe(0.3);
      } else {
        expect([0.1, 0.3, 0.5, 0.7, 0.9]).toContain(trial.config.temperature);
      }
    }
  });

  it('rejects bayesian trialConcurrency > 1', async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
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
        algorithm: 'bayesian',
        maxTrials: 4,
        trialConcurrency: 2,
      }),
    ).rejects.toThrow(/does not support trialConcurrency > 1/i);
  });

  it('preserves TrialContext isolation across concurrent trials', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c', 'd']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      const before = getTrialParam('model');
      await delay(trialConfig.config.model === 'a' ? 20 : 5);
      const currentConfig = getTrialConfig();
      const after = getTrialParam('model');
      return {
        metrics: {
          accuracy: before === after ? 1 : 0,
        },
        metadata: {
          before,
          after,
          current: currentConfig.model,
          trialId: TrialContext.getTrialId(),
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 4,
      trialConcurrency: 4,
    });

    expect(result.stopReason).toBe('completed');
    for (const trial of result.trials) {
      expect(trial.metrics.accuracy).toBe(1);
      expect(trial.metadata).toMatchObject({
        before: trial.config.model,
        after: trial.config.model,
        current: trial.config.model,
      });
    }
  });

  it('supports checkpoint/resume with the same final result as an uninterrupted run', async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), 'traigent-checkpoint-'));
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
        retries: param.int({ min: 0, max: 2 }),
      },
      objectives: ['accuracy', 'cost'],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      await delay(15);
      return {
        metrics: {
          accuracy:
            Number(trialConfig.config.retries) * 0.2 +
            (trialConfig.config.model === 'c' ? 0.6 : 0.3),
          cost: trialConfig.config.model === 'c' ? 0.5 : 0.1,
        },
      };
    });

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 35);

    const partial = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 9,
      signal: controller.signal,
      checkpoint: {
        key: 'resume-grid',
        dir: checkpointDir,
      },
    });

    const resumed = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 9,
      checkpoint: {
        key: 'resume-grid',
        dir: checkpointDir,
        resume: true,
      },
    });

    const uninterrupted = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 9,
    });

    expect(partial.stopReason).toBe('cancelled');
    expect(partial.trials.length).toBeGreaterThan(0);
    expect(resumed.bestConfig).toEqual(uninterrupted.bestConfig);
    expect(resumed.bestMetrics).toEqual(uninterrupted.bestMetrics);
    expect(resumed.stopReason).toEqual(uninterrupted.stopReason);
    expect(resumed.trials.map((trial) => trial.config)).toEqual(
      uninterrupted.trials.map((trial) => trial.config),
    );

    await rm(checkpointDir, { recursive: true, force: true });
  });

  it('stops with plateau when improvements stay below the configured threshold', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c', 'd', 'e']),
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

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 5,
      plateau: {
        window: 2,
        minImprovement: 0.01,
      },
    });

    expect(result.stopReason).toBe('plateau');
    expect(result.trials).toHaveLength(3);
  });
});
