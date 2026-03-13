import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it, vi } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import {
  TrialContext,
  getTrialConfig,
  getTrialParam,
  parseTvlSpec,
  optimize,
  param,
} from '../../../src/index.js';
import { runNativeOptimization } from '../../../src/optimization/native.js';
import type { NormalizedOptimizationSpec } from '../../../src/optimization/types.js';

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

function createNormalizedSpec(
  overrides: Partial<NormalizedOptimizationSpec> = {}
): NormalizedOptimizationSpec {
  return {
    configurationSpace: {
      model: {
        type: 'enum',
        values: ['a', 'b'],
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

describe('native optimize()', () => {
  it('runs grid optimization and selects the best config from built-in objectives', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy', 'cost'],
      execution: {
        contract: 'trial',
      },
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
    expect(result.bestMetrics).toEqual({
      accuracy: 0.6,
      cost: 0.1,
      total_cost: 0.1,
    });
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
      execution: {
        contract: 'trial',
      },
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
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost: trialConfig.config.model === 'a' ? 0.1 : trialConfig.config.model === 'b' ? 0.2 : 0.3,
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
      second.trials.map((trial) => trial.config)
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
      execution: {
        contract: 'trial',
      },
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
      0.00001, 0.0001, 0.001, 0.01, 0.1,
    ]);
    expect(result.stopReason).toBe('completed');
  });

  it('requires step for float grid search', async () => {
    const wrapped = optimize({
      configurationSpace: {
        temperature: param.float({ min: 0, max: 1, scale: 'linear' }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
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
        maxTrials: 2,
      })
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
      execution: {
        contract: 'trial',
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
      })
    ).rejects.toThrow(
      /budget\.maxCostUsd requires every trial to return numeric metrics\.total_cost or metrics\.cost/i
    );
  });

  it('accepts metrics.total_cost as the budget signal and normalizes cost aliases', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      objectives: ['cost'],
      budget: {
        maxCostUsd: 0.15,
      },
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        total_cost: trialConfig.config.model === 'a' ? 0.1 : 0.2,
        input_cost: 0.04,
        output_cost: trialConfig.config.model === 'a' ? 0.06 : 0.16,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.stopReason).toBe('budget');
    expect(result.totalCostUsd).toBeCloseTo(0.3, 10);
    expect(result.trials[0]?.metrics.cost).toBeCloseTo(0.1, 10);
    expect(result.trials[0]?.metrics.total_cost).toBeCloseTo(0.1, 10);
  });

  it('throws when optimize() is called on a wrapped function that does not return trial metrics', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => 'not-a-trial-result');

    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
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
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost: trialConfig.config.model === 'a' ? 0.1 : trialConfig.config.model === 'b' ? 0.2 : 0.3,
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

  it('stops on execution.maxTotalExamples and limits dataset subsets exactly', async () => {
    const seenSubsetSizes: number[] = [];
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
        maxTotalExamples: 5,
      },
      evaluation: {
        data: [{ id: 1 }, { id: 2 }, { id: 3 }],
      },
    })(async (trialConfig) => {
      seenSubsetSizes.push(trialConfig.dataset_subset.indices.length);
      return {
        metrics: {
          accuracy: 1,
        },
        metadata: {
          evaluatedRows: trialConfig.dataset_subset.indices.length,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.stopReason).toBe('maxExamples');
    expect(seenSubsetSizes).toEqual([3, 2]);
    expect(result.trials).toHaveLength(2);
  });

  it('merges defaultConfig into native trial configs and resulting bestConfig', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      defaultConfig: {
        region: 'eu',
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'b' && trialConfig.config.region === 'eu' ? 1 : 0,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ model: 'b', region: 'eu' });
    expect(result.trials.map((trial) => trial.config)).toEqual([
      { model: 'a', region: 'eu' },
      { model: 'b', region: 'eu' },
    ]);
  });

  it('skips pre-trial constraint failures without consuming trial slots', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
      },
      constraints: [(config) => config.model !== 'b'],
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'c' ? 1 : 0,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.stopReason).toBe('completed');
    expect(result.trials).toHaveLength(2);
    expect(result.trials.map((trial) => trial.config.model)).toEqual(['a', 'c']);
    expect(result.bestConfig).toEqual({ model: 'c' });
  });

  it('records rejected post-trial constraints while continuing optimization', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
      },
      constraints: [(_config, metrics) => Number(metrics.cost) < 0.25],
      objectives: ['cost'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost: trialConfig.config.model === 'a' ? 0.1 : trialConfig.config.model === 'b' ? 0.3 : 0.2,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 3,
    });

    expect(result.stopReason).toBe('completed');
    expect(result.totalCostUsd).toBeCloseTo(0.6, 10);
    expect(result.trials).toHaveLength(3);
    expect(result.trials[1]).toMatchObject({
      status: 'rejected',
      errorMessage: expect.stringMatching(/constraint/i),
    });
    expect(result.bestConfig).toEqual({ model: 'a' });
  });

  it('enforces safetyConstraints after evaluation and keeps successful trials selectable', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['safe', 'unsafe']),
      },
      safetyConstraints: [(_config, metrics) => Number(metrics.faithfulness) >= 0.9],
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'unsafe' ? 1 : 0.8,
        faithfulness: trialConfig.config.model === 'unsafe' ? 0.4 : 0.95,
        total_cost: 0.05,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.stopReason).toBe('completed');
    expect(result.trials.map((trial) => trial.status)).toEqual(['completed', 'rejected']);
    expect(result.bestConfig).toEqual({ model: 'safe' });
    expect(result.totalCostUsd).toBeCloseTo(0.1, 10);
  });

  it('throws when no valid configurations satisfy pre-trial constraints', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      constraints: [() => false],
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
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
        maxTrials: 2,
      })
    ).rejects.toThrow(/No valid configurations satisfy the configured constraints/i);
  });

  it('surfaces pre-trial constraint exceptions as validation errors', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      constraints: [
        () => {
          throw new Error('broken pre-check');
        },
      ],
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
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
      })
    ).rejects.toThrow(/broken pre-check/i);
  });

  it('marks thrown post-trial safety constraints as rejected trials without aborting the run', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['safe', 'unsafe']),
      },
      safetyConstraints: [
        (config) => {
          if (config.model === 'unsafe') {
            throw new Error('unsafe path');
          }
          return true;
        },
      ],
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'unsafe' ? 1 : 0.9,
        total_cost: 0.05,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.stopReason).toBe('completed');
    expect(result.trials[1]).toMatchObject({
      status: 'rejected',
      errorMessage: expect.stringMatching(/unsafe path/i),
    });
    expect(result.bestConfig).toEqual({ model: 'safe' });
  });

  it('aggregates repeated trials for stability while summing actual spent cost', async () => {
    let invocation = 0;
    const sequence = [0.1, 0.9, 0.2];
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['stable']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
        repsPerTrial: 3,
        repsAggregation: 'median',
      },
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async (trialConfig) => {
      const score = sequence[invocation] ?? sequence.at(-1)!;
      invocation += 1;
      return {
        metrics: {
          accuracy: score,
          total_cost: 0.1,
          latency: trialConfig.dataset_subset.indices.length,
        },
        metadata: {
          evaluatedRows: trialConfig.dataset_subset.indices.length,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.trials).toHaveLength(1);
    expect(result.bestMetrics?.accuracy).toBeCloseTo(0.2, 10);
    expect(result.bestMetrics?.total_cost).toBeCloseTo(0.1, 10);
    expect(result.bestMetrics?.cost).toBeCloseTo(0.1, 10);
    expect(result.totalCostUsd).toBeCloseTo(0.3, 10);
    expect(result.trials[0]?.metadata).toMatchObject({
      evaluatedRows: 6,
      repsPerTrial: 3,
      repsAggregation: 'median',
    });
  });

  it('preserves spent cost and evaluated examples when a repeated trial fails mid-run', async () => {
    let invocation = 0;
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['unstable']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
        repsPerTrial: 3,
        repsAggregation: 'mean',
      },
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async (trialConfig) => {
      invocation += 1;
      if (invocation === 2) {
        throw new Error('repeat boom');
      }

      return {
        metrics: {
          accuracy: 0.5,
          total_cost: 0.2,
        },
        metadata: {
          evaluatedRows: trialConfig.dataset_subset.indices.length,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.stopReason).toBe('error');
    expect(result.errorMessage).toContain('repeat boom');
    expect(result.totalCostUsd).toBeCloseTo(0.2, 10);
    expect(result.trials).toHaveLength(0);
  });

  it('returns a timeout stop reason when a trial exceeds timeoutMs', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['slow']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
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
      execution: {
        contract: 'trial',
      },
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
      execution: {
        contract: 'trial',
      },
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

  it('rejects bayesian trialConcurrency > 1', async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
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
        algorithm: 'bayesian',
        maxTrials: 4,
        trialConcurrency: 2,
      })
    ).rejects.toThrow(/does not support trialConcurrency > 1/i);
  });

  it('preserves TrialContext isolation across concurrent trials', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c', 'd']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
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
    let invocationCount = 0;
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
        retries: param.int({ min: 0, max: 2 }),
      },
      objectives: ['accuracy', 'cost'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      invocationCount += 1;
      await delay(invocationCount === 1 ? 5 : 250);
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
    setTimeout(() => controller.abort(), 25);

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
      uninterrupted.trials.map((trial) => trial.config)
    );

    await rm(checkpointDir, { recursive: true, force: true });
  }, 10000);

  it('supports bayesian checkpoint/resume by restoring sampler state', async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), 'traigent-bayes-checkpoint-'));
    const controller = new AbortController();
    let invocationCount = 0;
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: -1, max: 1 }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      invocationCount += 1;
      if (invocationCount === 2) {
        controller.abort();
      }
      return {
        metrics: {
          accuracy: 1 - Math.abs(Number(trialConfig.config.x)),
        },
      };
    });

    const interrupted = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 4,
      randomSeed: 7,
      signal: controller.signal,
      checkpoint: {
        key: 'resume-bayesian',
        dir: checkpointDir,
      },
    });

    expect(interrupted.stopReason).toBe('cancelled');
    expect(interrupted.trials.length).toBeGreaterThan(0);

    const resumed = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 4,
      randomSeed: 7,
      checkpoint: {
        key: 'resume-bayesian',
        dir: checkpointDir,
        resume: true,
      },
    });

    expect(resumed.trials).toHaveLength(4);
    expect(resumed.stopReason).toBe('maxTrials');

    await rm(checkpointDir, { recursive: true, force: true });
  });

  it('stops with plateau when improvements stay below the configured threshold', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c', 'd', 'e']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
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

  it('optimizes a plain agent function using scoringFunction and default dataset mapping', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [
          { input: 'What is 2+2?', output: '4' },
          { input: 'What is the capital of France?', output: 'Paris' },
        ],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (question: string) => {
      const model = String(getTrialParam('model', 'cheap'));
      if (model === 'best') {
        return question.includes('France') ? 'Paris' : '4';
      }
      return 'unknown';
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ model: 'best' });
    expect(result.bestMetrics).toEqual({ accuracy: 1, latency: expect.any(Number) });
  });

  it('supports parameter injection for high-level agent optimization', async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['plain', 'friendly']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'parameter',
      },
      evaluation: {
        data: [{ input: { question: 'hi' }, output: 'friendly:hi' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (input: { question: string }, config?: { tone?: string }) => {
      return `${config?.tone ?? 'plain'}:${input.question}`;
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ tone: 'friendly' });
    wrapped.applyBestConfig(result);
    await expect(wrapped({ question: 'hi' })).resolves.toBe('friendly:hi');
  });

  it('combines scoringFunction with metricFunctions and runtime metrics', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['cheap', 'best']),
      },
      objectives: ['accuracy', 'cost'],
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
        metricFunctions: {
          cost: (output) => (output === 'x' ? 0.1 : 0.4),
        },
      },
    })(async (value: string) => {
      return getTrialParam('model') === 'best' ? value : 'wrong';
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ model: 'best' });
    expect(result.bestMetrics?.accuracy).toBe(1);
    expect(result.bestMetrics?.cost).toBe(0.1);
  });

  it('fails the trial when the evaluator does not produce numeric objective metrics', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'y' }],
        metricFunctions: {
          cost: () => 0.1,
        },
      },
    })(async () => 'x');

    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/accuracy/i);
  });

  it('allows metricFunctions without expected output fields', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: [{ metric: 'cost', direction: 'minimize' }],
      evaluation: {
        data: [{ input: 'x' }],
        metricFunctions: {
          cost: (_output, _expectedOutput, runtimeMetrics) => runtimeMetrics.latency ?? 0.5,
        },
      },
    })(async () => 'x');

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.bestMetrics?.cost).toBeGreaterThanOrEqual(0);
  });

  it('allows customEvaluator without expected output fields', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x' }],
        customEvaluator: async ({ output, expectedOutput }) => ({
          accuracy: output === 'x' && expectedOutput === undefined ? 1 : 0,
        }),
      },
    })(async () => 'x');

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.bestMetrics?.accuracy).toBe(1);
  });

  it('surfaces scoringFunction errors as optimizer error results', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', output: 'x' }],
        scoringFunction: () => {
          throw new Error('boom');
        },
      },
    })(async () => 'x');

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(result.stopReason).toBe('error');
    expect(result.errorMessage).toContain('boom');
  });

  it('supports async loadData for agent optimization', async () => {
    const loadData = vi.fn(async () => [{ question: 'hi', answer: 'HI' }]);
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['quiet', 'loud']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'parameter',
      },
      evaluation: {
        loadData,
        inputField: 'question',
        expectedField: 'answer',
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (input: string, config?: { tone?: string }) =>
      config?.tone === 'loud' ? input.toUpperCase() : input.toLowerCase()
    );

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(loadData).toHaveBeenCalledTimes(1);
    expect(result.bestConfig).toEqual({ tone: 'loud' });
  });

  it('supports custom input and expected field resolution', async () => {
    const wrapped = optimize({
      configurationSpace: {
        tone: param.enum(['quiet', 'loud']),
      },
      objectives: ['accuracy'],
      injection: {
        mode: 'parameter',
      },
      evaluation: {
        data: [{ question: 'hello', answer: 'HELLO' }],
        inputField: 'question',
        expectedField: 'answer',
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
    })(async (input: string, config?: { tone?: string }) =>
      config?.tone === 'loud' ? input.toUpperCase() : input.toLowerCase()
    );

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
    });

    expect(result.bestConfig).toEqual({ tone: 'loud' });
  });

  it('validates optimize options and evaluation requirements', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(wrapped.optimize(undefined as unknown as never)).rejects.toThrow(
      /options are required/i
    );
    await expect(wrapped.optimize({ algorithm: 'bad' as never, maxTrials: 1 })).rejects.toThrow(
      /only supports algorithm/i
    );
    await expect(wrapped.optimize({ algorithm: 'grid', maxTrials: 0 })).rejects.toThrow(
      /positive integer/i
    );
    await expect(
      wrapped.optimize({ algorithm: 'grid', maxTrials: 1, randomSeed: -1 })
    ).rejects.toThrow(/randomSeed/i);
    await expect(
      wrapped.optimize({ algorithm: 'grid', maxTrials: 1, timeoutMs: 0 })
    ).rejects.toThrow(/timeoutMs/i);
    await expect(
      wrapped.optimize({ algorithm: 'grid', maxTrials: 1, trialConcurrency: 0 })
    ).rejects.toThrow(/trialConcurrency/i);
    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
        plateau: {
          window: 0,
          minImprovement: 0,
        },
      })
    ).rejects.toThrow(/plateau\.window/i);
    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
        plateau: {
          window: 1,
          minImprovement: -1,
        },
      })
    ).rejects.toThrow(/plateau\.minImprovement/i);
    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
        checkpoint: {
          key: '',
        },
      })
    ).rejects.toThrow(/checkpoint\.key/i);
    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
        checkpoint: {
          key: 'ok',
          dir: '',
        },
      })
    ).rejects.toThrow(/checkpoint\.dir/i);
  });

  it('rejects missing or empty evaluation data for the low-level trial contract', async () => {
    const missingData = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      missingData.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/requires spec\.evaluation\.data or spec\.evaluation\.loadData/i);

    const emptyData = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    await expect(
      emptyData.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/non-empty array/i);
  });

  it('rejects invalid numeric metrics from low-level trial functions', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: Number.NaN,
      },
    }));

    await expect(
      wrapped.optimize({
        algorithm: 'grid',
        maxTrials: 1,
      })
    ).rejects.toThrow(/trial metrics are invalid/i);
  });

  it('surfaces string throws as optimizer errors and uses explicit trial duration when provided', async () => {
    const failing = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => {
      throw 'bad-news';
    });

    const failure = await failing.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(failure.stopReason).toBe('error');
    expect(failure.errorMessage).toBe('bad-news');

    const durationWrapped = optimize({
      configurationSpace: {
        model: param.enum(['a']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
      duration: 12.5,
    }));

    const durationResult = await durationWrapped.optimize({
      algorithm: 'grid',
      maxTrials: 1,
    });

    expect(durationResult.trials[0]?.duration).toBe(12.5);
  });

  it('supports continuous random sampling and stops when a discrete bayesian space is exhausted', async () => {
    const randomWrapped = optimize({
      configurationSpace: {
        temperature: param.float({ min: 0, max: 1 }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: 1 - Math.abs(Number(trialConfig.config.temperature) - 0.5),
      },
    }));

    const randomResult = await randomWrapped.optimize({
      algorithm: 'random',
      maxTrials: 3,
      randomSeed: 7,
    });

    expect(randomResult.trials).toHaveLength(3);
    expect(randomResult.stopReason).toBe('maxTrials');

    const bayesianWrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'b' ? 1 : 0.5,
      },
    }));

    const bayesianResult = await bayesianWrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 10,
      randomSeed: 3,
    });

    expect(bayesianResult.stopReason).toBe('completed');
    expect(bayesianResult.trials).toHaveLength(2);
  });

  it('rejects mismatched checkpoint resume state', async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), 'traigent-checkpoint-mismatch-'));
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === 'b' ? 1 : 0,
      },
    }));

    await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 2,
      checkpoint: {
        key: 'resume-mismatch',
        dir: checkpointDir,
      },
    });

    await expect(
      wrapped.optimize({
        algorithm: 'random',
        maxTrials: 2,
        checkpoint: {
          key: 'resume-mismatch',
          dir: checkpointDir,
          resume: true,
        },
      })
    ).rejects.toThrow(/does not match the current spec\/options/i);

    await rm(checkpointDir, { recursive: true, force: true });
  });

  it('supports loadData, pre-aborted signals, and missing resume checkpoints in runNativeOptimization()', async () => {
    const checkpointDir = await mkdtemp(join(tmpdir(), 'traigent-checkpoint-empty-'));
    const loadData = vi.fn(async () => [{ id: 1 }]);
    const controller = new AbortController();
    controller.abort();

    const cancelled = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
        },
      }),
      createNormalizedSpec({
        evaluation: {
          loadData,
        },
      }),
      {
        algorithm: 'grid',
        maxTrials: 2,
        signal: controller.signal,
      }
    );

    expect(cancelled.stopReason).toBe('cancelled');
    expect(loadData).toHaveBeenCalledTimes(1);

    const resumedWithoutFile = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy: trialConfig.config.model === 'b' ? 1 : 0,
        },
      }),
      createNormalizedSpec(),
      {
        algorithm: 'grid',
        maxTrials: 2,
        checkpoint: {
          key: 'missing-file',
          dir: checkpointDir,
          resume: true,
        },
      }
    );

    expect(resumedWithoutFile.stopReason).toBe('completed');
    expect(resumedWithoutFile.trials).toHaveLength(2);

    await rm(checkpointDir, { recursive: true, force: true });
  });

  it('validates handcrafted invalid normalized specs inside runNativeOptimization()', async () => {
    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        createNormalizedSpec({
          configurationSpace: {
            retries: {
              type: 'int',
              min: 1,
              max: 16,
              scale: 'log',
            },
          },
        }),
        {
          algorithm: 'grid',
          maxTrials: 4,
        }
      )
    ).rejects.toThrow(/log-scaled int parameters to define a multiplicative step/i);

    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        createNormalizedSpec({
          configurationSpace: {
            temperature: {
              type: 'float',
              min: 0.1,
              max: 1,
              scale: 'log',
              step: 1,
            },
          },
        }),
        {
          algorithm: 'grid',
          maxTrials: 4,
        }
      )
    ).rejects.toThrow(/multiplicative step greater than 1/i);
  });

  it('completes random search when a discrete space is exhausted before maxTrials', async () => {
    const result = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy: trialConfig.config.model === 'b' ? 1 : 0,
        },
      }),
      createNormalizedSpec(),
      {
        algorithm: 'random',
        maxTrials: 10,
        randomSeed: 1,
      }
    );

    expect(result.stopReason).toBe('completed');
    expect(result.trials).toHaveLength(2);
  });

  it('covers additional handcrafted native parameter edge cases', async () => {
    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        createNormalizedSpec({
          configurationSpace: {
            retries: {
              type: 'int',
              min: 0,
              max: 2,
              step: 0.5,
            },
          },
        }),
        {
          algorithm: 'grid',
          maxTrials: 4,
        }
      )
    ).rejects.toThrow(/positive integer step/i);

    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        createNormalizedSpec({
          configurationSpace: {
            temperature: {
              type: 'float',
              min: 0,
              max: 1,
              step: -0.1,
            },
          },
        }),
        {
          algorithm: 'grid',
          maxTrials: 4,
        }
      )
    ).rejects.toThrow(/positive finite step/i);

    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            accuracy: 1,
          },
        }),
        createNormalizedSpec({
          configurationSpace: {
            temperature: {
              type: 'float',
              min: 0.1,
              max: 1,
              scale: 'log',
            },
          },
        }),
        {
          algorithm: 'grid',
          maxTrials: 4,
        }
      )
    ).rejects.toThrow(/log-scaled float parameters to define a multiplicative step/i);

    const linearIntResult = await runNativeOptimization(
      async (trialConfig) => ({
        metrics: {
          accuracy: Number(trialConfig.config.retries),
        },
      }),
      createNormalizedSpec({
        configurationSpace: {
          retries: {
            type: 'int',
            min: 0,
            max: 3,
            step: 2,
          },
        },
      }),
      {
        algorithm: 'grid',
        maxTrials: 10,
      }
    );

    expect(linearIntResult.trials.map((trial) => trial.config.retries)).toEqual([0, 2, 3]);
  });

  it('fails low-level trials that do not provide the configured objective metric', async () => {
    await expect(
      runNativeOptimization(
        async () => ({
          metrics: {
            cost: 0.1,
          },
        }),
        createNormalizedSpec(),
        {
          algorithm: 'grid',
          maxTrials: 2,
        }
      )
    ).rejects.toThrow(/missing numeric metric "accuracy"/i);
  });

  it('covers bayesian stop reasons for timeout, budget, and plateau', async () => {
    const timeoutResult = await runNativeOptimization(
      async () => {
        await delay(30);
        return {
          metrics: {
            accuracy: 1,
          },
        };
      },
      createNormalizedSpec({
        configurationSpace: {
          temperature: {
            type: 'float',
            min: 0,
            max: 1,
          },
        },
      }),
      {
        algorithm: 'bayesian',
        maxTrials: 2,
        timeoutMs: 1,
      }
    );

    expect(timeoutResult.stopReason).toBe('timeout');

    const budgetResult = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
          cost: 0.2,
        },
      }),
      createNormalizedSpec({
        budget: {
          maxCostUsd: 0.1,
        },
      }),
      {
        algorithm: 'bayesian',
        maxTrials: 5,
      }
    );

    expect(budgetResult.stopReason).toBe('budget');

    const plateauResult = await runNativeOptimization(
      async () => ({
        metrics: {
          accuracy: 1,
        },
      }),
      createNormalizedSpec({
        configurationSpace: {
          temperature: {
            type: 'float',
            min: 0,
            max: 1,
          },
        },
      }),
      {
        algorithm: 'bayesian',
        maxTrials: 8,
        plateau: {
          window: 2,
          minImprovement: 0.01,
        },
      }
    );

    expect(plateauResult.stopReason).toBe('plateau');
  });

  it('continues bayesian search after rejected trials from post-trial constraints', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
      },
      constraints: [(_config, metrics) => Number(metrics.cost) < 0.25],
      objectives: ['cost'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        cost: trialConfig.config.model === 'a' ? 0.1 : trialConfig.config.model === 'b' ? 0.3 : 0.2,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 3,
      randomSeed: 11,
    });

    expect(result.stopReason).toBe('maxTrials');
    expect(result.trials.some((trial) => trial.status === 'rejected')).toBe(true);
    expect(result.bestConfig).not.toEqual({ model: 'b' });
  });

  it('stops bayesian search on execution.maxTotalExamples', async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
        maxTotalExamples: 2,
      },
      evaluation: {
        data: [{ id: 1 }, { id: 2 }, { id: 3 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: Number(trialConfig.config.x),
      },
      metadata: {
        evaluatedRows: trialConfig.dataset_subset.indices.length,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 5,
      randomSeed: 3,
    });

    expect(result.stopReason).toBe('maxExamples');
    expect(result.trials).toHaveLength(1);
    expect(result.trials[0]?.metadata).toMatchObject({
      evaluatedRows: 2,
    });
  });

  it('stops bayesian search immediately when the sample budget cannot cover one repetition', async () => {
    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
        maxTotalExamples: 1,
        repsPerTrial: 2,
      },
      evaluation: {
        data: [{ id: 1 }, { id: 2 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 5,
      randomSeed: 4,
    });

    expect(result.stopReason).toBe('maxExamples');
    expect(result.trials).toHaveLength(0);
  });

  it('returns cancelled for bayesian search when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();

    const wrapped = optimize({
      configurationSpace: {
        x: param.float({ min: 0, max: 1 }),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: 2,
      signal: controller.signal,
    });

    expect(result.stopReason).toBe('cancelled');
    expect(result.errorMessage).toContain('cancelled');
    expect(result.trials).toHaveLength(0);
  });

  it('prefers in-band values for banded objectives', async () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: response_length
    type: int
    domain:
      range: [100, 220]
      step: 40
objectives:
  - name: response_length
    band:
      target: [130, 190]
exploration:
  strategy: grid
  budgets:
    max_trials: 10
`);

    const wrapped = optimize({
      ...parsed.spec,
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        response_length: Number(trialConfig.config.response_length),
      },
    }));

    const result = await wrapped.optimize({
      algorithm: parsed.optimizeOptions?.algorithm ?? 'grid',
      maxTrials: parsed.optimizeOptions?.maxTrials ?? 10,
    });

    expect(result.bestConfig).toEqual({ response_length: 140 });
    expect(result.bestMetrics).toEqual({ response_length: 140 });
  });

  it('uses promotionPolicy minEffect and tieBreakers when selecting the best trial', async () => {
    const wrapped = optimize({
      configurationSpace: {
        variant: param.enum(['a', 'b']),
      },
      objectives: [
        { metric: 'accuracy', direction: 'maximize' },
        { metric: 'latency', direction: 'minimize' },
      ],
      promotionPolicy: {
        dominance: 'epsilon_pareto',
        minEffect: {
          accuracy: 0.05,
          latency: 10,
        },
        tieBreakers: {
          cost: 'minimize',
        },
      },
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      if (trialConfig.config.variant === 'a') {
        return {
          metrics: {
            accuracy: 0.92,
            latency: 100,
            cost: 0.2,
          },
        };
      }

      return {
        metrics: {
          accuracy: 0.95,
          latency: 104,
          cost: 0.1,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ variant: 'b' });
    expect(result.bestMetrics).toEqual({
      accuracy: 0.95,
      latency: 104,
      cost: 0.1,
      total_cost: 0.1,
    });
    expect(result.promotionDecision).toMatchObject({
      decision: 'promote',
      method: 'deterministic',
      candidateTrialId: expect.any(String),
      incumbentTrialId: expect.any(String),
    });
    expect(result.promotionDecision?.reason).toMatch(/tie-breaker|deterministic/i);
    expect(result.reporting).toMatchObject({
      totalTrials: 2,
      completedTrials: 2,
      rejectedTrials: 0,
      evaluatedExamples: 2,
      promotion: {
        applied: true,
        bestTrialId: expect.any(String),
        bestTrialNumber: 2,
        decision: 'promote',
        method: 'deterministic',
        usedChanceConstraints: false,
        usedStatisticalComparison: false,
        usedTieBreakers: true,
      },
    });
  });

  it('rejects trial-contract candidates that fail promotionPolicy chance constraints', async () => {
    const wrapped = optimize({
      configurationSpace: {
        variant: param.enum(['safe', 'unsafe']),
      },
      objectives: ['accuracy'],
      promotionPolicy: {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.8,
            confidence: 0.95,
          },
        ],
      },
      execution: {
        contract: 'trial',
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      if (trialConfig.config.variant === 'safe') {
        return {
          metrics: {
            accuracy: 0.9,
          },
          metadata: {
            chanceConstraintCounts: {
              safety: { successes: 95, trials: 100 },
            },
          },
        };
      }

      return {
        metrics: {
          accuracy: 0.99,
        },
        metadata: {
          chanceConstraintCounts: {
            safety: { successes: 50, trials: 100 },
          },
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ variant: 'safe' });
    expect(result.promotionDecision?.decision).not.toBe('reject');
    expect(result.reporting).toMatchObject({
      totalTrials: 2,
      completedTrials: 1,
      rejectedTrials: 1,
      evaluatedExamples: 2,
      promotion: {
        applied: true,
        bestTrialId: expect.any(String),
        bestTrialNumber: 1,
        decision: 'reject',
        method: 'chance-constraints',
        usedChanceConstraints: true,
      },
    });
    expect(result.trials.find((trial) => trial.config.variant === 'unsafe')).toMatchObject({
      status: 'rejected',
      promotionDecision: {
        decision: 'reject',
        method: 'chance-constraints',
      },
    });
  });

  it('derives chance constraint counts from binary per-example metrics in agent mode', async () => {
    const wrapped = optimize({
      configurationSpace: {
        variant: param.enum(['safe', 'unsafe']),
      },
      objectives: ['accuracy'],
      promotionPolicy: {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.3,
            confidence: 0.8,
          },
        ],
      },
      evaluation: {
        data: [
          { input: 'a', expected_output: 'ok' },
          { input: 'b', expected_output: 'ok' },
          { input: 'c', expected_output: 'ok' },
          { input: 'd', expected_output: 'ok' },
        ],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
        metricFunctions: {
          safety: (output) => (output === 'ok' ? 1 : 0),
        },
      },
      injection: {
        mode: 'parameter',
      },
    })(async (_input, config) => (config?.variant === 'safe' ? 'ok' : 'bad'));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ variant: 'safe' });
    expect(result.trials.find((trial) => trial.config.variant === 'unsafe')).toMatchObject({
      status: 'rejected',
    });
  });

  it('reports baseline trial counts when no promotionPolicy is configured', async () => {
    const wrapped = optimize({
      configurationSpace: {
        variant: param.enum(['a', 'b']),
      },
      objectives: ['accuracy'],
      evaluation: {
        data: [{ input: 'x', expected_output: 'x' }],
        scoringFunction: (output, expectedOutput) => (output === expectedOutput ? 1 : 0),
      },
      injection: {
        mode: 'parameter',
      },
    })(async (_input, config) => (config?.variant === 'a' ? 'x' : 'y'));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.bestConfig).toEqual({ variant: 'a' });
    expect(result.reporting).toMatchObject({
      totalTrials: 2,
      completedTrials: 2,
      rejectedTrials: 0,
      evaluatedExamples: 2,
      promotion: {
        applied: false,
        bestTrialId: expect.any(String),
        bestTrialNumber: 1,
        decision: undefined,
        method: undefined,
        usedChanceConstraints: false,
        usedStatisticalComparison: false,
        usedTieBreakers: false,
      },
    });
  });

  it('stops on execution.maxWallclockMs for candidate-plan algorithms', async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(['a', 'b', 'c']),
      },
      objectives: ['accuracy'],
      execution: {
        contract: 'trial',
        maxWallclockMs: 20,
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      await delay(15);
      return {
        metrics: {
          accuracy: trialConfig.config.model === 'c' ? 1 : 0.5,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 10,
    });

    expect(result.stopReason).toBe('timeout');
    expect(result.errorMessage).toContain('wallclock');
    expect(result.trials.length).toBeGreaterThan(0);
    expect(result.trials.length).toBeLessThan(3);
  });
});
