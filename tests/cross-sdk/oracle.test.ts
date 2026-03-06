import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { optimize, param } from '../../src/index.js';

interface OracleFixture {
  case: string;
  spec: {
    configurationSpace: Record<string, any>;
    objectives: any[];
    budget?: { maxCostUsd: number };
  };
  normalized_stop_reason: string;
  best_config?: Record<string, unknown> | null;
  best_metrics?: Record<string, number> | null;
  total_cost_usd?: number | null;
  configs?: Array<Record<string, unknown>> | null;
  trial_count?: number | null;
  supported?: boolean;
  reason?: string | null;
}

interface OraclePayload {
  fixtures: OracleFixture[];
}

const pythonRepoRoot = resolve(process.cwd(), '../Traigent');
const oracleScript = resolve(
  pythonRepoRoot,
  'tests/cross_sdk_oracles/generate_native_js_oracles.py',
);

function loadOraclePayload():
  | { available: true; payload: OraclePayload }
  | { available: false; reason: string } {
  if (!existsSync(oracleScript)) {
    return {
      available: false,
      reason: `Missing oracle script: ${oracleScript}`,
    };
  }

  try {
    const stdout = execFileSync('python3', [oracleScript], {
      cwd: pythonRepoRoot,
      env: {
        ...process.env,
        PYTHONPATH: pythonRepoRoot,
      },
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    return {
      available: true,
      payload: JSON.parse(stdout) as OraclePayload,
    };
  } catch (error) {
    return {
      available: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

function getFixture(
  payload: OraclePayload,
  name: string,
): OracleFixture {
  const fixture = payload.fixtures.find((candidate) => candidate.case === name);
  if (!fixture) {
    throw new Error(`Missing oracle fixture "${name}".`);
  }
  return fixture;
}

function toParamDefinition(definition: Record<string, any>) {
  switch (definition.type) {
    case 'enum':
      return param.enum(definition.values);
    case 'int':
      return param.int({
        min: definition.min,
        max: definition.max,
        scale: definition.scale,
        step: definition.step,
      });
    case 'float':
      return param.float({
        min: definition.min,
        max: definition.max,
        scale: definition.scale,
        step: definition.step,
      });
    default:
      throw new Error(`Unsupported oracle parameter type: ${definition.type}`);
  }
}

const oracle = loadOraclePayload();
const maybeDescribe = oracle.available ? describe : describe.skip;

maybeDescribe('cross-SDK oracle parity', () => {
  const payload = oracle.available ? oracle.payload : { fixtures: [] };

  it('matches the Python-owned grid_3x3 oracle exactly', async () => {
    const fixture = getFixture(payload, 'grid_3x3');
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(([name, definition]) => [
          name,
          toParamDefinition(definition),
        ]),
      ),
      objectives: fixture.spec.objectives,
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          Number(trialConfig.config.alpha) + Number(trialConfig.config.beta) / 100,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 9,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.bestConfig).toEqual(fixture.best_config);
    expect(result.bestMetrics).toEqual(fixture.best_metrics);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it('matches the Python-owned random_seed_42 oracle exactly', async () => {
    const fixture = getFixture(payload, 'random_seed_42');
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(([name, definition]) => [
          name,
          toParamDefinition(definition),
        ]),
      ),
      objectives: fixture.spec.objectives,
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async () => ({
      metrics: {
        accuracy: 1,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: 'random',
      maxTrials: fixture.trial_count ?? 20,
      randomSeed: 42,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it('matches the Python-owned budget_cutoff oracle', async () => {
    const fixture = getFixture(payload, 'budget_cutoff');
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(([name, definition]) => [
          name,
          toParamDefinition(definition),
        ]),
      ),
      objectives: fixture.spec.objectives,
      budget: fixture.spec.budget,
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      const costTable: Record<string, number> = {
        a: 0.1,
        b: 0.15,
        c: 0.2,
        d: 0.25,
      };
      return {
        metrics: {
          cost: costTable[String(trialConfig.config.model)],
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'grid',
      maxTrials: 4,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.bestConfig).toEqual(fixture.best_config);
    expect(result.bestMetrics).toEqual(fixture.best_metrics);
    expect(result.totalCostUsd).toBeCloseTo(fixture.total_cost_usd ?? 0, 10);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it('stays within the Python bayesian reference envelope when optional deps are present', async () => {
    const fixture = getFixture(payload, 'bayesian_branin');
    if (fixture.supported === false) {
      return;
    }

    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(([name, definition]) => [
          name,
          toParamDefinition(definition),
        ]),
      ),
      objectives: fixture.spec.objectives,
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => {
      const x = Number(trialConfig.config.x);
      const y = Number(trialConfig.config.y);
      const a = 1;
      const b = 5.1 / (4 * Math.PI ** 2);
      const c = 5 / Math.PI;
      const r = 6;
      const s = 10;
      const t = 1 / (8 * Math.PI);
      const score =
        -(a * (y - b * x ** 2 + c * x - r) ** 2 + s * (1 - t) * Math.cos(x) + s);
      return {
        metrics: {
          score,
        },
      };
    });

    const result = await wrapped.optimize({
      algorithm: 'bayesian',
      maxTrials: fixture.trial_count ?? 24,
      randomSeed: 7,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(Number(result.bestMetrics?.score)).toBeGreaterThanOrEqual(
      Number(fixture.best_metrics?.score) * 0.95,
    );
  });
});

if (!oracle.available) {
  describe('cross-SDK oracle parity (skipped)', () => {
    it('records why the Python oracle suite is unavailable', () => {
      expect(oracle.reason).toBeTruthy();
    });
  });
}
