import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { optimize, param, parseTvlSpec } from "../../src/index.js";
import { objectiveScoreValue } from "../../src/optimization/hybrid.js";

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
}

interface OraclePayload {
  fixtures: OracleFixture[];
}

const pythonRepoRoot = resolve(process.cwd(), "../Traigent");
const backendRepoRoot = resolve(process.cwd(), "../TraigentBackend");
const backendPython = resolve(backendRepoRoot, ".venv/bin/python");
const oracleScript = resolve(
  pythonRepoRoot,
  "tests/cross_sdk_oracles/generate_native_js_oracles.py",
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
    const stdout = execFileSync("python3", [oracleScript], {
      cwd: pythonRepoRoot,
      env: {
        ...process.env,
        PYTHONPATH: pythonRepoRoot,
      },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
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

function getFixture(payload: OraclePayload, name: string): OracleFixture {
  const fixture = payload.fixtures.find((candidate) => candidate.case === name);
  if (!fixture) {
    throw new Error(`Missing oracle fixture "${name}".`);
  }
  return fixture;
}

function toParamDefinition(definition: Record<string, any>) {
  switch (definition.type) {
    case "enum":
      return param.enum(definition.values, {
        conditions: definition.conditions,
        default: definition.default,
      });
    case "int":
      return param.int({
        min: definition.min,
        max: definition.max,
        scale: definition.scale,
        step: definition.step,
        conditions: definition.conditions,
        default: definition.default,
      });
    case "float":
      return param.float({
        min: definition.min,
        max: definition.max,
        scale: definition.scale,
        step: definition.step,
        conditions: definition.conditions,
        default: definition.default,
      });
    default:
      throw new Error(`Unsupported oracle parameter type: ${definition.type}`);
  }
}

describe("cross-SDK backend-guided parity", () => {
  it("matches backend Python banded objective scoring for representative values", () => {
    if (!existsSync(backendPython)) {
      return;
    }

    const cases = [
      { value: 150, band: { low: 120, high: 180 } },
      { value: 120, band: { low: 120, high: 180 } },
      { value: 180, band: { low: 120, high: 180 } },
      { value: 100, band: { low: 120, high: 180 } },
      { value: 220, band: { low: 120, high: 180 } },
      { value: 97.5, band: { low: 95, high: 105 } },
      { value: 120.5, band: { low: 95, high: 105 } },
    ];

    const pythonStdout = execFileSync(
      backendPython,
      [
        "-c",
        [
          "import json, sys",
          "sys.path.insert(0, sys.argv[1])",
          "from src.services.traigent.interactive_session_service import _objective_score_value",
          "cases = json.loads(sys.stdin.read())",
          "scores = [_objective_score_value({'kind': 'banded', 'metric': 'response_length', 'band': case['band']}, case['value']) for case in cases]",
          "print(json.dumps(scores))",
        ].join("; "),
        backendRepoRoot,
      ],
      {
        cwd: backendRepoRoot,
        env: {
          ...process.env,
          PYTHONPATH: backendRepoRoot,
        },
        input: JSON.stringify(cases),
        encoding: "utf8",
        stdio: ["pipe", "pipe", "pipe"],
      },
    );

    const pythonScores = JSON.parse(
      pythonStdout.trim().split(/\r?\n/).at(-1) ?? "[]",
    ) as number[];

    const jsScores = cases.map((entry) =>
      objectiveScoreValue(entry.value, {
        kind: "banded",
        metric: "response_length",
        band: entry.band,
        bandTest: "TOST",
        bandAlpha: 0.05,
        weight: 1,
      }),
    );

    expect(jsScores).toEqual(pythonScores);
  });
});

const oracle = loadOraclePayload();
const maybeDescribe = oracle.available ? describe : describe.skip;

maybeDescribe("cross-SDK oracle parity", () => {
  const payload = oracle.available ? oracle.payload : { fixtures: [] };

  it("matches the Python TVL loader for banded objectives, tuple domains, defaults, and promotion policy", async () => {
    const source = `
tvars:
  - name: retrieval_pair
    type: tuple[str,str]
    domain:
      values:
        - ["dense", "rerank"]
        - ["bm25", "none"]
    default: ["dense", "rerank"]
  - name: scorer
    type: callable[Ranker]
    domain:
      values: ["rank.fast", "rank.safe"]
objectives:
  - name: response_length
    band:
      low: 120
      high: 180
    weight: 2
promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  adjust: BH
  min_effect:
    response_length: 0
`;

    const dir = await mkdtemp(join(tmpdir(), "traigent-tvl-parity-"));
    const specPath = join(dir, "demo.tvl.yaml");
    await writeFile(specPath, source, "utf8");

    const pythonSummary = JSON.parse(
      execFileSync(
        "python3",
        [
          "-c",
          [
            "import json, sys",
            "from traigent.tvl.spec_loader import load_tvl_spec",
            "artifact = load_tvl_spec(sys.argv[1])",
            "objectives = []",
            "for obj in (artifact.objective_schema.objectives if artifact.objective_schema else []):",
            "    entry = {'metric': obj.name, 'weight': obj.weight}",
            "    if getattr(obj, 'band', None) is not None:",
            "        entry.update({'kind': 'banded', 'band': {'low': obj.band.low, 'high': obj.band.high}, 'bandAlpha': obj.band_alpha})",
            "    else:",
            "        entry.update({'kind': 'standard', 'direction': obj.orientation})",
            "    objectives.append(entry)",
            "summary = {",
            "  'configurationSpace': artifact.configuration_space,",
            "  'defaultConfig': artifact.default_config,",
            "  'promotionPolicy': {",
            "    'dominance': artifact.promotion_policy.dominance,",
            "    'alpha': artifact.promotion_policy.alpha,",
            "    'adjust': artifact.promotion_policy.adjust,",
            "    'minEffect': artifact.promotion_policy.min_effect,",
            "  } if artifact.promotion_policy else None,",
            "  'objectives': objectives,",
            "}",
            "print(json.dumps(summary))",
          ].join("; "),
          specPath,
        ],
        {
          cwd: pythonRepoRoot,
          env: {
            ...process.env,
            PYTHONPATH: pythonRepoRoot,
          },
          encoding: "utf8",
          stdio: ["ignore", "pipe", "pipe"],
        },
      ),
    ) as Record<string, unknown>;

    const jsSummary = parseTvlSpec(source).spec;
    expect(jsSummary.configurationSpace).toEqual(pythonSummary.configurationSpace);
    expect(jsSummary.defaultConfig).toEqual(pythonSummary.defaultConfig);
    expect(jsSummary.promotionPolicy).toEqual(pythonSummary.promotionPolicy);
    expect(jsSummary.objectives).toEqual(pythonSummary.objectives);
  });

  it("matches the Python-owned grid_3x3 oracle exactly", async () => {
    const fixture = getFixture(payload, "grid_3x3");
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(
          ([name, definition]) => [name, toParamDefinition(definition)],
        ),
      ),
      objectives: fixture.spec.objectives,
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          Number(trialConfig.config.alpha) +
          Number(trialConfig.config.beta) / 100,
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 9,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.bestConfig).toEqual(fixture.best_config);
    expect(result.bestMetrics).toEqual(fixture.best_metrics);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it("matches the Python-owned random_seed_42 oracle exactly", async () => {
    const fixture = getFixture(payload, "random_seed_42");
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(
          ([name, definition]) => [name, toParamDefinition(definition)],
        ),
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
      mode: "native",
      algorithm: "random",
      maxTrials: fixture.trial_count ?? 20,
      randomSeed: 42,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it("matches the Python-owned budget_cutoff oracle", async () => {
    const fixture = getFixture(payload, "budget_cutoff");
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(
          ([name, definition]) => [name, toParamDefinition(definition)],
        ),
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
      mode: "native",
      algorithm: "grid",
      maxTrials: 4,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.bestConfig).toEqual(fixture.best_config);
    expect(result.bestMetrics).toEqual(fixture.best_metrics);
    expect(result.totalCostUsd).toBeCloseTo(fixture.total_cost_usd ?? 0, 10);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it("matches the Python-owned conditional_grid oracle exactly", async () => {
    const fixture = getFixture(payload, "conditional_grid");
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(
          ([name, definition]) => [name, toParamDefinition(definition)],
        ),
      ),
      objectives: fixture.spec.objectives,
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy:
          trialConfig.config.model === "gpt-4"
            ? Number(trialConfig.config.max_tokens) / 1000
            : Number(trialConfig.config.temperature),
      },
    }));

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: fixture.trial_count ?? 6,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.bestConfig).toEqual(fixture.best_config);
    expect(result.bestMetrics).toEqual(fixture.best_metrics);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it("matches the Python-owned conditional_random_seed_7 oracle exactly", async () => {
    const fixture = getFixture(payload, "conditional_random_seed_7");
    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(
          ([name, definition]) => [name, toParamDefinition(definition)],
        ),
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
      mode: "native",
      algorithm: "random",
      maxTrials: fixture.trial_count ?? 6,
      randomSeed: 7,
    });

    expect(result.stopReason).toBe(fixture.normalized_stop_reason);
    expect(result.trials.map((trial) => trial.config)).toEqual(fixture.configs);
  });

  it("stays within the Python bayesian reference envelope when optional support is available", async () => {
    const fixture = getFixture(payload, "bayesian_branin");
    if (fixture.supported === false) {
      return;
    }

    const wrapped = optimize({
      configurationSpace: Object.fromEntries(
        Object.entries(fixture.spec.configurationSpace).map(
          ([name, definition]) => [name, toParamDefinition(definition)],
        ),
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
      const score = -(
        a * (y - b * x ** 2 + c * x - r) ** 2 +
        s * (1 - t) * Math.cos(x) +
        s
      );
      return {
        metrics: {
          score,
        },
      };
    });

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "bayesian",
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
  describe("cross-SDK oracle parity (skipped)", () => {
    it("records why the Python oracle suite is unavailable", () => {
      expect(oracle.reason).toBeTruthy();
    });
  });
}
