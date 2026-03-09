import { describe, expect, it } from "vitest";

import {
  getOptimizationSpec,
  loadTvlSpec,
  optimize,
  param,
  parseTvlSpec,
  toHybridConfigSpace,
} from "../../../src/index.js";
import { ValidationError } from "../../../src/core/errors.js";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

describe("optimization spec helpers", () => {
  it("attaches metadata without changing the wrapped function behavior", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini", "gpt-4o"]),
      },
      objectives: ["accuracy"],
    })(async (value: string) => value.toUpperCase());

    expect(await wrapped("ok")).toBe("OK");
    expect(getOptimizationSpec(wrapped)).toMatchObject({
      configurationSpace: {
        model: {
          type: "enum",
          values: ["gpt-4o-mini", "gpt-4o"],
        },
      },
      objectives: [{ metric: "accuracy", direction: "maximize", weight: 1 }],
    });
    expect(typeof wrapped.optimize).toBe("function");
    expect(typeof wrapped.applyBestConfig).toBe("function");
    expect(typeof wrapped.currentConfig).toBe("function");
  });

  it("serializes config space to the hybrid tunables shape", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini", "gpt-4o"]),
        temperature: param.float({ min: 0, max: 1, scale: "linear" }),
        max_retries: param.int({ min: 0, max: 3, scale: "linear" }),
      },
      objectives: ["accuracy", "cost"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(toHybridConfigSpace(wrapped)).toEqual({
      tunables: [
        {
          name: "model",
          type: "enum",
          domain: { values: ["gpt-4o-mini", "gpt-4o"] },
        },
        {
          name: "temperature",
          type: "float",
          domain: { range: [0, 1] },
          scale: "linear",
        },
        {
          name: "max_retries",
          type: "int",
          domain: { range: [0, 3] },
          scale: "linear",
        },
      ],
      constraints: {},
    });
  });

  it("supports boolean parameters through param.bool()", () => {
    const wrapped = optimize({
      configurationSpace: {
        useCache: param.bool(),
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(getOptimizationSpec(wrapped)?.configurationSpace.useCache).toEqual({
      type: "enum",
      values: [false, true],
    });
  });

  it("rejects unknown string objectives", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["gpt-4o-mini"]),
        },
        objectives: ["quality"],
      }),
    ).toThrow(ValidationError);
  });

  it("rejects malformed objective bands and invalid promotion policy settings", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: [
          {
            metric: "response_length",
            band: { low: 1, high: 2, center: 3, tol: 1 } as never,
          },
        ],
      }),
    ).toThrow(/either low\/high or center\/tol/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        promotionPolicy: {
          tieBreakers: {
            accuracy: "sideways" as never,
          },
        },
      }),
    ).toThrow(/tieBreakers\.accuracy must be "maximize" or "minimize"/i);
  });

  it("accepts explicit objective objects for custom metrics", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: [{ metric: "quality_score", direction: "maximize" }],
    })(async () => ({ metrics: { quality_score: 0.9 } }));

    expect(getOptimizationSpec(wrapped)?.objectives).toEqual([
      {
        kind: "standard",
        metric: "quality_score",
        direction: "maximize",
        weight: 1,
      },
    ]);
  });

  it("accepts banded objective objects for TVL-style target bands", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini"]),
      },
      objectives: [
        {
          metric: "response_length",
          band: { center: 150, tol: 30 },
          alpha: 0.01,
        },
      ],
    })(async () => ({ metrics: { response_length: 155 } }));

    expect(getOptimizationSpec(wrapped)?.objectives).toEqual([
      {
        kind: "banded",
        metric: "response_length",
        band: { low: 120, high: 180 },
        bandTest: "TOST",
        bandAlpha: 0.01,
        weight: 1,
      },
    ]);
  });

  it("stores wrapper-local applied config without creating runtime global state", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini", "gpt-4o"]),
      },
      objectives: ["accuracy"],
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === "gpt-4o" ? 1 : 0.5,
      },
    }));

    expect(wrapped.currentConfig()).toBeUndefined();

    const result = await wrapped.optimize({
      mode: "native",
      algorithm: "grid",
      maxTrials: 2,
    });

    const applied = wrapped.applyBestConfig(result);
    expect(applied).toEqual({ model: "gpt-4o" });
    expect(wrapped.currentConfig()).toEqual({ model: "gpt-4o" });

    const emptyApplied = wrapped.applyBestConfig({
      ...result,
      bestConfig: null,
    });
    expect(emptyApplied).toBeUndefined();
    expect(wrapped.currentConfig()).toBeUndefined();
  });

  it("initializes currentConfig from defaultConfig and autoLoadBest", async () => {
    const loadPath = join(tmpdir(), `traigent-best-${Date.now()}.json`);
    await writeFile(
      loadPath,
      JSON.stringify({ model: "gpt-4o", temperature: 0.2 }),
      "utf8",
    );

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-4o-mini", "gpt-4o"]),
      },
      objectives: ["accuracy"],
      defaultConfig: {
        temperature: 0.7,
      },
      autoLoadBest: true,
      loadFrom: loadPath,
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(wrapped.currentConfig()).toEqual({
      temperature: 0.2,
      model: "gpt-4o",
    });
  });

  it("applies execution defaults from the spec when optimize() options omit mode", async () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
      },
      objectives: ["accuracy"],
      execution: {
        mode: "native",
      },
      evaluation: {
        data: [{ id: 1 }],
      },
    })(async (trialConfig) => ({
      metrics: {
        accuracy: trialConfig.config.model === "accurate" ? 1 : 0.5,
      },
    }));

    const result = await wrapped.optimize({
      algorithm: "grid",
      maxTrials: 2,
    });

    expect(result.mode).toBe("native");
    expect(result.bestConfig).toEqual({ model: "accurate" });
  });

  it("rejects log-scaled parameters with non-positive ranges or non-multiplicative steps", () => {
    expect(() => param.float({ min: 0, max: 1, scale: "log" })).toThrow(
      /require min\/max > 0/i,
    );

    expect(() =>
      param.float({ min: 0.001, max: 1, scale: "log", step: 1 }),
    ).toThrow(/require step to be greater than 1/i);
  });

  it("supports conditional parameters with equality conditions and default fallback", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-3.5", "gpt-4"]),
        max_tokens: param.int({
          min: 256,
          max: 1024,
          step: 256,
          conditions: { model: "gpt-4" },
          default: 512,
        }),
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(getOptimizationSpec(wrapped)?.configurationSpace.max_tokens).toEqual(
      {
        type: "int",
        min: 256,
        max: 1024,
        step: 256,
        scale: "linear",
        conditions: { model: "gpt-4" },
        default: 512,
      },
    );
  });

  it("rejects invalid ranges, condition values, and conditional defaults", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          temperature: param.float({
            min: 1,
            max: 0,
            step: 0.1,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/require max >= min/i);

    expect(() =>
      optimize({
        configurationSpace: {
          retries: param.int({
            min: 0,
            max: 3,
            step: 1.5 as never,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/require step to be an integer/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["cheap", "accurate"]),
          invalid: param.bool({
            conditions: { model: { nested: true } as never },
            default: false,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/only support string, number, or boolean equality values/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["cheap", "accurate"]),
          maxTokens: param.int({
            min: 64,
            max: 128,
            step: 64,
            conditions: { model: "accurate" },
            default: 32,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/default must fall within min\/max/i);
  });

  it("rejects invalid conditional dependency graphs", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          first: param.bool({
            conditions: { missing: true },
            default: false,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/unknown dependency "missing"/i);

    expect(() =>
      optimize({
        configurationSpace: {
          first: param.bool({
            conditions: { second: true },
            default: false,
          }),
          second: param.bool({
            conditions: { first: true },
            default: false,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/dependency cycles/i);
  });

  it("rejects malformed constraints, defaultConfig, and autoLoadBest settings", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        constraints: [] as never,
      }),
    ).toThrow(/constraints must be an object/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        defaultConfig: [] as never,
      }),
    ).toThrow(/defaultConfig must be an object/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        autoLoadBest: true,
      })(async () => ({ metrics: { accuracy: 1 } })),
    ).toThrow(/autoLoadBest requires loadFrom/i);
  });

  it("normalizes authored spec objects passed directly to getOptimizationSpec()", () => {
    const spec = {
      configurationSpace: {
        mode: param.enum(["safe", "fast"]),
        useCache: param.bool({
          conditions: { mode: "safe" },
          default: false,
        }),
      },
      objectives: [{ metric: "quality", direction: "maximize" as const }],
    };

    expect(getOptimizationSpec(spec)).toMatchObject({
      configurationSpace: {
        useCache: {
          values: [false, true],
          conditions: { mode: "safe" },
          default: false,
        },
      },
    });
  });

  it("handles missing and malformed auto-loaded configs", async () => {
    const missingPath = join(tmpdir(), `traigent-missing-${Date.now()}.json`);
    const missingWrapped = optimize({
      configurationSpace: {
        model: param.enum(["a"]),
      },
      objectives: ["accuracy"],
      defaultConfig: { temperature: 0.5 },
      autoLoadBest: true,
      loadFrom: missingPath,
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(missingWrapped.currentConfig()).toEqual({ temperature: 0.5 });

    const dir = await mkdtemp(join(tmpdir(), "traigent-autoload-"));
    const invalidJsonPath = join(dir, "invalid.json");
    await writeFile(invalidJsonPath, "{not-json", "utf8");

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        autoLoadBest: true,
        loadFrom: invalidJsonPath,
      })(async () => ({ metrics: { accuracy: 1 } })),
    ).toThrow(/Failed to load best config/i);

    const invalidObjectPath = join(dir, "invalid-object.json");
    await writeFile(invalidObjectPath, JSON.stringify(["bad"]), "utf8");

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        autoLoadBest: true,
        loadFrom: invalidObjectPath,
      })(async () => ({ metrics: { accuracy: 1 } })),
    ).toThrow(/must be a JSON object/i);
  });

  it("returns undefined for non-spec inputs and rejects toHybridConfigSpace() without a spec", () => {
    expect(getOptimizationSpec(null)).toBeUndefined();
    expect(getOptimizationSpec({ nope: true })).toBeUndefined();
    expect(() => toHybridConfigSpace({ nope: true })).toThrow(
      /requires a wrapped function or optimization spec/i,
    );
  });

  it("treats structurally equal conditional enum defaults as equal even when object key order differs", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["tuple"]),
        retrieval: {
          type: "enum",
          values: [{ provider: "dense", stage: "rerank" }],
          conditions: { model: "tuple" },
          default: { stage: "rerank", provider: "dense" },
        } as any,
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(getOptimizationSpec(wrapped)?.configurationSpace.retrieval).toEqual({
      type: "enum",
      values: [{ provider: "dense", stage: "rerank" }],
      conditions: { model: "tuple" },
      default: { stage: "rerank", provider: "dense" },
    });
  });

  it("rejects circular categorical values when validating conditional enum defaults", () => {
    const circular: unknown[] = [];
    circular.push(circular);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["tuple"]),
          retrieval: {
            type: "enum",
            values: [circular],
            conditions: { model: "tuple" },
            default: circular,
          } as any,
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/Circular optimization values are not supported/i);
  });

  it("normalizes budgets and structural / derived constraints", () => {
    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["cheap", "accurate"]),
      },
      objectives: ["accuracy"],
      budget: {
        maxCostUsd: 1.5,
        maxTrials: 12,
        maxWallclockMs: 30_000,
      },
      constraints: {
        structural: [
          {
            when: 'params.model == "accurate"',
            then: "True",
            errorMessage: "accurate must remain valid",
          },
        ],
        derived: [
          {
            require: "metrics.accuracy >= 0.8",
            errorMessage: "accuracy floor",
          },
        ],
      },
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(getOptimizationSpec(wrapped)).toMatchObject({
      budget: {
        maxCostUsd: 1.5,
        maxTrials: 12,
        maxWallclockMs: 30_000,
      },
      constraints: {
        structural: [
          {
            when: 'params.model == "accurate"',
            then: "True",
            errorMessage: "accurate must remain valid",
          },
        ],
        derived: [
          {
            require: "metrics.accuracy >= 0.8",
            errorMessage: "accuracy floor",
          },
        ],
      },
    });
  });

  it("parses a supported TVL subset into an optimization spec", async () => {
    const source = `
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap", "accurate"]
  - name: use_cache
    type: bool
    domain: {}
  - name: temperature
    type: float
    domain:
      range: [0.1, 0.9]
      step: 0.2
objectives:
  - name: accuracy
    direction: maximize
  - name: latency
    direction: minimize
    weight: 2
constraints:
  structural:
    - when: model = "accurate"
      then: temperature <= 0.5
  derived:
    - require: metrics.accuracy >= 0.8
exploration:
  strategy:
    type: tpe
  budgets:
    max_trials: 12
    max_spend_usd: 5
    max_wallclock_s: 60
`;

    const parsed = parseTvlSpec(source);
    expect(parsed.metadata.strategyType).toBe("tpe");
    expect(parsed.spec).toMatchObject({
      configurationSpace: {
        model: {
          type: "enum",
          values: ["cheap", "accurate"],
        },
        use_cache: {
          type: "enum",
          values: [false, true],
        },
        temperature: {
          type: "float",
          min: 0.1,
          max: 0.9,
          step: 0.2,
        },
      },
      budget: {
        maxTrials: 12,
        maxCostUsd: 5,
        maxWallclockMs: 60_000,
      },
      constraints: {
        structural: [
          {
            when: 'model = "accurate"',
            then: "temperature <= 0.5",
          },
        ],
        derived: [
          {
            require: "metrics.accuracy >= 0.8",
          },
        ],
      },
    });

    const path = join(tmpdir(), `traigent-spec-${Date.now()}.tvl.yaml`);
    await writeFile(path, source, "utf8");
    const loaded = await loadTvlSpec(path);
    expect(loaded.metadata.path).toBe(path);
    expect(loaded.spec.budget?.maxTrials).toBe(12);
  });

  it("rejects conditional parameters without a valid default fallback", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["gpt-3.5", "gpt-4"]),
          max_tokens: param.int({
            min: 256,
            max: 1024,
            step: 256,
            conditions: { model: "gpt-4" },
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/requires a default fallback value/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["gpt-3.5", "gpt-4"]),
          max_tokens: param.int({
            min: 256,
            max: 1024,
            step: 256,
            conditions: { model: "gpt-4" },
            default: 2048,
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/default must fall within min\/max/i);
  });

  it("rejects conditional dependency cycles and hybrid serialization for native-only conditionals", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          alpha: param.enum(["x", "y"], {
            conditions: { beta: "x" },
            default: "x",
          }),
          beta: param.enum(["x", "y"], {
            conditions: { alpha: "x" },
            default: "x",
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/cannot form dependency cycles/i);

    const wrapped = optimize({
      configurationSpace: {
        model: param.enum(["gpt-3.5", "gpt-4"]),
        max_tokens: param.int({
          min: 256,
          max: 1024,
          step: 256,
          conditions: { model: "gpt-4" },
          default: 512,
        }),
      },
      objectives: ["accuracy"],
    })(async () => ({ metrics: { accuracy: 1 } }));

    expect(() => toHybridConfigSpace(wrapped)).toThrow(
      /does not support conditional parameter "max_tokens" yet/i,
    );
  });

  it("normalizes raw optimization specs passed to getOptimizationSpec()", () => {
    const normalized = getOptimizationSpec({
      configurationSpace: {
        model: param.enum(["a", "b"]),
      },
      objectives: [{ metric: "quality", direction: "maximize", weight: 1.5 }],
    });

    expect(normalized).toEqual({
      configurationSpace: {
        model: {
          type: "enum",
          values: ["a", "b"],
          conditions: undefined,
          default: undefined,
        },
      },
      objectives: [
        {
          kind: "standard",
          metric: "quality",
          direction: "maximize",
          weight: 1.5,
        },
      ],
      budget: undefined,
      constraints: undefined,
      defaultConfig: undefined,
      promotionPolicy: undefined,
      execution: undefined,
      autoLoadBest: undefined,
      loadFrom: undefined,
      evaluation: undefined,
    });
  });

  it("rejects invalid optimization spec shapes", () => {
    expect(() =>
      optimize({
        configurationSpace: {} as never,
        objectives: ["accuracy"],
      }),
    ).toThrow(/at least one configuration parameter/i);

    expect(() =>
      optimize({
        configurationSpace: {
          "bad-name": param.enum(["a"]),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/must be a valid identifier-like key/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: [],
      }),
    ).toThrow(/requires at least one objective/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        evaluation: {
          data: [{ id: 1 }],
          loadData: async () => [{ id: 1 }],
        },
      }),
    ).toThrow(/either evaluation\.data or evaluation\.loadData/i);
  });

  it("rejects invalid parameter conditions and defaults", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a", "b"]),
          variant: param.enum(["x", "y"], {
            conditions: {} as never,
            default: "x",
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/conditions must be a non-empty object/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a", "b"]),
          variant: param.enum(["x", "y"], {
            conditions: { model: ["a"] as never },
            default: "x",
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/only support string, number, or boolean equality values/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a", "b"], {
            default: "a",
          }),
        },
        objectives: ["accuracy"],
      }),
    ).toThrow(/default requires conditions to be defined/i);
  });

  it("rejects invalid budgets and malformed constraints", () => {
    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        budget: {
          maxCostUsd: 0,
        },
      }),
    ).toThrow(/budget\.maxCostUsd must be a positive number/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        budget: {
          maxTrials: 0,
        },
      }),
    ).toThrow(/budget\.maxTrials must be a positive integer/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        budget: {
          maxWallclockMs: 0,
        },
      }),
    ).toThrow(/budget\.maxWallclockMs must be a positive integer/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        constraints: {} as never,
      }),
    ).toThrow(/must include at least one structural or derived constraint/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        constraints: {
          structural: [
            {
              require: "True",
              when: "True",
              then: "True",
            },
          ],
        },
      }),
    ).toThrow(/cannot mix require with when\/then/i);

    expect(() =>
      optimize({
        configurationSpace: {
          model: param.enum(["a"]),
        },
        objectives: ["accuracy"],
        constraints: {
          derived: [{} as never],
        },
      }),
    ).toThrow(/constraints\.derived\[0\]\.require must be a non-empty string/i);
  });
});
