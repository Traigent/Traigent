import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { loadTvlSpec, parseTvlSpec } from "../../../src/optimization/tvl.js";

describe("TVL parsing", () => {
  it("parses bool, enum, tuple, callable, int, float, budgets, defaults, and promotion metadata", () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: use_cache
    type: bool
    domain: {}
  - name: model
    type: enum
    domain:
      values: ["cheap", "accurate"]
    default: "accurate"
  - name: retrieval_pair
    type: tuple[str,str]
    domain:
      values:
        - ["dense", "rerank"]
        - ["bm25", "none"]
  - name: scorer
    type: callable[Ranker]
    domain:
      values: ["rank.fast", "rank.safe"]
  - name: retries
    type: int
    domain:
      range: [1, 5]
      step: 2
      scale: log
  - name: temperature
    type: float
    domain:
      range: [0.1, 0.9]
      step: 0.2
objectives:
  - name: accuracy
    direction: maximize
  - name: response_length
    band:
      low: 120
      high: 180
    weight: 2
    alpha: 0.01
constraints:
  structural:
    - expr: model = "cheap"
      error_message: fallback
  derived:
    - expr: metrics.accuracy >= 0.8
      errorMessage: score floor
exploration:
  strategy:
    type: nsga2
  budgets:
    max_trials: 9
    max_spend_usd: 3
    max_wallclock_s: 12
promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
  adjust: BH
  min_effect:
    accuracy: 0.01
  chance_constraints:
    - name: latency_slo
      threshold: 0.1
      confidence: 0.95
`);

    expect(parsed.metadata.strategyType).toBe("nsga2");
    expect(parsed.spec).toMatchObject({
      configurationSpace: {
        use_cache: { type: "enum", values: [false, true] },
        model: { type: "enum", values: ["cheap", "accurate"] },
        retrieval_pair: {
          type: "enum",
          values: [
            ["dense", "rerank"],
            ["bm25", "none"],
          ],
        },
        scorer: { type: "enum", values: ["rank.fast", "rank.safe"] },
        retries: { type: "int", min: 1, max: 5, step: 2, scale: "log" },
        temperature: {
          type: "float",
          min: 0.1,
          max: 0.9,
          step: 0.2,
          scale: "linear",
        },
      },
      budget: {
        maxTrials: 9,
        maxCostUsd: 3,
        maxWallclockMs: 12_000,
      },
      constraints: {
        structural: [{ require: 'model = "cheap"', errorMessage: "fallback" }],
        derived: [
          {
            require: "metrics.accuracy >= 0.8",
            errorMessage: "score floor",
          },
        ],
      },
      defaultConfig: {
        model: "accurate",
      },
      promotionPolicy: {
        dominance: "epsilon_pareto",
        alpha: 0.05,
        adjust: "BH",
        minEffect: { accuracy: 0.01 },
        chanceConstraints: [
          {
            name: "latency_slo",
            threshold: 0.1,
            confidence: 0.95,
          },
        ],
      },
    });
    expect(parsed.spec.objectives).toEqual([
      { kind: "standard", metric: "accuracy", direction: "maximize", weight: 1 },
      {
        kind: "banded",
        metric: "response_length",
        band: { low: 120, high: 180 },
        bandTest: "TOST",
        bandAlpha: 0.01,
        weight: 2,
      },
    ]);
  });

  it("loads from path and rejects missing path/source input", async () => {
    const dir = await mkdtemp(join(tmpdir(), "traigent-tvl-"));
    const path = join(dir, "demo.tvl.yml");
    await writeFile(
      path,
      `
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
`,
      "utf8",
    );

    const loaded = await loadTvlSpec({ path });
    expect(loaded.metadata.path).toBe(path);

    await expect(loadTvlSpec({})).rejects.toThrow(
      /requires either path or source/i,
    );
  });

  it("rejects malformed TVL sections with explicit validation errors", () => {
    expect(() => parseTvlSpec("[]")).toThrow(/must parse to an object/i);
    expect(() =>
      parseTvlSpec(`
tvars: nope
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/tvars must be a non-empty array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: []
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/domain\.values must be a non-empty array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: retries
    type: int
    domain:
      range: [0.1, 5]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/must use integers for int variables/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: retrieval_pair
    type: tuple[str,str]
    domain:
      values: []
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/non-empty array for tuple variables/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: scorer
    type: callable[Ranker]
    domain:
      values: [""]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/non-empty string for callable variables/i);
  });

  it("rejects invalid objectives, constraints, and budgets", () => {
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: sideways
`),
    ).toThrow(/direction must be "maximize" or "minimize"/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - 1
`),
    ).toThrow(/constraints\.structural\[0\] must be an object/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  derived:
    - {}
`),
    ).toThrow(/constraints\.derived\[0\] must provide require or expr/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_trials: 0
`),
    ).toThrow(/max_trials must be a positive integer/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_spend_usd: -1
`),
    ).toThrow(/max_spend_usd must be a positive number/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_wallclock_s: 0
`),
    ).toThrow(/max_wallclock_s must be a positive number/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: response_length
    band:
      low: 10
      high: 5
`),
    ).toThrow(/band\.low must be less than/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  adjust: bogus
`),
    ).toThrow(/promotion_policy\.adjust/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: response_length
    band:
      center: 100
      tol: 0
`),
    ).toThrow(/band\.tol must be positive/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints: []
`),
    ).toThrow(/constraints must be an object/i);
  });

  it("loads from inline source and supports expr aliases on structural and derived constraints", async () => {
    const loaded = await loadTvlSpec({
      source: `
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap", "accurate"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - expr: params.model == "cheap"
  derived:
    - expr: metrics.accuracy >= 0.8
`,
    });

    expect(loaded.spec.constraints).toEqual({
      structural: [{ require: 'params.model == "cheap"' }],
      derived: [{ require: "metrics.accuracy >= 0.8" }],
    });
  });

  it("rejects invalid TVL promotion-policy and objective details", () => {
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: response_length
    band:
      low: 10
      high: 20
    alpha: 1.2
`),
    ).toThrow(/alpha must be in \(0, 1\)/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: retries
    type: int
    domain:
      range: [1, 5]
      step: 1.5
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/step must be an integer/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: score
    type: float
    domain:
      range: [1]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/range must contain exactly two numeric values/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  min_effect: []
`),
    ).toThrow(/min_effect must be an object/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  chance_constraints:
    - name: latency
      threshold: 2
      confidence: 0.95
`),
    ).toThrow(/threshold must be in \[0, 1\]/i);
  });

  it("supports band center/tolerance, strategy objects, source loading, and legacy promotion metadata", async () => {
    const loaded = await loadTvlSpec({
      source: `
tvars:
  - name: temperature
    type: float
    domain:
      range: [0.1, 0.9]
objectives:
  - name: latency
    band:
      center: 100
      tol: 20
    test: TOST
    alpha: 0.1
exploration:
  strategy:
    type: grid
promotion:
  gate: manual_review
`,
    });

    expect(loaded.metadata.strategyType).toBe("grid");
    expect(loaded.spec.objectives).toEqual([
      {
        kind: "banded",
        metric: "latency",
        band: { low: 80, high: 120 },
        bandTest: "TOST",
        bandAlpha: 0.1,
        weight: 1,
      },
    ]);
  });

  it("supports require aliases and boolean-expression normalization", () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap", "accurate"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - require: params.model = "accurate" or params.model == "cheap"
  derived:
    - require: not (metrics.accuracy < 0.8)
`);

    expect(parsed.spec.constraints).toEqual({
      structural: [
        { require: 'params.model = "accurate" or params.model == "cheap"' },
      ],
      derived: [{ require: "not (metrics.accuracy < 0.8)" }],
    });
  });

  it("handles a minimal TVL spec without optional sections", () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
`);

    expect(parsed.metadata).toEqual({
      strategyType: undefined,
    });
    expect(parsed.spec.constraints).toBeUndefined();
    expect(parsed.spec.defaultConfig).toBeUndefined();
    expect(parsed.spec.budget).toBeUndefined();
    expect(parsed.spec.execution).toBeUndefined();
  });

  it("parses float variables without explicit steps and rejects malformed min_effect values", () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: temperature
    type: float
    domain:
      range: [0.1, 0.9]
objectives:
  - name: accuracy
    direction: maximize
`);

    expect(parsed.spec.configurationSpace.temperature).toEqual({
      type: "float",
      min: 0.1,
      max: 0.9,
      scale: "linear",
    });

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  min_effect: []
`),
    ).toThrow(/min_effect must be an object/i);
  });

  it("parses spec.version numbers and promotion tie breakers", () => {
    const parsed = parseTvlSpec(`
spec:
  version: 1
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  tie_breakers:
    accuracy: maximize
`);

    expect(parsed.metadata).toEqual({
      strategyType: undefined,
    });
    expect(parsed.spec.promotionPolicy).toEqual({
      tieBreakers: {
        accuracy: "maximize",
      },
    });
  });

  it("rejects invalid promotion policy and exploration budget shapes", () => {
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  dominance: strict_pareto
`),
    ).toThrow(/dominance must be "epsilon_pareto"/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  chance_constraints: {}
`),
    ).toThrow(/chance_constraints must be an array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_trials: 0
`),
    ).toThrow(/max_trials must be a positive integer/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_wallclock_s: 0
`),
    ).toThrow(/max_wallclock_s must be a positive number/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy:
    type: annealing
`),
    ).toThrow(/not supported by the JS SDK/i);
  });

  it("rejects malformed tvars, objectives, and constraint entry shapes", () => {
    expect(() => parseTvlSpec(`tvars: []`)).toThrow(
      /TVL tvars must be a non-empty array/i,
    );

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: choice
    type: tuple[str,str]
    domain:
      values:
        - []
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/must be a non-empty tuple-like array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: scorer
    type: callable[Ranker]
    domain:
      values: [""]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/must be a non-empty string for callable variables/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: []
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/domain\.values must be a non-empty array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - {}
`),
    ).toThrow(/objectives\[0\]\.name must be a non-empty string/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: latency
    band: []
`),
    ).toThrow(/objectives\[0\]\.band must be an object/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: latency
    band:
      low: 1
      high: 2
    test: WELCH
`),
    ).toThrow(/objectives\[0\]\.test must be "TOST"/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural: {}
`),
    ).toThrow(/constraints\.structural must be an array/i);
  });

  it("preserves optional constraint ids/messages and rejects malformed entry metadata", () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap", "accurate"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - id: choose_model
      expr: params.model == "cheap"
      errorMessage: use the cheaper model
  derived:
    - id: floor
      expr: metrics.accuracy >= 0.8
      error_message: accuracy floor
`);

    expect(parsed.spec.constraints).toEqual({
      structural: [
        {
          id: "choose_model",
          require: 'params.model == "cheap"',
          errorMessage: "use the cheaper model",
        },
      ],
      derived: [
        {
          id: "floor",
          require: "metrics.accuracy >= 0.8",
          errorMessage: "accuracy floor",
        },
      ],
    });

    expect(() =>
      parseTvlSpec(`
tvars:
  - {}
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/tvars\[0\]\.name must be a non-empty string/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: ""
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/tvars\[0\]\.type must be a non-empty string/i);
  });

  it("rejects malformed tie breakers and unsupported strategy metadata", () => {
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  tie_breakers: []
`),
    ).toThrow(/promotion_policy\.tie_breakers must be an object/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  tie_breakers:
    accuracy: sideways
`),
    ).toThrow(/tie_breakers\.accuracy must be "maximize" or "minimize"/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy:
    type: ParetoFrontier
`),
    ).toThrow(/not supported by the JS SDK/i);
  });
});
