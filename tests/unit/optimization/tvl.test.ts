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
});
