import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  getNativeTvlCompatibilityReport,
  loadTvlSpec,
  parseTvlSpec,
} from '../../../src/optimization/tvl.js';

describe('TVL loading', () => {
  it('parses the supported TVL subset into native spec + optimize options', () => {
    const parsed = parseTvlSpec(`
spec:
  id: rag-demo
  version: 0.9.0
metadata:
  owner: team@example.com
tvars:
  - name: use_cache
    type: bool
    domain: {}
  - name: model
    type: enum[str]
    domain: ["cheap", "accurate"]
    default: accurate
  - name: route
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
      target: [120, 180]
      test: TOST
      alpha: 0.01
constraints:
  structural:
    - expr: params.temperature <= 0.8
      id: temp_cap
      error_message: keep it stable
    - when: params.model == "accurate"
      then: params.retries <= 3
      id: premium_guard
  derived:
    - expr: metrics.accuracy >= 0.8
      id: score_floor
exploration:
  strategy: bayesian
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

    expect(parsed.moduleId).toBe('rag-demo');
    expect(parsed.tvlVersion).toBe('0.9.0');
    expect(parsed.metadata.strategyType).toBe('bayesian');
    expect(parsed.optimizeOptions).toEqual({
      algorithm: 'bayesian',
      maxTrials: 9,
    });
    expect(parsed.nativeCompatibility).toEqual(getNativeTvlCompatibilityReport());
    expect(parsed.nativeCompatibility.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          feature: 'promotion-policy',
          status: 'supported-with-reduced-semantics',
        }),
        expect.objectContaining({
          feature: 'hybrid-session-features',
          status: 'hybrid-only',
        }),
      ]),
    );
    expect(parsed.spec).toMatchObject({
      configurationSpace: {
        use_cache: { type: 'enum', values: [false, true] },
        model: { type: 'enum', values: ['cheap', 'accurate'] },
        route: {
          type: 'enum',
          values: [
            ['dense', 'rerank'],
            ['bm25', 'none'],
          ],
        },
        scorer: { type: 'enum', values: ['rank.fast', 'rank.safe'] },
        retries: { type: 'int', min: 1, max: 5, step: 2, scale: 'log' },
        temperature: {
          type: 'float',
          min: 0.1,
          max: 0.9,
          step: 0.2,
          scale: 'linear',
        },
      },
      defaultConfig: {
        model: 'accurate',
      },
      budget: {
        maxCostUsd: 3,
      },
      execution: {
        maxWallclockMs: 12_000,
      },
      promotionPolicy: {
        dominance: 'epsilon_pareto',
        alpha: 0.05,
        adjust: 'BH',
        minEffect: {
          accuracy: 0.01,
        },
        chanceConstraints: [
          {
            name: 'latency_slo',
            threshold: 0.1,
            confidence: 0.95,
          },
        ],
      },
    });
    expect(parsed.spec.objectives).toEqual([
      { metric: 'accuracy', direction: 'maximize' },
      {
        metric: 'response_length',
        direction: 'band',
        band: { low: 120, high: 180 },
      },
    ]);
    expect(parsed.spec.constraints).toHaveLength(3);
    expect(parsed.promotionPolicy).toEqual({
      dominance: 'epsilon_pareto',
      alpha: 0.05,
      adjust: 'BH',
      minEffect: {
        accuracy: 0.01,
      },
      chanceConstraints: [
        {
          name: 'latency_slo',
          threshold: 0.1,
          confidence: 0.95,
        },
      ],
    });
  });

  it('loads from a file path', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'traigent-tvl-'));
    const path = join(dir, 'demo.tvl.yml');
    await writeFile(
      path,
      `
spec:
  id: path-loaded
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
`,
      'utf8',
    );

    const loaded = await loadTvlSpec({ path });
    expect(loaded.moduleId).toBe('path-loaded');
    expect(loaded.metadata.path).toBe(path);
  });

  it('rejects loadTvlSpec calls without path or source', async () => {
    await expect(loadTvlSpec({})).rejects.toThrow(/requires either path or source/i);
  });

  it('compiles structural and derived constraints into runtime predicates', () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap", "accurate"]
  - name: retries
    type: int
    domain:
      range: [1, 5]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - when: params.model == "accurate"
      then: params.retries <= 3
  derived:
    - expr: metrics.accuracy >= 0.8
`);

    const [structural, derived] = parsed.spec.constraints ?? [];
    expect(structural?.({ model: 'accurate', retries: 2 })).toBe(true);
    expect(structural?.({ model: 'accurate', retries: 5 })).toBe(false);
    expect(structural?.({ model: 'cheap', retries: 5 })).toBe(true);
    expect(derived?.requiresMetrics).toBe(true);
    expect(derived?.({ model: 'cheap' }, { accuracy: 0.85 })).toBe(true);
    expect(derived?.({ model: 'cheap' }, { accuracy: 0.5 })).toBe(false);
  });

  it('supports band center/tolerance, strategy objects, source loading, and legacy promotion metadata', async () => {
    const loaded = await loadTvlSpec({
      source: `
spec:
  id: source-loaded
tvl_version: 0.9
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

    expect(loaded.moduleId).toBe('source-loaded');
    expect(loaded.tvlVersion).toBe('0.9');
    expect(loaded.optimizeOptions).toEqual({ algorithm: 'grid' });
    expect(loaded.spec.objectives).toEqual([
      {
        metric: 'latency',
        direction: 'band',
        band: { low: 80, high: 120 },
      },
    ]);
    expect(loaded.metadata.strategyType).toBe('grid');
    expect(loaded.metadata.legacyPromotion).toEqual({
      gate: 'manual_review',
    });
  });

  it('supports require aliases and boolean-expression normalization', () => {
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

    const [structural, derived] = parsed.spec.constraints ?? [];
    expect(structural?.({ model: 'accurate' })).toBe(true);
    expect(structural?.({ model: 'cheap' })).toBe(true);
    expect(structural?.({ model: 'other' })).toBe(false);
    expect(derived?.({ model: 'accurate' }, { accuracy: 0.9 })).toBe(true);
    expect(derived?.({ model: 'accurate' }, { accuracy: 0.5 })).toBe(false);
  });

  it('handles a minimal TVL spec without optional sections', () => {
    const parsed = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
`);

    expect(parsed.optimizeOptions).toBeUndefined();
    expect(parsed.promotionPolicy).toBeUndefined();
    expect(parsed.moduleId).toBeUndefined();
    expect(parsed.tvlVersion).toBeUndefined();
    expect(parsed.metadata).toEqual({
      strategyType: undefined,
    });
    expect(parsed.spec.constraints).toBeUndefined();
    expect(parsed.spec.defaultConfig).toBeUndefined();
    expect(parsed.spec.budget).toBeUndefined();
    expect(parsed.spec.execution).toBeUndefined();
  });

  it('parses float variables without explicit steps and rejects malformed min_effect values', () => {
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
      type: 'float',
      min: 0.1,
      max: 0.9,
      scale: 'linear',
    });

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
  min_effect: []
`),
    ).toThrow(/min_effect must be an object/i);

    const metadataOnly = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy: {}
`);
    expect(metadataOnly.metadata.strategyType).toBeUndefined();
  });

  it('parses spec.version numbers and promotion tie breakers', () => {
    const parsed = parseTvlSpec(`
spec:
  version: 1
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  tie_breakers:
    accuracy: maximize
`);

    expect(parsed.tvlVersion).toBe('1');
    expect(parsed.promotionPolicy).toEqual({
      tieBreakers: {
        accuracy: 'maximize',
      },
    });
  });

  it('rejects invalid promotion policy and exploration budget shapes', () => {
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
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy:
  chance_constraints:
    - 1
`),
    ).toThrow(/chance_constraints\[0\] must be an object/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
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
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_wallclock_s: -1
`),
    ).toThrow(/max_wallclock_s must be a positive number/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  budgets:
    max_spend_usd: 0
`),
    ).toThrow(/max_spend_usd must be a positive number/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy:
    type: annealing
`),
    ).toThrow(/not supported by the native JS SDK/i);

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
  adjust: bonferroni
`),
    ).toThrow(/adjust must be "none" or "BH"/i);

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
  min_effect:
    accuracy: -0.1
`),
    ).toThrow(/min_effect\.accuracy must be non-negative/i);

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
  chance_constraints:
    - name: latency
      threshold: 0.1
      confidence: 0
`),
    ).toThrow(/confidence must be in \(0, 1\]/i);

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
  alpha: 1.2
`),
    ).toThrow(/promotion_policy\.alpha must be in \(0, 1\)/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
promotion_policy: nope
`),
    ).toThrow(/promotion_policy must be an object/i);

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
  tie_breakers:
    accuracy: lower
`),
    ).toThrow(
      /promotion_policy\.tie_breakers\.accuracy must be "maximize" or "minimize"/i,
    );

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
  tie_breakers: []
`),
    ).toThrow(/tie_breakers must be an object/i);
  });

  it('rejects invalid TVL shapes for tuple, callable, and band alpha', () => {
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: route
    type: tuple[str,str]
    domain:
      values:
        - []
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/non-empty array/i);

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
    ).toThrow(/non-empty string/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: latency
    type: float
    domain:
      range: [10, 20]
objectives:
  - name: latency
    band:
      low: 12
      high: 18
      alpha: 1
`),
    ).toThrow(/band\.alpha must be in \(0, 1\)/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: retries
    type: int
    domain:
      range: [1.5, 5]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/must use integers for int variables/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - {}
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/tvars\[0\]\.name must be a non-empty string/i);
  });

  it('rejects additional malformed TVL edge shapes loudly', () => {
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: prompt
    type: "   "
    domain: {}
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/tvars\[0\]\.type must be a non-empty string/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: route
    type: tuple[str,str]
    domain:
      values:
        - rerank
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/must be a non-empty array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: scorer
    type: callable[Ranker]
    domain: {}
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/must provide a non-empty values array for callable variables/i);

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
    ).toThrow(/must provide a non-empty values array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: temperature
    type: float
    domain:
      range: [0.1]
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/range must contain exactly two numeric values/i);

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
    ).toThrow(/step must be an integer for int variables/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: stability
    type: float
    domain:
      range: [0.1, 1]
objectives:
  - name: stability
    band:
      target:
        center: 0.8
        tol: 0
`),
    ).toThrow(/target\.tol must be positive/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: stability
    type: float
    domain:
      range: [0.1, 1]
objectives:
  - name: stability
    band:
      low: 0.8
      high: 0.8
`),
    ).toThrow(/requires low < high/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: stability
    type: float
    domain:
      range: [0.1, 1]
objectives:
  - name: stability
    band: {}
`),
    ).toThrow(/must provide target, low\/high, or center\/tol/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
constraints: []
`),
    ).toThrow(/constraints must be an object/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural: {}
`),
    ).toThrow(/constraints\.structural must be an array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  derived: {}
`),
    ).toThrow(/constraints\.derived must be an array/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - id: missing_parts
`),
    ).toThrow(/must provide expr, require, or when\/then/i);

    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  derived:
    - id: missing_expr
`),
    ).toThrow(/must provide expr or require/i);
  });

  it('rejects unsupported or malformed TVL sections loudly', () => {
    expect(() => parseTvlSpec('[]')).toThrow(/must parse to an object/i);
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
  - name: prompt
    type: str
    domain: {}
objectives:
  - name: accuracy
    direction: maximize
`),
    ).toThrow(/not supported by the native JS SDK/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: stability
    band:
      target: [0.9]
`),
    ).toThrow(/must contain exactly two values/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy:
    type: annealing
`),
    ).toThrow(/exploration\.strategy type "annealing" is not supported/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy: pareto_optimal
`),
    ).toThrow(/hybrid\/server optimization/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
exploration:
  strategy:
    type: nsga2
`),
    ).toThrow(/hybrid\/server optimization/i);
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
  chance_constraints:
    - name: " "
      threshold: 1
      confidence: 0.9
`),
    ).toThrow(/chance_constraints\[0\]\.name must be a non-empty string/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - expr: process.exit(1)
`),
    ).toThrow(/unsupported syntax "CallExpression"/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - expr: params['constructor']
`),
    ).toThrow(/cannot use computed property access/i);
    expect(() =>
      parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain: ["a"]
objectives:
  - name: accuracy
    direction: maximize
constraints:
  structural:
    - expr: ({}).constructor.constructor("return process")()
`),
    ).toThrow(/unsupported syntax/i);
    expect(loadTvlSpec({})).rejects.toThrow(
      /requires either path or source/i,
    );
  });
});
