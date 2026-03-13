import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import { compileTvlConstraint } from '../../../src/optimization/tvl-expression.js';

describe('TVL expression compiler', () => {
  it('supports equality normalization and python-style logical keywords', () => {
    const constraint = compileTvlConstraint(
      'logical',
      'params.model = "accurate" and not (metrics.accuracy < 0.8)',
      undefined,
      'expr'
    );

    expect(constraint.requiresMetrics).toBe(true);
    expect(constraint({ model: 'accurate' }, { accuracy: 0.9 })).toBe(true);
    expect(constraint({ model: 'accurate' }, { accuracy: 0.5 })).toBe(false);
    expect(constraint({ model: 'cheap' }, { accuracy: 0.9 })).toBe(false);
  });

  it('supports implication constraints without metrics', () => {
    const constraint = compileTvlConstraint(
      'implication',
      'params.retries <= 3',
      undefined,
      'implication',
      'params.model = "accurate"'
    );

    expect(constraint.requiresMetrics).toBe(false);
    expect(constraint({ model: 'cheap', retries: 5 })).toBe(true);
    expect(constraint({ model: 'accurate', retries: 2 })).toBe(true);
    expect(constraint({ model: 'accurate', retries: 5 })).toBe(false);
  });

  it('supports arithmetic, unary operators, null, undefined, and logical disjunction', () => {
    const constraint = compileTvlConstraint(
      'mathy',
      '((params.retries + 1) % 2 == 0) or (+metrics.latency < 3 and -(params.offset) == -2) or params.optional == undefined or params.value != null',
      undefined,
      'expr'
    );

    expect(
      constraint(
        {
          retries: 1,
          offset: 2,
          optional: undefined,
          value: null,
        },
        {
          latency: 2,
        }
      )
    ).toBe(true);
  });

  it('treats nested property access on missing objects as undefined and supports implication guards that read metrics', () => {
    const nullable = compileTvlConstraint(
      'nullable',
      'params.missing.value == undefined',
      undefined,
      'expr'
    );
    expect(nullable({})).toBe(true);

    const implication = compileTvlConstraint(
      'metric-guard',
      'params.retries <= 2',
      undefined,
      'implication',
      'metrics.accuracy >= 0.8'
    );
    expect(implication.requiresMetrics).toBe(true);
    expect(implication({ retries: 5 }, { accuracy: 0.5 })).toBe(true);
    expect(implication({ retries: 5 }, { accuracy: 0.9 })).toBe(false);
  });

  it('supports the remaining comparison and arithmetic operators in the safe subset', () => {
    const constraint = compileTvlConstraint(
      'operators',
      'params.low <= metrics.high and params.high >= metrics.low and params.left !== params.right and params.mode != "cheap" and params.count / 2 == 3 and params.count * 2 >= 12 and params.count - 3 === 3',
      undefined,
      'expr'
    );

    expect(
      constraint(
        {
          low: 1,
          high: 10,
          left: 'a',
          right: 'b',
          mode: 'accurate',
          count: 6,
        },
        {
          high: 5,
          low: 1,
        }
      )
    ).toBe(true);
  });

  it('rejects unsupported identifiers, computed access, and call expressions', () => {
    expect(() => compileTvlConstraint('bad-id', 'process.exit(1)', undefined, 'expr')).toThrow(
      /unsupported syntax "CallExpression"/i
    );

    expect(() =>
      compileTvlConstraint('computed', 'params["constructor"]', undefined, 'expr')
    ).toThrow(/cannot use computed property access/i);

    expect(() => compileTvlConstraint('other-id', 'window.location', undefined, 'expr')).toThrow(
      /unsupported identifier "window"/i
    );
  });

  it('rejects unsupported operators and malformed expressions', () => {
    expect(() => compileTvlConstraint('bitwise', 'params.count & 1', undefined, 'expr')).toThrow(
      /unsupported binary operator "&"/i
    );

    expect(() =>
      compileTvlConstraint('unterminated', 'params.model = "x', undefined, 'expr')
    ).toThrow(/terminate string literals|could not be parsed/i);
  });

  it('wraps evaluation failures with the configured error message', () => {
    const constraint = compileTvlConstraint(
      'runtime-failure',
      'params.token + 1 > 0',
      'friendly message',
      'expr'
    );

    expect(() => constraint({ token: Symbol('boom') })).toThrow(/friendly message/i);
  });

  it('rejects unsupported syntax nodes such as template literals', () => {
    expect(() =>
      compileTvlConstraint('template', 'params.model === `${metrics.accuracy}`', undefined, 'expr')
    ).toThrow(ValidationError);
  });
});
