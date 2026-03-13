import { describe, expect, it } from 'vitest';

import { PythonRandom } from '../../../src/optimization/python-random.js';

describe('PythonRandom', () => {
  it('replays the same sequence after serialize/restore', () => {
    const original = new PythonRandom(42);
    const firstValues = [
      original.random(),
      original.randint(1, 10),
      original.uniform(-1, 1),
    ];

    const restored = new PythonRandom(original.serialize());
    const nextOriginal = [original.random(), original.randint(5, 9), original.choice(['a', 'b'])];
    const nextRestored = [restored.random(), restored.randint(5, 9), restored.choice(['a', 'b'])];

    expect(firstValues).toHaveLength(3);
    expect(nextRestored).toEqual(nextOriginal);
  });

  it('supports large numeric seeds and randBelow power-of-two limits', () => {
    const random = new PythonRandom(2 ** 40 + 12345);

    const samples = Array.from({ length: 8 }, () => random.randBelow(8));

    expect(samples.every((value) => Number.isInteger(value) && value >= 0 && value < 8)).toBe(
      true,
    );
  });

  it('validates constructor and public argument errors', () => {
    expect(
      () =>
        new PythonRandom({
          index: 0,
          state: [1, 2, 3],
        } as never),
    ).toThrow(/invalid serialized python random state/i);

    const random = new PythonRandom(0);

    expect(() => random.randBelow(0)).toThrow(/positive safe integer limit/i);
    expect(() => random.choice([])).toThrow(/non-empty array/i);
    expect(() => random.randint(10, 1)).toThrow(/safe integer bounds/i);
  });
});
