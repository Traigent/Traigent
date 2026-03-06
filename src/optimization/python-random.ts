interface SerializedPythonRandomState {
  index: number;
  state: number[];
}

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

function int32(value: number): number {
  return value >>> 0;
}

function multiply32(a: number, b: number): number {
  return Math.imul(a, b) >>> 0;
}

function bitLength(value: bigint): number {
  let remaining = value;
  let bits = 0;
  while (remaining > 0n) {
    remaining >>= 1n;
    bits += 1;
  }
  return bits;
}

function toSeedWords(seed: number): number[] {
  let remaining = BigInt(seed);
  if (remaining === 0n) {
    return [0];
  }

  const words: number[] = [];
  while (remaining > 0n) {
    words.push(Number(remaining & 0xffffffffn));
    remaining >>= 32n;
  }

  return words;
}

export class PythonRandom {
  private state: number[] = new Array(N).fill(0);
  private index = N;

  constructor(seed: number);
  constructor(serialized: SerializedPythonRandomState);
  constructor(seedOrState: number | SerializedPythonRandomState) {
    if (typeof seedOrState === 'number') {
      this.initByArray(toSeedWords(seedOrState));
      return;
    }

    if (
      !seedOrState ||
      !Array.isArray(seedOrState.state) ||
      seedOrState.state.length !== N ||
      !Number.isInteger(seedOrState.index)
    ) {
      throw new Error('Invalid serialized Python random state.');
    }

    this.state = seedOrState.state.map((value) => int32(value));
    this.index = seedOrState.index;
  }

  private initGenrand(seed: number): void {
    this.state[0] = int32(seed);
    for (let i = 1; i < N; i += 1) {
      const previous = this.state[i - 1]!;
      this.state[i] = int32(
        multiply32(previous ^ (previous >>> 30), 1812433253) + i,
      );
    }
    this.index = N;
  }

  private initByArray(words: number[]): void {
    this.initGenrand(19650218);

    let i = 1;
    let j = 0;
    let k = Math.max(N, words.length);

    for (; k > 0; k -= 1) {
      const previous = this.state[i - 1]!;
      this.state[i] = int32(
        (this.state[i]! ^
          multiply32(previous ^ (previous >>> 30), 1664525)) +
          words[j]! +
          j,
      );
      i += 1;
      j += 1;

      if (i >= N) {
        this.state[0] = this.state[N - 1]!;
        i = 1;
      }
      if (j >= words.length) {
        j = 0;
      }
    }

    for (k = N - 1; k > 0; k -= 1) {
      const previous = this.state[i - 1]!;
      this.state[i] = int32(
        (this.state[i]! ^
          multiply32(previous ^ (previous >>> 30), 1566083941)) - i,
      );
      i += 1;

      if (i >= N) {
        this.state[0] = this.state[N - 1]!;
        i = 1;
      }
    }

    this.state[0] = 0x80000000;
  }

  private twist(): void {
    for (let i = 0; i < N; i += 1) {
      const upper = this.state[i]! & UPPER_MASK;
      const lower = this.state[(i + 1) % N]! & LOWER_MASK;
      let next = this.state[(i + M) % N]! ^ ((upper | lower) >>> 1);
      if ((lower & 1) === 1) {
        next ^= MATRIX_A;
      }
      this.state[i] = int32(next);
    }
    this.index = 0;
  }

  private extractInt32(): number {
    if (this.index >= N) {
      this.twist();
    }

    let y = this.state[this.index]!;
    this.index += 1;

    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;

    return int32(y);
  }

  random(): number {
    const high = this.extractInt32() >>> 5;
    const low = this.extractInt32() >>> 6;
    return (high * 67108864 + low) / 9007199254740992;
  }

  private getRandBits(bits: number): bigint {
    if (!Number.isInteger(bits) || bits < 0) {
      throw new Error('bits must be a non-negative integer.');
    }
    if (bits === 0) {
      return 0n;
    }

    let remaining = bits;
    let value = 0n;
    let shift = 0n;

    while (remaining >= 32) {
      value |= BigInt(this.extractInt32()) << shift;
      remaining -= 32;
      shift += 32n;
    }

    if (remaining > 0) {
      value |= BigInt(this.extractInt32() >>> (32 - remaining)) << shift;
    }

    return value;
  }

  randBelow(limit: number): number {
    if (!Number.isSafeInteger(limit) || limit <= 0) {
      throw new Error('randBelow() requires a positive safe integer limit.');
    }

    const max = BigInt(limit);
    const bits = bitLength(max - 1n);
    let candidate = this.getRandBits(bits);
    while (candidate >= max) {
      candidate = this.getRandBits(bits);
    }
    return Number(candidate);
  }

  choice<T>(values: readonly T[]): T {
    if (values.length === 0) {
      throw new Error('choice() requires a non-empty array.');
    }
    return values[this.randBelow(values.length)]!;
  }

  randint(min: number, max: number): number {
    if (!Number.isSafeInteger(min) || !Number.isSafeInteger(max) || max < min) {
      throw new Error('randint() requires safe integer bounds with max >= min.');
    }
    return min + this.randBelow(max - min + 1);
  }

  uniform(min: number, max: number): number {
    return min + (max - min) * this.random();
  }

  serialize(): SerializedPythonRandomState {
    return {
      index: this.index,
      state: [...this.state],
    };
  }
}

export type { SerializedPythonRandomState };
