import { AsyncLocalStorage } from "node:async_hooks";

import type { Metrics } from "../dtos/trial.js";

type RuntimeMetricsStore = Record<string, number>;

const runtimeMetricsStorage = new AsyncLocalStorage<RuntimeMetricsStore>();

export async function withRuntimeMetricsCollector<T>(
  fn: () => Promise<T> | T,
): Promise<{ result: T; metrics: Metrics }> {
  const store: RuntimeMetricsStore = {};
  const result = await runtimeMetricsStorage.run(store, () => Promise.resolve(fn()));
  return {
    result,
    metrics: { ...store },
  };
}

export function recordRuntimeMetrics(metrics: Record<string, unknown>): void {
  const store = runtimeMetricsStorage.getStore();
  if (!store) {
    return;
  }

  for (const [key, value] of Object.entries(metrics)) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      continue;
    }
    store[key] = (store[key] ?? 0) + value;
  }
}
