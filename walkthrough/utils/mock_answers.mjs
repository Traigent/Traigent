export function modelAccuracy(config) {
  const model = String(
    config.providerModel ?? config.model ?? config.provider ?? "balanced",
  );
  if (model.includes("accurate") || model.includes("gpt-4o")) return 0.92;
  if (model.includes("balanced") || model.includes("mini")) return 0.82;
  return 0.68;
}

export function mockCost(config) {
  const model = String(
    config.providerModel ?? config.model ?? config.provider ?? "balanced",
  );
  if (model.includes("accurate") || model.includes("gpt-4o")) return 0.22;
  if (model.includes("balanced") || model.includes("mini")) return 0.11;
  return 0.05;
}

export function mockLatency(config) {
  const model = String(
    config.providerModel ?? config.model ?? config.provider ?? "balanced",
  );
  if (model.includes("accurate") || model.includes("gpt-4o")) return 1.1;
  if (model.includes("balanced") || model.includes("mini")) return 0.7;
  return 0.4;
}

export function mockText(row, config, scenario) {
  const accuracy = modelAccuracy(config);
  if (scenario === "custom_evaluator") {
    return accuracy >= 0.8
      ? `function solve() { return ${JSON.stringify(row.input.task)}; }`
      : "pass";
  }
  return accuracy >= 0.8 ? row.output : "unknown";
}
