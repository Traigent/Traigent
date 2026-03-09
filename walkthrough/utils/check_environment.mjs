#!/usr/bin/env node

import { resolveRealProviderConfig } from "./helpers.mjs";

const provider = resolveRealProviderConfig();

console.log("Traigent JS walkthrough environment");
console.log(`- OFFLINE: ${process.env.TRAIGENT_OFFLINE_MODE ?? "true"}`);
console.log(
  `- Hybrid backend: ${process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL ?? "not configured"}`,
);
console.log(
  `- Provider: ${provider ? provider.defaultModel : "no OpenAI/OpenRouter key found"}`,
);
