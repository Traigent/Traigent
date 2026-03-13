#!/usr/bin/env node

import { runExample } from '../../hybrid-optuna.mjs';

if (
  !(process.env.TRAIGENT_BACKEND_URL ?? process.env.TRAIGENT_API_URL) ||
  !process.env.TRAIGENT_API_KEY
) {
  console.log(
    '[skip] examples/core/hybrid-optuna/run.mjs requires TRAIGENT_BACKEND_URL or TRAIGENT_API_URL plus TRAIGENT_API_KEY.'
  );
  process.exit(0);
}

const result = await runExample();
console.log(JSON.stringify(result, null, 2));
