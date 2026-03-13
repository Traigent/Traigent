#!/usr/bin/env node

import { runExample } from '../../native-optimization.mjs';

const result = await runExample();
console.log(JSON.stringify(result, null, 2));
