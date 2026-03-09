#!/usr/bin/env node

import { parseTvlSpec } from "../../../dist/index.js";

const loaded = parseTvlSpec(`
tvars:
  - name: model
    type: enum[str]
    domain:
      values: ["cheap", "accurate"]
objectives:
  - name: accuracy
    direction: maximize
`);

console.log(JSON.stringify(loaded.spec, null, 2));
