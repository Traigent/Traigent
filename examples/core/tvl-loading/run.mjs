import { parseTvlSpec } from '../../../dist/index.js';

const loaded = parseTvlSpec(`
spec:
  id: tvl-demo
tvars:
  - name: model
    type: enum[str]
    domain: ["cheap", "accurate"]
    default: accurate
  - name: temperature
    type: float
    domain:
      range: [0.0, 0.8]
      step: 0.4
objectives:
  - name: accuracy
    direction: maximize
  - name: response_length
    band:
      target: [120, 180]
constraints:
  structural:
    - expr: params.temperature <= 0.8
exploration:
  strategy: random
  budgets:
    max_trials: 5
    max_spend_usd: 2
    max_wallclock_s: 30
promotion_policy:
  dominance: epsilon_pareto
  alpha: 0.05
`);

console.log('TVL spec:');
console.log(JSON.stringify(loaded.spec, null, 2));
console.log('Optimize options:');
console.log(JSON.stringify(loaded.optimizeOptions, null, 2));
console.log('Promotion policy:');
console.log(JSON.stringify(loaded.promotionPolicy, null, 2));
console.log('Native compatibility:');
console.log(JSON.stringify(loaded.nativeCompatibility, null, 2));
