import { trainDqnPolicy } from '../lib/skygrid/rl-policy';
import { writeFileSync } from 'node:fs';

const episodes = Number(process.argv[2] ?? 8_000);
const seed = Number(process.argv[3] ?? 20_261_125);
const outputPath = process.argv[4];
const trained = trainDqnPolicy(episodes, seed);

const output = JSON.stringify(
  {
    weights: trained.policy.toWeights(),
    stats: trained.stats,
  },
  null,
  2,
);
if (outputPath) {
  writeFileSync(outputPath, output);
  process.stdout.write(`${outputPath}\n`);
} else {
  process.stdout.write(output);
}
