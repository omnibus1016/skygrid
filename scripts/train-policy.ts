import { CandidateDqn, type PolicyWeights } from '../lib/skygrid/rl-policy';
import { trainDqnPolicy } from '../lib/skygrid/training';
import { readFileSync, writeFileSync } from 'node:fs';

const episodes = Number(process.argv[2] ?? 8_000);
const seed = Number(process.argv[3] ?? 20_261_125);
const outputPath = process.argv[4];
const initialPath = process.argv[5];
const initialPolicy = initialPath
  ? (CandidateDqn.fromWeights(
      (
        JSON.parse(readFileSync(initialPath, 'utf8')) as {
          weights: PolicyWeights;
        }
      ).weights,
    ) ?? undefined)
  : undefined;
const trained = trainDqnPolicy(episodes, seed, undefined, initialPolicy);

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
