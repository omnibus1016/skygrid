import { trainDqnPolicy } from '../lib/skygrid/rl-policy';

const episodes = Number(process.argv[2] ?? 8_000);
const seed = Number(process.argv[3] ?? 20_261_125);
const trained = trainDqnPolicy(episodes, seed);

process.stdout.write(
  JSON.stringify(
    {
      weights: trained.policy.toWeights(),
      stats: trained.stats,
    },
    null,
    2,
  ),
);
