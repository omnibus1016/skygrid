import { readFileSync, writeFileSync } from 'node:fs';

import type { PolicyWeights } from '../lib/skygrid/rl-policy';
import type { PolicyStats } from '../lib/skygrid/types';

const inputPath = process.argv[2];
const outputPath = process.argv[3] ?? 'lib/skygrid/pretrained-policy.ts';
if (!inputPath) throw new Error('학습 모델 JSON 경로가 필요합니다.');

const trained = JSON.parse(readFileSync(inputPath, 'utf8')) as {
  weights: PolicyWeights;
  stats: PolicyStats;
};
const weights = { ...trained.weights, version: 5 as const };
const stats: PolicyStats = {
  ...trained.stats,
  modelVersion: 5,
  trainingScope: `${trained.stats.episodes.toLocaleString()}개 실제 비행 시뮬레이터 기반 복합 시나리오`,
  rewardHistory: trained.stats.rewardHistory.filter(
    (point) => point.episode === 1 || point.episode % 200 === 0,
  ),
  validation: trained.stats.validation
    ? {
        ...trained.stats.validation,
        passed: trained.stats.validation.improvementVsBest >= 0,
      }
    : undefined,
  operationalValidation: undefined,
};

const source = `import type { PolicyWeights } from './rl-policy';
import type { PolicyStats } from './types';

export const PRETRAINED_POLICY_WEIGHTS: PolicyWeights = ${JSON.stringify(weights, null, 2)};

export const PRETRAINED_POLICY_STATS: PolicyStats = ${JSON.stringify(stats, null, 2)};
`;

writeFileSync(outputPath, source);
process.stdout.write(`${outputPath}\n`);
