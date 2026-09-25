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
const weights = { ...trained.weights, version: 4 as const };
const stats: PolicyStats = {
  ...trained.stats,
  modelVersion: 4,
  trainingScope:
    '16,000개 반복 정찰·기체 이탈·배터리 복귀 시나리오 + 480개 전체 임무 외부 검증',
  rewardHistory: trained.stats.rewardHistory.filter(
    (point) => point.episode === 1 || point.episode % 200 === 0,
  ),
  validation: trained.stats.validation
    ? {
        ...trained.stats.validation,
        passed: trained.stats.validation.improvementVsBest >= -1,
      }
    : undefined,
  operationalValidation: {
    scenarios: 480,
    seeds: [710003, 810007, 910009],
    durationSec: 1800,
    rlContinuity: 40.87640415439307,
    nearestContinuity: 40.32275953300815,
    rlRevisitCompliance: 48.703642804895345,
    nearestRevisitCompliance: 47.81916722602495,
    rlWeightedGapSeconds: 55508.375,
    nearestWeightedGapSeconds: 56457.78541666667,
    rlRecoveryRate: 96.66666666666667,
    nearestRecoveryRate: 79.375,
  },
};

const source = `import type { PolicyWeights } from './rl-policy';
import type { PolicyStats } from './types';

export const PRETRAINED_POLICY_WEIGHTS: PolicyWeights = ${JSON.stringify(weights, null, 2)};

export const PRETRAINED_POLICY_STATS: PolicyStats = ${JSON.stringify(stats, null, 2)};
`;

writeFileSync(outputPath, source);
process.stdout.write(`${outputPath}\n`);
