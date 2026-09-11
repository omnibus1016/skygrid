import { readFileSync } from 'node:fs';

import { CandidateDqn } from '../lib/skygrid/rl-policy';
import {
  createMission,
  runBatchEvaluation,
  runScenarioComparison,
} from '../lib/skygrid/simulation';
import type { PlannerKind, ScenarioConfig } from '../lib/skygrid/types';

const modelPath = process.argv[2];
if (!modelPath) throw new Error('모델 JSON 경로가 필요합니다.');
const saved = JSON.parse(readFileSync(modelPath, 'utf8')) as {
  weights?: unknown;
};
const policy = CandidateDqn.fromWeights(saved.weights);
if (!policy) throw new Error('모델 가중치를 읽지 못했습니다.');

const config: ScenarioConfig = {
  droneCount: 4,
  waypointCount: 20,
  durationSec: 1_800,
  failureAt: 360,
  failureDroneId: 'UAV-02',
  planner: 'rl',
  simRate: 4,
  sensorRadiusM: 110,
  randomSeed: 20_261_125,
};

const planners: PlannerKind[] = ['nearest', 'priority', 'rl'];
const totals = Object.fromEntries(
  planners.map((planner) => [
    planner,
    {
      continuity: 0,
      revisitCompliance: 0,
      weightedGapSeconds: 0,
      recoverySeconds: 0,
      recovered: 0,
    },
  ]),
) as Record<
  PlannerKind,
  {
    continuity: number;
    revisitCompliance: number;
    weightedGapSeconds: number;
    recoverySeconds: number;
    recovered: number;
  }
>;

const fullRuns = Number(process.argv[3] ?? 40);
for (let run = 0; run < fullRuns; run += 1) {
  const runConfig = {
    ...config,
    randomSeed: config.randomSeed + run * 97,
  };
  const initial = createMission(runConfig, policy);
  const comparison = runScenarioComparison(
    initial,
    runConfig,
    policy,
    runConfig.durationSec,
  );
  for (const result of comparison) {
    const total = totals[result.planner];
    total.continuity += result.continuity;
    total.revisitCompliance += result.revisitCompliance;
    total.weightedGapSeconds += result.weightedGapSeconds;
    if (result.recoverySeconds !== null) {
      total.recoverySeconds += result.recoverySeconds;
      total.recovered += 1;
    }
  }
}

process.stdout.write(
  JSON.stringify(
    {
      routeAssignment: runBatchEvaluation(config, policy, 80),
      fullMission: planners.map((planner) => ({
        planner,
        continuity: totals[planner].continuity / fullRuns,
        revisitCompliance: totals[planner].revisitCompliance / fullRuns,
        weightedGapSeconds: totals[planner].weightedGapSeconds / fullRuns,
        recoverySeconds: totals[planner].recovered
          ? totals[planner].recoverySeconds / totals[planner].recovered
          : null,
        recoveryRate: (totals[planner].recovered / fullRuns) * 100,
      })),
    },
    null,
    2,
  ),
);
