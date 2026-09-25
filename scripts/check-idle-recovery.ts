import { PRETRAINED_POLICY_WEIGHTS } from '../lib/skygrid/pretrained-policy';
import { CandidateDqn } from '../lib/skygrid/rl-policy';
import { advanceMission, createMission } from '../lib/skygrid/simulation';
import type { ScenarioConfig } from '../lib/skygrid/types';

const policy = CandidateDqn.fromWeights(PRETRAINED_POLICY_WEIGHTS);
if (!policy) throw new Error('기본 모델을 불러오지 못했습니다.');

const config: ScenarioConfig = {
  droneCount: 4,
  waypointCount: 20,
  durationSec: 1_800,
  failureAt: 360,
  failureDroneId: 'UAV-02',
  planner: 'rl',
  simRate: 1,
  sensorRadiusM: 110,
  randomSeed: 920_011,
};
let mission = { ...createMission(config, policy), running: true };
const idleStreak = new Map<string, number>();
const maximumIdleStreak = new Map<string, number>();

while (!mission.completed) {
  mission = advanceMission(mission, 1, config, policy);
  const routeCarriers = mission.drones.filter(
    (drone) =>
      drone.status !== 'failed' && drone.routeIndex < drone.route.length,
  ).length;
  for (const drone of mission.drones) {
    const unexplainedIdle =
      drone.status === 'ready' &&
      drone.turnaroundRemainingSec <= 0 &&
      drone.routeIndex >= drone.route.length &&
      mission.waypoints.length > routeCarriers;
    const streak = unexplainedIdle ? (idleStreak.get(drone.id) ?? 0) + 1 : 0;
    idleStreak.set(drone.id, streak);
    maximumIdleStreak.set(
      drone.id,
      Math.max(maximumIdleStreak.get(drone.id) ?? 0, streak),
    );
  }
}

const maximumSeconds = Math.max(0, ...maximumIdleStreak.values());
if (maximumSeconds > 1) {
  throw new Error(`설명되지 않은 기지 대기 ${maximumSeconds}초`);
}
process.stdout.write(
  JSON.stringify(
    {
      passed: true,
      maximumUnexplainedBaseIdleSeconds: maximumSeconds,
      replans: mission.replanCount,
      completedVisits: mission.waypoints.reduce(
        (sum, waypoint) => sum + waypoint.visitedCount,
        0,
      ),
    },
    null,
    2,
  ),
);
