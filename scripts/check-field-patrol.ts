import { CandidateDqn } from '../lib/skygrid/rl-policy';
import {
  PRETRAINED_POLICY_WEIGHTS,
} from '../lib/skygrid/pretrained-policy';
import {
  createMission,
  reportFieldWaypointVisit,
} from '../lib/skygrid/simulation';
import type { MissionState, ScenarioConfig } from '../lib/skygrid/types';

const loadedPolicy = CandidateDqn.fromWeights(PRETRAINED_POLICY_WEIGHTS);
if (!loadedPolicy) throw new Error('사전학습 모델을 불러오지 못했습니다.');
const policy: CandidateDqn = loadedPolicy;

function config(droneCount: number, seed: number): ScenarioConfig {
  return {
    droneCount,
    waypointCount: 4,
    durationSec: 1_800,
    failureAt: 900,
    failureDroneId: 'UAV-02',
    planner: 'rl',
    simRate: 1,
    sensorRadiusM: 110,
    randomSeed: seed,
  };
}

function nextAssignedDrone(mission: MissionState) {
  return mission.drones.find(
    (drone) =>
      drone.status !== 'failed' &&
      Boolean(drone.route[drone.routeIndex]),
  );
}

function runPatrol(droneCount: number, seed: number, rounds: number) {
  let mission = createMission(config(droneCount, seed), policy);
  const waypointCount = mission.waypoints.length;
  const sequence: string[] = [];
  for (let round = 0; round < rounds; round += 1) {
    const seen = new Set<string>();
    for (let step = 0; step < waypointCount; step += 1) {
      const drone = nextAssignedDrone(mission);
      if (!drone) throw new Error('정찰 가능한 기체 경로가 사라졌습니다.');
      const waypointId = drone.route[drone.routeIndex];
      if (seen.has(waypointId)) {
        throw new Error(
          `${droneCount}대 조건에서 전체 순회 전 ${waypointId}가 반복되었습니다: ${sequence.join('>')}`,
        );
      }
      seen.add(waypointId);
      sequence.push(waypointId);
      const result = reportFieldWaypointVisit(
        mission,
        drone.id,
        waypointId,
        'rl',
        policy,
      );
      if (!result) throw new Error(`${waypointId} 완료 보고에 실패했습니다.`);
      mission = { ...result.state, time: result.state.time + 180 };
    }
    if (seen.size !== waypointCount) {
      throw new Error(
        `${droneCount}대 조건에서 모든 지점을 방문하지 못했습니다.`,
      );
    }
  }
  const visits = mission.waypoints.map((waypoint) => waypoint.visitedCount);
  if (Math.max(...visits) - Math.min(...visits) > 1) {
    throw new Error(
      `${droneCount}대 조건에서 방문 편차가 1회를 초과했습니다: ${visits.join(',')}`,
    );
  }
  return { droneCount, sequence, visits };
}

const results = [runPatrol(1, 41, 3), runPatrol(2, 73, 3)];
for (const result of results) {
  console.log(
    JSON.stringify({
      drones: result.droneCount,
      sequence: result.sequence,
      visits: result.visits,
      passed: true,
    }),
  );
}
