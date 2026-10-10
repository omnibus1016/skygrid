import { CandidateDqn } from '../lib/skygrid/rl-policy';
import { PRETRAINED_POLICY_WEIGHTS } from '../lib/skygrid/pretrained-policy';
import {
  assignFieldRoutes,
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
    // This regression isolates round-robin field reporting from range and
    // obstacle constraints, which are covered by check-research-pipeline.
    layout: 'ring',
    mixedFleet: false,
    noFlyZoneCount: 0,
  };
}

function nextAssignedDrone(mission: MissionState) {
  return mission.drones
    .filter(
      (drone) =>
        drone.status !== 'failed' && Boolean(drone.route[drone.routeIndex]),
    )
    .sort((a, b) => {
      const targetA = mission.waypoints.find(
        (waypoint) => waypoint.id === a.route[a.routeIndex],
      );
      const targetB = mission.waypoints.find(
        (waypoint) => waypoint.id === b.route[b.routeIndex],
      );
      if (!targetA || !targetB) return 0;
      return (
        targetA.visitedCount - targetB.visitedCount ||
        targetA.lastVisited + targetA.revisitSec -
          (targetB.lastVisited + targetB.revisitSec)
      );
    })[0];
}

function runPatrol(droneCount: number, seed: number, rounds: number) {
  let mission = createMission(config(droneCount, seed), policy);
  const initialPlan = assignFieldRoutes(
    mission.drones,
    mission.waypoints,
    mission.time,
    'rl',
    policy,
  );
  mission = {
    ...mission,
    drones: initialPlan.drones,
    waypoints: initialPlan.waypoints,
  };
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
      const operational = mission.drones.filter(
        (candidate) => candidate.status !== 'failed',
      );
      const unassigned = operational.filter(
        (candidate) => !candidate.route[candidate.routeIndex],
      );
      if (waypointCount >= operational.length && unassigned.length) {
        throw new Error(
          `${droneCount}대 조건에서 ${unassigned.map((candidate) => candidate.id).join(',')} 경로가 비었습니다.`,
        );
      }
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

function runCommittedTargetRegression() {
  let mission = createMission(config(2, 151), policy);
  const initialPlan = assignFieldRoutes(
    mission.drones,
    mission.waypoints,
    mission.time,
    'rl',
    policy,
  );
  mission = {
    ...mission,
    drones: initialPlan.drones,
    waypoints: initialPlan.waypoints,
  };

  const reportingDrone = mission.drones[0];
  const committedDrone = mission.drones[1];
  const completedWaypointId = reportingDrone.route[reportingDrone.routeIndex];
  const committedWaypointId = committedDrone.route[committedDrone.routeIndex];
  if (!completedWaypointId || !committedWaypointId) {
    throw new Error('두 기체의 초기 현장 목표를 배정하지 못했습니다.');
  }

  const result = reportFieldWaypointVisit(
    mission,
    reportingDrone.id,
    completedWaypointId,
    'rl',
    policy,
  );
  if (!result) throw new Error('현장 정찰 완료 보고에 실패했습니다.');

  const after = result.state.drones.find(
    (drone) => drone.id === committedDrone.id,
  );
  const targetAfterReport = after?.route[after.routeIndex];
  if (targetAfterReport !== committedWaypointId) {
    throw new Error(
      `${reportingDrone.id} 완료 보고 후 ${committedDrone.id}의 이동 중 목표가 ${committedWaypointId}에서 ${targetAfterReport ?? '미배정'}(으)로 변경되었습니다.`,
    );
  }

  return {
    reportingDrone: reportingDrone.id,
    completedWaypoint: completedWaypointId,
    committedDrone: committedDrone.id,
    committedWaypoint: committedWaypointId,
    preserved: true,
  };
}

const results = [
  runPatrol(1, 41, 3),
  runPatrol(2, 73, 3),
  runPatrol(3, 97, 3),
];
console.log(JSON.stringify(runCommittedTargetRegression()));
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
