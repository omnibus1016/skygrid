import { haversineMeters } from '../lib/skygrid/geo';
import { obstacleAwareDistance } from '../lib/skygrid/pathfinding';
import { CandidateDqn } from '../lib/skygrid/rl-policy';
import {
  createMission,
  theoreticalCapacityEstimate,
} from '../lib/skygrid/simulation';
import { timeBasedDiscount } from '../lib/skygrid/training';
import type { ScenarioConfig, Waypoint } from '../lib/skygrid/types';

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

const policy = new CandidateDqn();
const baseConfig: ScenarioConfig = {
  droneCount: 4,
  waypointCount: 20,
  durationSec: 600,
  failureAt: 240,
  failureDroneId: 'UAV-02',
  planner: 'rl',
  simRate: 1,
  sensorRadiusM: 110,
  randomSeed: 91_027,
  layout: 'mixed',
  mixedFleet: true,
  noFlyZoneCount: 2,
};

const loadFactors = [0.5, 0.8, 1, 1.2];
const missions = loadFactors.map((loadFactor) =>
  createMission({ ...baseConfig, loadFactor }, policy),
);
for (let index = 1; index < missions.length; index += 1) {
  assert(
    missions[index].drones.length >= missions[index - 1].drones.length,
    '부하율이 증가했지만 기체 수가 감소했습니다.',
  );
}
const representative = missions[1];
assert(representative.noFlyZones.length === 2, 'NFZ 생성 수가 다릅니다.');
assert(
  new Set(representative.drones.map((drone) => drone.profileId)).size > 1,
  '다기종 편성이 생성되지 않았습니다.',
);
assert(
  representative.waypoints.every(
    (waypoint) => waypoint.revisitSec >= 120 && waypoint.revisitSec <= 300,
  ),
  '재방문 주기가 120~300초 범위를 벗어났습니다.',
);

const colocated: Waypoint[] = [1, 2].map((index) => ({
  id: `T-${index}`,
  lat: 36.35,
  lng: 127.85,
  priority: index,
  revisitSec: 120,
  dwellSec: 60,
  lastVisited: 0,
  visitedCount: 0,
}));
const capacity = theoreticalCapacityEstimate(colocated, 1, 10);
assert(
  Math.abs(capacity.minimumRequiredDrones - 1) < 1e-9,
  '이론 최소 기체 수 계산이 수학적 기준과 다릅니다.',
);
assert(
  Math.abs(timeBasedDiscount(60) - 0.97) < 1e-12 &&
    Math.abs(timeBasedDiscount(120) - 0.97 ** 2) < 1e-12,
  '시간 기준 할인율이 gamma^Delta_t와 다릅니다.',
);

const start = { lat: 36.35, lng: 127.82 };
const end = { lat: 36.35, lng: 127.88 };
const zone = {
  id: 'TEST-NFZ',
  name: '시험 제한구역',
  points: [
    { lat: 36.345, lng: 127.845 },
    { lat: 36.355, lng: 127.845 },
    { lat: 36.355, lng: 127.855 },
    { lat: 36.345, lng: 127.855 },
  ],
};
assert(
  obstacleAwareDistance(start, end, [zone]) > haversineMeters(start, end),
  'NFZ 우회 거리가 직선 거리보다 길지 않습니다.',
);

console.log(
  JSON.stringify({
    loadFactors: missions.map((mission) => ({
      requested: loadFactors[missions.indexOf(mission)],
      actualDrones: mission.drones.length,
      calculatedLoadFactor: mission.capacityEstimate.loadFactor,
    })),
    minimumRequiredDrones: capacity.minimumRequiredDrones,
    discount60: timeBasedDiscount(60),
    discount120: timeBasedDiscount(120),
    passed: true,
  }),
);
