import { haversineMeters, moveToward, polylineDistance } from './geo';
import { mulberry32 } from './random';
import { assignRoutes, type CandidateDqn } from './rl-policy';
import type {
  BatchResult,
  Drone,
  MissionEvent,
  MissionMetrics,
  MissionState,
  PlannerKind,
  ScenarioConfig,
  Waypoint,
} from './types';

const CENTER = { lat: 36.6219, lng: 127.5032 };
const COLORS = [
  '#67e8f9',
  '#fbbf62',
  '#a78bfa',
  '#64d39b',
  '#f472b6',
  '#b4c8d2',
  '#fb7185',
  '#60a5fa',
];
const MODELS = [
  {
    model: 'DJI MAVIC PRO',
    speedMps: 10,
    turnRateDps: 120,
    consumptionPerKm: 9.6,
  },
  {
    model: 'DJI AVATA 2',
    speedMps: 12,
    turnRateDps: 180,
    consumptionPerKm: 13.8,
  },
  {
    model: 'SIM SCOUT-S',
    speedMps: 11,
    turnRateDps: 140,
    consumptionPerKm: 10.4,
  },
];

function makeEvent(
  time: number,
  kind: MissionEvent['kind'],
  title: string,
  detail: string,
): MissionEvent {
  return {
    id: `${time}-${kind}-${Math.random().toString(36).slice(2, 7)}`,
    time,
    kind,
    title,
    detail,
  };
}

export function createMission(
  config: ScenarioConfig,
  policy: CandidateDqn,
): MissionState {
  const random = mulberry32(config.randomSeed);
  const drones: Drone[] = Array.from(
    { length: config.droneCount },
    (_, index) => {
      const profile = MODELS[index % MODELS.length];
      const angle = (index / config.droneCount) * Math.PI * 2;
      return {
        id: `UAV-${String(index + 1).padStart(2, '0')}`,
        model: profile.model,
        color: COLORS[index % COLORS.length],
        lat: CENTER.lat + Math.sin(angle) * 0.003,
        lng: CENTER.lng + Math.cos(angle) * 0.004,
        speedMps: profile.speedMps,
        turnRateDps: profile.turnRateDps,
        battery: 82 - index * 2,
        reserveBattery: 22,
        consumptionPerKm: profile.consumptionPerKm,
        status: 'active',
        route: [],
        routeIndex: 0,
        totalDistanceM: 0,
        heading: 0,
      };
    },
  );
  const waypoints: Waypoint[] = Array.from(
    { length: config.waypointCount },
    (_, index) => {
      const ring = index % 3;
      const angle =
        (index / config.waypointCount) * Math.PI * 2 + random() * 0.35;
      const radiusLat = 0.0048 + ring * 0.0015 + random() * 0.0012;
      const radiusLng = radiusLat * 1.3;
      return {
        id: `RP-${String(index + 1).padStart(2, '0')}`,
        lat: CENTER.lat + Math.sin(angle) * radiusLat,
        lng: CENTER.lng + Math.cos(angle) * radiusLng,
        priority: 1 + Math.floor(random() * 5),
        revisitSec: 90 + Math.floor(random() * 120),
        dwellSec: 6 + Math.floor(random() * 8),
        lastVisited: 0,
        visitedCount: 0,
      };
    },
  );
  const noFlyZones = [
    {
      id: 'NFZ-01',
      name: '비행 제한구역',
      points: [
        { lat: CENTER.lat + 0.0004, lng: CENTER.lng - 0.0022 },
        { lat: CENTER.lat + 0.0031, lng: CENTER.lng - 0.0004 },
        { lat: CENTER.lat + 0.0015, lng: CENTER.lng + 0.0018 },
        { lat: CENTER.lat - 0.0012, lng: CENTER.lng + 0.0002 },
      ],
    },
  ];
  const planned = assignRoutes(drones, waypoints, 0, config.planner, policy);
  return {
    time: 0,
    running: false,
    completed: false,
    failureTriggered: false,
    drones: planned.drones,
    waypoints: planned.waypoints,
    noFlyZones,
    events: [
      makeEvent(
        0,
        'system',
        '시나리오 초기화',
        `${config.droneCount}대 · 정찰지점 ${config.waypointCount}개`,
      ),
    ],
    weightedGapSeconds: 0,
    replanCount: 0,
    inferenceMs: 0,
  };
}

export function triggerFailure(
  state: MissionState,
  droneId: string,
  planner: PlannerKind,
  policy: CandidateDqn,
  reason = '운용자 이탈 판정',
): MissionState {
  const start = performance.now();
  const failed = state.drones.find(
    (drone) => drone.id === droneId && drone.status === 'active',
  );
  if (!failed) return state;
  const orphanedWaypoints = state.waypoints.filter(
    (waypoint) => waypoint.assignedDrone === droneId,
  ).length;
  const drones = state.drones.map((drone) =>
    drone.id === droneId
      ? { ...drone, status: 'failed' as const, route: [], routeIndex: 0 }
      : drone,
  );
  const waypoints = state.waypoints.map((waypoint) =>
    waypoint.assignedDrone === droneId
      ? { ...waypoint, assignedDrone: undefined }
      : { ...waypoint },
  );
  const planned = assignRoutes(drones, waypoints, state.time, planner, policy);
  const inferenceMs = Math.max(1, performance.now() - start);
  return {
    ...state,
    failureTriggered: true,
    drones: planned.drones,
    waypoints: planned.waypoints,
    replanCount: state.replanCount + 1,
    inferenceMs,
    events: [
      makeEvent(
        state.time,
        'replan',
        'AI 임무 재계획 완료',
        `${orphanedWaypoints}개 정찰지점 재할당 · ${planner.toUpperCase()} · ${inferenceMs.toFixed(1)} ms`,
      ),
      makeEvent(
        state.time,
        'failure',
        `${droneId} 임무 이탈`,
        `${reason} · ${failed.model} · 잔여 ${failed.battery.toFixed(0)}%`,
      ),
      ...state.events,
    ].slice(0, 18),
  };
}

export function restoreDrone(
  state: MissionState,
  droneId: string,
  planner: PlannerKind,
  policy: CandidateDqn,
): MissionState {
  const restored = state.drones.find(
    (drone) => drone.id === droneId && drone.status === 'failed',
  );
  if (!restored) return state;
  const drones = state.drones.map((drone) =>
    drone.id === droneId
      ? { ...drone, status: 'active' as const, route: [], routeIndex: 0 }
      : drone,
  );
  const planned = assignRoutes(
    drones,
    state.waypoints,
    state.time,
    planner,
    policy,
  );
  return {
    ...state,
    failureTriggered: planned.drones.some((drone) => drone.status === 'failed'),
    drones: planned.drones,
    waypoints: planned.waypoints,
    replanCount: state.replanCount + 1,
    events: [
      makeEvent(
        state.time,
        'system',
        `${droneId} 임무 복귀`,
        '복귀 기체를 포함하여 전체 정찰경로를 다시 계산했습니다.',
      ),
      ...state.events,
    ].slice(0, 18),
  };
}

/**
 * 현장 운용에서는 웹 앱이 DJI 기체를 직접 제어하거나 위치를 수신하지 않는다.
 * 따라서 임무 시계와 공백 지표만 갱신하며, 위치·배터리·정찰 완료는 운용자가 입력한다.
 */
export function advanceFieldMission(
  state: MissionState,
  deltaSeconds: number,
): MissionState {
  if (!state.running || state.completed) return state;
  const time = state.time + deltaSeconds;
  const weightedGapRate = state.waypoints.reduce(
    (sum, waypoint) =>
      sum +
      (time - waypoint.lastVisited > waypoint.revisitSec
        ? waypoint.priority
        : 0),
    0,
  );
  return {
    ...state,
    time,
    weightedGapSeconds:
      state.weightedGapSeconds + weightedGapRate * deltaSeconds,
  };
}

export function advanceMission(
  state: MissionState,
  deltaSeconds: number,
  config: ScenarioConfig,
  policy: CandidateDqn,
): MissionState {
  if (!state.running || state.completed) return state;
  let next = { ...state, time: state.time + deltaSeconds };
  if (!state.failureTriggered && next.time >= config.failureAt)
    next = triggerFailure(next, config.failureDroneId, config.planner, policy);

  const waypointMap = new Map(
    next.waypoints.map((waypoint) => [waypoint.id, { ...waypoint }]),
  );
  const visitEvents: MissionEvent[] = [];
  const drones = next.drones.map((drone) => {
    if (drone.status !== 'active' || drone.routeIndex >= drone.route.length)
      return { ...drone };
    const target = waypointMap.get(drone.route[drone.routeIndex]);
    if (!target) return { ...drone, routeIndex: drone.routeIndex + 1 };
    const travel = drone.speedMps * deltaSeconds;
    const distance = haversineMeters(drone, target);
    const moved = Math.min(travel, distance);
    const position = moveToward(drone, target, moved);
    const batteryUse = (moved / 1000) * drone.consumptionPerKm;
    const updated = {
      ...drone,
      ...position,
      battery: Math.max(0, drone.battery - batteryUse),
      totalDistanceM: drone.totalDistanceM + moved,
    };
    if (distance <= travel + 1) {
      target.lastVisited = next.time;
      target.visitedCount += 1;
      waypointMap.set(target.id, target);
      updated.routeIndex += 1;
      visitEvents.push(
        makeEvent(
          next.time,
          'visit',
          `${drone.id} ${target.id} 정찰`,
          `중요도 ${target.priority} · 방문 ${target.visitedCount}회`,
        ),
      );
    }
    if (updated.battery <= updated.reserveBattery) updated.status = 'returning';
    return updated;
  });

  const waypoints = [...waypointMap.values()];
  const weightedGapRate = waypoints.reduce(
    (sum, waypoint) =>
      sum +
      (next.time - waypoint.lastVisited > waypoint.revisitSec
        ? waypoint.priority
        : 0),
    0,
  );
  const active = drones.filter((drone) => drone.status === 'active');
  const allRoutesDone =
    active.length > 0 &&
    active.every((drone) => drone.routeIndex >= drone.route.length);
  if (allRoutesDone) {
    const planned = assignRoutes(
      drones,
      waypoints,
      next.time,
      config.planner,
      policy,
    );
    next = {
      ...next,
      drones: planned.drones,
      waypoints: planned.waypoints,
      replanCount: next.replanCount + 1,
    };
  } else {
    next = { ...next, drones, waypoints };
  }
  return {
    ...next,
    completed: next.time >= 600 || active.length === 0,
    weightedGapSeconds:
      next.weightedGapSeconds + weightedGapRate * deltaSeconds,
    events: [...visitEvents.reverse(), ...next.events].slice(0, 18),
  };
}

export function missionMetrics(state: MissionState): MissionMetrics {
  const priorityTotal = state.waypoints.reduce(
    (sum, waypoint) => sum + waypoint.priority,
    0,
  );
  const currentCoveredPriority = state.waypoints.reduce(
    (sum, waypoint) =>
      sum +
      (state.time - waypoint.lastVisited <= waypoint.revisitSec
        ? waypoint.priority
        : 0),
    0,
  );
  const visited = state.waypoints.filter(
    (waypoint) => waypoint.visitedCount > 0,
  ).length;
  const continuity = priorityTotal
    ? (currentCoveredPriority / priorityTotal) * 100
    : 100;
  const failure = [...state.events]
    .reverse()
    .find((event) => event.kind === 'failure');
  const firstVisitAfterFailure = failure
    ? [...state.events]
        .reverse()
        .find((event) => event.kind === 'visit' && event.time >= failure.time)
    : undefined;
  return {
    continuity,
    coverage: state.waypoints.length
      ? (visited / state.waypoints.length) * 100
      : 0,
    activeDrones: state.drones.filter((drone) => drone.status === 'active')
      .length,
    weightedGapSeconds: state.weightedGapSeconds,
    recoverySeconds:
      failure && firstVisitAfterFailure
        ? firstVisitAfterFailure.time - failure.time
        : null,
    totalDistanceKm:
      state.drones.reduce((sum, drone) => sum + drone.totalDistanceM, 0) / 1000,
    averageBattery: state.drones.length
      ? state.drones.reduce((sum, drone) => sum + drone.battery, 0) /
        state.drones.length
      : 0,
    completedVisits: state.waypoints.reduce(
      (sum, waypoint) => sum + waypoint.visitedCount,
      0,
    ),
  };
}

function scoreAssignedPlan(
  drones: Drone[],
  waypoints: Waypoint[],
): Omit<BatchResult, 'planner' | 'label'> {
  let weightedGap = 0;
  let distanceM = 0;
  let completed = 0;
  for (const drone of drones.filter((item) => item.status === 'active')) {
    let position = { lat: drone.lat, lng: drone.lng };
    let elapsed = 0;
    let battery = drone.battery;
    for (const waypointId of drone.route) {
      const waypoint = waypoints.find((item) => item.id === waypointId);
      if (!waypoint) continue;
      const segment = haversineMeters(position, waypoint);
      const use = (segment / 1000) * drone.consumptionPerKm;
      if (battery - use < drone.reserveBattery) break;
      elapsed += segment / drone.speedMps;
      weightedGap +=
        Math.max(0, elapsed - waypoint.revisitSec) * waypoint.priority;
      distanceM += segment;
      battery -= use;
      completed += 1;
      position = waypoint;
    }
  }
  const completion = waypoints.length
    ? (completed / waypoints.length) * 100
    : 0;
  const continuity = Math.max(
    0,
    100 -
      weightedGap /
        Math.max(
          1,
          waypoints.reduce((sum, waypoint) => sum + waypoint.priority, 0) * 5,
        ),
  );
  return { continuity, weightedGap, distance: distanceM / 1000, completion };
}

export function runBatchEvaluation(
  config: ScenarioConfig,
  policy: CandidateDqn,
  runs = 80,
): BatchResult[] {
  const planners: { kind: PlannerKind; label: string }[] = [
    { kind: 'nearest', label: '최근접 우선' },
    { kind: 'priority', label: '중요도 우선' },
    { kind: 'rl', label: '강화학습' },
  ];
  return planners.map(({ kind, label }) => {
    const totals = {
      continuity: 0,
      weightedGap: 0,
      distance: 0,
      completion: 0,
    };
    for (let run = 0; run < runs; run += 1) {
      const mission = createMission(
        { ...config, planner: kind, randomSeed: config.randomSeed + run * 17 },
        policy,
      );
      const failedDrones = mission.drones.map((drone) =>
        drone.id === config.failureDroneId
          ? { ...drone, status: 'failed' as const }
          : drone,
      );
      const planned = assignRoutes(
        failedDrones,
        mission.waypoints,
        config.failureAt,
        kind,
        policy,
      );
      const result = scoreAssignedPlan(planned.drones, planned.waypoints);
      totals.continuity += result.continuity;
      totals.weightedGap += result.weightedGap;
      totals.distance += result.distance;
      totals.completion += result.completion;
    }
    return {
      planner: kind,
      label,
      continuity: totals.continuity / runs,
      weightedGap: totals.weightedGap / runs,
      distance: totals.distance / runs,
      completion: totals.completion / runs,
    };
  });
}

export function plannedRoutePoints(
  drone: Drone,
  waypoints: Waypoint[],
): { lat: number; lng: number }[] {
  return [
    drone,
    ...(drone.route
      .slice(drone.routeIndex)
      .map((id) => waypoints.find((waypoint) => waypoint.id === id))
      .filter(Boolean) as Waypoint[]),
  ];
}

export function plannedDistance(drone: Drone, waypoints: Waypoint[]): number {
  return polylineDistance(plannedRoutePoints(drone, waypoints));
}
