import { haversineMeters, moveToward, polylineDistance } from './geo';
import { mulberry32 } from './random';
import {
  assignRoutes,
  estimateSortieEnergy,
  type CandidateDqn,
} from './rl-policy';
import type {
  BatchResult,
  Drone,
  MissionEvent,
  MissionMetrics,
  MissionState,
  PlannerKind,
  ScenarioConfig,
  ScenarioComparisonResult,
  Waypoint,
} from './types';

const CENTER = { lat: 36.6219, lng: 127.5032 };
const BASE = {
  id: 'BASE-01',
  name: '무인기 전개기지',
  lat: CENTER.lat - 0.0094,
  lng: CENTER.lng - 0.0118,
};
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
    loiterConsumptionPerMin: 1.05,
    maxBattery: 92,
    reserveBattery: 20,
    turnaroundSec: 45,
  },
  {
    model: 'DJI AVATA 2',
    speedMps: 12,
    turnRateDps: 180,
    consumptionPerKm: 13.8,
    loiterConsumptionPerMin: 1.45,
    maxBattery: 90,
    reserveBattery: 22,
    turnaroundSec: 50,
  },
  {
    model: 'SIM SCOUT-S',
    speedMps: 11,
    turnRateDps: 140,
    consumptionPerKm: 10.4,
    loiterConsumptionPerMin: 1.15,
    maxBattery: 90,
    reserveBattery: 20,
    turnaroundSec: 40,
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

function continuityAt(time: number, waypoints: Waypoint[]): number {
  const priorityTotal = waypoints.reduce(
    (sum, waypoint) => sum + waypoint.priority,
    0,
  );
  if (!priorityTotal) return 100;
  const coveredPriority = waypoints.reduce(
    (sum, waypoint) =>
      sum +
      (time - waypoint.lastVisited <= waypoint.revisitSec
        ? waypoint.priority
        : 0),
    0,
  );
  return (coveredPriority / priorityTotal) * 100;
}

function headingTo(from: Drone, to: { lat: number; lng: number }): number {
  const y = Math.sin(((to.lng - from.lng) * Math.PI) / 180);
  const x = Math.sin(((to.lat - from.lat) * Math.PI) / 180);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

function returnEnergy(drone: Drone): number {
  return (
    (haversineMeters(drone, {
      lat: drone.homeLat,
      lng: drone.homeLng,
    }) /
      1000) *
    drone.consumptionPerKm
  );
}

function sendHome(drone: Drone): Drone {
  return {
    ...drone,
    status: 'returning',
    phase: 'return',
    route: [],
    routeIndex: 0,
    dwellRemainingSec: 0,
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
      const stagingOffset = (index - (config.droneCount - 1) / 2) * 0.00007;
      return {
        id: `UAV-${String(index + 1).padStart(2, '0')}`,
        model: profile.model,
        color: COLORS[index % COLORS.length],
        lat: BASE.lat + stagingOffset,
        lng: BASE.lng + stagingOffset * 0.7,
        homeLat: BASE.lat,
        homeLng: BASE.lng,
        speedMps: profile.speedMps,
        turnRateDps: profile.turnRateDps,
        battery: profile.maxBattery,
        reserveBattery: profile.reserveBattery,
        consumptionPerKm: profile.consumptionPerKm,
        loiterConsumptionPerMin: profile.loiterConsumptionPerMin,
        maxBattery: profile.maxBattery,
        turnaroundSec: profile.turnaroundSec,
        turnaroundRemainingSec: 0,
        dwellRemainingSec: 0,
        phase: 'base',
        sortieCount: 0,
        minimumBattery: profile.maxBattery,
        status: 'ready',
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
        revisitSec: 120 + Math.floor(random() * 180),
        dwellSec: 20 + Math.floor(random() * 31),
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
    base: { ...BASE },
    noFlyZones,
    events: [
      makeEvent(
        0,
        'system',
        '가상 임무 준비',
        `${config.droneCount}대 기지 대기 · 정찰지점 ${config.waypointCount}개`,
      ),
    ],
    weightedGapSeconds: 0,
    continuityIntegral: 0,
    continuityObservationSeconds: 0,
    postFailureContinuityIntegral: 0,
    postFailureObservationSeconds: 0,
    minimumPostFailureContinuity: 100,
    revisitChecks: 0,
    onTimeRevisits: 0,
    returnCount: 0,
    reserveViolations: 0,
    failureTime: null,
    orphanedWaypointIds: [],
    recoveredAt: null,
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
    (drone) => drone.id === droneId && drone.status !== 'failed',
  );
  if (!failed) return state;
  const orphanedWaypointIds = [
    ...new Set([
      ...failed.route.slice(failed.routeIndex),
      ...state.waypoints
        .filter((waypoint) => waypoint.assignedDrone === droneId)
        .map((waypoint) => waypoint.id),
    ]),
  ];
  const drones = state.drones.map((drone) =>
    drone.id === droneId
      ? {
          ...drone,
          status: 'failed' as const,
          route: [],
          routeIndex: 0,
          dwellRemainingSec: 0,
        }
      : drone,
  );
  const waypoints = state.waypoints.map((waypoint) => ({
    ...waypoint,
    assignedDrone: undefined,
  }));
  const planned = assignRoutes(drones, waypoints, state.time, planner, policy);
  const inferenceMs = Math.max(0.1, performance.now() - start);
  const currentContinuity = continuityAt(state.time, state.waypoints);
  return {
    ...state,
    failureTriggered: true,
    failureTime: state.failureTime ?? state.time,
    orphanedWaypointIds,
    recoveredAt: orphanedWaypointIds.length ? null : state.time,
    minimumPostFailureContinuity: Math.min(
      state.minimumPostFailureContinuity,
      currentContinuity,
    ),
    drones: planned.drones,
    waypoints: planned.waypoints,
    replanCount: state.replanCount + 1,
    inferenceMs,
    events: [
      makeEvent(
        state.time,
        'replan',
        '잔여 기체 경로 재계산',
        `${orphanedWaypointIds.length}개 담당지점 재배정 · ${inferenceMs.toFixed(1)} ms`,
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
      ? {
          ...drone,
          lat: drone.homeLat,
          lng: drone.homeLng,
          battery: drone.maxBattery,
          status: 'ready' as const,
          phase: 'base' as const,
          route: [],
          routeIndex: 0,
          dwellRemainingSec: 0,
          turnaroundRemainingSec: 0,
        }
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
    drones: planned.drones,
    waypoints: planned.waypoints,
    replanCount: state.replanCount + 1,
    events: [
      makeEvent(
        state.time,
        'system',
        `${droneId} 기지 복구`,
        '배터리 교체 완료 · 다음 출격 경로 배정',
      ),
      ...state.events,
    ].slice(0, 18),
  };
}

export function advanceFieldMission(
  state: MissionState,
  deltaSeconds: number,
): MissionState {
  if (!state.running || state.completed) return state;
  const time = state.time + deltaSeconds;
  const continuity = continuityAt(time, state.waypoints);
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
    continuityIntegral: state.continuityIntegral + continuity * deltaSeconds,
    continuityObservationSeconds:
      state.continuityObservationSeconds + deltaSeconds,
    postFailureContinuityIntegral:
      state.postFailureContinuityIntegral +
      (state.failureTime === null ? 0 : continuity * deltaSeconds),
    postFailureObservationSeconds:
      state.postFailureObservationSeconds +
      (state.failureTime === null ? 0 : deltaSeconds),
    minimumPostFailureContinuity:
      state.failureTime === null
        ? state.minimumPostFailureContinuity
        : Math.min(state.minimumPostFailureContinuity, continuity),
  };
}

export function advanceMission(
  state: MissionState,
  deltaSeconds: number,
  config: ScenarioConfig,
  policy: CandidateDqn,
): MissionState {
  if (!state.running || state.completed) return state;
  let next: MissionState = { ...state, time: state.time + deltaSeconds };
  if (!state.failureTriggered && next.time >= config.failureAt) {
    next = triggerFailure(
      next,
      config.failureDroneId,
      config.planner,
      policy,
      '설정된 이탈 시점',
    );
  }

  const waypointMap = new Map(
    next.waypoints.map((waypoint) => [waypoint.id, { ...waypoint }]),
  );
  const tickEvents: MissionEvent[] = [];
  let returnCount = next.returnCount;
  let reserveViolations = next.reserveViolations;
  let revisitChecks = next.revisitChecks;
  let onTimeRevisits = next.onTimeRevisits;

  let drones = next.drones.map((drone): Drone => {
    if (drone.status === 'failed') return { ...drone };

    if (drone.status === 'ready') {
      const remaining = Math.max(
        0,
        drone.turnaroundRemainingSec - deltaSeconds,
      );
      if (remaining > 0) {
        return { ...drone, turnaroundRemainingSec: remaining, phase: 'base' };
      }
      const replenished = {
        ...drone,
        lat: drone.homeLat,
        lng: drone.homeLng,
        battery: drone.maxBattery,
        turnaroundRemainingSec: 0,
        phase: 'base' as const,
      };
      if (drone.turnaroundRemainingSec > 0) {
        tickEvents.push(
          makeEvent(
            next.time,
            'system',
            `${drone.id} 출격 준비 완료`,
            `배터리 ${drone.maxBattery.toFixed(0)}%`,
          ),
        );
      }
      if (replenished.routeIndex < replenished.route.length) {
        tickEvents.push(
          makeEvent(
            next.time,
            'system',
            `${drone.id} 기지 출격`,
            `목표 ${replenished.route[replenished.routeIndex]} · ${replenished.sortieCount + 1}차 출격`,
          ),
        );
        return {
          ...replenished,
          status: 'active',
          phase: 'transit',
          sortieCount: replenished.sortieCount + 1,
        };
      }
      return replenished;
    }

    if (drone.status === 'returning') {
      const home = { lat: drone.homeLat, lng: drone.homeLng };
      const travel = drone.speedMps * deltaSeconds;
      const distance = haversineMeters(drone, home);
      const moved = Math.min(travel, distance);
      const position = moveToward(drone, home, moved);
      const battery = Math.max(
        0,
        drone.battery - (moved / 1000) * drone.consumptionPerKm,
      );
      const updated: Drone = {
        ...drone,
        ...position,
        heading: headingTo(drone, home),
        battery,
        minimumBattery: Math.min(drone.minimumBattery, battery),
        totalDistanceM: drone.totalDistanceM + moved,
      };
      if (distance <= travel + 1) {
        returnCount += 1;
        tickEvents.push(
          makeEvent(
            next.time,
            'system',
            `${drone.id} 기지 도착`,
            `잔여 ${battery.toFixed(0)}% · 배터리 교체 ${drone.turnaroundSec}초`,
          ),
        );
        return {
          ...updated,
          ...home,
          status: 'ready',
          phase: 'base',
          route: [],
          routeIndex: 0,
          turnaroundRemainingSec: drone.turnaroundSec,
        };
      }
      return updated;
    }

    const target = waypointMap.get(drone.route[drone.routeIndex]);
    if (!target) {
      return { ...drone, phase: 'transit' };
    }

    if (drone.phase === 'dwell') {
      const used = (drone.loiterConsumptionPerMin * deltaSeconds) / 60;
      const battery = Math.max(0, drone.battery - used);
      const remaining = Math.max(0, drone.dwellRemainingSec - deltaSeconds);
      const updated: Drone = {
        ...drone,
        battery,
        minimumBattery: Math.min(drone.minimumBattery, battery),
        dwellRemainingSec: remaining,
      };
      if (remaining <= 0) {
        const age = next.time - target.lastVisited;
        if (target.visitedCount > 0) {
          revisitChecks += 1;
          if (age <= target.revisitSec) onTimeRevisits += 1;
        }
        target.lastVisited = next.time;
        target.visitedCount += 1;
        waypointMap.set(target.id, target);
        tickEvents.push(
          makeEvent(
            next.time,
            'visit',
            `${drone.id} ${target.id} 정찰 완료`,
            `${target.dwellSec}초 체공 · 방문 ${target.visitedCount}회`,
          ),
        );
        return {
          ...updated,
          phase: 'transit',
          dwellRemainingSec: 0,
          routeIndex: drone.routeIndex + 1,
        };
      }
      return updated;
    }

    const required = estimateSortieEnergy(drone, target);
    if (drone.battery - required < drone.reserveBattery) {
      if (drone.battery - returnEnergy(drone) < drone.reserveBattery) {
        reserveViolations += 1;
      }
      tickEvents.push(
        makeEvent(
          next.time,
          'warning',
          `${drone.id} 기지 복귀`,
          `목표 정찰·귀환 후 예비전력 확보 불가 · 잔여 ${drone.battery.toFixed(0)}%`,
        ),
      );
      return sendHome(drone);
    }

    const travel = drone.speedMps * deltaSeconds;
    const distance = haversineMeters(drone, target);
    const moved = Math.min(travel, distance);
    const position = moveToward(drone, target, moved);
    const battery = Math.max(
      0,
      drone.battery - (moved / 1000) * drone.consumptionPerKm,
    );
    const updated: Drone = {
      ...drone,
      ...position,
      heading: headingTo(drone, target),
      battery,
      minimumBattery: Math.min(drone.minimumBattery, battery),
      totalDistanceM: drone.totalDistanceM + moved,
    };
    if (distance <= travel + 1) {
      tickEvents.push(
        makeEvent(
          next.time,
          'system',
          `${drone.id} ${target.id} 정찰 시작`,
          `${target.dwellSec}초 체공`,
        ),
      );
      return {
        ...updated,
        phase: 'dwell',
        dwellRemainingSec: target.dwellSec,
      };
    }
    return updated;
  });

  let waypoints = [...waypointMap.values()];
  const needsPlan = drones.some(
    (drone) =>
      (drone.status === 'active' &&
        drone.phase !== 'dwell' &&
        drone.routeIndex >= drone.route.length) ||
      (drone.status === 'ready' &&
        drone.turnaroundRemainingSec <= 0 &&
        drone.routeIndex >= drone.route.length),
  );

  if (needsPlan && waypoints.length) {
    const start = performance.now();
    const planned = assignRoutes(
      drones,
      waypoints,
      next.time,
      config.planner,
      policy,
    );
    const activeAtBase = (drone: Drone) =>
      haversineMeters(drone, {
        lat: drone.homeLat,
        lng: drone.homeLng,
      }) < 3;
    drones = planned.drones.map((drone) => {
      if (
        drone.status === 'active' &&
        drone.route.length === 0 &&
        !activeAtBase(drone)
      ) {
        return sendHome(drone);
      }
      if (
        drone.status === 'active' &&
        drone.route.length === 0 &&
        activeAtBase(drone)
      ) {
        return { ...drone, status: 'ready', phase: 'base' };
      }
      if (
        drone.status === 'active' &&
        drone.phase !== 'dwell' &&
        drone.route.length > 0
      ) {
        return { ...drone, phase: 'transit' };
      }
      return drone;
    });
    waypoints = planned.waypoints;
    next = {
      ...next,
      replanCount: next.replanCount + 1,
      inferenceMs: Math.max(0.1, performance.now() - start),
    };
  }

  const continuity = continuityAt(next.time, waypoints);
  const weightedGapRate = waypoints.reduce(
    (sum, waypoint) =>
      sum +
      (next.time - waypoint.lastVisited > waypoint.revisitSec
        ? waypoint.priority
        : 0),
    0,
  );
  let recoveredAt = next.recoveredAt;
  if (
    recoveredAt === null &&
    next.failureTime !== null &&
    next.orphanedWaypointIds.length > 0 &&
    next.orphanedWaypointIds.every(
      (id) => (waypointMap.get(id)?.lastVisited ?? -1) >= next.failureTime!,
    )
  ) {
    recoveredAt = next.time;
    tickEvents.push(
      makeEvent(
        next.time,
        'system',
        '이탈 기체 담당구역 회복',
        `${(next.time - next.failureTime).toFixed(0)}초`,
      ),
    );
  }

  const noAvailableDrones = drones.every((drone) => drone.status === 'failed');
  const completed = next.time >= config.durationSec || noAvailableDrones;
  if (completed && !next.completed) {
    tickEvents.push(
      makeEvent(
        Math.min(next.time, config.durationSec),
        'system',
        '가상 임무 종료',
        noAvailableDrones ? '운용 가능 기체 없음' : '설정 임무시간 도달',
      ),
    );
  }

  return {
    ...next,
    time: Math.min(next.time, config.durationSec),
    drones,
    waypoints,
    completed,
    running: completed ? false : next.running,
    weightedGapSeconds:
      next.weightedGapSeconds + weightedGapRate * deltaSeconds,
    continuityIntegral: next.continuityIntegral + continuity * deltaSeconds,
    continuityObservationSeconds:
      next.continuityObservationSeconds + deltaSeconds,
    postFailureContinuityIntegral:
      next.postFailureContinuityIntegral +
      (next.failureTime === null ? 0 : continuity * deltaSeconds),
    postFailureObservationSeconds:
      next.postFailureObservationSeconds +
      (next.failureTime === null ? 0 : deltaSeconds),
    minimumPostFailureContinuity:
      next.failureTime === null
        ? next.minimumPostFailureContinuity
        : Math.min(next.minimumPostFailureContinuity, continuity),
    revisitChecks,
    onTimeRevisits,
    returnCount,
    reserveViolations,
    recoveredAt,
    events: [...tickEvents.reverse(), ...next.events].slice(0, 18),
  };
}

export function missionMetrics(state: MissionState): MissionMetrics {
  const visited = state.waypoints.filter(
    (waypoint) => waypoint.visitedCount > 0,
  ).length;
  const continuity = continuityAt(state.time, state.waypoints);
  const averageContinuity = state.continuityObservationSeconds
    ? state.continuityIntegral / state.continuityObservationSeconds
    : continuity;
  const operationalDrones = state.drones.filter(
    (drone) => drone.status !== 'failed',
  );
  const priorityTime =
    state.waypoints.reduce((sum, waypoint) => sum + waypoint.priority, 0) *
    Math.max(1, state.time);
  return {
    continuity,
    averageContinuity,
    postFailureAverageContinuity: state.postFailureObservationSeconds
      ? state.postFailureContinuityIntegral /
        state.postFailureObservationSeconds
      : null,
    minimumPostFailureContinuity:
      state.failureTime === null
        ? continuity
        : state.minimumPostFailureContinuity,
    revisitCompliance: Math.max(
      0,
      100 * (1 - state.weightedGapSeconds / Math.max(1, priorityTime)),
    ),
    coverage: state.waypoints.length
      ? (visited / state.waypoints.length) * 100
      : 0,
    activeDrones: state.drones.filter(
      (drone) => drone.status === 'active' || drone.status === 'returning',
    ).length,
    weightedGapSeconds: state.weightedGapSeconds,
    recoverySeconds:
      state.failureTime !== null && state.recoveredAt !== null
        ? state.recoveredAt - state.failureTime
        : null,
    totalDistanceKm:
      state.drones.reduce((sum, drone) => sum + drone.totalDistanceM, 0) / 1000,
    averageBattery: operationalDrones.length
      ? operationalDrones.reduce((sum, drone) => sum + drone.battery, 0) /
        operationalDrones.length
      : 0,
    minimumBattery: operationalDrones.length
      ? Math.min(...operationalDrones.map((drone) => drone.minimumBattery))
      : 0,
    returnCount: state.returnCount,
    reserveViolations: state.reserveViolations,
    sortieCount: state.drones.reduce(
      (sum, drone) => sum + drone.sortieCount,
      0,
    ),
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
  for (const drone of drones.filter(
    (item) => item.status === 'active' || item.status === 'ready',
  )) {
    let position = { lat: drone.lat, lng: drone.lng };
    let elapsed = 0;
    let battery = drone.battery;
    for (const waypointId of drone.route) {
      const waypoint = waypoints.find((item) => item.id === waypointId);
      if (!waypoint) continue;
      const segment = haversineMeters(position, waypoint);
      const transitUse = (segment / 1000) * drone.consumptionPerKm;
      const dwellUse = (waypoint.dwellSec / 60) * drone.loiterConsumptionPerMin;
      const homeUse =
        (haversineMeters(waypoint, {
          lat: drone.homeLat,
          lng: drone.homeLng,
        }) /
          1000) *
        drone.consumptionPerKm;
      if (battery - transitUse - dwellUse - homeUse < drone.reserveBattery)
        break;
      elapsed += segment / drone.speedMps + waypoint.dwellSec;
      weightedGap +=
        Math.max(0, elapsed - waypoint.revisitSec) * waypoint.priority;
      distanceM += segment;
      battery -= transitUse + dwellUse;
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
    { kind: 'rl', label: '다중 에이전트 DQN' },
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

export function cloneMissionState(state: MissionState): MissionState {
  return {
    ...state,
    base: { ...state.base },
    drones: state.drones.map((drone) => ({
      ...drone,
      route: [...drone.route],
    })),
    waypoints: state.waypoints.map((waypoint) => ({ ...waypoint })),
    noFlyZones: state.noFlyZones.map((zone) => ({
      ...zone,
      points: zone.points.map((point) => ({ ...point })),
    })),
    orphanedWaypointIds: [...state.orphanedWaypointIds],
    events: [...state.events],
  };
}

export function runScenarioComparison(
  initialState: MissionState,
  config: ScenarioConfig,
  policy: CandidateDqn,
  durationSeconds = config.durationSec,
): ScenarioComparisonResult[] {
  const planners: { kind: PlannerKind; label: string }[] = [
    { kind: 'nearest', label: '최근접 우선' },
    { kind: 'priority', label: '중요도 우선' },
    { kind: 'rl', label: '다중 에이전트 DQN' },
  ];
  const validFailureDroneId =
    initialState.drones.find((drone) => drone.id === config.failureDroneId)
      ?.id ??
    initialState.drones.find((drone) => drone.status !== 'failed')?.id;

  return planners.map(({ kind, label }) => {
    const base = cloneMissionState(initialState);
    const drones = base.drones.map((drone) => ({
      ...drone,
      status:
        drone.status === 'failed'
          ? ('failed' as const)
          : drone.phase === 'base'
            ? ('ready' as const)
            : ('active' as const),
      route: [],
      routeIndex: 0,
      dwellRemainingSec: 0,
      turnaroundRemainingSec: 0,
    }));
    const waypoints = base.waypoints.map((waypoint) => ({
      ...waypoint,
      assignedDrone: undefined,
    }));
    const plannerConfig: ScenarioConfig = {
      ...config,
      durationSec: base.time + durationSeconds,
      planner: kind,
      ...(validFailureDroneId ? { failureDroneId: validFailureDroneId } : {}),
    };
    const planned = assignRoutes(drones, waypoints, base.time, kind, policy);
    let state: MissionState = {
      ...base,
      running: true,
      completed: false,
      failureTriggered: false,
      drones: planned.drones,
      waypoints: planned.waypoints,
      events: [],
      weightedGapSeconds: 0,
      continuityIntegral: 0,
      continuityObservationSeconds: 0,
      postFailureContinuityIntegral: 0,
      postFailureObservationSeconds: 0,
      minimumPostFailureContinuity: 100,
      revisitChecks: 0,
      onTimeRevisits: 0,
      returnCount: 0,
      reserveViolations: 0,
      failureTime: null,
      orphanedWaypointIds: [],
      recoveredAt: null,
      replanCount: 0,
      inferenceMs: 0,
    };
    const endTime = state.time + durationSeconds;
    while (state.running && !state.completed && state.time < endTime) {
      state = advanceMission(
        state,
        Math.min(1, endTime - state.time),
        plannerConfig,
        policy,
      );
    }
    const metrics = missionMetrics(state);
    return {
      planner: kind,
      label,
      continuity:
        metrics.postFailureAverageContinuity ?? metrics.averageContinuity,
      revisitCompliance: metrics.revisitCompliance,
      minimumPostFailureContinuity: metrics.minimumPostFailureContinuity,
      coverage: metrics.coverage,
      weightedGapSeconds: metrics.weightedGapSeconds,
      recoverySeconds: metrics.recoverySeconds,
      distanceKm: metrics.totalDistanceKm,
      averageBattery: metrics.averageBattery,
      returnCount: metrics.returnCount,
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
