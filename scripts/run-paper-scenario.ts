import { assignRoutes, CandidateDqn } from '../lib/skygrid/rl-policy';
import { PRETRAINED_POLICY_WEIGHTS } from '../lib/skygrid/pretrained-policy';
import {
  advanceMission,
  createMission,
  missionMetrics,
  triggerFailure,
} from '../lib/skygrid/simulation';
import type {
  MissionMetrics,
  MissionState,
  PlannerKind,
  ScenarioConfig,
} from '../lib/skygrid/types';

const RUNS = Number(process.argv[2] ?? 100);
const BASE_SEED = Number(process.argv[3] ?? 20_261_125);
const STEP_SECONDS = 10;
const FIRST_FAILURE_SECONDS = 20 * 60;
const SECOND_FAILURE_SECONDS = 40 * 60;
const DURATION_SECONDS = 60 * 60;
const GENERATOR_CENTER = { lat: 36.35, lng: 127.85 };
const KAIST_BASE = { lat: 36.3722, lng: 127.3604 };
const CAMPUS_SCALE = 0.25;

const policy = CandidateDqn.fromWeights(PRETRAINED_POLICY_WEIGHTS);
if (!policy) throw new Error('사전학습 DQN 모델을 불러오지 못했습니다.');

const planners: PlannerKind[] = ['nearest', 'priority', 'rl'];

type RunResult = {
  planner: PlannerKind;
  seed: number;
  metrics: MissionMetrics;
  recoverySeconds: number[];
  recoveryRate: number;
};

function remainingRouteLength(state: MissionState, droneId: string): number {
  const drone = state.drones.find((candidate) => candidate.id === droneId);
  return drone ? Math.max(0, drone.route.length - drone.routeIndex) : -1;
}

function chooseFailureDrone(
  state: MissionState,
  preferredId: string,
): string {
  const preferred = state.drones.find(
    (drone) => drone.id === preferredId && drone.status !== 'failed',
  );
  if (preferred && remainingRouteLength(state, preferred.id) > 0) {
    return preferred.id;
  }
  return (
    state.drones
      .filter((drone) => drone.status !== 'failed')
      .sort(
        (a, b) =>
          remainingRouteLength(state, b.id) - remainingRouteLength(state, a.id) ||
          a.id.localeCompare(b.id),
      )[0]?.id ?? preferredId
  );
}

function runMission(
  planner: PlannerKind,
  seed: number,
): RunResult {
  const config: ScenarioConfig = {
    droneCount: 4,
    waypointCount: 6,
    durationSec: DURATION_SECONDS,
    // 두 이탈은 아래에서 같은 시각에 직접 적용한다.
    failureAt: DURATION_SECONDS + STEP_SECONDS,
    failureDroneId: 'UAV-02',
    planner,
    simRate: 20,
    sensorRadiusM: 20,
    randomSeed: seed,
    layout: 'mixed',
    mixedFleet: true,
    noFlyZoneCount: 0,
  };
  const generated = createMission(config, policy);
  const waypoints = generated.waypoints.map((waypoint) => ({
    ...waypoint,
    lat:
      KAIST_BASE.lat +
      (waypoint.lat - GENERATOR_CENTER.lat) * CAMPUS_SCALE,
    lng:
      KAIST_BASE.lng +
      (waypoint.lng - GENERATOR_CENTER.lng) * CAMPUS_SCALE,
    assignedDrone: undefined,
  }));
  const drones = generated.drones.map((drone, index) => {
    const offset = (index - (generated.drones.length - 1) / 2) * 0.000025;
    return {
      ...drone,
      lat: KAIST_BASE.lat + offset,
      lng: KAIST_BASE.lng + offset * 0.7,
      homeLat: KAIST_BASE.lat,
      homeLng: KAIST_BASE.lng,
      route: [],
      routeIndex: 0,
    };
  });
  const planned = assignRoutes(
    drones,
    waypoints,
    0,
    planner,
    policy,
    [],
  );
  let state: MissionState = {
    ...generated,
    base: {
      id: 'BASE-01',
      name: 'KAIST 시험기지',
      ...KAIST_BASE,
      configured: true,
    },
    noFlyZones: [],
    drones: planned.drones,
    waypoints: planned.waypoints,
    running: true,
  };
  const failures = [
    { time: FIRST_FAILURE_SECONDS, preferredId: 'UAV-02' },
    { time: SECOND_FAILURE_SECONDS, preferredId: 'UAV-04' },
  ];
  const recoveries: {
    time: number;
    waypointIds: string[];
    recoveredAt: number | null;
  }[] = [];
  let failureIndex = 0;

  while (state.running && !state.completed) {
    const delta = Math.min(STEP_SECONDS, DURATION_SECONDS - state.time);
    state = advanceMission(state, delta, config, policy);

    if (
      failureIndex < failures.length &&
      state.time >= failures[failureIndex].time
    ) {
      const failure = failures[failureIndex];
      const droneId = chooseFailureDrone(state, failure.preferredId);
      state = triggerFailure(
        state,
        droneId,
        planner,
        policy,
        `${failure.time / 60}분 모의 이탈`,
      );
      recoveries.push({
        time: failure.time,
        waypointIds: [...state.orphanedWaypointIds],
        recoveredAt: state.orphanedWaypointIds.length ? null : failure.time,
      });
      failureIndex += 1;
    }

    for (const recovery of recoveries) {
      if (recovery.recoveredAt !== null) continue;
      const recovered = recovery.waypointIds.every((waypointId) => {
        const waypoint = state.waypoints.find(
          (candidate) => candidate.id === waypointId,
        );
        return waypoint !== undefined && waypoint.lastVisited >= recovery.time;
      });
      if (recovered) recovery.recoveredAt = state.time;
    }
  }

  const recoveredSeconds = recoveries
    .filter(
      (recovery): recovery is typeof recovery & { recoveredAt: number } =>
        recovery.recoveredAt !== null,
    )
    .map((recovery) => recovery.recoveredAt - recovery.time);

  return {
    planner,
    seed,
    metrics: missionMetrics(state),
    recoverySeconds: recoveredSeconds,
    recoveryRate: recoveries.length
      ? (recoveredSeconds.length / recoveries.length) * 100
      : 100,
  };
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleSd(values: number[]): number {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - average) ** 2, 0) /
      (values.length - 1),
  );
}

function summary(values: number[]) {
  return {
    mean: mean(values),
    sd: sampleSd(values),
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

const results: RunResult[] = [];
for (let run = 0; run < RUNS; run += 1) {
  const seed = BASE_SEED + run * 17;
  for (const planner of planners) results.push(runMission(planner, seed));
}

const byPlanner = Object.fromEntries(
  planners.map((planner) => {
    const rows = results.filter((result) => result.planner === planner);
    const recoveryValues = rows.flatMap((row) => row.recoverySeconds);
    return [
      planner,
      {
        continuity: summary(
          rows.map(
            (row) =>
              row.metrics.postFailureAverageContinuity ??
              row.metrics.averageContinuity,
          ),
        ),
        weightedGapSeconds: summary(
          rows.map((row) => row.metrics.weightedGapSeconds),
        ),
        recoverySeconds: summary(recoveryValues),
        distanceKm: summary(rows.map((row) => row.metrics.totalDistanceKm)),
        reserveViolations: summary(
          rows.map((row) => row.metrics.reserveViolations),
        ),
        recoveryRate: mean(rows.map((row) => row.recoveryRate)),
        completedVisits: summary(
          rows.map((row) => row.metrics.completedVisits),
        ),
      },
    ];
  }),
);

const paired = Object.fromEntries(
  planners.map((planner) => [
    planner,
    results
      .filter((result) => result.planner === planner)
      .sort((a, b) => a.seed - b.seed)
      .map((result) => ({
        seed: result.seed,
        continuity:
          result.metrics.postFailureAverageContinuity ??
          result.metrics.averageContinuity,
        weightedGapSeconds: result.metrics.weightedGapSeconds,
        recoverySeconds: result.recoverySeconds.length
          ? mean(result.recoverySeconds)
          : null,
        distanceKm: result.metrics.totalDistanceKm,
        reserveViolations: result.metrics.reserveViolations,
      })),
  ]),
);

process.stdout.write(
  JSON.stringify(
    {
      settings: {
        runs: RUNS,
        baseSeed: BASE_SEED,
        seedIncrement: 17,
        durationSeconds: DURATION_SECONDS,
        failureSeconds: [FIRST_FAILURE_SECONDS, SECOND_FAILURE_SECONDS],
        stepSeconds: STEP_SECONDS,
        droneCount: 4,
        waypointCount: 6,
        mixedFleet: true,
        noFlyZoneCount: 0,
        campusCenter: KAIST_BASE,
        campusScale: CAMPUS_SCALE,
        dwellSeconds: [20, 80],
        revisitSeconds: [120, 300],
        initialBatteryPercent: 100,
        reserveBatteryPercent: 20,
      },
      byPlanner,
      paired,
    },
    null,
    2,
  ),
);
