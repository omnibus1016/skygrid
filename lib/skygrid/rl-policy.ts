import {
  DRONE_PROFILES,
  getDroneProfile,
  profileSimulationValues,
} from './drone-profiles';
import { haversineMeters } from './geo';
import { mulberry32 } from './random';
import type {
  Drone,
  GeoPoint,
  PlannerKind,
  PolicyStats,
  Waypoint,
} from './types';

/*
 * The policy is a centralized candidate-value network. One action chooses a
 * (drone, waypoint) pair for the fleet. Training and inference share the same
 * geographic, battery, dwell, revisit and return-to-base calculations.
 */
const MODEL_VERSION = 2 as const;
const INPUTS = 16;
const HIDDEN = 36;
const DISTANCE_SCALE_KM = 12;
const NETWORK_SCORE_WEIGHT = 1.5;

type NextCandidate = {
  features: number[];
  prior: number;
};

type Experience = {
  features: number[];
  reward: number;
  next: NextCandidate[];
  done: boolean;
};

type CandidateContext = {
  features: number[];
  distanceM: number;
  transitSeconds: number;
  visitSeconds: number;
  projectedAgeRatio: number;
  expectedBatteryUse: number;
  reserveMargin: number;
  relativeAdvantage: number;
  localDensity: number;
};

type TrainingScenario = {
  drones: Drone[];
  tasks: Waypoint[];
  elapsedByDrone: number[];
  missionTime: number;
};

type TrainingCandidate = CandidateContext & {
  droneIndex: number;
  taskIndex: number;
  prior: number;
};

export interface PolicyTrainingContext {
  base: GeoPoint;
  drones: Drone[];
  waypoints: Waypoint[];
  missionTime: number;
  durationSec: number;
  failureAt: number;
}

export interface TrainingProgress {
  episode: number;
  totalEpisodes: number;
  reward: number;
  stage: 'training' | 'validation';
}

export interface PolicyWeights {
  version: typeof MODEL_VERSION;
  inputs: number;
  hidden: number;
  w1: number[][];
  b1: number[];
  w2: number[];
  b2: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function mean(values: number[]): number {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function candidateFeatureVector({
  distanceKm,
  priority,
  urgency,
  battery,
  expectedBatteryUse,
  usableBattery,
  remaining,
  overdue,
  teamSize,
  averageBattery,
  relativeAdvantage,
  localDensity,
  reserveMargin,
  transitRatio,
  dwellRatio,
  returnDistanceKm,
}: {
  distanceKm: number;
  priority: number;
  urgency: number;
  battery: number;
  expectedBatteryUse: number;
  usableBattery: number;
  remaining: number;
  overdue: boolean;
  teamSize: number;
  averageBattery: number;
  relativeAdvantage: number;
  localDensity: number;
  reserveMargin: number;
  transitRatio: number;
  dwellRatio: number;
  returnDistanceKm: number;
}): number[] {
  return [
    1,
    clamp(distanceKm / DISTANCE_SCALE_KM, 0, 1),
    clamp(priority / 5, 0, 1),
    clamp(urgency / 2, 0, 1),
    clamp(battery, 0, 1),
    clamp(expectedBatteryUse / Math.max(0.05, usableBattery), 0, 2) / 2,
    clamp(remaining / 24, 0, 1),
    overdue ? 1 : 0,
    clamp(teamSize / 8, 0, 1),
    clamp(averageBattery, 0, 1),
    clamp(relativeAdvantage, 0, 1),
    clamp(localDensity / 6, 0, 1),
    clamp(reserveMargin, -0.25, 1) / 1.25 + 0.2,
    clamp(transitRatio / 2, 0, 1),
    clamp(dwellRatio, 0, 1),
    clamp(returnDistanceKm / DISTANCE_SCALE_KM, 0, 1),
  ];
}

export class CandidateDqn {
  private w1: number[][];
  private b1: number[];
  private w2: number[];
  private b2: number;

  constructor(random = mulberry32(41)) {
    const scale = Math.sqrt(2 / INPUTS);
    this.w1 = Array.from({ length: HIDDEN }, () =>
      Array.from({ length: INPUTS }, () => (random() - 0.5) * scale),
    );
    this.b1 = Array(HIDDEN).fill(0);
    this.w2 = Array(HIDDEN).fill(0);
    this.b2 = 0;
  }

  clone(): CandidateDqn {
    const copy = new CandidateDqn(() => 0.5);
    copy.w1 = this.w1.map((row) => [...row]);
    copy.b1 = [...this.b1];
    copy.w2 = [...this.w2];
    copy.b2 = this.b2;
    return copy;
  }

  toWeights(): PolicyWeights {
    return {
      version: MODEL_VERSION,
      inputs: INPUTS,
      hidden: HIDDEN,
      w1: this.w1.map((row) => [...row]),
      b1: [...this.b1],
      w2: [...this.w2],
      b2: this.b2,
    };
  }

  static fromWeights(value: unknown): CandidateDqn | null {
    if (!value || typeof value !== 'object') return null;
    const weights = value as Partial<PolicyWeights>;
    if (
      weights.version !== MODEL_VERSION ||
      weights.inputs !== INPUTS ||
      weights.hidden !== HIDDEN ||
      !Array.isArray(weights.w1) ||
      !Array.isArray(weights.b1) ||
      !Array.isArray(weights.w2) ||
      typeof weights.b2 !== 'number' ||
      weights.w1.length !== HIDDEN ||
      weights.b1.length !== HIDDEN ||
      weights.w2.length !== HIDDEN ||
      weights.w1.some(
        (row) =>
          !Array.isArray(row) ||
          row.length !== INPUTS ||
          row.some((weight) => !Number.isFinite(weight)),
      ) ||
      weights.b1.some((bias) => !Number.isFinite(bias)) ||
      weights.w2.some((weight) => !Number.isFinite(weight)) ||
      !Number.isFinite(weights.b2)
    )
      return null;

    const policy = new CandidateDqn(() => 0.5);
    policy.w1 = weights.w1.map((row) => [...row]);
    policy.b1 = [...weights.b1];
    policy.w2 = [...weights.w2];
    policy.b2 = weights.b2;
    return policy;
  }

  predict(features: number[]): number {
    const hidden = this.w1.map((row, index) =>
      Math.max(
        0,
        row.reduce(
          (sum, weight, input) => sum + weight * features[input],
          this.b1[index],
        ),
      ),
    );
    return hidden.reduce(
      (sum, value, index) => sum + value * this.w2[index],
      this.b2,
    );
  }

  train(features: number[], target: number, learningRate: number): number {
    const preActivation = this.w1.map((row, index) =>
      row.reduce(
        (sum, weight, input) => sum + weight * features[input],
        this.b1[index],
      ),
    );
    const hidden = preActivation.map((value) => Math.max(0, value));
    const prediction = hidden.reduce(
      (sum, value, index) => sum + value * this.w2[index],
      this.b2,
    );
    const error = clamp(prediction - target, -8, 8);
    const gradient = clamp(error, -1, 1);
    const previousW2 = [...this.w2];

    for (let h = 0; h < HIDDEN; h += 1)
      this.w2[h] -= learningRate * gradient * hidden[h];
    this.b2 -= learningRate * gradient;

    for (let h = 0; h < HIDDEN; h += 1) {
      if (preActivation[h] <= 0) continue;
      const hiddenGradient = clamp(gradient * previousW2[h], -1, 1);
      for (let input = 0; input < INPUTS; input += 1)
        this.w1[h][input] -= learningRate * hiddenGradient * features[input];
      this.b1[h] -= learningRate * hiddenGradient;
    }
    return Math.abs(error) <= 1 ? 0.5 * error * error : Math.abs(error) - 0.5;
  }
}

function candidateContext(
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
  team: Drone[],
  pending: Waypoint[],
): CandidateContext {
  const fleet = [
    drone,
    ...team.filter(
      (candidate) =>
        candidate.id !== drone.id &&
        (candidate.status === 'active' || candidate.status === 'ready'),
    ),
  ];
  const distanceM = haversineMeters(drone, waypoint);
  const returnDistanceM = haversineMeters(waypoint, {
    lat: drone.homeLat,
    lng: drone.homeLng,
  });
  const age = Math.max(0, missionTime - waypoint.lastVisited);
  const transitSeconds = distanceM / Math.max(1, drone.speedMps);
  const visitSeconds = transitSeconds + waypoint.dwellSec;
  const urgency = clamp(age / Math.max(1, waypoint.revisitSec), 0, 2);
  const projectedAgeRatio = clamp(
    (age + visitSeconds) / Math.max(1, waypoint.revisitSec),
    0,
    3,
  );
  const expectedBatteryUse = estimateSortieEnergy(drone, waypoint);
  const usableBattery = Math.max(5, drone.battery - drone.reserveBattery);
  const reserveMargin =
    (drone.battery - expectedBatteryUse - drone.reserveBattery) /
    Math.max(5, drone.maxBattery - drone.reserveBattery);
  const peers = fleet.slice(1);
  const nearestPeerDistance = peers.length
    ? Math.min(...peers.map((peer) => haversineMeters(peer, waypoint)))
    : distanceM * 2;
  const relativeAdvantage =
    peers.length === 0
      ? 1
      : clamp(nearestPeerDistance / Math.max(25, distanceM), 0, 2) / 2;
  const localDensity = pending.filter(
    (candidate) =>
      candidate.id !== waypoint.id &&
      haversineMeters(candidate, waypoint) < 1_000,
  ).length;
  const averageBattery =
    fleet.reduce((sum, candidate) => sum + candidate.battery / 100, 0) /
    Math.max(1, fleet.length);

  return {
    features: candidateFeatureVector({
      distanceKm: distanceM / 1_000,
      priority: waypoint.priority,
      urgency,
      battery: drone.battery / 100,
      expectedBatteryUse: expectedBatteryUse / 100,
      usableBattery: usableBattery / 100,
      remaining,
      overdue: age > waypoint.revisitSec,
      teamSize: fleet.length,
      averageBattery,
      relativeAdvantage,
      localDensity,
      reserveMargin,
      transitRatio: visitSeconds / Math.max(1, waypoint.revisitSec),
      dwellRatio: waypoint.dwellSec / Math.max(1, waypoint.revisitSec),
      returnDistanceKm: returnDistanceM / 1_000,
    }),
    distanceM,
    transitSeconds,
    visitSeconds,
    projectedAgeRatio,
    expectedBatteryUse,
    reserveMargin,
    relativeAdvantage,
    localDensity,
  };
}

function operationalPrior(
  waypoint: Waypoint,
  context: CandidateContext,
): number {
  const deadlinePressure = Math.min(1, context.projectedAgeRatio);
  const overdue = Math.max(0, context.projectedAgeRatio - 1);
  const missionValue =
    waypoint.priority * (1 + 1.4 * deadlinePressure + 2 * overdue);
  const serviceMinutes = Math.max(0.25, context.visitSeconds / 60);
  const efficientMissionValue = missionValue / (0.65 + serviceMinutes * 0.8);
  const energyCost = context.expectedBatteryUse * 0.02;
  const reservePenalty = Math.max(0, 0.2 - context.reserveMargin) * 4;
  const allocationBonus = context.relativeAdvantage * 0.8;
  const clusterBonus = Math.min(4, context.localDensity) * 0.15;
  return (
    efficientMissionValue +
    allocationBonus +
    clusterBonus -
    energyCost -
    reservePenalty
  );
}

function rlCandidateScore(
  policy: CandidateDqn,
  waypoint: Waypoint,
  context: CandidateContext,
): number {
  return (
    operationalPrior(waypoint, context) +
    policy.predict(context.features) * NETWORK_SCORE_WEIGHT
  );
}

export function missionFeatures(
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
  team: Drone[] = [],
  pending: Waypoint[] = [],
): number[] {
  return candidateContext(
    drone,
    waypoint,
    missionTime,
    remaining,
    team,
    pending,
  ).features;
}

export function estimateSortieEnergy(drone: Drone, waypoint: Waypoint): number {
  const transitEnergy =
    (haversineMeters(drone, waypoint) / 1000) * drone.consumptionPerKm;
  const dwellEnergy = (waypoint.dwellSec / 60) * drone.loiterConsumptionPerMin;
  const returnEnergy =
    (haversineMeters(waypoint, {
      lat: drone.homeLat,
      lng: drone.homeLng,
    }) /
      1000) *
    drone.consumptionPerKm;
  return transitEnergy + dwellEnergy + returnEnergy;
}

export function plannerScore(
  kind: PlannerKind,
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
  policy: CandidateDqn,
  team: Drone[] = [],
  pending: Waypoint[] = [],
): number {
  const distanceM = haversineMeters(drone, waypoint);
  const age = Math.max(0, missionTime - waypoint.lastVisited);
  if (kind === 'nearest') return -distanceM;
  if (kind === 'priority')
    return waypoint.priority * 1_000 + age * 2 - distanceM * 0.35;
  const context = candidateContext(
    drone,
    waypoint,
    missionTime,
    remaining,
    team,
    pending,
  );
  return rlCandidateScore(policy, waypoint, context);
}

export function assignRoutes(
  drones: Drone[],
  waypoints: Waypoint[],
  missionTime: number,
  kind: PlannerKind,
  policy: CandidateDqn,
): { drones: Drone[]; waypoints: Waypoint[] } {
  const active = drones
    .filter((drone) => drone.status === 'active' || drone.status === 'ready')
    .map((drone) => ({ ...drone, route: [], routeIndex: 0 }));
  const virtual: Drone[] = active.map((drone) => ({ ...drone }));
  const virtualElapsed = virtual.map(() => 0);
  const pending = waypoints.map((waypoint) => ({ ...waypoint }));

  for (let index = 0; index < virtual.length; index += 1) {
    const drone = virtual[index];
    const source = drones.find((candidate) => candidate.id === drone.id);
    if (source?.phase !== 'dwell' && source?.phase !== 'transit') continue;
    const currentId = source.route[source.routeIndex];
    const targetIndex = pending.findIndex((target) => target.id === currentId);
    if (targetIndex < 0) continue;
    const target = pending[targetIndex];
    const forwardEnergy =
      source.phase === 'dwell'
        ? (Math.max(0, source.dwellRemainingSec) / 60) *
          drone.loiterConsumptionPerMin
        : (haversineMeters(source, target) / 1000) * drone.consumptionPerKm +
          (target.dwellSec / 60) * drone.loiterConsumptionPerMin;
    const returnEnergy =
      (haversineMeters(target, {
        lat: drone.homeLat,
        lng: drone.homeLng,
      }) /
        1000) *
      drone.consumptionPerKm;
    if (drone.battery - forwardEnergy - returnEnergy < drone.reserveBattery)
      continue;
    pending.splice(targetIndex, 1);
    drone.route.push(target.id);
    drone.battery -= forwardEnergy;
    virtualElapsed[index] +=
      source.phase === 'dwell'
        ? Math.max(0, source.dwellRemainingSec)
        : haversineMeters(source, target) / Math.max(1, drone.speedMps) +
          target.dwellSec;
    drone.lat = target.lat;
    drone.lng = target.lng;
    target.assignedDrone = drone.id;
    const original = waypoints.find((waypoint) => waypoint.id === target.id);
    if (original) original.assignedDrone = drone.id;
  }

  while (pending.length && virtual.length) {
    let best: {
      droneIndex: number;
      targetIndex: number;
      score: number;
    } | null = null;
    for (let droneIndex = 0; droneIndex < virtual.length; droneIndex += 1) {
      for (
        let targetIndex = 0;
        targetIndex < pending.length;
        targetIndex += 1
      ) {
        const drone = virtual[droneIndex];
        const target = pending[targetIndex];
        const requiredEnergy = estimateSortieEnergy(drone, target);
        if (drone.battery - requiredEnergy < drone.reserveBattery) continue;
        const score = plannerScore(
          kind,
          drone,
          target,
          missionTime + virtualElapsed[droneIndex],
          pending.length,
          policy,
          virtual,
          pending,
        );
        if (!best || score > best.score)
          best = { droneIndex, targetIndex, score };
      }
    }
    if (!best) break;
    const drone = virtual[best.droneIndex];
    const [target] = pending.splice(best.targetIndex, 1);
    drone.route.push(target.id);
    const transitDistance = haversineMeters(drone, target);
    virtualElapsed[best.droneIndex] +=
      transitDistance / Math.max(1, drone.speedMps) + target.dwellSec;
    drone.battery -=
      (transitDistance / 1000) * drone.consumptionPerKm +
      (target.dwellSec / 60) * drone.loiterConsumptionPerMin;
    drone.lat = target.lat;
    drone.lng = target.lng;
    target.assignedDrone = drone.id;
    const source = waypoints.find((waypoint) => waypoint.id === target.id);
    if (source) source.assignedDrone = drone.id;
  }

  const routes = new Map(virtual.map((drone) => [drone.id, drone.route]));
  return {
    drones: drones.map((drone) =>
      routes.has(drone.id)
        ? { ...drone, route: routes.get(drone.id) ?? [], routeIndex: 0 }
        : drone,
    ),
    waypoints: waypoints.map((waypoint) => ({ ...waypoint })),
  };
}

function offsetPoint(
  center: GeoPoint,
  distanceKm: number,
  angle: number,
): GeoPoint {
  const northKm = Math.sin(angle) * distanceKm;
  const eastKm = Math.cos(angle) * distanceKm;
  return {
    lat: center.lat + northKm / 111.32,
    lng:
      center.lng +
      eastKm / Math.max(20, 111.32 * Math.cos((center.lat * Math.PI) / 180)),
  };
}

function makeTrainingDrone(
  id: string,
  profileId: string,
  base: GeoPoint,
  position: GeoPoint,
  battery: number,
): Drone {
  const performance = profileSimulationValues(getDroneProfile(profileId));
  const availableBattery = clamp(
    battery,
    performance.reserveBattery + 5,
    performance.maxBattery,
  );
  return {
    id,
    ...performance,
    color: '#67e8f9',
    ...position,
    homeLat: base.lat,
    homeLng: base.lng,
    battery: availableBattery,
    turnaroundRemainingSec: 0,
    dwellRemainingSec: 0,
    phase: 'transit',
    sortieCount: 1,
    minimumBattery: availableBattery,
    status: 'active',
    route: [],
    routeIndex: 0,
    totalDistanceM: 0,
    heading: 0,
  };
}

function createTrainingScenario(
  random: () => number,
  context?: PolicyTrainingContext,
): TrainingScenario {
  const hasCurrentSetup = Boolean(
    context && context.drones.length >= 2 && context.waypoints.length >= 3,
  );
  const useCurrentSetup = hasCurrentSetup && random() < 0.7;
  const base = context?.base ?? { lat: 36.35, lng: 127.85 };
  const contextRadiusKm = context?.waypoints.length
    ? Math.max(
        0.8,
        ...context.waypoints.map(
          (waypoint) => haversineMeters(base, waypoint) / 1_000,
        ),
      )
    : 4;
  const radiusKm = clamp(contextRadiusKm * (0.8 + random() * 0.5), 0.8, 12);
  const contextStart = Math.max(
    context?.missionTime ?? 0,
    context?.failureAt ?? 0,
  );
  const contextEnd = Math.max(contextStart + 60, context?.durationSec ?? 1_800);
  const missionTime = useCurrentSetup
    ? contextStart + random() * (contextEnd - contextStart)
    : 180 + random() * 1_500;

  let drones: Drone[];
  if (useCurrentSetup && context) {
    drones = context.drones
      .filter((drone) => drone.status !== 'failed')
      .map((source, index) => {
        const position =
          random() < 0.72
            ? offsetPoint(base, random() * radiusKm, random() * Math.PI * 2)
            : base;
        return makeTrainingDrone(
          `TRAIN-UAV-${index + 1}`,
          source.profileId,
          base,
          position,
          source.reserveBattery + 15 + random() * 65,
        );
      });
  } else {
    const initialCount = 2 + Math.floor(random() * 6);
    drones = Array.from({ length: initialCount }, (_, index) => {
      const profile =
        DRONE_PROFILES[Math.floor(random() * DRONE_PROFILES.length)];
      const position =
        random() < 0.72
          ? offsetPoint(base, random() * radiusKm, random() * Math.PI * 2)
          : base;
      return makeTrainingDrone(
        `TRAIN-UAV-${index + 1}`,
        profile.id,
        base,
        position,
        38 + random() * 62,
      );
    });
  }

  // Every episode represents a replanning state after one aircraft leaves.
  if (drones.length > 1) drones.splice(Math.floor(random() * drones.length), 1);
  if (!drones.length) {
    const profile = DRONE_PROFILES[0];
    drones.push(makeTrainingDrone('TRAIN-UAV-1', profile.id, base, base, 80));
  }

  let tasks: Waypoint[];
  if (useCurrentSetup && context) {
    tasks = context.waypoints.map((source, index) => {
      const jitter = offsetPoint(
        source,
        random() * Math.max(0.08, radiusKm * 0.12),
        random() * Math.PI * 2,
      );
      const revisitSec = clamp(
        Math.round(source.revisitSec * (0.8 + random() * 0.4)),
        60,
        1_200,
      );
      const age = random() * revisitSec * 1.7;
      return {
        ...source,
        id: `TRAIN-RP-${index + 1}`,
        ...jitter,
        priority: clamp(
          source.priority + (random() < 0.25 ? (random() < 0.5 ? -1 : 1) : 0),
          1,
          5,
        ),
        revisitSec,
        dwellSec: clamp(
          Math.round(source.dwellSec * (0.8 + random() * 0.4)),
          10,
          180,
        ),
        lastVisited: Math.max(0, missionTime - age),
        visitedCount: random() < 0.8 ? 1 : 0,
        assignedDrone: undefined,
      };
    });
  } else {
    const taskCount = 6 + Math.floor(random() * 15);
    tasks = Array.from({ length: taskCount }, (_, index) => {
      const revisitSec = 90 + Math.round(random() * 510);
      const point = offsetPoint(
        base,
        0.35 + random() * radiusKm,
        random() * Math.PI * 2,
      );
      return {
        id: `TRAIN-RP-${index + 1}`,
        ...point,
        priority: 1 + Math.floor(random() * 5),
        revisitSec,
        dwellSec: 15 + Math.round(random() * 75),
        lastVisited: Math.max(0, missionTime - random() * revisitSec * 1.7),
        visitedCount: random() < 0.8 ? 1 : 0,
      };
    });
  }

  return {
    drones,
    tasks,
    elapsedByDrone: drones.map(() => 0),
    missionTime,
  };
}

function feasibleTrainingCandidates(
  scenario: TrainingScenario,
): TrainingCandidate[] {
  const candidates: TrainingCandidate[] = [];
  for (
    let droneIndex = 0;
    droneIndex < scenario.drones.length;
    droneIndex += 1
  ) {
    const drone = scenario.drones[droneIndex];
    for (let taskIndex = 0; taskIndex < scenario.tasks.length; taskIndex += 1) {
      const task = scenario.tasks[taskIndex];
      const context = candidateContext(
        drone,
        task,
        scenario.missionTime + scenario.elapsedByDrone[droneIndex],
        scenario.tasks.length,
        scenario.drones,
        scenario.tasks,
      );
      if (drone.battery - context.expectedBatteryUse < drone.reserveBattery)
        continue;
      candidates.push({
        droneIndex,
        taskIndex,
        ...context,
        prior: operationalPrior(task, context),
      });
    }
  }
  return candidates;
}

function applyTrainingAction(
  scenario: TrainingScenario,
  selected: TrainingCandidate,
): { reward: number; quality: number; priority: number } {
  const drone = scenario.drones[selected.droneIndex];
  const task = scenario.tasks[selected.taskIndex];
  const visitTime =
    scenario.missionTime +
    scenario.elapsedByDrone[selected.droneIndex] +
    selected.visitSeconds;
  const ageAtVisit = Math.max(0, visitTime - task.lastVisited);
  const latenessRatio =
    Math.max(0, ageAtVisit - task.revisitSec) / Math.max(1, task.revisitSec);
  const timeliness = clamp(1 - latenessRatio, 0, 1);
  const energyUsed =
    (selected.distanceM / 1_000) * drone.consumptionPerKm +
    (task.dwellSec / 60) * drone.loiterConsumptionPerMin;
  const reward =
    selected.prior / 5 +
    task.priority * timeliness * 0.7 +
    selected.reserveMargin * 0.4;

  scenario.elapsedByDrone[selected.droneIndex] += selected.visitSeconds;
  drone.battery = clamp(drone.battery - energyUsed, 0, drone.maxBattery);
  drone.lat = task.lat;
  drone.lng = task.lng;
  scenario.tasks.splice(selected.taskIndex, 1);
  return {
    reward,
    quality: task.priority * (0.3 + timeliness * 0.7),
    priority: task.priority,
  };
}

function plannerTrainingScore(
  kind: PlannerKind,
  policy: CandidateDqn,
  candidate: TrainingCandidate,
  task: Waypoint,
  missionTime: number,
): number {
  if (kind === 'nearest') return -candidate.distanceM;
  if (kind === 'priority') {
    const age = Math.max(0, missionTime - task.lastVisited);
    return task.priority * 1_000 + age * 2 - candidate.distanceM * 0.35;
  }
  return (
    candidate.prior + policy.predict(candidate.features) * NETWORK_SCORE_WEIGHT
  );
}

function evaluateScenario(
  source: TrainingScenario,
  kind: PlannerKind,
  policy: CandidateDqn,
): number {
  const scenario: TrainingScenario = {
    missionTime: source.missionTime,
    elapsedByDrone: [...source.elapsedByDrone],
    drones: source.drones.map((drone) => ({ ...drone, route: [] })),
    tasks: source.tasks.map((task) => ({ ...task })),
  };
  const totalPriority = scenario.tasks.reduce(
    (sum, task) => sum + task.priority,
    0,
  );
  let quality = 0;
  let servedPriority = 0;
  while (scenario.tasks.length) {
    const candidates = feasibleTrainingCandidates(scenario);
    if (!candidates.length) break;
    const selected = candidates.reduce((best, candidate, index) => {
      const task = scenario.tasks[candidate.taskIndex];
      const bestTask = scenario.tasks[candidates[best].taskIndex];
      const score = plannerTrainingScore(
        kind,
        policy,
        candidate,
        task,
        scenario.missionTime + scenario.elapsedByDrone[candidate.droneIndex],
      );
      const bestScore = plannerTrainingScore(
        kind,
        policy,
        candidates[best],
        bestTask,
        scenario.missionTime +
          scenario.elapsedByDrone[candidates[best].droneIndex],
      );
      return score > bestScore ? index : best;
    }, 0);
    const result = applyTrainingAction(scenario, candidates[selected]);
    quality += result.quality;
    servedPriority += result.priority;
  }
  const completion = totalPriority ? servedPriority / totalPriority : 1;
  const timeliness = totalPriority ? quality / totalPriority : 1;
  return clamp((completion * 0.35 + timeliness * 0.65) * 100, 0, 100);
}

export function validatePolicy(
  policy: CandidateDqn,
  context: PolicyTrainingContext | undefined,
  seed: number,
  scenarios = 48,
): NonNullable<PolicyStats['validation']> {
  const random = mulberry32(seed);
  const scores: Record<PlannerKind, number[]> = {
    rl: [],
    nearest: [],
    priority: [],
  };
  for (let index = 0; index < scenarios; index += 1) {
    const scenario = createTrainingScenario(random, context);
    scores.nearest.push(evaluateScenario(scenario, 'nearest', policy));
    scores.priority.push(evaluateScenario(scenario, 'priority', policy));
    scores.rl.push(evaluateScenario(scenario, 'rl', policy));
  }
  const rlScore = mean(scores.rl);
  const nearestScore = mean(scores.nearest);
  const priorityScore = mean(scores.priority);
  const improvementVsBest = rlScore - Math.max(nearestScore, priorityScore);
  return {
    scenarios,
    rlScore,
    nearestScore,
    priorityScore,
    improvementVsBest,
    passed: improvementVsBest >= 0,
  };
}

function* trainDqnPolicyGenerator(
  episodes = 2_000,
  seed = 2026,
  context?: PolicyTrainingContext,
): Generator<
  TrainingProgress,
  { policy: CandidateDqn; stats: PolicyStats },
  void
> {
  const random = mulberry32(seed);
  const policy = new CandidateDqn(random);
  let targetPolicy = policy.clone();
  let bestPolicy = policy.clone();
  let bestValidationScore = -Infinity;
  const replay: Experience[] = [];
  const rewardHistory: { episode: number; reward: number }[] = [];
  const recentRewards: number[] = [];
  let finalLoss = 0;
  let epsilon = 0.9;

  for (let episode = 1; episode <= episodes; episode += 1) {
    const scenario = createTrainingScenario(random, context);
    let episodeReward = 0;

    while (scenario.tasks.length) {
      const candidates = feasibleTrainingCandidates(scenario);
      if (!candidates.length) {
        episodeReward -= scenario.tasks.length * 0.5;
        break;
      }
      let action = 0;
      if (random() < epsilon) {
        action = Math.floor(random() * candidates.length);
      } else {
        action = candidates.reduce(
          (best, candidate, index) =>
            candidate.prior +
              policy.predict(candidate.features) * NETWORK_SCORE_WEIGHT >
            candidates[best].prior +
              policy.predict(candidates[best].features) * NETWORK_SCORE_WEIGHT
              ? index
              : best,
          0,
        );
      }

      const selected = candidates[action];
      const result = applyTrainingAction(scenario, selected);
      const nextCandidates = feasibleTrainingCandidates(scenario);
      const done = scenario.tasks.length === 0 || nextCandidates.length === 0;
      const strandedPenalty =
        !nextCandidates.length && scenario.tasks.length
          ? scenario.tasks.reduce((sum, task) => sum + task.priority, 0) * 0.35
          : 0;
      const transitionReward = result.reward - strandedPenalty;
      replay.push({
        features: selected.features,
        reward: transitionReward,
        next: nextCandidates.map((candidate) => ({
          features: candidate.features,
          prior: candidate.prior,
        })),
        done,
      });
      if (replay.length > 12_000) replay.shift();
      episodeReward += transitionReward;

      const sampleCount = Math.min(6, replay.length);
      let batchLoss = 0;
      for (let sample = 0; sample < sampleCount; sample += 1) {
        const experience = replay[Math.floor(random() * replay.length)];
        let nextValue = 0;
        if (!experience.done && experience.next.length) {
          const bestNext = experience.next.reduce(
            (best, candidate, index) =>
              candidate.prior +
                policy.predict(candidate.features) * NETWORK_SCORE_WEIGHT >
              experience.next[best].prior +
                policy.predict(experience.next[best].features) *
                  NETWORK_SCORE_WEIGHT
                ? index
                : best,
            0,
          );
          nextValue = targetPolicy.predict(experience.next[bestNext].features);
        }
        batchLoss += policy.train(
          experience.features,
          experience.reward + 0.92 * nextValue,
          0.001,
        );
      }
      finalLoss = batchLoss / Math.max(1, sampleCount);
    }

    recentRewards.push(episodeReward);
    if (recentRewards.length > 60) recentRewards.shift();
    epsilon = Math.max(0.04, epsilon * 0.9965);
    if (episode % 40 === 0) targetPolicy = policy.clone();
    if (episode === 1 || episode % 50 === 0 || episode === episodes) {
      rewardHistory.push({ episode, reward: mean(recentRewards) });
    }
    if (episode % 250 === 0 || episode === episodes) {
      const checkpoint = validatePolicy(policy, context, seed + 70_000, 18);
      if (checkpoint.rlScore > bestValidationScore) {
        bestValidationScore = checkpoint.rlScore;
        bestPolicy = policy.clone();
      }
    }
    yield {
      episode,
      totalEpisodes: episodes,
      reward: episodeReward,
      stage: 'training',
    };
  }

  yield {
    episode: episodes,
    totalEpisodes: episodes,
    reward: mean(recentRewards),
    stage: 'validation',
  };
  const validation = validatePolicy(bestPolicy, context, seed + 90_000, 64);
  return {
    policy: bestPolicy,
    stats: {
      episodes,
      averageReward: mean(
        rewardHistory.slice(-20).map((checkpoint) => checkpoint.reward),
      ),
      finalLoss,
      epsilon,
      rewardHistory,
      modelVersion: MODEL_VERSION,
      trainingScope:
        context && context.drones.length >= 2 && context.waypoints.length >= 3
          ? '현재 가상실험 구성 70% + 일반화 시나리오 30%'
          : 'DJI 5개 기체 프로파일 일반화 시나리오',
      validation,
    },
  };
}

export function trainDqnPolicy(
  episodes = 2_000,
  seed = 2026,
  context?: PolicyTrainingContext,
): { policy: CandidateDqn; stats: PolicyStats } {
  const trainer = trainDqnPolicyGenerator(episodes, seed, context);
  let step = trainer.next();
  while (!step.done) step = trainer.next();
  return step.value;
}

export async function trainDqnPolicyAsync(
  episodes = 2_000,
  seed = 2026,
  onProgress?: (progress: TrainingProgress) => void,
  context?: PolicyTrainingContext,
): Promise<{ policy: CandidateDqn; stats: PolicyStats }> {
  const trainer = trainDqnPolicyGenerator(episodes, seed, context);
  let step = trainer.next();
  let lastProgress: TrainingProgress | null = null;

  while (!step.done) {
    lastProgress = step.value as TrainingProgress;
    for (let count = 0; count < 4; count += 1) {
      step = trainer.next();
      if (step.done) break;
      lastProgress = step.value as TrainingProgress;
      if (lastProgress.stage === 'validation') break;
    }
    if (lastProgress) onProgress?.(lastProgress);
    if (step.done) break;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }

  return step.value;
}
