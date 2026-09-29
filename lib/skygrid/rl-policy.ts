import { haversineMeters } from './geo';
import { obstacleAwareDistance } from './pathfinding';
import { mulberry32 } from './random';
import type {
  Drone,
  GeoPoint,
  NoFlyZone,
  PlannerKind,
  Waypoint,
} from './types';

/*
 * The policy is a centralized candidate-value network. One action chooses a
 * (drone, waypoint) pair for the fleet. Training and inference share the same
 * geographic, battery, dwell, revisit and return-to-base calculations.
 */
const MODEL_VERSION = 5 as const;
const INPUTS = 20;
const HIDDEN = 64;

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
  workloadSeconds: number;
  routeLoad: number;
};

export interface PlanningDecision {
  features: number[];
  alternatives: PlanningAlternative[];
  deltaTimeSec: number;
  weightedGapReward: number;
}

export interface PlanningAlternative {
  features: number[];
  deltaTimeSec: number;
  weightedGapReward: number;
}

export interface PlanningTrace {
  missionTime: number;
  decisions: PlanningDecision[];
}

export interface RoutePlanningOptions {
  epsilon?: number;
  random?: () => number;
  onPlan?: (trace: PlanningTrace) => void;
  useGapOracle?: boolean;
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

function integratedWeightedGap(
  waypoints: Waypoint[],
  startTime: number,
  durationSec: number,
): number {
  return waypoints.reduce((sum, waypoint) => {
    const deadline = waypoint.lastVisited + waypoint.revisitSec;
    const overdueAtStart = Math.max(0, startTime - deadline);
    if (overdueAtStart > 0) {
      return (
        sum +
        (waypoint.priority *
          (overdueAtStart * durationSec + 0.5 * durationSec ** 2)) /
          Math.max(1, waypoint.revisitSec)
      );
    }
    const untilOverdue = deadline - startTime;
    const overdueDuration = Math.max(0, durationSec - untilOverdue);
    return (
      sum +
      (waypoint.priority * (0.5 * overdueDuration ** 2)) /
        Math.max(1, waypoint.revisitSec)
    );
  }, 0);
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
  routeLoad,
  workloadRatio,
  relativeWorkload,
  slackRatio,
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
  routeLoad: number;
  workloadRatio: number;
  relativeWorkload: number;
  slackRatio: number;
}): number[] {
  const logarithmicDistance = (distanceKm: number) =>
    clamp(Math.log1p(Math.max(0, distanceKm)) / Math.log1p(80), 0, 1);
  return [
    1,
    logarithmicDistance(distanceKm),
    clamp(priority / 5, 0, 1),
    clamp(urgency / 2, 0, 1),
    clamp(battery, 0, 1),
    clamp(expectedBatteryUse / Math.max(0.05, usableBattery), 0, 2) / 2,
    clamp(remaining / 40, 0, 1),
    overdue ? 1 : 0,
    clamp(teamSize / 12, 0, 1),
    clamp(averageBattery, 0, 1),
    clamp(relativeAdvantage, 0, 1),
    clamp(localDensity / 6, 0, 1),
    clamp(reserveMargin, -0.25, 1) / 1.25 + 0.2,
    clamp(transitRatio / 2, 0, 1),
    clamp(dwellRatio, 0, 1),
    logarithmicDistance(returnDistanceKm),
    clamp(routeLoad / 10, 0, 1),
    clamp(workloadRatio / 3, 0, 1),
    clamp(relativeWorkload / 3, 0, 1),
    clamp((slackRatio + 2) / 3, 0, 1),
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
  workloadSeconds = 0,
  averageWorkloadSeconds = 0,
  noFlyZones: NoFlyZone[] = [],
): CandidateContext {
  const fleet = [
    drone,
    ...team.filter(
      (candidate) =>
        candidate.id !== drone.id &&
        (candidate.status === 'active' || candidate.status === 'ready'),
    ),
  ];
  const distanceM = obstacleAwareDistance(drone, waypoint, noFlyZones);
  const returnDistanceM = obstacleAwareDistance(
    waypoint,
    { lat: drone.homeLat, lng: drone.homeLng },
    noFlyZones,
  );
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
    ? Math.min(
        ...peers.map((peer) =>
          obstacleAwareDistance(peer, waypoint, noFlyZones),
        ),
      )
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
  const workloadRatio = workloadSeconds / Math.max(30, waypoint.revisitSec);
  const relativeWorkload =
    workloadSeconds / Math.max(30, averageWorkloadSeconds || workloadSeconds);
  const slackRatio =
    (waypoint.revisitSec - age - visitSeconds) /
    Math.max(1, waypoint.revisitSec);
  const routeLoad = drone.route.length;

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
      routeLoad,
      workloadRatio,
      relativeWorkload,
      slackRatio,
    }),
    distanceM,
    transitSeconds,
    visitSeconds,
    projectedAgeRatio,
    expectedBatteryUse,
    reserveMargin,
    relativeAdvantage,
    localDensity,
    workloadSeconds,
    routeLoad,
  };
}

function rlCandidateScore(
  policy: CandidateDqn,
  context: CandidateContext,
): number {
  return policy.predict(context.features);
}

export function missionFeatures(
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
  team: Drone[] = [],
  pending: Waypoint[] = [],
  workloadSeconds = 0,
  averageWorkloadSeconds = 0,
  noFlyZones: NoFlyZone[] = [],
): number[] {
  return candidateContext(
    drone,
    waypoint,
    missionTime,
    remaining,
    team,
    pending,
    workloadSeconds,
    averageWorkloadSeconds,
    noFlyZones,
  ).features;
}

export function estimateSortieEnergy(
  drone: Drone,
  waypoint: Waypoint,
  noFlyZones: NoFlyZone[] = [],
): number {
  const transitEnergy =
    (obstacleAwareDistance(drone, waypoint, noFlyZones) / 1000) *
    drone.consumptionPerKm;
  const dwellEnergy = (waypoint.dwellSec / 60) * drone.loiterConsumptionPerMin;
  const returnEnergy =
    (obstacleAwareDistance(
      waypoint,
      { lat: drone.homeLat, lng: drone.homeLng },
      noFlyZones,
    ) /
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
  workloadSeconds = 0,
  averageWorkloadSeconds = 0,
  noFlyZones: NoFlyZone[] = [],
): number {
  const distanceM = obstacleAwareDistance(drone, waypoint, noFlyZones);
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
    workloadSeconds,
    averageWorkloadSeconds,
    noFlyZones,
  );
  return rlCandidateScore(policy, context);
}

type RoutePlan = { drones: Drone[]; waypoints: Waypoint[] };

function routeDistanceMeters(
  drone: Drone,
  route: string[],
  waypointById: Map<string, Waypoint>,
  noFlyZones: NoFlyZone[],
): number {
  let position: GeoPoint = drone;
  let distance = 0;
  for (const waypointId of route) {
    const waypoint = waypointById.get(waypointId);
    if (!waypoint) continue;
    distance += obstacleAwareDistance(position, waypoint, noFlyZones);
    position = waypoint;
  }
  distance += obstacleAwareDistance(
    position,
    { lat: drone.homeLat, lng: drone.homeLng },
    noFlyZones,
  );
  return distance;
}

function refineRouteOrder(
  drone: Drone,
  route: string[],
  waypointById: Map<string, Waypoint>,
  noFlyZones: NoFlyZone[],
): string[] {
  if (route.length < 3) return route;
  let best = [...route];
  let bestDistance = routeDistanceMeters(drone, best, waypointById, noFlyZones);
  let improved = true;
  let pass = 0;
  while (improved && pass < 8) {
    improved = false;
    pass += 1;
    // The learned policy's first urgent target is retained. 2-opt only
    // shortens the remainder of the sortie and therefore does not overwrite
    // the policy's immediate decision.
    for (let start = 1; start < best.length - 1; start += 1) {
      for (let end = start + 1; end < best.length; end += 1) {
        const candidate = [
          ...best.slice(0, start),
          ...best.slice(start, end + 1).reverse(),
          ...best.slice(end + 1),
        ];
        const distance = routeDistanceMeters(
          drone,
          candidate,
          waypointById,
          noFlyZones,
        );
        if (distance + 0.1 >= bestDistance) continue;
        best = candidate;
        bestDistance = distance;
        improved = true;
      }
    }
  }
  return best;
}

function buildRoutePlan(
  drones: Drone[],
  waypoints: Waypoint[],
  missionTime: number,
  kind: PlannerKind,
  policy: CandidateDqn,
  noFlyZones: NoFlyZone[],
  options?: RoutePlanningOptions,
): RoutePlan {
  const active = drones
    .filter((drone) => drone.status === 'active' || drone.status === 'ready')
    .map((drone) => ({ ...drone, route: [], routeIndex: 0 }));
  const virtual: Drone[] = active.map((drone) => ({ ...drone }));
  const virtualElapsed = virtual.map(() => 0);
  const waypointCopies: Waypoint[] = waypoints.map((waypoint) => ({
    ...waypoint,
    assignedDrone: undefined,
  }));
  const pending = [...waypointCopies];
  const decisions: PlanningDecision[] = [];

  type ScoredCandidate = {
    droneIndex: number;
    targetIndex: number;
    score: number;
    context: CandidateContext;
    weightedGapReward: number;
  };

  const assignCandidate = (
    selected: ScoredCandidate,
    alternatives: ScoredCandidate[],
  ) => {
    const { droneIndex, targetIndex, context } = selected;
    const drone = virtual[droneIndex];
    const [target] = pending.splice(targetIndex, 1);
    if (kind === 'rl') {
      decisions.push({
        features: [...context.features],
        alternatives: alternatives.map((candidate) => ({
          features: [...candidate.context.features],
          deltaTimeSec: Math.max(1, candidate.context.visitSeconds),
          weightedGapReward: candidate.weightedGapReward,
        })),
        deltaTimeSec: Math.max(1, context.visitSeconds),
        weightedGapReward: selected.weightedGapReward,
      });
    }
    drone.route.push(target.id);
    const transitDistance = context.distanceM;
    virtualElapsed[droneIndex] +=
      transitDistance / Math.max(1, drone.speedMps) + target.dwellSec;
    drone.battery -=
      (transitDistance / 1000) * drone.consumptionPerKm +
      (target.dwellSec / 60) * drone.loiterConsumptionPerMin;
    drone.lat = target.lat;
    drone.lng = target.lng;
    target.assignedDrone = drone.id;
  };

  const candidateSet = (droneIndices: number[]): ScoredCandidate[] => {
    const candidates: ScoredCandidate[] = [];
    const averageWorkload = mean(virtualElapsed);
    for (const droneIndex of droneIndices) {
      for (
        let targetIndex = 0;
        targetIndex < pending.length;
        targetIndex += 1
      ) {
        const drone = virtual[droneIndex];
        const target = pending[targetIndex];
        const requiredEnergy = estimateSortieEnergy(drone, target, noFlyZones);
        if (drone.battery - requiredEnergy < drone.reserveBattery) continue;
        const context = candidateContext(
          drone,
          target,
          missionTime + virtualElapsed[droneIndex],
          pending.length,
          virtual,
          pending,
          virtualElapsed[droneIndex],
          averageWorkload,
          noFlyZones,
        );
        const decisionTime = missionTime + virtualElapsed[droneIndex];
        const priorityTotal = Math.max(
          1,
          pending.reduce((sum, waypoint) => sum + waypoint.priority, 0),
        );
        const normalizedFleetDelay =
          integratedWeightedGap(
            pending.filter((waypoint) => waypoint.id !== target.id),
            decisionTime,
            context.visitSeconds,
          ) / Math.max(1, priorityTotal * context.visitSeconds);
        const visitTime = decisionTime + context.visitSeconds;
        const lookaheadSec = Math.max(300, target.revisitSec * 1.5);
        const noServiceGap = integratedWeightedGap(
          [target],
          decisionTime,
          lookaheadSec,
        );
        const preVisitDuration = Math.min(context.visitSeconds, lookaheadSec);
        const gapBeforeVisit = integratedWeightedGap(
          [target],
          decisionTime,
          preVisitDuration,
        );
        const remainingLookahead = Math.max(
          0,
          lookaheadSec - context.visitSeconds,
        );
        const resetTarget = { ...target, lastVisited: visitTime };
        const gapAfterVisit = integratedWeightedGap(
          [resetTarget],
          visitTime,
          remainingLookahead,
        );
        const marginalGapReduction =
          (noServiceGap - gapBeforeVisit - gapAfterVisit) /
          Math.max(1, priorityTotal * lookaheadSec);
        const weightedGapReward =
          marginalGapReduction * 20 -
          normalizedFleetDelay * 4 -
          Math.max(0, 0.1 - context.reserveMargin) * 2;
        const score =
          kind === 'nearest'
            ? -context.distanceM
            : kind === 'priority'
              ? target.priority * 1_000 +
                Math.max(0, decisionTime - target.lastVisited) * 2 -
                context.distanceM * 0.35
              : options?.useGapOracle
                ? weightedGapReward
                : policy.predict(context.features);
        candidates.push({
          droneIndex,
          targetIndex,
          score,
          context,
          weightedGapReward,
        });
      }
    }
    return candidates;
  };

  const chooseCandidate = (
    droneIndices: number[],
  ): { selected: ScoredCandidate; alternatives: ScoredCandidate[] } | null => {
    const alternatives = candidateSet(droneIndices);
    if (!alternatives.length) return null;
    const random = options?.random ?? Math.random;
    const explore = kind === 'rl' && random() < (options?.epsilon ?? 0);
    const selected = explore
      ? alternatives[Math.floor(random() * alternatives.length)]
      : alternatives.reduce((best, candidate) =>
          candidate.score > best.score ? candidate : best,
        );
    return { selected, alternatives };
  };

  for (let index = 0; index < virtual.length; index += 1) {
    const drone = virtual[index];
    const source = drones.find((candidate) => candidate.id === drone.id);
    if (source?.phase !== 'dwell' && source?.phase !== 'transit') continue;
    const currentId = source.route[source.routeIndex];
    const targetIndex = pending.findIndex((target) => target.id === currentId);
    if (targetIndex < 0) continue;
    const target = pending[targetIndex];
    const forwardDistance = obstacleAwareDistance(source, target, noFlyZones);
    const forwardEnergy =
      source.phase === 'dwell'
        ? (Math.max(0, source.dwellRemainingSec) / 60) *
          drone.loiterConsumptionPerMin
        : (forwardDistance / 1000) * drone.consumptionPerKm +
          (target.dwellSec / 60) * drone.loiterConsumptionPerMin;
    const returnEnergy =
      (obstacleAwareDistance(
        target,
        { lat: drone.homeLat, lng: drone.homeLng },
        noFlyZones,
      ) /
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
        : forwardDistance / Math.max(1, drone.speedMps) + target.dwellSec;
    drone.lat = target.lat;
    drone.lng = target.lng;
    target.assignedDrone = drone.id;
  }

  if (pending.length) {
    const unseeded = new Set(
      virtual
        .map((drone, index) => ({ drone, index }))
        .filter(({ drone }) => drone.route.length === 0)
        .map(({ index }) => index),
    );
    while (pending.length && unseeded.size) {
      const choice = chooseCandidate([...unseeded]);
      if (!choice) break;
      assignCandidate(choice.selected, choice.alternatives);
      unseeded.delete(choice.selected.droneIndex);
    }
  }

  while (pending.length && virtual.length) {
    const choice = chooseCandidate(virtual.map((_, index) => index));
    if (!choice) break;
    assignCandidate(choice.selected, choice.alternatives);
  }

  const waypointById = new Map(
    waypointCopies.map((waypoint) => [waypoint.id, waypoint]),
  );
  const routes = new Map(
    virtual.map((drone) => [
      drone.id,
      refineRouteOrder(
        drones.find((source) => source.id === drone.id) ?? drone,
        drone.route,
        waypointById,
        noFlyZones,
      ),
    ]),
  );
  const assignment = new Map(
    waypointCopies.map((waypoint) => [waypoint.id, waypoint.assignedDrone]),
  );
  const result = {
    drones: drones.map((drone) =>
      routes.has(drone.id)
        ? { ...drone, route: routes.get(drone.id) ?? [], routeIndex: 0 }
        : drone,
    ),
    waypoints: waypoints.map((waypoint) => ({
      ...waypoint,
      assignedDrone: assignment.get(waypoint.id),
    })),
  };
  options?.onPlan?.({ missionTime, decisions });
  return result;
}

export function assignRoutes(
  drones: Drone[],
  waypoints: Waypoint[],
  missionTime: number,
  kind: PlannerKind,
  policy: CandidateDqn,
  noFlyZones: NoFlyZone[] = [],
  options?: RoutePlanningOptions,
): RoutePlan {
  return buildRoutePlan(
    drones,
    waypoints,
    missionTime,
    kind,
    policy,
    noFlyZones,
    options,
  );
}
