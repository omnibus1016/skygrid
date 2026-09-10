import { haversineMeters } from './geo';
import { mulberry32 } from './random';
import type { Drone, PlannerKind, PolicyStats, Waypoint } from './types';

/*
 * The policy is a centralized multi-agent DQN:
 * one decision selects a (drone, waypoint) pair for the whole team.
 * This keeps the model small enough to train in the browser while letting
 * every candidate see the current state of the other available drones.
 */
const INPUTS = 13;
const HIDDEN = 24;

type Experience = {
  features: number[];
  reward: number;
  next: number[][];
  done: boolean;
};

interface TrainingTask {
  x: number;
  y: number;
  priority: number;
  age: number;
  deadline: number;
}

interface TrainingDrone {
  x: number;
  y: number;
  battery: number;
  reserveBattery: number;
  consumptionPerKm: number;
}

interface Candidate {
  droneIndex: number;
  taskIndex: number;
  features: number[];
}

export interface TrainingProgress {
  episode: number;
  totalEpisodes: number;
  reward: number;
}

export interface PolicyWeights {
  version: 1;
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

function distance2d(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function candidateFeatureVector({
  distance,
  priority,
  urgency,
  battery,
  expectedBatteryUse,
  reserveSpan,
  remaining,
  overdue,
  teamSize,
  averageBattery,
  relativeAdvantage,
  localDensity,
}: {
  distance: number;
  priority: number;
  urgency: number;
  battery: number;
  expectedBatteryUse: number;
  reserveSpan: number;
  remaining: number;
  overdue: boolean;
  teamSize: number;
  averageBattery: number;
  relativeAdvantage: number;
  localDensity: number;
}): number[] {
  return [
    1,
    clamp(distance / 1.42, 0, 1),
    clamp(priority / 5, 0, 1),
    clamp(urgency, 0, 2) / 2,
    clamp(battery, 0, 1),
    clamp(expectedBatteryUse / Math.max(0.15, reserveSpan), 0, 1.5) / 1.5,
    clamp(remaining / 30, 0, 1),
    overdue ? 1 : 0,
    clamp(teamSize / 6, 0, 1),
    clamp(averageBattery, 0, 1),
    clamp(relativeAdvantage, 0, 1),
    clamp(localDensity / 4, 0, 1),
    clamp(
      (battery - expectedBatteryUse - (battery - reserveSpan)) /
        Math.max(0.15, reserveSpan),
      0,
      1,
    ),
  ];
}

export class CandidateDqn {
  private w1: number[][];
  private b1: number[];
  private w2: number[];
  private b2: number;

  constructor(random = mulberry32(41)) {
    this.w1 = Array.from({ length: HIDDEN }, () =>
      Array.from({ length: INPUTS }, () => (random() - 0.5) * 0.28),
    );
    this.b1 = Array(HIDDEN).fill(0);
    this.w2 = Array.from({ length: HIDDEN }, () => (random() - 0.5) * 0.22);
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
      version: 1,
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
      weights.version !== 1 ||
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
          row.some((weight) => typeof weight !== 'number'),
      ) ||
      weights.b1.some((bias) => typeof bias !== 'number') ||
      weights.w2.some((weight) => typeof weight !== 'number')
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
    const error = clamp(prediction - target, -10, 10);
    const previousW2 = [...this.w2];

    for (let h = 0; h < HIDDEN; h += 1)
      this.w2[h] -= learningRate * error * hidden[h];
    this.b2 -= learningRate * error;

    for (let h = 0; h < HIDDEN; h += 1) {
      if (preActivation[h] <= 0) continue;
      const gradient = error * previousW2[h];
      for (let input = 0; input < INPUTS; input += 1)
        this.w1[h][input] -= learningRate * gradient * features[input];
      this.b1[h] -= learningRate * gradient;
    }
    return error * error;
  }
}

function trainingFeatures(
  task: TrainingTask,
  taskIndex: number,
  droneIndex: number,
  team: TrainingDrone[],
  tasks: TrainingTask[],
): number[] {
  const drone = team[droneIndex];
  const distance = distance2d(drone, task);
  const urgency = clamp(task.age / task.deadline, 0, 2);
  const expectedBatteryUse = distance * drone.consumptionPerKm;
  const reserveSpan = Math.max(0.2, drone.battery - drone.reserveBattery);
  const peers = team.filter((_, index) => index !== droneIndex);
  const nearestPeerDistance = peers.length
    ? Math.min(...peers.map((peer) => distance2d(peer, task)))
    : distance * 2;
  const relativeAdvantage =
    peers.length === 0
      ? 1
      : clamp(nearestPeerDistance / Math.max(0.03, distance), 0, 2) / 2;
  const localDensity = tasks.filter(
    (candidate, index) =>
      index !== taskIndex && distance2d(candidate, task) < 0.28,
  ).length;
  const averageBattery =
    team.reduce((sum, candidate) => sum + candidate.battery, 0) /
    Math.max(1, team.length);

  return candidateFeatureVector({
    distance,
    priority: task.priority,
    urgency,
    battery: drone.battery,
    expectedBatteryUse,
    reserveSpan,
    remaining: tasks.length,
    overdue: task.age > task.deadline,
    teamSize: team.length,
    averageBattery,
    relativeAdvantage,
    localDensity,
  });
}

function feasibleTrainingCandidates(
  team: TrainingDrone[],
  tasks: TrainingTask[],
): Candidate[] {
  const candidates: Candidate[] = [];
  for (let droneIndex = 0; droneIndex < team.length; droneIndex += 1) {
    const drone = team[droneIndex];
    for (let taskIndex = 0; taskIndex < tasks.length; taskIndex += 1) {
      const task = tasks[taskIndex];
      const distance = distance2d(drone, task);
      const use = distance * drone.consumptionPerKm;
      if (drone.battery - use < drone.reserveBattery) continue;
      candidates.push({
        droneIndex,
        taskIndex,
        features: trainingFeatures(task, taskIndex, droneIndex, team, tasks),
      });
    }
  }
  return candidates;
}

function* trainDqnPolicyGenerator(
  episodes = 700,
  seed = 2026,
): Generator<
  TrainingProgress,
  { policy: CandidateDqn; stats: PolicyStats },
  void
> {
  const random = mulberry32(seed);
  const policy = new CandidateDqn(random);
  let targetPolicy = policy.clone();
  const replay: Experience[] = [];
  const rewardHistory: { episode: number; reward: number }[] = [];
  const recentRewards: number[] = [];
  let finalLoss = 0;
  let epsilon = 0.95;

  for (let episode = 1; episode <= episodes; episode += 1) {
    const team: TrainingDrone[] = Array.from(
      { length: 2 + Math.floor(random() * 5) },
      () => ({
        x: random(),
        y: random(),
        battery: 0.68 + random() * 0.32,
        reserveBattery: 0.16,
        consumptionPerKm: 0.27 + random() * 0.1,
      }),
    );
    const tasks: TrainingTask[] = Array.from(
      { length: 10 + Math.floor(random() * 15) },
      () => ({
        x: random(),
        y: random(),
        priority: 1 + Math.floor(random() * 5),
        age: random() * 180,
        deadline: 70 + random() * 170,
      }),
    );
    let episodeReward = 0;

    while (tasks.length) {
      const candidates = feasibleTrainingCandidates(team, tasks);
      if (!candidates.length) {
        episodeReward -= tasks.length * 0.9;
        break;
      }
      let action = 0;
      if (random() < epsilon) {
        action = Math.floor(random() * candidates.length);
      } else {
        action = candidates.reduce(
          (best, candidate, index) =>
            policy.predict(candidate.features) >
            policy.predict(candidates[best].features)
              ? index
              : best,
          0,
        );
      }

      const selected = candidates[action];
      const drone = team[selected.droneIndex];
      const task = tasks[selected.taskIndex];
      const distance = distance2d(drone, task);
      const energyUse = distance * drone.consumptionPerKm;
      const urgency = clamp(task.age / task.deadline, 0, 2);
      const peers = team.filter((_, index) => index !== selected.droneIndex);
      const nearestPeerDistance = peers.length
        ? Math.min(...peers.map((peer) => distance2d(peer, task)))
        : distance * 2;
      const relativeAdvantage =
        peers.length === 0
          ? 1
          : clamp(nearestPeerDistance / Math.max(0.03, distance), 0, 2) / 2;
      const reserveMargin = clamp(
        (drone.battery - energyUse - drone.reserveBattery) /
          Math.max(0.15, drone.battery - drone.reserveBattery),
        0,
        1,
      );
      const reward =
        task.priority * 1.55 +
        urgency * 3.2 +
        relativeAdvantage * 1.4 +
        reserveMargin * 0.7 -
        distance * 4.2 -
        energyUse * 2.4;

      drone.x = task.x;
      drone.y = task.y;
      drone.battery = clamp(drone.battery - energyUse - 0.008, 0, 1);
      tasks.splice(selected.taskIndex, 1);
      const transitSeconds = distance * 145;
      tasks.forEach((remainingTask) => {
        remainingTask.age += transitSeconds;
      });

      const nextCandidates = feasibleTrainingCandidates(team, tasks);
      const done = tasks.length === 0 || nextCandidates.length === 0;
      const missedTaskPenalty =
        !nextCandidates.length && tasks.length ? tasks.length * 0.9 : 0;
      const transitionReward = reward - missedTaskPenalty;
      replay.push({
        features: selected.features,
        reward: transitionReward,
        next: nextCandidates.map((candidate) => candidate.features),
        done,
      });
      if (replay.length > 8_000) replay.shift();
      episodeReward += transitionReward;

      const sampleCount = Math.min(24, replay.length);
      let batchLoss = 0;
      for (let sample = 0; sample < sampleCount; sample += 1) {
        const experience = replay[Math.floor(random() * replay.length)];
        const nextValue =
          experience.done || !experience.next.length
            ? 0
            : Math.max(
                ...experience.next.map((features) =>
                  targetPolicy.predict(features),
                ),
              );
        batchLoss += policy.train(
          experience.features,
          experience.reward + 0.94 * nextValue,
          0.0014,
        );
      }
      finalLoss = batchLoss / Math.max(1, sampleCount);
    }

    recentRewards.push(episodeReward);
    if (recentRewards.length > 50) recentRewards.shift();
    epsilon = Math.max(0.05, epsilon * 0.994);
    if (episode % 50 === 0) targetPolicy = policy.clone();
    if (episode === 1 || episode % 25 === 0 || episode === episodes) {
      rewardHistory.push({
        episode,
        reward:
          recentRewards.reduce((sum, value) => sum + value, 0) /
          recentRewards.length,
      });
    }
    yield {
      episode,
      totalEpisodes: episodes,
      reward: episodeReward,
    };
  }

  return {
    policy,
    stats: {
      episodes,
      averageReward:
        recentRewards.reduce((sum, value) => sum + value, 0) /
        recentRewards.length,
      finalLoss,
      epsilon,
      rewardHistory,
    },
  };
}

export function trainDqnPolicy(
  episodes = 700,
  seed = 2026,
): { policy: CandidateDqn; stats: PolicyStats } {
  const trainer = trainDqnPolicyGenerator(episodes, seed);
  let step = trainer.next();
  while (!step.done) step = trainer.next();
  return step.value;
}

export async function trainDqnPolicyAsync(
  episodes = 700,
  seed = 2026,
  onProgress?: (progress: TrainingProgress) => void,
): Promise<{ policy: CandidateDqn; stats: PolicyStats }> {
  const trainer = trainDqnPolicyGenerator(episodes, seed);
  let step = trainer.next();
  let lastProgress: TrainingProgress | null = null;

  while (!step.done) {
    lastProgress = step.value as TrainingProgress;
    for (let count = 0; count < 8; count += 1) {
      step = trainer.next();
      if (step.done) break;
      lastProgress = step.value as TrainingProgress;
    }
    if (lastProgress) onProgress?.(lastProgress);
    if (step.done) break;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }

  if (lastProgress && lastProgress.episode < episodes)
    onProgress?.({
      episode: episodes,
      totalEpisodes: episodes,
      reward: lastProgress.reward,
    });
  return step.value;
}

export function missionFeatures(
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
  team: Drone[] = [],
  pending: Waypoint[] = [],
): number[] {
  const fleet = [
    drone,
    ...team.filter(
      (candidate) =>
        candidate.id !== drone.id &&
        (candidate.status === 'active' || candidate.status === 'ready'),
    ),
  ];
  const distanceM = haversineMeters(drone, waypoint);
  const age = Math.max(0, missionTime - waypoint.lastVisited);
  const urgency = clamp(age / waypoint.revisitSec, 0, 2);
  const expectedBatteryUse = (distanceM / 1000) * drone.consumptionPerKm;
  const reserveSpan = Math.max(5, drone.battery - drone.reserveBattery);
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
      haversineMeters(candidate, waypoint) < 500,
  ).length;
  const averageBattery =
    fleet.reduce((sum, candidate) => sum + candidate.battery / 100, 0) /
    Math.max(1, fleet.length);

  return candidateFeatureVector({
    distance: distanceM / 1_000,
    priority: waypoint.priority,
    urgency,
    battery: drone.battery / 100,
    expectedBatteryUse: expectedBatteryUse / 100,
    reserveSpan: reserveSpan / 100,
    remaining,
    overdue: age > waypoint.revisitSec,
    teamSize: fleet.length,
    averageBattery,
    relativeAdvantage,
    localDensity,
  });
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
  return policy.predict(
    missionFeatures(drone, waypoint, missionTime, remaining, team, pending),
  );
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
  const pending = waypoints.map((waypoint) => ({ ...waypoint }));

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
        const use =
          (haversineMeters(drone, target) / 1000) * drone.consumptionPerKm;
        if (drone.battery - use < drone.reserveBattery) continue;
        const score = plannerScore(
          kind,
          drone,
          target,
          missionTime,
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
    drone.battery -= (transitDistance / 1000) * drone.consumptionPerKm;
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
