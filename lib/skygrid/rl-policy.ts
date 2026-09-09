import { haversineMeters } from './geo';
import { mulberry32 } from './random';
import type { Drone, PlannerKind, PolicyStats, Waypoint } from './types';

const INPUTS = 8;
const HIDDEN = 14;
type Experience = {
  features: number[];
  reward: number;
  next: number[][];
  done: boolean;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export class CandidateDqn {
  private w1: number[][];
  private b1: number[];
  private w2: number[];
  private b2: number;

  constructor(random = mulberry32(41)) {
    this.w1 = Array.from({ length: HIDDEN }, () =>
      Array.from({ length: INPUTS }, () => (random() - 0.5) * 0.34),
    );
    this.b1 = Array(HIDDEN).fill(0);
    this.w2 = Array.from({ length: HIDDEN }, () => (random() - 0.5) * 0.28);
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
    const error = clamp(prediction - target, -12, 12);
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

interface TrainingTask {
  x: number;
  y: number;
  priority: number;
  age: number;
  deadline: number;
}

function trainingFeatures(
  task: TrainingTask,
  x: number,
  y: number,
  battery: number,
  remaining: number,
): number[] {
  const distance = Math.hypot(task.x - x, task.y - y);
  const urgency = clamp(task.age / task.deadline, 0, 2) / 2;
  const energyCost =
    clamp((distance * 0.32) / Math.max(0.15, battery), 0, 1.5) / 1.5;
  return [
    1,
    clamp(distance / 1.42, 0, 1),
    task.priority / 5,
    urgency,
    battery,
    energyCost,
    clamp(remaining / 12, 0, 1),
    task.age > task.deadline ? 1 : 0,
  ];
}

export function trainDqnPolicy(
  episodes = 650,
  seed = 2026,
): { policy: CandidateDqn; stats: PolicyStats } {
  const random = mulberry32(seed);
  const policy = new CandidateDqn(random);
  let targetPolicy = policy.clone();
  const replay: Experience[] = [];
  const rewardHistory: { episode: number; reward: number }[] = [];
  const recentRewards: number[] = [];
  let finalLoss = 0;
  let epsilon = 0.9;

  for (let episode = 1; episode <= episodes; episode += 1) {
    const taskCount = 6 + Math.floor(random() * 7);
    const tasks: TrainingTask[] = Array.from({ length: taskCount }, () => ({
      x: random(),
      y: random(),
      priority: 1 + Math.floor(random() * 5),
      age: random() * 160,
      deadline: 70 + random() * 150,
    }));
    let x = random();
    let y = random();
    let battery = 0.55 + random() * 0.45;
    let episodeReward = 0;

    while (tasks.length && battery > 0.14) {
      const candidates = tasks.map((task) =>
        trainingFeatures(task, x, y, battery, tasks.length),
      );
      let action = 0;
      if (random() < epsilon) action = Math.floor(random() * candidates.length);
      else
        action = candidates.reduce(
          (best, features, index) =>
            policy.predict(features) > policy.predict(candidates[best])
              ? index
              : best,
          0,
        );

      const selected = tasks[action];
      const distance = Math.hypot(selected.x - x, selected.y - y);
      const transitSeconds = distance * 145;
      const urgency = clamp(selected.age / selected.deadline, 0, 2);
      const reserveViolation = battery - distance * 0.32 < 0.16;
      const reward =
        selected.priority * 1.65 +
        urgency * 2.8 -
        distance * 5.4 -
        (reserveViolation ? 8 : 0);

      x = selected.x;
      y = selected.y;
      battery = clamp(battery - distance * 0.32 - 0.008, 0, 1);
      tasks.splice(action, 1);
      tasks.forEach((task) => {
        task.age += transitSeconds;
      });
      const next = tasks.map((task) =>
        trainingFeatures(task, x, y, battery, tasks.length),
      );
      replay.push({
        features: candidates[action],
        reward,
        next,
        done: tasks.length === 0 || battery <= 0.14,
      });
      if (replay.length > 2_400) replay.shift();
      episodeReward += reward;

      const sampleCount = Math.min(10, replay.length);
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
          experience.reward + 0.92 * nextValue,
          0.0018,
        );
      }
      finalLoss = batchLoss / Math.max(1, sampleCount);
    }

    recentRewards.push(episodeReward);
    if (recentRewards.length > 50) recentRewards.shift();
    epsilon = Math.max(0.04, epsilon * 0.9935);
    if (episode % 40 === 0) targetPolicy = policy.clone();
    if (episode === 1 || episode % 25 === 0 || episode === episodes) {
      rewardHistory.push({
        episode,
        reward:
          recentRewards.reduce((sum, value) => sum + value, 0) /
          recentRewards.length,
      });
    }
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

export function missionFeatures(
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
): number[] {
  const distanceM = haversineMeters(drone, waypoint);
  const age = Math.max(0, missionTime - waypoint.lastVisited);
  const urgency = clamp(age / waypoint.revisitSec, 0, 2) / 2;
  const expectedBatteryUse = (distanceM / 1000) * drone.consumptionPerKm;
  return [
    1,
    clamp(distanceM / 2_000, 0, 1),
    waypoint.priority / 5,
    urgency,
    drone.battery / 100,
    clamp(
      expectedBatteryUse / Math.max(5, drone.battery - drone.reserveBattery),
      0,
      1.5,
    ) / 1.5,
    clamp(remaining / 30, 0, 1),
    age > waypoint.revisitSec ? 1 : 0,
  ];
}

export function plannerScore(
  kind: PlannerKind,
  drone: Drone,
  waypoint: Waypoint,
  missionTime: number,
  remaining: number,
  policy: CandidateDqn,
): number {
  const distanceM = haversineMeters(drone, waypoint);
  const age = Math.max(0, missionTime - waypoint.lastVisited);
  if (kind === 'nearest') return -distanceM;
  if (kind === 'priority')
    return waypoint.priority * 1_000 + age * 2 - distanceM * 0.35;
  return policy.predict(
    missionFeatures(drone, waypoint, missionTime, remaining),
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
