import { mulberry32 } from './random';
import {
  CandidateDqn,
  type PlanningTrace,
  type RoutePlanningOptions,
} from './rl-policy';
import { advanceMission, createMission, missionMetrics } from './simulation';
import type {
  Drone,
  GeoPoint,
  PlannerKind,
  PolicyStats,
  ScenarioConfig,
  Waypoint,
} from './types';

type Experience = {
  features: number[];
  reward: number;
  next: number[][];
  done: boolean;
  deltaTimeSec: number;
};

type RewardWindow = {
  trace: PlanningTrace;
  rewardIntegral: number;
  elapsedSec: number;
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

const LOAD_FACTORS = [0.5, 0.8, 1, 1.2] as const;
const SIMULATION_STEP_SEC = 10;
const BASE_GAMMA_PER_MINUTE = 0.97;

function mean(values: number[]): number {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

export function timeBasedDiscount(
  deltaTimeSec: number,
  gammaPerMinute = BASE_GAMMA_PER_MINUTE,
): number {
  return Math.pow(gammaPerMinute, Math.max(0, deltaTimeSec) / 60);
}

function episodeConfig(
  random: () => number,
  seed: number,
  context?: PolicyTrainingContext,
): ScenarioConfig {
  const contextWaypoints = context?.waypoints.length ?? 0;
  const waypointCount =
    contextWaypoints >= 4 && random() < 0.35
      ? Math.max(4, Math.min(30, contextWaypoints))
      : 10 + Math.floor(random() * 17);
  const durationSec = context
    ? Math.max(480, Math.min(1_200, context.durationSec))
    : 480 + Math.floor(random() * 241);
  return {
    droneCount: Math.max(1, context?.drones.length ?? 4),
    waypointCount,
    durationSec,
    failureAt: Math.round(durationSec * (0.35 + random() * 0.3)),
    failureDroneId: 'UAV-02',
    planner: 'rl',
    simRate: 1,
    sensorRadiusM: 110,
    randomSeed: seed,
    loadFactor: LOAD_FACTORS[Math.floor(random() * LOAD_FACTORS.length)],
    layout: random() < 0.55 ? 'clustered' : 'mixed',
    mixedFleet: true,
    noFlyZoneCount: 1 + Math.floor(random() * 2),
  };
}

function traceExperiences(
  window: RewardWindow,
  nextTrace: PlanningTrace | null,
  terminal: boolean,
): Experience[] {
  if (!window.trace.decisions.length) return [];
  const normalizedReward = window.elapsedSec
    ? window.rewardIntegral / window.elapsedSec
    : 0;
  return window.trace.decisions.flatMap((decision, index) => {
    const nextDecision = window.trace.decisions[index + 1];
    const crossPlanDecision = nextTrace?.decisions[0];
    const next = (
      nextDecision?.alternatives ??
      crossPlanDecision?.alternatives ??
      []
    ).map((alternative) => alternative.features);
    const done =
      terminal &&
      index === window.trace.decisions.length - 1 &&
      !crossPlanDecision;
    const selectedExperience: Experience = {
      features: decision.features,
      reward: decision.weightedGapReward * 0.75 + normalizedReward * 0.25,
      next,
      done,
      deltaTimeSec: decision.deltaTimeSec,
    };
    const availableAlternatives = decision.alternatives.filter((alternative) =>
      alternative.features.some(
        (value, featureIndex) => value !== decision.features[featureIndex],
      ),
    );
    const sampledAlternatives = Array.from(
      { length: Math.min(16, availableAlternatives.length) },
      (_, sampleIndex) =>
        availableAlternatives[
          Math.min(
            availableAlternatives.length - 1,
            Math.floor(
              (sampleIndex * availableAlternatives.length) /
                Math.min(16, availableAlternatives.length),
            ),
          )
        ],
    );
    const counterfactuals = sampledAlternatives.map(
      (alternative): Experience => ({
        features: alternative.features,
        reward: alternative.weightedGapReward * 0.75 + normalizedReward * 0.25,
        next,
        done,
        deltaTimeSec: alternative.deltaTimeSec,
      }),
    );
    return [selectedExperience, ...counterfactuals];
  });
}

function simulateTrainingEpisode(
  policy: CandidateDqn,
  random: () => number,
  config: ScenarioConfig,
  epsilon: number,
): { reward: number; experiences: Experience[] } {
  const traces: PlanningTrace[] = [];
  const planningOptions: RoutePlanningOptions = {
    epsilon,
    random,
    onPlan: (trace) => {
      if (trace.decisions.length) traces.push(trace);
    },
  };
  let mission = createMission(config, policy, planningOptions);
  const failureDrone = mission.drones.find((drone) => drone.route.length > 0);
  const runConfig = {
    ...config,
    failureDroneId: failureDrone?.id ?? mission.drones[0]?.id ?? 'UAV-01',
  };
  mission = { ...mission, running: true };
  let active: RewardWindow | null = traces.length
    ? { trace: traces.shift()!, rewardIntegral: 0, elapsedSec: 0 }
    : null;
  const experiences: Experience[] = [];
  let episodeRewardIntegral = 0;
  let episodeElapsed = 0;

  while (mission.running && !mission.completed) {
    const delta = Math.min(
      SIMULATION_STEP_SEC,
      runConfig.durationSec - mission.time,
    );
    if (delta <= 0) break;
    const beforeGap = mission.weightedGapSeconds;
    mission = advanceMission(
      mission,
      delta,
      runConfig,
      policy,
      planningOptions,
    );
    const priorityTotal = Math.max(
      1,
      mission.waypoints.reduce((sum, waypoint) => sum + waypoint.priority, 0),
    );
    const gapRate =
      (mission.weightedGapSeconds - beforeGap) /
      Math.max(1, delta * priorityTotal);
    // The reward is the achieved fraction of a zero-gap patrol envelope.
    // Increasing overdue duration raises gapRate continuously, rather than
    // producing a one-off visit bonus.
    const intervalReward = 1 - gapRate * 4;
    episodeRewardIntegral += intervalReward * delta;
    episodeElapsed += delta;
    if (active) {
      active.rewardIntegral += intervalReward * delta;
      active.elapsedSec += delta;
    }

    while (traces.length) {
      const nextTrace = traces.shift()!;
      if (active)
        experiences.push(...traceExperiences(active, nextTrace, false));
      active = { trace: nextTrace, rewardIntegral: 0, elapsedSec: 0 };
    }
  }
  if (active) experiences.push(...traceExperiences(active, null, true));
  return {
    reward: episodeElapsed ? episodeRewardIntegral / episodeElapsed : 0,
    experiences,
  };
}

function evaluatePlanner(
  policy: CandidateDqn,
  config: ScenarioConfig,
  planner: PlannerKind,
): number {
  const runConfig = { ...config, planner };
  let mission = createMission(runConfig, policy);
  const failureDrone = mission.drones.find((drone) => drone.route.length > 0);
  const configured = {
    ...runConfig,
    failureDroneId: failureDrone?.id ?? mission.drones[0]?.id ?? 'UAV-01',
  };
  mission = { ...mission, running: true };
  while (mission.running && !mission.completed) {
    mission = advanceMission(
      mission,
      Math.min(SIMULATION_STEP_SEC, configured.durationSec - mission.time),
      configured,
      policy,
    );
  }
  return missionMetrics(mission).upperBoundAttainment;
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
    const config = episodeConfig(random, seed + index * 97, context);
    scores.nearest.push(evaluatePlanner(policy, config, 'nearest'));
    scores.priority.push(evaluatePlanner(policy, config, 'priority'));
    scores.rl.push(evaluatePlanner(policy, config, 'rl'));
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

function replayUpdate(
  policy: CandidateDqn,
  targetPolicy: CandidateDqn,
  replay: Experience[],
  random: () => number,
  sampleCount: number,
): number {
  let loss = 0;
  for (let sample = 0; sample < sampleCount; sample += 1) {
    const experience = replay[Math.floor(random() * replay.length)];
    let nextValue = 0;
    if (!experience.done && experience.next.length) {
      const bestNext = experience.next.reduce((best, features) =>
        policy.predict(features) > policy.predict(best) ? features : best,
      );
      nextValue = targetPolicy.predict(bestNext);
    }
    const target =
      clamp(experience.reward, -4, 4) +
      (experience.done
        ? 0
        : timeBasedDiscount(experience.deltaTimeSec) * nextValue);
    loss += policy.train(experience.features, target, 0.0002);
  }
  return loss / Math.max(1, sampleCount);
}

function fitImmediateGapValue(
  policy: CandidateDqn,
  replay: Experience[],
  random: () => number,
  sampleCount: number,
): void {
  for (let sample = 0; sample < sampleCount; sample += 1) {
    const experience = replay[Math.floor(random() * replay.length)];
    policy.train(experience.features, clamp(experience.reward, -4, 4), 0.001);
  }
}

function* trainDqnPolicyGenerator(
  episodes = 2_000,
  seed = 2026,
  context?: PolicyTrainingContext,
  initialPolicy?: CandidateDqn,
): Generator<
  TrainingProgress,
  { policy: CandidateDqn; stats: PolicyStats },
  void
> {
  const random = mulberry32(seed);
  const policy = initialPolicy?.clone() ?? new CandidateDqn(random);
  let targetPolicy = policy.clone();
  let bestPolicy = policy.clone();
  let bestValidationScore = -Infinity;
  const replay: Experience[] = [];
  const rewardHistory: { episode: number; reward: number }[] = [];
  const recentRewards: number[] = [];
  let finalLoss = 0;
  let epsilon = initialPolicy ? 0.25 : 0.85;

  for (let episode = 1; episode <= episodes; episode += 1) {
    const config = episodeConfig(random, seed + episode * 131, context);
    const result = simulateTrainingEpisode(policy, random, config, epsilon);
    replay.push(...result.experiences);
    if (replay.length > 30_000) replay.splice(0, replay.length - 30_000);
    if (replay.length) {
      fitImmediateGapValue(
        policy,
        replay,
        random,
        Math.min(256, replay.length),
      );
      finalLoss = replayUpdate(
        policy,
        targetPolicy,
        replay,
        random,
        Math.min(32, replay.length),
      );
    }
    recentRewards.push(result.reward);
    if (recentRewards.length > 80) recentRewards.shift();
    epsilon = Math.max(0.03, epsilon * 0.996);
    if (episode % 30 === 0) targetPolicy = policy.clone();
    if (episode === 1 || episode % 25 === 0 || episode === episodes) {
      rewardHistory.push({ episode, reward: mean(recentRewards) });
    }
    if (episode % 250 === 0 || episode === episodes) {
      const checkpoint = validatePolicy(
        policy,
        context,
        seed + 70_000,
        Math.min(16, Math.max(6, Math.round(episodes / 150))),
      );
      if (checkpoint.improvementVsBest > bestValidationScore) {
        bestValidationScore = checkpoint.improvementVsBest;
        bestPolicy = policy.clone();
      }
    }
    yield {
      episode,
      totalEpisodes: episodes,
      reward: result.reward,
      stage: 'training',
    };
  }

  yield {
    episode: episodes,
    totalEpisodes: episodes,
    reward: mean(recentRewards),
    stage: 'validation',
  };
  const validation = validatePolicy(
    bestPolicy,
    context,
    seed + 90_000,
    Math.min(48, Math.max(8, Math.round(episodes / 50))),
  );
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
      modelVersion: 5,
      trainingScope: '실제 일괄 배정·2-opt·비행 시뮬레이터 기반 복합 시나리오',
      validation,
    },
  };
}

export function trainDqnPolicy(
  episodes = 2_000,
  seed = 2026,
  context?: PolicyTrainingContext,
  initialPolicy?: CandidateDqn,
): { policy: CandidateDqn; stats: PolicyStats } {
  const trainer = trainDqnPolicyGenerator(
    episodes,
    seed,
    context,
    initialPolicy,
  );
  let step = trainer.next();
  while (!step.done) step = trainer.next();
  return step.value;
}

export async function trainDqnPolicyAsync(
  episodes = 2_000,
  seed = 2026,
  onProgress?: (progress: TrainingProgress) => void,
  context?: PolicyTrainingContext,
  initialPolicy?: CandidateDqn,
): Promise<{ policy: CandidateDqn; stats: PolicyStats }> {
  const trainer = trainDqnPolicyGenerator(
    episodes,
    seed,
    context,
    initialPolicy,
  );
  let step = trainer.next();
  while (!step.done) {
    onProgress?.(step.value);
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    step = trainer.next();
  }
  return step.value;
}
