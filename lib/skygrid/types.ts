export type AppMode =
  | 'field'
  | 'analysis'
  | 'simulation'
  | 'evaluation'
  | 'training';
export type PlannerKind = 'rl' | 'nearest' | 'priority';
export type DroneStatus = 'ready' | 'active' | 'failed' | 'returning';
export type DronePhase = 'base' | 'transit' | 'dwell' | 'return';

export interface GeoPoint {
  lat: number;
  lng: number;
}

export interface MissionBase extends GeoPoint {
  id: string;
  name: string;
  configured: boolean;
}

export interface Waypoint extends GeoPoint {
  id: string;
  priority: number;
  revisitSec: number;
  dwellSec: number;
  assignedDrone?: string;
  lastVisited: number;
  visitedCount: number;
}

export interface NoFlyZone {
  id: string;
  name: string;
  points: GeoPoint[];
}

export interface Drone extends GeoPoint {
  id: string;
  model: string;
  color: string;
  homeLat: number;
  homeLng: number;
  speedMps: number;
  turnRateDps: number;
  battery: number;
  reserveBattery: number;
  consumptionPerKm: number;
  loiterConsumptionPerMin: number;
  maxBattery: number;
  turnaroundSec: number;
  turnaroundRemainingSec: number;
  dwellRemainingSec: number;
  phase: DronePhase;
  sortieCount: number;
  minimumBattery: number;
  status: DroneStatus;
  route: string[];
  routeIndex: number;
  totalDistanceM: number;
  heading: number;
}

export interface MissionEvent {
  id: string;
  time: number;
  kind: 'system' | 'visit' | 'replan' | 'warning' | 'failure' | 'upload';
  title: string;
  detail: string;
}

export interface ScenarioConfig {
  droneCount: number;
  waypointCount: number;
  durationSec: number;
  failureAt: number;
  failureDroneId: string;
  planner: PlannerKind;
  simRate: number;
  sensorRadiusM: number;
  randomSeed: number;
}

export interface MissionState {
  time: number;
  running: boolean;
  completed: boolean;
  failureTriggered: boolean;
  drones: Drone[];
  waypoints: Waypoint[];
  base: MissionBase;
  noFlyZones: NoFlyZone[];
  events: MissionEvent[];
  weightedGapSeconds: number;
  continuityIntegral: number;
  continuityObservationSeconds: number;
  postFailureContinuityIntegral: number;
  postFailureObservationSeconds: number;
  minimumPostFailureContinuity: number;
  revisitChecks: number;
  onTimeRevisits: number;
  returnCount: number;
  reserveViolations: number;
  failureTime: number | null;
  orphanedWaypointIds: string[];
  recoveredAt: number | null;
  replanCount: number;
  inferenceMs: number;
}

export interface MissionMetrics {
  continuity: number;
  averageContinuity: number;
  postFailureAverageContinuity: number | null;
  minimumPostFailureContinuity: number;
  revisitCompliance: number;
  coverage: number;
  activeDrones: number;
  weightedGapSeconds: number;
  recoverySeconds: number | null;
  totalDistanceKm: number;
  averageBattery: number;
  minimumBattery: number;
  returnCount: number;
  reserveViolations: number;
  sortieCount: number;
  completedVisits: number;
}

export interface PolicyStats {
  episodes: number;
  averageReward: number;
  finalLoss: number;
  epsilon: number;
  rewardHistory: { episode: number; reward: number }[];
}

export interface FlightLogPoint extends GeoPoint {
  timeMs: number;
  altitudeM: number;
  speedMps: number;
  verticalSpeedMps: number;
  batteryPercent: number;
  satellites: number;
  heading: number;
  flightMode: string;
}

export interface FlightLog {
  id: string;
  fileName: string;
  droneName: string;
  color: string;
  points: FlightLogPoint[];
  plannedPath?: GeoPoint[];
}

export interface LogMetrics {
  durationSec: number;
  distanceKm: number;
  avgSpeedMps: number;
  maxSpeedMps: number;
  maxAltitudeM: number;
  batteryUsed: number;
  batteryPerKm: number;
  sampleCount: number;
  sampleRateHz: number;
  routeErrorM: number | null;
}

export interface BatchResult {
  planner: PlannerKind;
  label: string;
  continuity: number;
  weightedGap: number;
  distance: number;
  completion: number;
}

export interface ScenarioComparisonResult {
  planner: PlannerKind;
  label: string;
  continuity: number;
  revisitCompliance: number;
  minimumPostFailureContinuity: number;
  coverage: number;
  weightedGapSeconds: number;
  recoverySeconds: number | null;
  distanceKm: number;
  averageBattery: number;
  returnCount: number;
}
