export type AppMode =
  | 'field'
  | 'analysis'
  | 'simulation'
  | 'evaluation'
  | 'training';
export type PlannerKind = 'rl' | 'nearest' | 'priority';
export type DroneStatus = 'ready' | 'active' | 'failed' | 'returning';

export interface GeoPoint {
  lat: number;
  lng: number;
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
  noFlyZones: NoFlyZone[];
  events: MissionEvent[];
  weightedGapSeconds: number;
  replanCount: number;
  inferenceMs: number;
}

export interface MissionMetrics {
  continuity: number;
  coverage: number;
  activeDrones: number;
  weightedGapSeconds: number;
  recoverySeconds: number | null;
  totalDistanceKm: number;
  averageBattery: number;
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
  coverage: number;
  weightedGapSeconds: number;
  recoverySeconds: number | null;
  distanceKm: number;
  averageBattery: number;
}
