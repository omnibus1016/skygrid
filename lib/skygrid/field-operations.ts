import { bearingDegrees, haversineMeters } from './geo';
import type {
  Drone,
  DroneProfileId,
  GeoPoint,
  MissionState,
  ScenarioConfig,
} from './types';

export const RESEARCH_FLEET_PROFILE_IDS: DroneProfileId[] = [
  'dji-mavic-pro',
  'dji-avata-2',
];

export interface MissionPlanFile {
  schema: 'skygrid.mission-plan.v1';
  createdAt: string;
  config: Pick<ScenarioConfig, 'planner' | 'durationSec'>;
  base: MissionState['base'];
  waypoints: MissionState['waypoints'];
  drones: Array<
    Pick<
      Drone,
      'id' | 'profileId' | 'model' | 'speedMps' | 'reserveBattery' | 'route'
    >
  >;
}

export function plannedPathForDrone(
  mission: Pick<MissionState, 'base' | 'waypoints'>,
  drone: Pick<Drone, 'route' | 'routeIndex' | 'lat' | 'lng'>,
): GeoPoint[] {
  const targets = drone.route
    .slice(drone.routeIndex)
    .map((id) => mission.waypoints.find((waypoint) => waypoint.id === id))
    .filter(Boolean) as GeoPoint[];
  return [{ lat: drone.lat, lng: drone.lng }, ...targets];
}

export function buildMissionPlanFile(
  mission: MissionState,
  config: ScenarioConfig,
): MissionPlanFile {
  return {
    schema: 'skygrid.mission-plan.v1',
    createdAt: new Date().toISOString(),
    config: { planner: config.planner, durationSec: config.durationSec },
    base: mission.base,
    waypoints: mission.waypoints,
    drones: mission.drones.map((drone) => ({
      id: drone.id,
      profileId: drone.profileId,
      model: drone.model,
      speedMps: drone.speedMps,
      reserveBattery: drone.reserveBattery,
      route: drone.route,
    })),
  };
}

export function parseMissionPlanFile(text: string): MissionPlanFile {
  const parsed = JSON.parse(text) as Partial<MissionPlanFile>;
  if (
    parsed.schema !== 'skygrid.mission-plan.v1' ||
    !parsed.base ||
    !Array.isArray(parsed.waypoints) ||
    !Array.isArray(parsed.drones)
  ) {
    throw new Error('SKYGRID 시험계획 JSON 형식이 아닙니다.');
  }
  return parsed as MissionPlanFile;
}

export function plannedPathFromPlan(
  plan: MissionPlanFile,
  profileId: DroneProfileId,
): GeoPoint[] {
  const drone =
    plan.drones.find((item) => item.profileId === profileId) ?? plan.drones[0];
  if (!drone) return [];
  const points = drone.route
    .map((id) => plan.waypoints.find((waypoint) => waypoint.id === id))
    .filter(Boolean) as GeoPoint[];
  return [{ lat: plan.base.lat, lng: plan.base.lng }, ...points];
}

const LITCHI_HEADER = [
  'latitude',
  'longitude',
  'altitude(m)',
  'heading(deg)',
  'curvesize(m)',
  'rotationdir',
  'gimbalmode',
  'gimbalpitchangle',
  ...Array.from({ length: 15 }, (_, index) => [
    `actiontype${index + 1}`,
    `actionparam${index + 1}`,
  ]).flat(),
  'altitudemode',
  'speed(m/s)',
  'poi_latitude',
  'poi_longitude',
  'poi_altitude(m)',
  'poi_altitudemode',
  'photo_timeinterval',
  'photo_distinterval',
];

export function buildLitchiCsv(
  mission: MissionState,
  drone: Drone,
  altitudeM = 30,
): string {
  const rows = drone.route
    .slice(drone.routeIndex)
    .map((id) => mission.waypoints.find((waypoint) => waypoint.id === id))
    .filter(Boolean)
    .map((waypoint) => {
      const actions = [0, waypoint!.dwellSec * 1000];
      while (actions.length < 30) actions.push(actions.length % 2 ? 0 : -1);
      return [
        waypoint!.lat.toFixed(7),
        waypoint!.lng.toFixed(7),
        altitudeM,
        0,
        0.2,
        0,
        0,
        0,
        ...actions,
        0,
        drone.speedMps.toFixed(1),
        0,
        0,
        0,
        0,
        -1,
        -1,
      ].join(',');
    });
  if (rows.length < 2)
    throw new Error('Litchi 임무에는 정찰지점이 2개 이상 필요합니다.');
  return [LITCHI_HEADER.join(','), ...rows].join('\n');
}

export function buildGuidanceCsv(
  mission: MissionState,
  drone: Drone,
  altitudeM = 30,
): string {
  const header =
    'sequence,point_id,latitude,longitude,altitude_m,speed_mps,dwell_sec,priority,revisit_sec,leg_distance_m,bearing_deg';
  let previous: GeoPoint = { lat: mission.base.lat, lng: mission.base.lng };
  const rows = drone.route
    .slice(drone.routeIndex)
    .map((id, index) => {
      const waypoint = mission.waypoints.find((item) => item.id === id);
      if (!waypoint) return null;
      const row = [
        index + 1,
        waypoint.id,
        waypoint.lat.toFixed(7),
        waypoint.lng.toFixed(7),
        altitudeM,
        drone.speedMps.toFixed(1),
        waypoint.dwellSec,
        waypoint.priority,
        waypoint.revisitSec,
        haversineMeters(previous, waypoint).toFixed(1),
        bearingDegrees(previous, waypoint).toFixed(1),
      ].join(',');
      previous = waypoint;
      return row;
    })
    .filter(Boolean);
  return [header, ...rows].join('\n');
}

export function buildExperimentProtocolCsv(): string {
  const header =
    'profile,aircraft,mission_min,altitude_m,cruise_mps,points,dwell_sec,revisit_sec,failure_time_sec,failure_aircraft,planner,repeats,purpose';
  const rows = [
    [
      'CAL-M',
      'DJI Mavic Pro',
      4,
      30,
      6,
      2,
      10,
      180,
      '-',
      '-',
      'AI',
      3,
      '100 m 왕복·체공 소모율 보정',
    ],
    [
      'CAL-A',
      'DJI Avata 2',
      4,
      30,
      5,
      2,
      10,
      180,
      '-',
      '-',
      'AI',
      3,
      '100 m 왕복·체공 소모율 보정',
    ],
    [
      'EXP-01',
      'Mavic+Avata 2',
      10,
      30,
      '6/5',
      6,
      10,
      180,
      240,
      'Mavic',
      'AI/최근접/중요도',
      2,
      '균일 중요도·Mavic 이탈',
    ],
    [
      'EXP-02',
      'Mavic+Avata 2',
      10,
      30,
      '6/5',
      6,
      10,
      180,
      240,
      'Avata 2',
      'AI/최근접/중요도',
      2,
      '균일 중요도·Avata 2 이탈',
    ],
    [
      'EXP-03',
      'Mavic+Avata 2',
      10,
      30,
      '6/5',
      6,
      10,
      '120/180',
      240,
      'Mavic',
      'AI/최근접/중요도',
      2,
      '고중요도 지점 재방문 압박',
    ],
    [
      'EXP-04',
      'Mavic+Avata 2',
      10,
      30,
      '6/5',
      6,
      20,
      150,
      240,
      'Avata 2',
      'AI/최근접/중요도',
      2,
      '체공시간 증가',
    ],
    [
      'EXP-05',
      'Mavic+Avata 2',
      10,
      30,
      '6/5',
      8,
      10,
      '120/180',
      240,
      'Mavic',
      'AI/최근접/중요도',
      2,
      '지점 수 증가',
    ],
  ];
  return [header, ...rows.map((row) => row.join(','))].join('\n');
}
