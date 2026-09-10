import type { DroneProfileId } from './types';

export interface DroneProfile {
  id: DroneProfileId;
  model: string;
  cruiseSpeedMps: number;
  maxSpeedMps: number;
  maxFlightTimeMin: number;
  batteryCapacityMah: number;
  batteryEnergyWh: number;
  takeoffWeightG: number;
  maxWindResistanceMps: number;
  transmissionRangeKm: number;
  simulationTurnRateDps: number;
  batterySwapSec: number;
  sourceUrl: string;
}

export const DEFAULT_DRONE_PROFILE_ID: DroneProfileId = 'dji-mavic-4-pro';

export const DRONE_PROFILES: readonly DroneProfile[] = [
  {
    id: 'dji-mavic-4-pro',
    model: 'DJI Mavic 4 Pro',
    cruiseSpeedMps: 9,
    maxSpeedMps: 25,
    maxFlightTimeMin: 51,
    batteryCapacityMah: 6654,
    batteryEnergyWh: 95.3,
    takeoffWeightG: 1063,
    maxWindResistanceMps: 12,
    transmissionRangeKm: 30,
    simulationTurnRateDps: 120,
    batterySwapSec: 45,
    sourceUrl: 'https://store.dji.com/uk/product/dji-mavic-4-pro?vid=189291',
  },
  {
    id: 'dji-air-3s',
    model: 'DJI Air 3S',
    cruiseSpeedMps: 9,
    maxSpeedMps: 21,
    maxFlightTimeMin: 45,
    batteryCapacityMah: 4276,
    batteryEnergyWh: 62.5,
    takeoffWeightG: 724,
    maxWindResistanceMps: 12,
    transmissionRangeKm: 20,
    simulationTurnRateDps: 130,
    batterySwapSec: 45,
    sourceUrl: 'https://www.dji.com/air-3s/specs',
  },
  {
    id: 'dji-mini-5-pro',
    model: 'DJI Mini 5 Pro',
    cruiseSpeedMps: 6,
    maxSpeedMps: 18,
    maxFlightTimeMin: 36,
    batteryCapacityMah: 2788,
    batteryEnergyWh: 19.52,
    takeoffWeightG: 249,
    maxWindResistanceMps: 12,
    transmissionRangeKm: 20,
    simulationTurnRateDps: 140,
    batterySwapSec: 40,
    sourceUrl: 'https://store.dji.com/uk/product/dji-mini-5-pro',
  },
  {
    id: 'dji-mavic-pro',
    model: 'DJI Mavic Pro',
    cruiseSpeedMps: 6.94,
    maxSpeedMps: 18.06,
    maxFlightTimeMin: 27,
    batteryCapacityMah: 3830,
    batteryEnergyWh: 43.6,
    takeoffWeightG: 734,
    maxWindResistanceMps: 10.7,
    transmissionRangeKm: 7,
    simulationTurnRateDps: 120,
    batterySwapSec: 45,
    sourceUrl: 'https://www.dji.com/mavic/info',
  },
  {
    id: 'dji-avata-2',
    model: 'DJI Avata 2',
    cruiseSpeedMps: 6,
    maxSpeedMps: 27,
    maxFlightTimeMin: 23,
    batteryCapacityMah: 2150,
    batteryEnergyWh: 31.7,
    takeoffWeightG: 377,
    maxWindResistanceMps: 10.7,
    transmissionRangeKm: 13,
    simulationTurnRateDps: 180,
    batterySwapSec: 50,
    sourceUrl: 'https://www.dji.com/avata-2/specs',
  },
] as const;

export function getDroneProfile(profileId: string): DroneProfile {
  return (
    DRONE_PROFILES.find((profile) => profile.id === profileId) ??
    DRONE_PROFILES[0]
  );
}

export function profileSimulationValues(profile: DroneProfile) {
  const nominalDistanceKm =
    (profile.cruiseSpeedMps * profile.maxFlightTimeMin * 60) / 1000;
  return {
    profileId: profile.id,
    model: profile.model,
    speedMps: profile.cruiseSpeedMps,
    turnRateDps: profile.simulationTurnRateDps,
    battery: 100,
    reserveBattery: 20,
    consumptionPerKm: 100 / nominalDistanceKm,
    loiterConsumptionPerMin: 100 / profile.maxFlightTimeMin,
    maxBattery: 100,
    turnaroundSec: profile.batterySwapSec,
  };
}
