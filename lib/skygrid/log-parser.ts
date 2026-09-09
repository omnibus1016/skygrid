import { haversineMeters, pointToSegmentMeters, polylineDistance } from './geo';
import type {
  FlightLog,
  FlightLogPoint,
  GeoPoint,
  LogMetrics,
  MissionState,
} from './types';

const LOG_COLORS = ['#67e8f9', '#fbbf62', '#a78bfa', '#64d39b'];

function splitCsvLine(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let current = '';
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const character = line[i];
    if (character === '"') {
      if (quoted && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else quoted = !quoted;
    } else if (character === delimiter && !quoted) {
      cells.push(current.trim());
      current = '';
    } else current += character;
  }
  cells.push(current.trim());
  return cells;
}

function normalizeHeader(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s_.()[\]/-]/g, '');
}

function findIndex(headers: string[], aliases: string[]): number {
  const normalized = headers.map(normalizeHeader);
  return normalized.findIndex((header) => aliases.includes(header));
}

function numberAt(row: string[], index: number, fallback = 0): number {
  if (index < 0) return fallback;
  const value = Number.parseFloat(row[index]?.replace(/[^\d+-.eE]/g, '') ?? '');
  return Number.isFinite(value) ? value : fallback;
}

export function parseFlightCsv(
  text: string,
  fileName: string,
  logIndex = 0,
): FlightLog {
  const lines = text
    .replace(/^\uFEFF/, '')
    .split(/\r?\n/)
    .filter((line) => line.trim());
  if (lines.length < 3)
    throw new Error('CSV에 분석 가능한 데이터 행이 없습니다.');
  const delimiter =
    (lines[0].match(/;/g)?.length ?? 0) > (lines[0].match(/,/g)?.length ?? 0)
      ? ';'
      : ',';
  const headers = splitCsvLine(lines[0], delimiter);
  const indexes = {
    timeMs: findIndex(headers, [
      'timemillisecond',
      'timems',
      'elapsedms',
      'timestampms',
    ]),
    timeSec: findIndex(headers, ['timesec', 'seconds', 'elapsedtime', 'time']),
    datetime: findIndex(headers, ['datetimeutc', 'datetime', 'timestamp']),
    lat: findIndex(headers, ['latitude', 'lat', 'osdlatitude']),
    lng: findIndex(headers, ['longitude', 'lng', 'lon', 'osdlongitude']),
    altitude: findIndex(headers, [
      'heightabovetakeoff',
      'altitudeabovesealevel',
      'altitude',
      'height',
      'relativealtitude',
    ]),
    speed: findIndex(headers, [
      'speed',
      'horizontalspeed',
      'groundspeed',
      'velocity',
    ]),
    verticalSpeed: findIndex(headers, ['zspeed', 'verticalspeed', 'vspeed']),
    battery: findIndex(headers, ['batterypercent', 'battery', 'chargepercent']),
    satellites: findIndex(headers, [
      'satellites',
      'satellitecount',
      'gpssatellites',
    ]),
    heading: findIndex(headers, [
      'compassheadingdegrees',
      'heading',
      'yaw',
      'gimbalheadingdegrees',
    ]),
    mode: findIndex(headers, ['flycstate', 'flightmode', 'mode']),
  };
  if (indexes.lat < 0 || indexes.lng < 0)
    throw new Error(
      '위도·경도 열을 찾지 못했습니다. AirData CSV 형식을 권장합니다.',
    );

  const rows = lines.slice(1).map((line) => splitCsvLine(line, delimiter));
  let firstEpoch: number | null = null;
  const points: FlightLogPoint[] = [];
  for (const row of rows) {
    const lat = numberAt(row, indexes.lat, Number.NaN);
    const lng = numberAt(row, indexes.lng, Number.NaN);
    if (
      !Number.isFinite(lat) ||
      !Number.isFinite(lng) ||
      Math.abs(lat) > 90 ||
      Math.abs(lng) > 180 ||
      (lat === 0 && lng === 0)
    )
      continue;
    let timeMs =
      indexes.timeMs >= 0
        ? numberAt(row, indexes.timeMs)
        : numberAt(row, indexes.timeSec) * 1000;
    if (indexes.datetime >= 0 && indexes.timeMs < 0 && indexes.timeSec < 0) {
      const epoch = Date.parse(row[indexes.datetime]);
      if (Number.isFinite(epoch)) {
        firstEpoch ??= epoch;
        timeMs = epoch - firstEpoch;
      }
    }
    points.push({
      timeMs,
      lat,
      lng,
      altitudeM: numberAt(row, indexes.altitude),
      speedMps: numberAt(row, indexes.speed, Number.NaN),
      verticalSpeedMps: numberAt(row, indexes.verticalSpeed),
      batteryPercent: numberAt(row, indexes.battery, 100),
      satellites: numberAt(row, indexes.satellites),
      heading: numberAt(row, indexes.heading),
      flightMode:
        indexes.mode >= 0 ? row[indexes.mode] || 'UNKNOWN' : 'UNKNOWN',
    });
  }
  if (points.length < 2) throw new Error('유효한 GPS 표본이 2개 미만입니다.');
  points.sort((a, b) => a.timeMs - b.timeMs);
  const offset = points[0].timeMs;
  points.forEach((point, index) => {
    point.timeMs -= offset;
    if (!Number.isFinite(point.speedMps)) {
      const previous = points[Math.max(0, index - 1)];
      const seconds = Math.max(0.1, (point.timeMs - previous.timeMs) / 1000);
      point.speedMps =
        index === 0 ? 0 : haversineMeters(previous, point) / seconds;
    }
  });
  const stride = Math.max(1, Math.floor(points.length / 8_000));
  return {
    id: `LOG-${Date.now()}-${logIndex}`,
    fileName,
    droneName: fileName.replace(/\.[^.]+$/, '').slice(0, 28),
    color: LOG_COLORS[logIndex % LOG_COLORS.length],
    points: points.filter(
      (_, index) => index % stride === 0 || index === points.length - 1,
    ),
  };
}

export function calculateLogMetrics(
  log: FlightLog,
  plannedPath: GeoPoint[] = [],
): LogMetrics {
  const first = log.points[0];
  const last = log.points[log.points.length - 1];
  const durationSec = Math.max(0, (last.timeMs - first.timeMs) / 1000);
  const distanceKm = polylineDistance(log.points) / 1000;
  const speedValues = log.points
    .map((point) => point.speedMps)
    .filter(Number.isFinite);
  let routeErrorM: number | null = null;
  if (plannedPath.length >= 2) {
    const sampled = log.points.filter(
      (_, index) =>
        index % Math.max(1, Math.floor(log.points.length / 400)) === 0,
    );
    routeErrorM =
      sampled.reduce((sum, point) => {
        let minimum = Infinity;
        for (let i = 1; i < plannedPath.length; i += 1)
          minimum = Math.min(
            minimum,
            pointToSegmentMeters(point, plannedPath[i - 1], plannedPath[i]),
          );
        return sum + minimum;
      }, 0) / Math.max(1, sampled.length);
  }
  const batteryUsed = Math.max(0, first.batteryPercent - last.batteryPercent);
  return {
    durationSec,
    distanceKm,
    avgSpeedMps:
      speedValues.reduce((sum, speed) => sum + speed, 0) /
      Math.max(1, speedValues.length),
    maxSpeedMps: Math.max(0, ...speedValues),
    maxAltitudeM: Math.max(0, ...log.points.map((point) => point.altitudeM)),
    batteryUsed,
    batteryPerKm: distanceKm > 0.02 ? batteryUsed / distanceKm : 0,
    sampleCount: log.points.length,
    sampleRateHz: durationSec > 0 ? log.points.length / durationSec : 0,
    routeErrorM,
  };
}

export function createDemoLogs(mission: MissionState): FlightLog[] {
  const generated = mission.drones.slice(0, 2).map((drone, droneIndex) => {
    const assignedTargets = drone.route
      .map((id) => mission.waypoints.find((waypoint) => waypoint.id === id))
      .filter(Boolean) as GeoPoint[];
    const fallbackTargets = mission.waypoints
      .filter((_, index) => index % 2 === droneIndex)
      .slice(0, 6);
    const targets = assignedTargets.length ? assignedTargets : fallbackTargets;
    const path = [drone, ...targets];
    const points: FlightLogPoint[] = [];
    let elapsed = 0;
    let battery = drone.battery;
    const endSegment =
      droneIndex === 1
        ? Math.max(1, Math.ceil((path.length - 1) * 0.58))
        : path.length - 1;
    for (let segment = 1; segment <= endSegment; segment += 1) {
      const start = path[segment - 1];
      const end = path[segment];
      const distance = haversineMeters(start, end);
      const seconds = Math.max(1, distance / drone.speedMps);
      const samples = Math.max(4, Math.floor(seconds * 2));
      for (let sample = 0; sample < samples; sample += 1) {
        const fraction = sample / samples;
        const wobble = Math.sin((elapsed + sample) * 0.7) * 0.000012;
        points.push({
          timeMs: elapsed * 1000,
          lat: start.lat + (end.lat - start.lat) * fraction + wobble,
          lng: start.lng + (end.lng - start.lng) * fraction - wobble * 0.7,
          altitudeM: 40 + Math.sin(fraction * Math.PI) * 4,
          speedMps: drone.speedMps * (0.92 + Math.sin(sample * 0.4) * 0.04),
          verticalSpeedMps: Math.sin(fraction * Math.PI * 2) * 0.2,
          batteryPercent: battery,
          satellites: 17 + ((sample + droneIndex) % 4),
          heading: 0,
          flightMode: 'GPS',
        });
        elapsed += seconds / samples;
        battery -= (distance / samples / 1000) * drone.consumptionPerKm;
      }
    }
    return {
      id: `DEMO-${drone.id}`,
      fileName: `${drone.id.toLowerCase()}_demo.csv`,
      droneName: `${drone.id} · ${drone.model}`,
      color: drone.color,
      points,
      plannedPath: path,
    };
  });
  if (
    generated.length >= 2 &&
    generated[0].points.length &&
    generated[1].points.length
  ) {
    const survivor = generated[0];
    const failed = generated[1];
    const failedEnd = failed.points.at(-1)!;
    const failureTime = failedEnd.timeMs;
    const handoverStartIndex = survivor.points.findIndex(
      (point) => point.timeMs >= failureTime,
    );
    if (handoverStartIndex > 0) {
      const retained = survivor.points.slice(0, handoverStartIndex);
      const start = retained.at(-1)!;
      const distance = haversineMeters(start, failedEnd);
      const seconds = Math.max(20, distance / 10);
      const sampleCount = Math.ceil(seconds * 2);
      for (let sample = 1; sample <= sampleCount; sample += 1) {
        const fraction = sample / sampleCount;
        retained.push({
          ...start,
          timeMs: failureTime + fraction * seconds * 1000,
          lat: start.lat + (failedEnd.lat - start.lat) * fraction,
          lng: start.lng + (failedEnd.lng - start.lng) * fraction,
          altitudeM: 42,
          speedMps: 10,
          batteryPercent:
            start.batteryPercent - fraction * (distance / 1000) * 9.6,
        });
      }
      survivor.points = retained;
      survivor.plannedPath = [...(survivor.plannedPath ?? []), failedEnd];
    }
  }
  return generated;
}

export function estimateHandoverSeconds(logs: FlightLog[]): number | null {
  if (logs.length < 2) return null;
  const sorted = [...logs].sort(
    (a, b) => a.points.at(-1)!.timeMs - b.points.at(-1)!.timeMs,
  );
  const failed = sorted[0];
  const survivor = sorted[1];
  const failureTime = failed.points.at(-1)!.timeMs;
  const failurePoint = failed.points.at(-1)!;
  const recovery = survivor.points.find(
    (point) =>
      point.timeMs >= failureTime &&
      haversineMeters(point, failurePoint) <= 120,
  );
  return recovery ? (recovery.timeMs - failureTime) / 1000 : null;
}
