import type { GeoPoint, NoFlyZone } from './types';
import { haversineMeters, moveToward, polylineDistance } from './geo';

interface GridNode {
  x: number;
  y: number;
}

const distanceCache = new Map<string, number>();

function distanceCacheKey(
  start: GeoPoint,
  end: GeoPoint,
  zones: NoFlyZone[],
): string {
  const point = (value: GeoPoint) =>
    `${value.lat.toFixed(5)},${value.lng.toFixed(5)}`;
  const a = point(start);
  const b = point(end);
  const zoneKey = zones
    .map(
      (zone) =>
        `${zone.id}:${zone.points
          .map((value) => `${value.lat.toFixed(4)},${value.lng.toFixed(4)}`)
          .join(';')}`,
    )
    .join('|');
  return `${zoneKey}|${a < b ? `${a}|${b}` : `${b}|${a}`}`;
}

function pointInPolygon(point: GeoPoint, polygon: GeoPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    const intersects =
      a.lat > point.lat !== b.lat > point.lat &&
      point.lng <
        ((b.lng - a.lng) * (point.lat - a.lat)) / (b.lat - a.lat) + a.lng;
    if (intersects) inside = !inside;
  }
  return inside;
}

function orientation(a: GeoPoint, b: GeoPoint, c: GeoPoint): number {
  return (b.lng - a.lng) * (c.lat - a.lat) - (b.lat - a.lat) * (c.lng - a.lng);
}

function segmentsIntersect(
  a: GeoPoint,
  b: GeoPoint,
  c: GeoPoint,
  d: GeoPoint,
): boolean {
  const abC = orientation(a, b, c);
  const abD = orientation(a, b, d);
  const cdA = orientation(c, d, a);
  const cdB = orientation(c, d, b);
  return abC * abD <= 0 && cdA * cdB <= 0;
}

function segmentCrossesZone(
  start: GeoPoint,
  end: GeoPoint,
  zone: NoFlyZone,
): boolean {
  if (pointInPolygon(start, zone.points) || pointInPolygon(end, zone.points))
    return true;
  return zone.points.some((point, index) =>
    segmentsIntersect(
      start,
      end,
      point,
      zone.points[(index + 1) % zone.points.length],
    ),
  );
}

function simplify(path: GeoPoint[]): GeoPoint[] {
  if (path.length < 3) return path;
  const result = [path[0]];
  for (let i = 1; i < path.length - 1; i += 1) {
    const a = path[i - 1];
    const b = path[i];
    const c = path[i + 1];
    const cross =
      (b.lng - a.lng) * (c.lat - b.lat) - (b.lat - a.lat) * (c.lng - b.lng);
    if (Math.abs(cross) > 1e-10) result.push(b);
  }
  result.push(path[path.length - 1]);
  return result;
}

export function aStarRoute(
  start: GeoPoint,
  end: GeoPoint,
  zones: NoFlyZone[],
  resolution = 34,
): GeoPoint[] {
  if (
    !zones.length ||
    !zones.some((zone) => segmentCrossesZone(start, end, zone))
  )
    return [start, end];
  const allPoints = [start, end, ...zones.flatMap((zone) => zone.points)];
  const minLat = Math.min(...allPoints.map((point) => point.lat)) - 0.002;
  const maxLat = Math.max(...allPoints.map((point) => point.lat)) + 0.002;
  const minLng = Math.min(...allPoints.map((point) => point.lng)) - 0.002;
  const maxLng = Math.max(...allPoints.map((point) => point.lng)) + 0.002;
  const width = resolution;
  const height = resolution;
  const toGrid = (point: GeoPoint): GridNode => ({
    x: Math.max(
      0,
      Math.min(
        width - 1,
        Math.round(((point.lng - minLng) / (maxLng - minLng)) * (width - 1)),
      ),
    ),
    y: Math.max(
      0,
      Math.min(
        height - 1,
        Math.round(((point.lat - minLat) / (maxLat - minLat)) * (height - 1)),
      ),
    ),
  });
  const toGeo = (node: GridNode): GeoPoint => ({
    lat: minLat + (node.y / (height - 1)) * (maxLat - minLat),
    lng: minLng + (node.x / (width - 1)) * (maxLng - minLng),
  });
  const source = toGrid(start);
  const destination = toGrid(end);
  const key = (node: GridNode) => `${node.x},${node.y}`;
  const blocked = new Set<string>();
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const point = toGeo({ x, y });
      if (zones.some((zone) => pointInPolygon(point, zone.points)))
        blocked.add(`${x},${y}`);
    }
  }
  blocked.delete(key(source));
  blocked.delete(key(destination));

  const open: { node: GridNode; score: number }[] = [
    { node: source, score: 0 },
  ];
  const previous = new Map<string, GridNode>();
  const gScore = new Map<string, number>([[key(source), 0]]);
  const visited = new Set<string>();
  const directions = [-1, 0, 1]
    .flatMap((dx) => [-1, 0, 1].map((dy) => ({ dx, dy })))
    .filter(({ dx, dy }) => dx || dy);
  const heuristic = (node: GridNode) =>
    Math.hypot(node.x - destination.x, node.y - destination.y);

  while (open.length) {
    open.sort((a, b) => a.score - b.score);
    const current = open.shift()!.node;
    const currentKey = key(current);
    if (visited.has(currentKey)) continue;
    visited.add(currentKey);
    if (current.x === destination.x && current.y === destination.y) {
      const gridPath = [current];
      let cursor = currentKey;
      while (previous.has(cursor)) {
        const parent = previous.get(cursor)!;
        gridPath.push(parent);
        cursor = key(parent);
      }
      gridPath.reverse();
      const route = gridPath.map(toGeo);
      route[0] = start;
      route[route.length - 1] = end;
      return simplify(route);
    }

    for (const { dx, dy } of directions) {
      const next = { x: current.x + dx, y: current.y + dy };
      if (
        next.x < 0 ||
        next.y < 0 ||
        next.x >= width ||
        next.y >= height ||
        blocked.has(key(next))
      )
        continue;
      const diagonal = dx !== 0 && dy !== 0;
      const tentative =
        (gScore.get(currentKey) ?? Infinity) + (diagonal ? Math.SQRT2 : 1);
      if (tentative >= (gScore.get(key(next)) ?? Infinity)) continue;
      previous.set(key(next), current);
      gScore.set(key(next), tentative);
      open.push({ node: next, score: tentative + heuristic(next) });
    }
  }
  return [start, end];
}

export function obstacleAwareDistance(
  start: GeoPoint,
  end: GeoPoint,
  zones: NoFlyZone[] = [],
): number {
  const key = distanceCacheKey(start, end, zones);
  const cached = distanceCache.get(key);
  if (cached !== undefined) return cached;
  const distance = polylineDistance(aStarRoute(start, end, zones));
  if (distanceCache.size > 50_000) distanceCache.clear();
  distanceCache.set(key, distance);
  return distance;
}

export function moveAlongObstacleAwareRoute(
  start: GeoPoint,
  end: GeoPoint,
  distanceM: number,
  zones: NoFlyZone[] = [],
): { position: GeoPoint; distanceMoved: number; headingTarget: GeoPoint } {
  const path = aStarRoute(start, end, zones);
  let remaining = Math.max(0, distanceM);
  let position = start;
  let distanceMoved = 0;
  let headingTarget = path[1] ?? end;
  for (let index = 1; index < path.length && remaining > 0; index += 1) {
    const target = path[index];
    const legDistance = haversineMeters(position, target);
    headingTarget = target;
    if (legDistance <= remaining) {
      position = target;
      remaining -= legDistance;
      distanceMoved += legDistance;
      continue;
    }
    position = moveToward(position, target, remaining);
    distanceMoved += remaining;
    remaining = 0;
  }
  return { position, distanceMoved, headingTarget };
}
