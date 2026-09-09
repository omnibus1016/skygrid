import type { GeoPoint, NoFlyZone } from './types';

interface GridNode { x: number; y: number }

function pointInPolygon(point: GeoPoint, polygon: GeoPoint[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    const intersects = ((a.lat > point.lat) !== (b.lat > point.lat))
      && point.lng < ((b.lng - a.lng) * (point.lat - a.lat)) / (b.lat - a.lat) + a.lng;
    if (intersects) inside = !inside;
  }
  return inside;
}

function simplify(path: GeoPoint[]): GeoPoint[] {
  if (path.length < 3) return path;
  const result = [path[0]];
  for (let i = 1; i < path.length - 1; i += 1) {
    const a = path[i - 1];
    const b = path[i];
    const c = path[i + 1];
    const cross = (b.lng - a.lng) * (c.lat - b.lat) - (b.lat - a.lat) * (c.lng - b.lng);
    if (Math.abs(cross) > 1e-10) result.push(b);
  }
  result.push(path[path.length - 1]);
  return result;
}

export function aStarRoute(start: GeoPoint, end: GeoPoint, zones: NoFlyZone[], resolution = 34): GeoPoint[] {
  if (!zones.length) return [start, end];
  const allPoints = [start, end, ...zones.flatMap((zone) => zone.points)];
  const minLat = Math.min(...allPoints.map((point) => point.lat)) - 0.002;
  const maxLat = Math.max(...allPoints.map((point) => point.lat)) + 0.002;
  const minLng = Math.min(...allPoints.map((point) => point.lng)) - 0.002;
  const maxLng = Math.max(...allPoints.map((point) => point.lng)) + 0.002;
  const width = resolution;
  const height = resolution;
  const toGrid = (point: GeoPoint): GridNode => ({
    x: Math.max(0, Math.min(width - 1, Math.round(((point.lng - minLng) / (maxLng - minLng)) * (width - 1)))),
    y: Math.max(0, Math.min(height - 1, Math.round(((point.lat - minLat) / (maxLat - minLat)) * (height - 1)))),
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
      if (zones.some((zone) => pointInPolygon(point, zone.points))) blocked.add(`${x},${y}`);
    }
  }
  blocked.delete(key(source));
  blocked.delete(key(destination));

  const open: { node: GridNode; score: number }[] = [{ node: source, score: 0 }];
  const previous = new Map<string, GridNode>();
  const gScore = new Map<string, number>([[key(source), 0]]);
  const visited = new Set<string>();
  const directions = [-1, 0, 1].flatMap((dx) => [-1, 0, 1].map((dy) => ({ dx, dy }))).filter(({ dx, dy }) => dx || dy);
  const heuristic = (node: GridNode) => Math.hypot(node.x - destination.x, node.y - destination.y);

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
      if (next.x < 0 || next.y < 0 || next.x >= width || next.y >= height || blocked.has(key(next))) continue;
      const diagonal = dx !== 0 && dy !== 0;
      const tentative = (gScore.get(currentKey) ?? Infinity) + (diagonal ? Math.SQRT2 : 1);
      if (tentative >= (gScore.get(key(next)) ?? Infinity)) continue;
      previous.set(key(next), current);
      gScore.set(key(next), tentative);
      open.push({ node: next, score: tentative + heuristic(next) });
    }
  }
  return [start, end];
}
