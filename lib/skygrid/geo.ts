import type { GeoPoint } from './types';

const EARTH_RADIUS_M = 6_371_000;

export function haversineMeters(a: GeoPoint, b: GeoPoint): number {
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const dLat = lat2 - lat1;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(h));
}

export function bearingDegrees(a: GeoPoint, b: GeoPoint): number {
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const y = Math.sin(dLng) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

export function moveToward(a: GeoPoint, b: GeoPoint, distanceM: number): GeoPoint {
  const total = haversineMeters(a, b);
  if (total <= distanceM || total === 0) return { lat: b.lat, lng: b.lng };
  const fraction = distanceM / total;
  return {
    lat: a.lat + (b.lat - a.lat) * fraction,
    lng: a.lng + (b.lng - a.lng) * fraction,
  };
}

export function polylineDistance(points: GeoPoint[]): number {
  let distance = 0;
  for (let i = 1; i < points.length; i += 1) distance += haversineMeters(points[i - 1], points[i]);
  return distance;
}

export function pointToSegmentMeters(point: GeoPoint, start: GeoPoint, end: GeoPoint): number {
  const latScale = 111_320;
  const lngScale = 111_320 * Math.cos((point.lat * Math.PI) / 180);
  const px = 0;
  const py = 0;
  const ax = (start.lng - point.lng) * lngScale;
  const ay = (start.lat - point.lat) * latScale;
  const bx = (end.lng - point.lng) * lngScale;
  const by = (end.lat - point.lat) * latScale;
  const dx = bx - ax;
  const dy = by - ay;
  const denom = dx * dx + dy * dy;
  const t = denom === 0 ? 0 : Math.max(0, Math.min(1, -((ax - px) * dx + (ay - py) * dy) / denom));
  return Math.hypot(ax + t * dx, ay + t * dy);
}

export function formatMissionTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${secs}`;
}
