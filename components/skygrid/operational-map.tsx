'use client';

import { Fragment, useEffect, useMemo } from 'react';
import L from 'leaflet';
import {
  Circle,
  MapContainer,
  Marker,
  Polygon,
  Polyline,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet';

import { aStarRoute } from '@/lib/skygrid/pathfinding';
import type {
  AppMode,
  FlightLog,
  GeoPoint,
  MissionState,
} from '@/lib/skygrid/types';

interface OperationalMapProps {
  mode: AppMode;
  mapBase: 'satellite' | 'street';
  mission: MissionState;
  logs: FlightLog[];
  selectedDroneId: string;
  selectedWaypointId: string;
  interaction:
    | 'inspect'
    | 'add-base'
    | 'add-waypoint'
    | 'add-drone'
    | 'move-drone';
  onMapClick: (point: GeoPoint) => void;
  onMapContextMenu: (
    point: GeoPoint,
    position: { x: number; y: number },
    waypointId?: string,
    droneId?: string,
  ) => void;
  onSelectDrone: (id: string) => void;
  onSelectWaypoint: (id: string) => void;
}

function MapClick({
  onClick,
  onContextMenu,
}: {
  onClick: (point: GeoPoint) => void;
  onContextMenu: (point: GeoPoint, position: { x: number; y: number }) => void;
}) {
  useMapEvents({
    click: (event) => onClick(event.latlng),
    contextmenu: (event) => {
      event.originalEvent.preventDefault();
      onContextMenu(event.latlng, {
        x: event.containerPoint.x,
        y: event.containerPoint.y,
      });
    },
  });
  return null;
}

interface ViewportBounds {
  minLat: number;
  maxLat: number;
  minLng: number;
  maxLng: number;
}

function MapViewport({ minLat, maxLat, minLng, maxLng }: ViewportBounds) {
  const map = useMap();
  useEffect(() => {
    window.setTimeout(() => map.invalidateSize(), 40);
    map.fitBounds(
      L.latLngBounds([
        [minLat, minLng],
        [maxLat, maxLng],
      ]),
      {
        paddingTopLeft: [58, 76],
        paddingBottomRight: [58, 126],
        maxZoom: 16,
      },
    );
  }, [map, minLat, maxLat, minLng, maxLng]);
  return null;
}

function droneIcon(
  id: string,
  color: string,
  heading: number,
  failed: boolean,
  selected: boolean,
): L.DivIcon {
  return L.divIcon({
    className: 'skygrid-leaflet-icon',
    html: `<div class="uav-map-marker ${failed ? 'is-failed' : ''} ${selected ? 'is-selected' : ''}" style="--uav-color:${color};--heading:${heading}deg">
      <span class="uav-pulse"></span><span class="uav-body"><i></i><b></b></span><em>${id}</em>
    </div>`,
    iconSize: [76, 54],
    iconAnchor: [38, 19],
  });
}

interface RouteAssignment {
  droneId: string;
  color: string;
  order: number;
  next: boolean;
}

function shortDroneId(id: string): string {
  const numeric = Number.parseInt(id.replace(/\D/g, ''), 10);
  return Number.isFinite(numeric) ? `U${numeric}` : id;
}

function waypointIcon(
  id: string,
  assignment: RouteAssignment | undefined,
  overdue: boolean,
  selected: boolean,
): L.DivIcon {
  const color = assignment?.color ?? (overdue ? '#ff786a' : '#73d9ed');
  const classes = [
    'route-point-marker',
    assignment?.next ? 'is-next' : '',
    overdue ? 'is-overdue' : '',
    selected ? 'is-selected' : '',
  ]
    .filter(Boolean)
    .join(' ');
  const owner = assignment ? shortDroneId(assignment.droneId) : '—';
  return L.divIcon({
    className: 'skygrid-leaflet-icon',
    html: `<div class="${classes}" style="--route-color:${color}">
      <span>${assignment?.order ?? '·'}</span>
      <div><b>${id}</b><i>${owner}${assignment?.next ? ' · 다음' : ''}</i></div>
    </div>`,
    iconSize: [70, 42],
    iconAnchor: [15, 15],
  });
}

function baseIcon(): L.DivIcon {
  return L.divIcon({
    className: 'skygrid-leaflet-icon',
    html: `<div class="uav-base-marker"><span></span><b>BASE</b></div>`,
    iconSize: [52, 52],
    iconAnchor: [26, 26],
  });
}

export default function OperationalMap({
  mode,
  mapBase,
  mission,
  logs,
  selectedDroneId,
  selectedWaypointId,
  interaction,
  onMapClick,
  onMapContextMenu,
  onSelectDrone,
  onSelectWaypoint,
}: OperationalMapProps) {
  const routeAssignments = useMemo(() => {
    const assignments = new Map<string, RouteAssignment>();
    for (const drone of mission.drones) {
      if (drone.status === 'failed' || drone.status === 'returning') continue;
      drone.route.slice(drone.routeIndex).forEach((waypointId, index) => {
        if (assignments.has(waypointId)) return;
        assignments.set(waypointId, {
          droneId: drone.id,
          color: drone.color,
          order: index + 1,
          next: index === 0,
        });
      });
    }
    return assignments;
  }, [mission.drones]);

  const routeLines = useMemo(
    () =>
      mission.drones.flatMap((drone) => {
        if (drone.status === 'failed') return [];
        if (drone.status === 'returning') {
          const points = aStarRoute(drone, mission.base, mission.noFlyZones);
          return [
            {
              id: drone.id,
              color: drone.color,
              points,
              nextPoints: points,
              returning: true,
            },
          ];
        }
        const remaining = drone.route
          .slice(drone.routeIndex)
          .map((id) => mission.waypoints.find((waypoint) => waypoint.id === id))
          .filter(Boolean) as GeoPoint[];
        if (!remaining.length) return [];
        const points = [drone, ...remaining, mission.base];
        const line: GeoPoint[] = [];
        for (let i = 1; i < points.length; i += 1) {
          const segment = aStarRoute(
            points[i - 1],
            points[i],
            mission.noFlyZones,
          );
          line.push(...(i === 1 ? segment : segment.slice(1)));
        }
        const nextPoints = aStarRoute(
          drone,
          remaining[0],
          mission.noFlyZones,
        );
        return line.length
          ? [
              {
                id: drone.id,
                color: drone.color,
                points: line,
                nextPoints,
                returning: false,
              },
            ]
          : [];
      }),
    [mission.drones, mission.waypoints, mission.base, mission.noFlyZones],
  );

  const operationalPoints = [
    ...(mission.base.configured ? [mission.base] : []),
    ...mission.waypoints,
    ...mission.drones,
  ];
  const mapPoints =
    mode === 'analysis' && logs.length
      ? logs.flatMap((log) =>
          log.points.filter(
            (_, index) =>
              index % Math.max(1, Math.floor(log.points.length / 200)) === 0,
          ),
        )
      : operationalPoints;
  const viewportBounds: ViewportBounds | null = mapPoints.length
    ? {
        minLat: Math.min(...mapPoints.map((point) => point.lat)),
        maxLat: Math.max(...mapPoints.map((point) => point.lat)),
        minLng: Math.min(...mapPoints.map((point) => point.lng)),
        maxLng: Math.max(...mapPoints.map((point) => point.lng)),
      }
    : null;
  const selectedDrone = mission.drones.find(
    (drone) => drone.id === selectedDroneId,
  );

  return (
    <MapContainer
      center={[36.25, 127.8]}
      zoom={7}
      zoomControl={false}
      attributionControl
      className={`tactical-map map-base-${mapBase} interaction-${interaction}`}
    >
      {mapBase === 'satellite' ? (
        <TileLayer
          key="satellite"
          attribution="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
          url="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />
      ) : (
        <TileLayer
          key="street"
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          subdomains="abc"
          maxZoom={19}
        />
      )}
      {viewportBounds && <MapViewport {...viewportBounds} />}
      <MapClick onClick={onMapClick} onContextMenu={onMapContextMenu} />

      {mission.noFlyZones.map((zone) => (
        <Polygon
          key={zone.id}
          positions={zone.points.map(
            (point) => [point.lat, point.lng] as [number, number],
          )}
          pathOptions={{
            color: '#ff786a',
            weight: 1.5,
            dashArray: '7 6',
            fillColor: '#ff5544',
            fillOpacity: 0.1,
          }}
        >
          <Tooltip
            direction="center"
            permanent
            className="tactical-tooltip danger-tooltip"
          >
            {zone.id} · 제한
          </Tooltip>
        </Polygon>
      ))}

      {mode !== 'analysis' &&
        routeLines.map((route) => (
          <Polyline
            key={route.id}
            positions={route.points.map(
              (point) => [point.lat, point.lng] as [number, number],
            )}
            pathOptions={{
              color: route.color,
              weight: 1.6,
              opacity: 0.58,
              dashArray: '6 7',
            }}
          />
        ))}

      {mode !== 'analysis' &&
        routeLines.map((route) => (
          <Polyline
            key={`${route.id}-next`}
            positions={route.nextPoints.map(
              (point) => [point.lat, point.lng] as [number, number],
            )}
            pathOptions={{
              color: route.returning ? '#f2c46e' : route.color,
              weight: 3.8,
              opacity: 0.96,
            }}
          />
        ))}

      {mode !== 'analysis' && mission.base.configured && (
        <Marker
          position={[mission.base.lat, mission.base.lng]}
          icon={baseIcon()}
          interactive={false}
          zIndexOffset={-200}
        >
          <Tooltip
            direction="top"
            offset={[0, -24]}
            className="tactical-tooltip base-tooltip"
          >
            <strong>{mission.base.id}</strong>
            <br />
            {mission.base.name}
          </Tooltip>
        </Marker>
      )}

      {mission.waypoints.map((waypoint) => {
        const overdue =
          mission.time - waypoint.lastVisited > waypoint.revisitSec;
        const selected = waypoint.id === selectedWaypointId;
        const assignment = routeAssignments.get(waypoint.id);
        return (
          <Marker
            key={waypoint.id}
            position={[waypoint.lat, waypoint.lng]}
            icon={waypointIcon(
              waypoint.id,
              assignment,
              overdue,
              selected,
            )}
            zIndexOffset={assignment?.next ? 300 : selected ? 200 : 0}
            eventHandlers={{
              click: () => onSelectWaypoint(waypoint.id),
              contextmenu: (event) => {
                event.originalEvent.preventDefault();
                event.originalEvent.stopPropagation();
                onMapContextMenu(
                  waypoint,
                  {
                    x: event.containerPoint.x,
                    y: event.containerPoint.y,
                  },
                  waypoint.id,
                );
              },
            }}
          >
            <Tooltip
              direction="top"
              offset={[0, -7]}
              className="tactical-tooltip"
            >
              <strong>{waypoint.id}</strong>
              <br />
              {assignment
                ? `${assignment.droneId} · ${assignment.order}번째 방문`
                : '담당 기체 미배정'}
              <br />
              중요도 {waypoint.priority} · 정찰 {waypoint.dwellSec}초
              <br />
              재방문 {waypoint.revisitSec}초
            </Tooltip>
          </Marker>
        );
      })}

      {mode !== 'analysis' &&
        mission.drones.map((drone) => (
          <Marker
            key={drone.id}
            position={[drone.lat, drone.lng]}
            icon={droneIcon(
              drone.id,
              drone.color,
              drone.heading,
              drone.status === 'failed',
              drone.id === selectedDroneId,
            )}
            eventHandlers={{
              click: () => onSelectDrone(drone.id),
              contextmenu: (event) => {
                event.originalEvent.preventDefault();
                event.originalEvent.stopPropagation();
                onMapContextMenu(
                  drone,
                  {
                    x: event.containerPoint.x,
                    y: event.containerPoint.y,
                  },
                  undefined,
                  drone.id,
                );
              },
            }}
          >
            <Tooltip
              direction="right"
              offset={[18, 0]}
              className="tactical-tooltip"
            >
              <strong>{drone.id}</strong>
              <br />
              {drone.model}
              <br />
              BAT {drone.battery.toFixed(1)}%
              <br />
              {drone.phase === 'dwell'
                ? `정찰 중 ${Math.ceil(drone.dwellRemainingSec)}초`
                : drone.status === 'returning'
                  ? '기지 복귀 중'
                  : drone.status === 'ready'
                    ? drone.turnaroundRemainingSec > 0
                      ? `출격 준비 ${Math.ceil(drone.turnaroundRemainingSec)}초`
                      : '기지 대기'
                    : '목표 이동 중'}
            </Tooltip>
          </Marker>
        ))}

      {selectedDrone && mode !== 'analysis' && (
        <Circle
          center={[selectedDrone.lat, selectedDrone.lng]}
          radius={110}
          pathOptions={{
            color: selectedDrone.color,
            weight: 1,
            fillColor: selectedDrone.color,
            fillOpacity: 0.035,
            dashArray: '4 6',
          }}
        />
      )}

      {mode === 'analysis' &&
        logs.map((log) => (
          <Fragment key={log.id}>
            {log.plannedPath && (
              <Polyline
                positions={log.plannedPath.map(
                  (point) => [point.lat, point.lng] as [number, number],
                )}
                pathOptions={{
                  color: log.color,
                  weight: 1.4,
                  opacity: 0.45,
                  dashArray: '7 7',
                }}
              />
            )}
            <Polyline
              positions={log.points.map(
                (point) => [point.lat, point.lng] as [number, number],
              )}
              pathOptions={{ color: log.color, weight: 3, opacity: 0.88 }}
            >
              <Tooltip sticky className="tactical-tooltip">
                {log.droneName} · {log.points.length.toLocaleString()} samples
              </Tooltip>
            </Polyline>
          </Fragment>
        ))}
    </MapContainer>
  );
}
