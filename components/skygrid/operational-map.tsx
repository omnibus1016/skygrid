'use client';

import { Fragment, useEffect, useMemo } from 'react';
import L from 'leaflet';
import {
  Circle,
  CircleMarker,
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
  interaction: 'inspect' | 'add-waypoint' | 'move-drone';
  onMapClick: (point: GeoPoint) => void;
  onMapContextMenu: (
    point: GeoPoint,
    position: { x: number; y: number },
    waypointId?: string,
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
      { padding: [52, 52], maxZoom: 16 },
    );
  }, [map, minLat, maxLat, minLng, maxLng]);
  return null;
}

function droneIcon(
  color: string,
  heading: number,
  failed: boolean,
  selected: boolean,
): L.DivIcon {
  return L.divIcon({
    className: 'skygrid-leaflet-icon',
    html: `<div class="uav-map-marker ${failed ? 'is-failed' : ''} ${selected ? 'is-selected' : ''}" style="--uav-color:${color};--heading:${heading}deg">
      <span class="uav-pulse"></span><span class="uav-body"><i></i><b></b></span>
    </div>`,
    iconSize: [38, 38],
    iconAnchor: [19, 19],
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
  const routeLines = useMemo(
    () =>
      mission.drones.flatMap((drone) => {
        if (drone.status === 'failed') return [];
        const remaining = drone.route
          .slice(drone.routeIndex)
          .map((id) => mission.waypoints.find((waypoint) => waypoint.id === id))
          .filter(Boolean) as GeoPoint[];
        const points = [drone, ...remaining];
        const line: GeoPoint[] = [];
        for (let i = 1; i < points.length; i += 1) {
          const segment = aStarRoute(
            points[i - 1],
            points[i],
            mission.noFlyZones,
          );
          line.push(...(i === 1 ? segment : segment.slice(1)));
        }
        return line.length
          ? [{ id: drone.id, color: drone.color, points: line }]
          : [];
      }),
    [mission.drones, mission.waypoints, mission.noFlyZones],
  );

  const mapPoints =
    mode === 'analysis' && logs.length
      ? logs.flatMap((log) =>
          log.points.filter(
            (_, index) =>
              index % Math.max(1, Math.floor(log.points.length / 200)) === 0,
          ),
        )
      : mission.waypoints.length
        ? mission.waypoints
        : mission.drones;
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
      center={[36.6219, 127.5032]}
      zoom={15}
      zoomControl
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
              weight: 2.4,
              opacity: 0.88,
              dashArray: '9 8',
            }}
          />
        ))}

      {mission.waypoints.map((waypoint) => {
        const overdue =
          mission.time - waypoint.lastVisited > waypoint.revisitSec;
        const selected = waypoint.id === selectedWaypointId;
        return (
          <CircleMarker
            key={waypoint.id}
            center={[waypoint.lat, waypoint.lng]}
            radius={5 + waypoint.priority * 0.55 + (selected ? 2 : 0)}
            pathOptions={{
              color: selected
                ? '#ffffff'
                : overdue
                  ? '#ff786a'
                  : waypoint.priority >= 4
                    ? '#fbbf62'
                    : '#73d9ed',
              fillColor: overdue ? '#501f1b' : '#0b2730',
              fillOpacity: 0.9,
              weight: selected ? 3 : 1.5,
            }}
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
              중요도 {waypoint.priority} · 재방문 {waypoint.revisitSec}초
            </Tooltip>
          </CircleMarker>
        );
      })}

      {mode !== 'analysis' &&
        mission.drones.map((drone) => (
          <Marker
            key={drone.id}
            position={[drone.lat, drone.lng]}
            icon={droneIcon(
              drone.color,
              drone.heading,
              drone.status === 'failed',
              drone.id === selectedDroneId,
            )}
            eventHandlers={{ click: () => onSelectDrone(drone.id) }}
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
