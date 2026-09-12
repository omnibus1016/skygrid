'use client';

import dynamic from 'next/dynamic';
import {
  ChangeEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  AlertTriangle,
  BatteryMedium,
  Bot,
  BrainCircuit,
  CheckCircle2,
  CircleDot,
  Clock3,
  Crosshair,
  Database,
  Download,
  FileChartColumn,
  Gauge,
  LocateFixed,
  List,
  Map as MapIcon,
  MapPinPlus,
  Pause,
  Play,
  RefreshCw,
  Route,
  Satellite,
  ShieldCheck,
  Sparkles,
  StepForward,
  Trash2,
  Upload,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { formatMissionTime } from '@/lib/skygrid/geo';
import {
  DEFAULT_DRONE_PROFILE_ID,
  DRONE_PROFILES,
  getDroneProfile,
  profileSimulationValues,
} from '@/lib/skygrid/drone-profiles';
import {
  calculateLogMetrics,
  createDemoLogs,
  estimateHandoverSeconds,
  parseFlightCsv,
} from '@/lib/skygrid/log-parser';
import {
  CandidateDqn,
  assignRoutes,
  plannerScore,
  trainDqnPolicyAsync,
} from '@/lib/skygrid/rl-policy';
import {
  PRETRAINED_POLICY_STATS,
  PRETRAINED_POLICY_WEIGHTS,
} from '@/lib/skygrid/pretrained-policy';
import {
  advanceFieldMission,
  advanceMission,
  cloneMissionState,
  createMission,
  missionMetrics,
  restoreDrone,
  runBatchEvaluation,
  runScenarioComparison,
  triggerFailure,
} from '@/lib/skygrid/simulation';
import type {
  AppMode,
  Drone,
  DroneProfileId,
  FlightLog,
  GeoPoint,
  MissionState,
  PlannerKind,
  PolicyStats,
  ScenarioConfig,
  ScenarioComparisonResult,
  Waypoint,
} from '@/lib/skygrid/types';

const OperationalMap = dynamic(() => import('./operational-map'), {
  ssr: false,
  loading: () => (
    <div className="map-loading">
      <Crosshair className="animate-spin" /> 작전 지도를 초기화하는 중
    </div>
  ),
});

const DEFAULT_CONFIG: ScenarioConfig = {
  droneCount: 0,
  waypointCount: 0,
  durationSec: 600,
  failureAt: 90,
  failureDroneId: 'UAV-02',
  planner: 'rl',
  simRate: 4,
  sensorRadiusM: 110,
  randomSeed: 20261125,
};

const EVALUATION_CONFIG: ScenarioConfig = {
  ...DEFAULT_CONFIG,
  droneCount: 4,
  waypointCount: 20,
};

const PLANNER_LABEL: Record<PlannerKind, string> = {
  rl: '물리 제약 결합 DQN',
  nearest: '최근접 우선',
  priority: '중요도 우선',
};

function droneOperationalLabel(drone: Drone): string {
  if (drone.status === 'failed') return '이탈';
  if (drone.status === 'returning') return '기지 복귀';
  if (drone.status === 'ready')
    return drone.turnaroundRemainingSec > 0 ? '출격 준비' : '기지 대기';
  if (drone.phase === 'dwell') return '지점 정찰';
  return '목표 이동';
}

type PolicyBundle = {
  policy: CandidateDqn;
  stats: PolicyStats;
  origin: 'pretrained' | 'custom';
  trainedAt: string | null;
};

const POLICY_STORAGE_KEY = 'skygrid-policy-v2';

function createPretrainedPolicyBundle(): PolicyBundle {
  const policy = CandidateDqn.fromWeights(PRETRAINED_POLICY_WEIGHTS);
  if (!policy) throw new Error('기본 정책 가중치를 불러오지 못했습니다.');
  return {
    policy,
    stats: PRETRAINED_POLICY_STATS,
    origin: 'pretrained',
    trainedAt: null,
  };
}

function isPolicyStats(value: unknown): value is PolicyStats {
  if (!value || typeof value !== 'object') return false;
  const stats = value as Partial<PolicyStats>;
  return (
    typeof stats.episodes === 'number' &&
    typeof stats.averageReward === 'number' &&
    typeof stats.finalLoss === 'number' &&
    typeof stats.epsilon === 'number' &&
    Array.isArray(stats.rewardHistory)
  );
}

function readSavedPolicyBundle(): PolicyBundle | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(POLICY_STORAGE_KEY);
    if (!raw) return null;
    const saved = JSON.parse(raw) as {
      weights?: unknown;
      stats?: unknown;
      trainedAt?: unknown;
    };
    const policy = CandidateDqn.fromWeights(saved.weights);
    if (!policy || !isPolicyStats(saved.stats)) return null;
    return {
      policy,
      stats: saved.stats,
      origin: 'custom',
      trainedAt: typeof saved.trainedAt === 'string' ? saved.trainedAt : null,
    };
  } catch {
    return null;
  }
}

function savePolicyBundle(bundle: PolicyBundle): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(
      POLICY_STORAGE_KEY,
      JSON.stringify({
        weights: bundle.policy.toWeights(),
        stats: bundle.stats,
        trainedAt: bundle.trainedAt,
      }),
    );
  } catch {
    // Private browsing or a full storage quota should not stop the mission UI.
  }
}

type MapContextMenu = {
  x: number;
  y: number;
  point: GeoPoint;
  waypointId?: string;
  droneId?: string;
};

function ResultRow({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  tone?: 'neutral' | 'amber' | 'red';
}) {
  return (
    <div className={`result-row tone-${tone}`}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  tone = 'cyan',
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone?: 'cyan' | 'amber' | 'red';
}) {
  return (
    <div className={`metric-card tone-${tone}`}>
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function downloadText(
  fileName: string,
  text: string,
  type = 'text/csv;charset=utf-8',
) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function SkygridApp() {
  const [policyBundle, setPolicyBundle] = useState<PolicyBundle>(() =>
    createPretrainedPolicyBundle(),
  );
  const policyStats = policyBundle.stats;
  const [mode, setMode] = useState<AppMode>('simulation');
  const [config, setConfig] = useState<ScenarioConfig>(DEFAULT_CONFIG);
  const [mission, setMission] = useState(() =>
    createMission(DEFAULT_CONFIG, policyBundle.policy),
  );
  const [selectedDroneId, setSelectedDroneId] = useState('UAV-01');
  const [selectedWaypointId, setSelectedWaypointId] = useState('RP-01');
  const [selectedMapEntity, setSelectedMapEntity] = useState<
    'drone' | 'waypoint'
  >('waypoint');
  const [interaction, setInteraction] = useState<
    'inspect' | 'add-base' | 'add-waypoint' | 'add-drone' | 'move-drone'
  >('inspect');
  const [logs, setLogs] = useState<FlightLog[]>(() => createDemoLogs(mission));
  const [selectedLogId, setSelectedLogId] = useState('DEMO-UAV-01');
  const [uploadError, setUploadError] = useState('');
  const [batchResults, setBatchResults] = useState(() =>
    runBatchEvaluation(EVALUATION_CONFIG, policyBundle.policy, 48),
  );
  const [comparisonBaseline, setComparisonBaseline] =
    useState<MissionState | null>(null);
  const [comparisonConfig, setComparisonConfig] =
    useState<ScenarioConfig | null>(null);
  const [scenarioComparison, setScenarioComparison] = useState<
    ScenarioComparisonResult[] | null
  >(null);
  const [busyAction, setBusyAction] = useState<
    'train' | 'batch' | 'compare' | null
  >(null);
  const [trainingEpisodes, setTrainingEpisodes] = useState(5_000);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStage, setTrainingStage] = useState('');
  const [trainingNotice, setTrainingNotice] = useState('');
  const [dropoutDroneId, setDropoutDroneId] = useState('UAV-02');
  const [dropoutReason, setDropoutReason] = useState('통신 두절');
  const [mapBase, setMapBase] = useState<'satellite' | 'street'>('satellite');
  const [mapContextMenu, setMapContextMenu] = useState<MapContextMenu | null>(
    null,
  );
  const [droneStatusOpen, setDroneStatusOpen] = useState(false);
  const [pendingDronePoint, setPendingDronePoint] = useState<GeoPoint | null>(
    null,
  );
  const [pendingDroneProfileId, setPendingDroneProfileId] =
    useState<DroneProfileId>(DEFAULT_DRONE_PROFILE_ID);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const saved = readSavedPolicyBundle();
    if (!saved) return;
    const timer = window.setTimeout(() => {
      setPolicyBundle(saved);
      setMission(createMission(DEFAULT_CONFIG, saved.policy));
      setBatchResults(runBatchEvaluation(EVALUATION_CONFIG, saved.policy, 48));
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const metrics = useMemo(() => missionMetrics(mission), [mission]);
  const hasMissionResults = mission.time > 0 || mission.completed;
  const selectedDrone =
    mission.drones.find((drone) => drone.id === selectedDroneId) ??
    mission.drones[0];
  const selectedWaypoint =
    mission.waypoints.find((waypoint) => waypoint.id === selectedWaypointId) ??
    mission.waypoints[0];
  const deleteTargetLabel =
    selectedMapEntity === 'drone'
      ? (selectedDrone?.id ?? '선택 기체')
      : (selectedWaypoint?.id ?? '선택지점');
  const canDeleteSelectedEntity =
    selectedMapEntity === 'drone'
      ? Boolean(selectedDrone)
      : Boolean(selectedWaypoint);
  const missionReady =
    mission.base.configured &&
    mission.drones.length > 0 &&
    mission.waypoints.length > 0;
  const activeDropoutDroneId = mission.drones.some(
    (drone) => drone.id === dropoutDroneId && drone.status !== 'failed',
  )
    ? dropoutDroneId
    : (mission.drones.find((drone) => drone.status !== 'failed')?.id ?? '');
  const selectedLog = logs.find((log) => log.id === selectedLogId) ?? logs[0];
  const selectedLogMetrics = useMemo(
    () =>
      selectedLog
        ? calculateLogMetrics(selectedLog, selectedLog.plannedPath ?? [])
        : null,
    [selectedLog],
  );
  const handoverSeconds = useMemo(() => estimateHandoverSeconds(logs), [logs]);
  const flightSeries = useMemo(() => {
    if (!selectedLog) return [];
    const stride = Math.max(1, Math.floor(selectedLog.points.length / 180));
    return selectedLog.points
      .filter((_, index) => index % stride === 0)
      .map((point) => ({
        time: Math.round(point.timeMs / 1000),
        speed: Number(point.speedMps.toFixed(2)),
        battery: Number(point.batteryPercent.toFixed(1)),
        altitude: Number(point.altitudeM.toFixed(1)),
      }));
  }, [selectedLog]);

  const candidateScores = useMemo(() => {
    if (!selectedDrone) return [];
    return mission.waypoints
      .map((waypoint) => ({
        waypoint,
        score: plannerScore(
          'rl',
          selectedDrone,
          waypoint,
          mission.time,
          mission.waypoints.length,
          policyBundle.policy,
          mission.drones,
          mission.waypoints,
        ),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);
  }, [
    mission.drones,
    mission.waypoints,
    mission.time,
    selectedDrone,
    policyBundle.policy,
  ]);

  useEffect(() => {
    if (!mission.running) return;
    const timer = window.setInterval(() => {
      setMission((current) =>
        mode === 'field'
          ? advanceFieldMission(current, 0.25)
          : advanceMission(
              current,
              0.25 * config.simRate,
              config,
              policyBundle.policy,
            ),
      );
    }, 250);
    return () => window.clearInterval(timer);
  }, [mission.running, config, mode, policyBundle.policy]);

  const applyScenario = useCallback(
    (nextConfig = config) => {
      const next = createMission(nextConfig, policyBundle.policy);
      setMission(next);
      setSelectedDroneId(next.drones[0]?.id ?? '');
      setSelectedWaypointId(next.waypoints[0]?.id ?? '');
      setComparisonBaseline(null);
      setComparisonConfig(null);
      setScenarioComparison(null);
      setInteraction('inspect');
    },
    [config, policyBundle.policy],
  );

  const clearScenario = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
    applyScenario(DEFAULT_CONFIG);
  }, [applyScenario]);

  const replaySimulation = useCallback(() => {
    if (!missionReady) return;
    const base = comparisonBaseline
      ? cloneMissionState(comparisonBaseline)
      : createMission(config, policyBundle.policy);
    const replay = {
      ...base,
      running: true,
      completed: false,
      events: [
        {
          id: `replay-${Date.now()}`,
          time: 0,
          kind: 'system' as const,
          title: '가상 정찰 임무 재시작',
          detail: `초기 위치·배터리·정찰 상태 복원 · ${config.simRate}배속`,
        },
        ...base.events,
      ].slice(0, 18),
    };
    setMission(replay);
    setSelectedDroneId(replay.drones[0]?.id ?? '');
    setSelectedWaypointId(replay.waypoints[0]?.id ?? '');
    setSelectedMapEntity('waypoint');
    setInteraction('inspect');
    setComparisonBaseline(cloneMissionState(base));
    setComparisonConfig({ ...config });
    setScenarioComparison(null);
  }, [comparisonBaseline, config, missionReady, policyBundle.policy]);

  const applySimulationSettings = useCallback(() => {
    const start = performance.now();
    const durationSec = Math.max(60, config.durationSec);
    const failureAt = Math.min(
      config.failureAt,
      Math.max(30, durationSec - 10),
    );
    const failureDroneId =
      mission.drones.find((drone) => drone.id === config.failureDroneId)?.id ??
      mission.drones.find((drone) => drone.status === 'active')?.id ??
      config.failureDroneId;
    setConfig((currentConfig) => ({
      ...currentConfig,
      durationSec,
      failureAt,
      failureDroneId,
    }));
    setComparisonBaseline(null);
    setComparisonConfig(null);
    setScenarioComparison(null);
    setMission((current) => {
      const planned = assignRoutes(
        current.drones,
        current.waypoints,
        current.time,
        config.planner,
        policyBundle.policy,
      );
      return {
        ...current,
        drones: planned.drones,
        waypoints: planned.waypoints,
        replanCount: current.replanCount + 1,
        inferenceMs: Math.max(1, performance.now() - start),
        events: [
          {
            id: `settings-${Date.now()}`,
            time: current.time,
            kind: 'system' as const,
            title: '실험 설정 적용',
            detail: `임무 ${formatMissionTime(durationSec)} · 이탈 ${formatMissionTime(failureAt)} · ${config.simRate}배속 · ${PLANNER_LABEL[config.planner]}`,
          },
          ...current.events,
        ].slice(0, 18),
      };
    });
  }, [
    config.durationSec,
    config.failureAt,
    config.failureDroneId,
    config.planner,
    config.simRate,
    mission.drones,
    policyBundle.policy,
  ]);

  const replanNow = useCallback(
    (planner = config.planner) => {
      const start = performance.now();
      setMission((current) => {
        const planned = assignRoutes(
          current.drones,
          current.waypoints,
          current.time,
          planner,
          policyBundle.policy,
        );
        return {
          ...current,
          drones: planned.drones,
          waypoints: planned.waypoints,
          replanCount: current.replanCount + 1,
          inferenceMs: Math.max(1, performance.now() - start),
          events: [
            {
              id: `manual-${Date.now()}`,
              time: current.time,
              kind: 'replan' as const,
              title: '운용자 요청 재계획 완료',
              detail: PLANNER_LABEL[planner],
            },
            ...current.events,
          ].slice(0, 18),
        };
      });
    },
    [config.planner, policyBundle.policy],
  );

  const executeDropout = useCallback(() => {
    if (!activeDropoutDroneId) return;
    setMission((current) =>
      triggerFailure(
        current,
        activeDropoutDroneId,
        config.planner,
        policyBundle.policy,
        dropoutReason,
      ),
    );
    setSelectedDroneId(activeDropoutDroneId);
  }, [
    activeDropoutDroneId,
    dropoutReason,
    config.planner,
    policyBundle.policy,
  ]);

  const recoverSelectedDrone = useCallback(() => {
    if (!selectedDroneId) return;
    setMission((current) =>
      restoreDrone(
        current,
        selectedDroneId,
        config.planner,
        policyBundle.policy,
      ),
    );
  }, [selectedDroneId, config.planner, policyBundle.policy]);

  const toggleMission = useCallback(() => {
    if (!mission.running && !missionReady) return;
    if (
      !mission.running &&
      !mission.completed &&
      mode === 'simulation' &&
      !comparisonBaseline
    ) {
      setComparisonBaseline(cloneMissionState(mission));
      setComparisonConfig({ ...config });
      setScenarioComparison(null);
    }
    setMission((current) => {
      const running = !current.running;
      if (running && (!current.drones.length || !current.waypoints.length)) {
        return current;
      }
      return {
        ...current,
        running,
        events: [
          {
            id: `clock-${Date.now()}`,
            time: current.time,
            kind: 'system' as const,
            title: running
              ? mode === 'field'
                ? '현장 임무 기록 시작'
                : '가상 정찰 임무 시작'
              : '임무 일시 정지',
            detail:
              mode === 'field'
                ? '기체 위치와 정찰 결과는 운용자 입력으로만 갱신됩니다.'
                : `시뮬레이션 ${config.simRate}배속`,
          },
          ...current.events,
        ].slice(0, 18),
      };
    });
  }, [comparisonBaseline, config, mission, missionReady, mode]);

  const stepSimulation = useCallback(() => {
    if (!missionReady) return;
    if (!comparisonBaseline) {
      setComparisonBaseline(cloneMissionState(mission));
      setComparisonConfig({ ...config });
    }
    setMission((current) => ({
      ...(current.drones.length && current.waypoints.length
        ? advanceMission(
            { ...current, running: true },
            10,
            config,
            policyBundle.policy,
          )
        : current),
      running: false,
    }));
  }, [comparisonBaseline, config, mission, missionReady, policyBundle.policy]);

  const addWaypointAt = useCallback(
    (point: GeoPoint) => {
      const nextNumber =
        Math.max(
          0,
          ...mission.waypoints.map((waypoint) =>
            Number.parseInt(waypoint.id.replace(/\D/g, ''), 10),
          ),
        ) + 1;
      const waypointId = `RP-${String(nextNumber).padStart(2, '0')}`;
      setSelectedWaypointId(waypointId);
      setSelectedMapEntity('waypoint');
      setConfig((current) => ({
        ...current,
        waypointCount: Math.max(
          current.waypointCount,
          mission.waypoints.length + 1,
        ),
      }));
      setMission((current) => {
        const waypoint: Waypoint = {
          id: waypointId,
          ...point,
          priority: 4,
          revisitSec: 150,
          dwellSec: 30,
          lastVisited: current.time - 151,
          visitedCount: 0,
        };
        const waypoints = [...current.waypoints, waypoint];
        const planned = assignRoutes(
          current.drones,
          waypoints,
          current.time,
          config.planner,
          policyBundle.policy,
        );
        return {
          ...current,
          drones: planned.drones,
          waypoints: planned.waypoints,
          replanCount: current.replanCount + 1,
          events: [
            {
              id: `waypoint-add-${Date.now()}`,
              time: current.time,
              kind: 'replan' as const,
              title: `${waypoint.id} 정찰지점 추가`,
              detail: '정찰지점 추가 · 경로 갱신',
            },
            ...current.events,
          ].slice(0, 18),
        };
      });
      setInteraction('inspect');
      setMapContextMenu(null);
    },
    [mission.waypoints, config.planner, policyBundle.policy],
  );

  const addDroneAt = useCallback(
    (point: GeoPoint, profileId: DroneProfileId) => {
      const nextNumber =
        Math.max(
          0,
          ...mission.drones.map((drone) =>
            Number.parseInt(drone.id.replace(/\D/g, ''), 10),
          ),
        ) + 1;
      const droneId = `UAV-${String(nextNumber).padStart(2, '0')}`;
      const profile = getDroneProfile(profileId);
      const performance = profileSimulationValues(profile);
      const newDrone: Drone = {
        id: droneId,
        ...performance,
        color: ['#67e8f9', '#fbbf62', '#a78bfa', '#64d39b', '#f472b6'][
          nextNumber % 5
        ],
        ...point,
        homeLat: mission.base.lat,
        homeLng: mission.base.lng,
        turnaroundRemainingSec: 0,
        dwellRemainingSec: 0,
        phase: 'transit',
        sortieCount: 1,
        minimumBattery: performance.maxBattery,
        status: 'active',
        route: [],
        routeIndex: 0,
        totalDistanceM: 0,
        heading: 0,
      };
      setSelectedDroneId(droneId);
      setSelectedMapEntity('drone');
      setConfig((current) => ({
        ...current,
        droneCount: Math.max(current.droneCount, mission.drones.length + 1),
      }));
      setMission((current) => {
        const drones = [...current.drones, newDrone];
        const planned = assignRoutes(
          drones,
          current.waypoints,
          current.time,
          config.planner,
          policyBundle.policy,
        );
        return {
          ...current,
          drones: planned.drones,
          waypoints: planned.waypoints,
          replanCount: current.replanCount + 1,
          events: [
            {
              id: `drone-add-${Date.now()}`,
              time: current.time,
              kind: 'replan' as const,
              title: `${droneId} 기체 추가`,
              detail: `${profile.model} · 경로 갱신`,
            },
            ...current.events,
          ].slice(0, 18),
        };
      });
      setInteraction('inspect');
      setMapContextMenu(null);
      setPendingDronePoint(null);
    },
    [mission.drones, mission.base, config.planner, policyBundle.policy],
  );

  const requestDronePlacement = useCallback((point: GeoPoint) => {
    setPendingDroneProfileId(DEFAULT_DRONE_PROFILE_ID);
    setPendingDronePoint(point);
    setMapContextMenu(null);
  }, []);

  const setBaseAt = useCallback(
    (point: GeoPoint) => {
      setMission((current) => {
        const base = { ...current.base, ...point, configured: true };
        const drones = current.drones.map((drone, index) => {
          const stagedAtBase =
            drone.status === 'ready' && drone.phase === 'base';
          const offset = (index - (current.drones.length - 1) / 2) * 0.00007;
          return {
            ...drone,
            homeLat: point.lat,
            homeLng: point.lng,
            ...(stagedAtBase
              ? {
                  lat: point.lat + offset,
                  lng: point.lng + offset * 0.7,
                }
              : {}),
          };
        });
        const planned = assignRoutes(
          drones,
          current.waypoints,
          current.time,
          config.planner,
          policyBundle.policy,
        );
        return {
          ...current,
          base,
          drones: planned.drones,
          waypoints: planned.waypoints,
          replanCount: current.replanCount + 1,
          events: [
            {
              id: `base-${Date.now()}`,
              time: current.time,
              kind: 'system' as const,
              title: `${base.id} 위치 지정`,
              detail: `${point.lat.toFixed(5)}, ${point.lng.toFixed(5)}`,
            },
            ...current.events,
          ].slice(0, 18),
        };
      });
      setInteraction('inspect');
      setMapContextMenu(null);
    },
    [config.planner, policyBundle.policy],
  );

  const handleMapClick = useCallback(
    (point: GeoPoint) => {
      if (interaction === 'add-waypoint') {
        addWaypointAt(point);
      } else if (interaction === 'add-drone') {
        requestDronePlacement(point);
      } else if (interaction === 'add-base') {
        setBaseAt(point);
      } else if (interaction === 'move-drone' && selectedDroneId) {
        setMission((current) => {
          const drones = current.drones.map((drone) =>
            drone.id === selectedDroneId
              ? {
                  ...drone,
                  ...point,
                  status:
                    drone.status === 'failed'
                      ? ('failed' as const)
                      : ('active' as const),
                  phase:
                    drone.status === 'failed'
                      ? drone.phase
                      : ('transit' as const),
                  turnaroundRemainingSec: 0,
                  sortieCount:
                    drone.status === 'failed'
                      ? drone.sortieCount
                      : Math.max(1, drone.sortieCount),
                }
              : drone,
          );
          const planned = assignRoutes(
            drones,
            current.waypoints,
            current.time,
            config.planner,
            policyBundle.policy,
          );
          return {
            ...current,
            drones: planned.drones,
            waypoints: planned.waypoints,
            replanCount: current.replanCount + 1,
            events: [
              {
                id: `position-${Date.now()}`,
                time: current.time,
                kind: 'system' as const,
                title: `${selectedDroneId} 위치 업데이트`,
                detail: `${point.lat.toFixed(6)}, ${point.lng.toFixed(6)} · 경로 갱신`,
              },
              ...current.events,
            ].slice(0, 18),
          };
        });
        setInteraction('inspect');
        setMapContextMenu(null);
      }
    },
    [
      interaction,
      selectedDroneId,
      addWaypointAt,
      requestDronePlacement,
      setBaseAt,
      config.planner,
      policyBundle.policy,
    ],
  );

  const deleteWaypointById = useCallback(
    (waypointId: string) => {
      if (!waypointId) return;
      setSelectedWaypointId(
        mission.waypoints.find((waypoint) => waypoint.id !== waypointId)?.id ??
          '',
      );
      setSelectedMapEntity('waypoint');
      setConfig((current) => ({
        ...current,
        waypointCount: Math.max(0, mission.waypoints.length - 1),
      }));
      setMission((current) => {
        const waypoints = current.waypoints.filter(
          (waypoint) => waypoint.id !== waypointId,
        );
        const planned = assignRoutes(
          current.drones,
          waypoints,
          current.time,
          config.planner,
          policyBundle.policy,
        );
        return {
          ...current,
          running: current.running && waypoints.length > 0,
          drones: planned.drones,
          waypoints: planned.waypoints,
          replanCount: current.replanCount + 1,
          events: [
            {
              id: `waypoint-delete-${Date.now()}`,
              time: current.time,
              kind: 'warning' as const,
              title: `${waypointId} 정찰지점 삭제`,
              detail: '잔여 정찰지점 · 경로 갱신',
            },
            ...current.events,
          ].slice(0, 18),
        };
      });
      setMapContextMenu(null);
    },
    [mission.waypoints, config.planner, policyBundle.policy],
  );

  const deleteSelectedWaypoint = useCallback(() => {
    if (selectedWaypointId) deleteWaypointById(selectedWaypointId);
  }, [selectedWaypointId, deleteWaypointById]);

  const deleteDroneById = useCallback(
    (droneId: string) => {
      if (!droneId) return;
      const remainingDrones = mission.drones.filter(
        (drone) => drone.id !== droneId,
      );
      setSelectedDroneId(remainingDrones[0]?.id ?? '');
      setSelectedMapEntity('drone');
      setInteraction('inspect');
      setConfig((current) => ({
        ...current,
        droneCount: remainingDrones.length,
      }));
      setMission((current) => {
        const drones = current.drones.filter((drone) => drone.id !== droneId);
        const waypoints = current.waypoints.map((waypoint) =>
          waypoint.assignedDrone === droneId
            ? { ...waypoint, assignedDrone: undefined }
            : waypoint,
        );
        const planned = assignRoutes(
          drones,
          waypoints,
          current.time,
          config.planner,
          policyBundle.policy,
        );
        return {
          ...current,
          running: current.running && drones.length > 0,
          drones: planned.drones,
          waypoints: planned.waypoints,
          replanCount: current.replanCount + 1,
          events: [
            {
              id: `drone-delete-${Date.now()}`,
              time: current.time,
              kind: 'warning' as const,
              title: `${droneId} 기체 삭제`,
              detail: '남은 기체에 정찰지점을 다시 배정했습니다.',
            },
            ...current.events,
          ].slice(0, 18),
        };
      });
      setMapContextMenu(null);
    },
    [mission.drones, config.planner, policyBundle.policy],
  );

  const handleMapContextMenu = useCallback(
    (
      point: GeoPoint,
      position: { x: number; y: number },
      waypointId?: string,
      droneId?: string,
    ) => {
      if (waypointId) {
        setSelectedWaypointId(waypointId);
        setSelectedMapEntity('waypoint');
      }
      if (droneId) {
        setSelectedDroneId(droneId);
        setSelectedMapEntity('drone');
      }
      setMapContextMenu({ ...position, point, waypointId, droneId });
    },
    [],
  );

  const closeMapContextMenu = useCallback(() => {
    setMapContextMenu(null);
  }, []);

  const deleteSelectedMapEntity = useCallback(() => {
    if (selectedMapEntity === 'drone') {
      if (selectedDroneId) deleteDroneById(selectedDroneId);
      return;
    }
    if (selectedWaypointId) deleteWaypointById(selectedWaypointId);
  }, [
    deleteDroneById,
    deleteWaypointById,
    selectedDroneId,
    selectedMapEntity,
    selectedWaypointId,
  ]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping =
        target?.tagName === 'INPUT' ||
        target?.tagName === 'TEXTAREA' ||
        target?.tagName === 'SELECT' ||
        target?.isContentEditable;
      if (
        !isTyping &&
        mode !== 'analysis' &&
        (event.key === 'Backspace' || event.key === 'Delete')
      ) {
        event.preventDefault();
        deleteSelectedMapEntity();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [deleteSelectedMapEntity, mode]);

  const reportWaypointVisit = useCallback(() => {
    if (!selectedWaypointId) return;
    setMission((current) => {
      const waypoints = current.waypoints.map((waypoint) =>
        waypoint.id === selectedWaypointId
          ? {
              ...waypoint,
              lastVisited: current.time,
              visitedCount: waypoint.visitedCount + 1,
            }
          : waypoint,
      );
      const planned = assignRoutes(
        current.drones,
        waypoints,
        current.time,
        config.planner,
        policyBundle.policy,
      );
      return {
        ...current,
        drones: planned.drones,
        waypoints: planned.waypoints,
        replanCount: current.replanCount + 1,
        events: [
          {
            id: `visit-report-${Date.now()}`,
            time: current.time,
            kind: 'visit' as const,
            title: `${selectedWaypointId} 정찰 완료 보고`,
            detail: `${selectedDroneId || '운용 기체'} 관측 결과 반영 · 후속 경로 갱신`,
          },
          ...current.events,
        ].slice(0, 18),
      };
    });
  }, [
    selectedWaypointId,
    selectedDroneId,
    config.planner,
    policyBundle.policy,
  ]);

  const updateWaypoint = useCallback(
    (patch: Partial<Waypoint>) => {
      if (!selectedWaypointId) return;
      setMission((current) => ({
        ...current,
        waypoints: current.waypoints.map((waypoint) =>
          waypoint.id === selectedWaypointId
            ? { ...waypoint, ...patch }
            : waypoint,
        ),
      }));
    },
    [selectedWaypointId],
  );

  const updateSelectedDroneBattery = useCallback(
    (battery: number) => {
      setMission((current) => ({
        ...current,
        drones: current.drones.map((drone) =>
          drone.id === selectedDroneId ? { ...drone, battery } : drone,
        ),
      }));
    },
    [selectedDroneId],
  );

  const handleFiles = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const files = [...(event.target.files ?? [])];
      if (!files.length) return;
      const parsed: FlightLog[] = [];
      const errors: string[] = [];
      for (let index = 0; index < files.length; index += 1) {
        try {
          parsed.push(
            parseFlightCsv(await files[index].text(), files[index].name, index),
          );
        } catch (error) {
          errors.push(
            `${files[index].name}: ${error instanceof Error ? error.message : '분석 실패'}`,
          );
        }
      }
      if (parsed.length) {
        setLogs(parsed);
        setSelectedLogId(parsed[0].id);
        setMission((current) => ({
          ...current,
          events: [
            {
              id: `upload-${Date.now()}`,
              time: current.time,
              kind: 'upload',
              title: '비행 로그 분석 완료',
              detail: `${parsed.length}개 파일 · ${parsed.reduce((sum, log) => sum + log.points.length, 0).toLocaleString()} 표본`,
            },
            ...current.events,
          ],
        }));
      }
      setUploadError(errors.join(' / '));
      event.target.value = '';
    },
    [],
  );

  const trainAgain = useCallback(() => {
    setBusyAction('train');
    setTrainingProgress(0);
    setTrainingStage('학습 환경 생성');
    setTrainingNotice('');
    window.setTimeout(() => {
      void (async () => {
        try {
          const trainingContext = {
            base: { lat: mission.base.lat, lng: mission.base.lng },
            drones: mission.drones,
            waypoints: mission.waypoints,
            missionTime: mission.time,
            durationSec: config.durationSec,
            failureAt: config.failureAt,
          };
          const next = await trainDqnPolicyAsync(
            trainingEpisodes,
            config.randomSeed + policyStats.episodes,
            (progress) => {
              setTrainingStage(
                progress.stage === 'validation'
                  ? '미사용 시나리오 검증'
                  : '가중치 학습',
              );
              setTrainingProgress(
                Math.round((progress.episode / progress.totalEpisodes) * 100),
              );
            },
            trainingContext,
          );
          const validation = next.stats.validation;
          if (!validation?.passed) {
            setTrainingNotice(
              `검증 점수 ${validation?.rlScore.toFixed(1) ?? '-'} · 기준 성능을 넘지 못해 현재 모델 유지`,
            );
            return;
          }
          const nextBundle: PolicyBundle = {
            ...next,
            origin: 'custom',
            trainedAt: new Date().toISOString(),
          };
          savePolicyBundle(nextBundle);
          setPolicyBundle(nextBundle);
          setTrainingProgress(100);
          setTrainingStage('검증 통과');
          setTrainingNotice(
            `검증 ${validation.scenarios}개 · 기준 대비 +${validation.improvementVsBest.toFixed(1)}점 · 새 모델 적용`,
          );
          setMission((current) => {
            const planned = assignRoutes(
              current.drones,
              current.waypoints,
              current.time,
              config.planner,
              next.policy,
            );
            return {
              ...current,
              drones: planned.drones,
              waypoints: planned.waypoints,
              replanCount: current.replanCount + 1,
              events: [
                {
                  id: 'policy-' + Date.now(),
                  time: current.time,
                  kind: 'replan' as const,
                  title: '새 모델 적용',
                  detail:
                    trainingEpisodes.toLocaleString() +
                    '회 학습 모델 · 전체 경로 갱신',
                },
                ...current.events,
              ].slice(0, 18),
            };
          });
        } catch {
          setTrainingProgress(0);
          setTrainingStage('');
          setTrainingNotice('학습을 완료하지 못했습니다. 다시 실행해 주세요.');
        } finally {
          setBusyAction(null);
        }
      })();
    }, 40);
  }, [
    config.planner,
    config.randomSeed,
    config.durationSec,
    config.failureAt,
    mission,
    policyStats.episodes,
    trainingEpisodes,
  ]);

  const restoreDefaultPolicy = useCallback(() => {
    const next = createPretrainedPolicyBundle();
    window.localStorage.removeItem(POLICY_STORAGE_KEY);
    setPolicyBundle(next);
    setMission((current) => {
      const planned = assignRoutes(
        current.drones,
        current.waypoints,
        current.time,
        config.planner,
        next.policy,
      );
      return {
        ...current,
        drones: planned.drones,
        waypoints: planned.waypoints,
        replanCount: current.replanCount + 1,
        events: [
          {
            id: 'policy-default-' + Date.now(),
            time: current.time,
            kind: 'replan' as const,
            title: '기본 모델 복원',
            detail: '기본 가중치 · 전체 경로 갱신',
          },
          ...current.events,
        ].slice(0, 18),
      };
    });
  }, [config.planner]);

  const runBatch = useCallback(() => {
    setBusyAction('batch');
    window.setTimeout(() => {
      setBatchResults(
        runBatchEvaluation(
          config.droneCount && config.waypointCount
            ? config
            : EVALUATION_CONFIG,
          policyBundle.policy,
          100,
        ),
      );
      setBusyAction(null);
    }, 40);
  }, [config, policyBundle.policy]);

  const runCurrentScenarioComparison = useCallback(() => {
    if (!comparisonBaseline || !comparisonConfig) return;
    setBusyAction('compare');
    window.setTimeout(() => {
      setScenarioComparison(
        runScenarioComparison(
          comparisonBaseline,
          comparisonConfig,
          policyBundle.policy,
          Math.max(1, mission.time - comparisonBaseline.time),
        ),
      );
      setBusyAction(null);
    }, 40);
  }, [comparisonBaseline, comparisonConfig, mission.time, policyBundle.policy]);

  const changeMode = useCallback(
    (nextMode: AppMode) => {
      if (nextMode === mode) return;
      if (nextMode === 'field') {
        const fieldConfig = {
          ...config,
          droneCount: 2,
          failureDroneId: 'UAV-02',
          simRate: 1,
        };
        setConfig(fieldConfig);
        applyScenario(fieldConfig);
      } else if (nextMode === 'simulation' && mode === 'field') {
        setConfig(DEFAULT_CONFIG);
        applyScenario(DEFAULT_CONFIG);
      }
      setMode(nextMode);
    },
    [applyScenario, config, mode],
  );

  const exportAnalysis = useCallback(() => {
    const rows = [
      'log,duration_sec,distance_km,avg_speed_mps,max_altitude_m,battery_used_pct,battery_pct_per_km,samples',
    ];
    logs.forEach((log) => {
      const result = calculateLogMetrics(log);
      rows.push(
        [
          log.droneName,
          result.durationSec.toFixed(1),
          result.distanceKm.toFixed(3),
          result.avgSpeedMps.toFixed(2),
          result.maxAltitudeM.toFixed(1),
          result.batteryUsed.toFixed(1),
          result.batteryPerKm.toFixed(2),
          result.sampleCount,
        ].join(','),
      );
    });
    downloadText('skygrid-flight-analysis.csv', rows.join('\n'));
  }, [logs]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const register = async () => {
      await context.registerTool(
        {
          name: 'skygrid_configure_scenario',
          title: 'SKYGRID 시나리오 설정',
          description: '기체 수·정찰지점 수·이탈 시점 설정',
          inputSchema: {
            type: 'object',
            properties: {
              droneCount: { type: 'number', minimum: 0, maximum: 10 },
              waypointCount: { type: 'number', minimum: 0, maximum: 30 },
              durationSec: { type: 'number', minimum: 60, maximum: 7200 },
              failureAt: { type: 'number', minimum: 30, maximum: 7140 },
            },
            required: ['droneCount', 'waypointCount', 'failureAt'],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          execute: async (input: unknown) => {
            const value = input as {
              droneCount: number;
              waypointCount: number;
              durationSec?: number;
              failureAt: number;
            };
            const durationSec = Math.min(
              7200,
              Math.max(60, value.durationSec ?? config.durationSec),
            );
            const next = {
              ...config,
              droneCount: value.droneCount,
              waypointCount: value.waypointCount,
              durationSec,
              failureAt: Math.min(
                value.failureAt,
                Math.max(30, durationSec - 10),
              ),
              failureDroneId: 'UAV-02',
            };
            setConfig(next);
            applyScenario(next);
            setMode('simulation');
            return { status: 'configured', ...value };
          },
        },
        { signal: lifecycle.signal },
      );
      await context.registerTool(
        {
          name: 'skygrid_start_simulation',
          title: 'SKYGRID 시뮬레이션 시작',
          description: '현재 가상 임무 시작',
          inputSchema: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          execute: async () => {
            setMode('simulation');
            setMission((current) => ({ ...current, running: true }));
            return { status: 'running' };
          },
        },
        { signal: lifecycle.signal },
      );
      await context.registerTool(
        {
          name: 'skygrid_trigger_dropout',
          title: '무인기 이탈 처리',
          description: '지정 기체 이탈 처리 · 잔여 임무 재계획',
          inputSchema: {
            type: 'object',
            properties: { droneId: { type: 'string' } },
            required: ['droneId'],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          execute: async (input: unknown) => {
            const droneId = (input as { droneId: string }).droneId;
            setMission((current) =>
              triggerFailure(
                current,
                droneId,
                config.planner,
                policyBundle.policy,
              ),
            );
            return { status: 'replanned', droneId, planner: config.planner };
          },
        },
        { signal: lifecycle.signal },
      );
    };
    void register().catch(() => undefined);
    return () => lifecycle.abort();
  }, [applyScenario, config, policyBundle.policy]);

  return (
    <main className="min-h-screen bg-background text-foreground">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.txt,text/csv,text/plain"
        multiple
        className="sr-only"
        onChange={handleFiles}
      />
      <header className="mission-header">
        <div className="brand-lockup">
          <div className="brand-insignia" aria-label="대한민국 공군과 KAIST">
            {/* oxlint-disable-next-line next/no-img-element -- relative public path supports both localhost and GitHub Pages */}
            <img
              className="header-rokaf"
              src="brand/rokaf-emblem.png"
              alt="대한민국 공군 표장"
            />
            {/* oxlint-disable-next-line next/no-img-element -- relative public path supports both localhost and GitHub Pages */}
            <img
              className="header-kaist"
              src="brand/kaist-wordmark.png"
              alt="KAIST"
            />
          </div>
          <div>
            <div className="brand-name">SKYGRID</div>
            <div className="brand-sub">군집 무인기 정찰 경로 최적화 플랫폼</div>
          </div>
        </div>
        <Tabs
          value={mode}
          onValueChange={(value) => changeMode(value as AppMode)}
          className="hidden lg:flex"
        >
          <TabsList variant="line" className="h-14 gap-1">
            <TabsTrigger value="field" className="mission-tab">
              현장 운용
            </TabsTrigger>
            <TabsTrigger value="analysis" className="mission-tab">
              비행 검증
            </TabsTrigger>
            <TabsTrigger value="simulation" className="mission-tab">
              임무 모의
            </TabsTrigger>
            <TabsTrigger value="evaluation" className="mission-tab">
              성능 평가
            </TabsTrigger>
            <TabsTrigger value="training" className="mission-tab">
              모델 학습
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      {mode === 'training' ? (
        <TrainingWorkspace
          policyBundle={policyBundle}
          trainingEpisodes={trainingEpisodes}
          trainingProgress={trainingProgress}
          trainingStage={trainingStage}
          trainingNotice={trainingNotice}
          currentSetupReady={
            mission.drones.length >= 2 && mission.waypoints.length >= 3
          }
          onTrainingEpisodesChange={setTrainingEpisodes}
          onTrain={trainAgain}
          onRestore={restoreDefaultPolicy}
          busy={busyAction === 'train'}
        />
      ) : mode === 'evaluation' ? (
        <EvaluationWorkspace
          results={batchResults}
          onRun={runBatch}
          busy={busyAction === 'batch'}
          scenarioResults={scenarioComparison}
          onCompare={runCurrentScenarioComparison}
          comparisonReady={
            Boolean(comparisonBaseline) &&
            !mission.running &&
            mission.time > (comparisonBaseline?.time ?? 0)
          }
          comparisonBusy={busyAction === 'compare'}
          missionTime={mission.time}
        />
      ) : (
        <section className="mission-shell">
          <aside className="left-rail">
            {mode === 'simulation' && (
              <>
                <div className="rail-section">
                  <div className="section-heading">
                    <span>임무 모의</span>
                    <span>{formatMissionTime(mission.time)}</span>
                  </div>
                  <div className="button-pair">
                    <Button
                      className="primary-command"
                      onClick={
                        mission.completed ? replaySimulation : toggleMission
                      }
                      disabled={!missionReady}
                    >
                      {mission.completed ? (
                        <RefreshCw />
                      ) : mission.running ? (
                        <Pause />
                      ) : (
                        <Play />
                      )}
                      {mission.completed
                        ? '다시 실행'
                        : mission.running
                          ? '일시 정지'
                          : '모의 시작'}
                    </Button>
                    <Button
                      variant="outline"
                      className="icon-command"
                      size="icon"
                      onClick={clearScenario}
                      aria-label="작전지도 비우기"
                      title="작전지도 초기화"
                    >
                      <RefreshCw />
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    className="mt-2 h-9 w-full justify-start"
                    onClick={stepSimulation}
                    disabled={
                      mission.running || mission.completed || !missionReady
                    }
                  >
                    <StepForward /> 10초 진행
                  </Button>
                </div>

                <DropoutControl
                  drones={mission.drones}
                  droneId={activeDropoutDroneId}
                  reason={dropoutReason}
                  onDroneChange={setDropoutDroneId}
                  onReasonChange={setDropoutReason}
                  onExecute={executeDropout}
                />

                <div className="rail-section parameter-panel">
                  <div className="section-heading">
                    <span>모의 조건</span>
                  </div>
                  <ParameterSlider
                    label="초기 기체"
                    value={config.droneCount}
                    suffix="대"
                    min={0}
                    max={10}
                    step={1}
                    onChange={(value) =>
                      setConfig((current) => ({
                        ...current,
                        droneCount: value,
                        failureDroneId: value < 2 ? 'UAV-01' : 'UAV-02',
                      }))
                    }
                  />
                  <ParameterSlider
                    label="초기 정찰지점"
                    value={config.waypointCount}
                    suffix="개"
                    min={0}
                    max={30}
                    step={1}
                    onChange={(value) =>
                      setConfig((current) => ({
                        ...current,
                        waypointCount: value,
                      }))
                    }
                  />
                  <ParameterSlider
                    label="이탈 시점"
                    value={config.failureAt}
                    suffix="초"
                    min={30}
                    max={Math.max(30, config.durationSec - 10)}
                    step={10}
                    onChange={(value) =>
                      setConfig((current) => ({ ...current, failureAt: value }))
                    }
                  />
                  <ParameterSlider
                    label="임무 시간"
                    value={Math.round(config.durationSec / 60)}
                    suffix="분"
                    min={1}
                    max={120}
                    step={1}
                    onChange={(value) =>
                      setConfig((current) => {
                        const durationSec = value * 60;
                        return {
                          ...current,
                          durationSec,
                          failureAt: Math.min(
                            current.failureAt,
                            Math.max(30, durationSec - 10),
                          ),
                        };
                      })
                    }
                  />
                  <ParameterSlider
                    label="모의 배속"
                    value={config.simRate}
                    suffix="×"
                    min={1}
                    max={20}
                    step={1}
                    onChange={(value) =>
                      setConfig((current) => ({ ...current, simRate: value }))
                    }
                  />
                  <label className="control-label" htmlFor="planner-select">
                    재계획 알고리즘
                  </label>
                  <Select
                    value={config.planner}
                    onValueChange={(value) =>
                      setConfig((current) => ({
                        ...current,
                        planner: value as PlannerKind,
                      }))
                    }
                  >
                    <SelectTrigger
                      id="planner-select"
                      className="control-select"
                    >
                      <SelectValue>{PLANNER_LABEL[config.planner]}</SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rl">물리 제약 결합 DQN</SelectItem>
                      <SelectItem value="nearest">최근접 우선</SelectItem>
                      <SelectItem value="priority">중요도 우선</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    className="mt-3 h-9 w-full"
                    onClick={applySimulationSettings}
                  >
                    <CheckCircle2 /> 설정 적용
                  </Button>
                </div>

                <WaypointEditor
                  selected={selectedWaypoint}
                  onChange={updateWaypoint}
                  onDelete={deleteSelectedWaypoint}
                  onReplan={() => replanNow()}
                />
              </>
            )}

            {mode === 'field' && (
              <>
                <div className="rail-section">
                  <div className="section-heading">
                    <span>현장 임무 기록</span>
                    <span>{formatMissionTime(mission.time)}</span>
                  </div>
                  <div className="button-pair">
                    <Button
                      className="primary-command"
                      onClick={toggleMission}
                      disabled={!mission.running && !missionReady}
                    >
                      {mission.running ? <Pause /> : <Play />}
                      {mission.running ? '기록 정지' : '기록 시작'}
                    </Button>
                    <Button
                      variant="outline"
                      size="icon"
                      className="icon-command"
                      onClick={() => applyScenario()}
                      aria-label="현장 임무 초기화"
                      title="초기 시나리오로 기체와 정찰지점을 새로 생성"
                    >
                      <RefreshCw />
                    </Button>
                  </div>
                </div>

                <DropoutControl
                  drones={mission.drones}
                  droneId={activeDropoutDroneId}
                  reason={dropoutReason}
                  onDroneChange={setDropoutDroneId}
                  onReasonChange={setDropoutReason}
                  onExecute={executeDropout}
                />

                {selectedDrone && (
                  <div className="rail-section">
                    <div className="section-heading">
                      <span>기체 상태</span>
                      <span>{mission.drones.length}대</span>
                    </div>
                    <div className="aircraft-selector">
                      {mission.drones.map((drone) => (
                        <button
                          key={drone.id}
                          type="button"
                          className={
                            drone.id === selectedDroneId ? 'active' : ''
                          }
                          onClick={() => setSelectedDroneId(drone.id)}
                        >
                          <i style={{ background: drone.color }} />
                          <span>{drone.id}</span>
                          <b>
                            {drone.status === 'failed'
                              ? '이탈'
                              : `${drone.battery.toFixed(0)}%`}
                          </b>
                        </button>
                      ))}
                    </div>
                    <div className="selected-aircraft">
                      <span
                        className="drone-dot"
                        style={{
                          color: selectedDrone.color,
                          borderColor: selectedDrone.color,
                        }}
                      >
                        <Satellite />
                      </span>
                      <div>
                        <strong>{selectedDrone.model}</strong>
                        <span>
                          {droneOperationalLabel(selectedDrone)} ·{' '}
                          {selectedDrone.speedMps} m/s
                        </span>
                      </div>
                    </div>
                    <ParameterSlider
                      label="현재 배터리"
                      value={Math.round(selectedDrone.battery)}
                      suffix="%"
                      min={10}
                      max={100}
                      step={1}
                      onChange={updateSelectedDroneBattery}
                    />
                    <Button
                      className="mt-3 h-10 w-full"
                      onClick={() => replanNow('rl')}
                      disabled={selectedDrone.status === 'failed'}
                    >
                      <BrainCircuit /> 경로 갱신
                    </Button>
                    {selectedDrone.status === 'failed' && (
                      <Button
                        variant="outline"
                        className="mt-2 h-9 w-full"
                        onClick={recoverSelectedDrone}
                      >
                        <RefreshCw /> 선택 기체 임무 복귀
                      </Button>
                    )}
                  </div>
                )}

                <WaypointEditor
                  selected={selectedWaypoint}
                  onChange={updateWaypoint}
                  onDelete={deleteSelectedWaypoint}
                  onVisit={reportWaypointVisit}
                  onReplan={() => replanNow()}
                />
              </>
            )}

            {mode === 'analysis' && (
              <>
                <div className="rail-section">
                  <div className="section-heading">
                    <span>비행 로그</span>
                    <span>{logs.length}개 로그</span>
                  </div>
                  <div className="button-stack">
                    <Button
                      className="primary-command justify-start"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload /> DJI CSV 불러오기
                    </Button>
                    <Button
                      variant="outline"
                      className="justify-start"
                      onClick={() => {
                        const demo = createDemoLogs(mission);
                        setLogs(demo);
                        setSelectedLogId(demo[0]?.id ?? '');
                        setUploadError('');
                      }}
                    >
                      <Sparkles /> 검증용 예제 데이터
                    </Button>
                  </div>
                  {uploadError && (
                    <div className="error-notice">
                      <AlertTriangle />
                      {uploadError}
                    </div>
                  )}
                </div>
                <div className="rail-section flex-1">
                  <div className="section-heading">
                    <span>로그 목록</span>
                    <FileChartColumn size={14} />
                  </div>
                  <div className="log-list">
                    {logs.map((log) => {
                      const result = calculateLogMetrics(log);
                      return (
                        <button
                          key={log.id}
                          type="button"
                          aria-label={`${log.droneName} 비행 로그 선택`}
                          aria-pressed={selectedLog?.id === log.id}
                          className={`log-item ${selectedLog?.id === log.id ? 'active' : ''}`}
                          onClick={() => setSelectedLogId(log.id)}
                        >
                          <span
                            className="log-color"
                            style={{ background: log.color }}
                          />
                          <div>
                            <strong>{log.droneName}</strong>
                            <span>
                              {result.sampleCount.toLocaleString()} samples ·{' '}
                              {formatMissionTime(result.durationSec)}
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
                <Button
                  variant="outline"
                  className="h-10"
                  onClick={exportAnalysis}
                  disabled={!logs.length}
                >
                  <Download /> 분석표 CSV 내보내기
                </Button>
              </>
            )}
          </aside>

          <section className="map-stage">
            <div className="map-control-stack">
              <div className="map-toolbar">
                <div className="flex items-center gap-2">
                  {mode !== 'analysis' &&
                    (mission.base.configured ||
                      mission.drones.length > 0 ||
                      mission.waypoints.length > 0) && (
                      <Badge
                        variant="outline"
                        className={
                          mission.failureTriggered
                            ? 'event-badge danger'
                            : 'event-badge'
                        }
                      >
                        <CircleDot />{' '}
                        {mission.failureTriggered
                          ? '기체 이탈 · 재계획 완료'
                          : mode === 'field'
                            ? '운용자 상태 입력 대기'
                            : missionReady
                              ? `자동 이탈 예정 ${formatMissionTime(config.failureAt)}`
                              : '작전구역 편집 중'}
                      </Badge>
                    )}
                  {mode === 'analysis' && (
                    <Badge variant="outline" className="event-badge">
                      <Database /> 비행기록 {logs.length}개 중첩
                    </Badge>
                  )}
                </div>
                <div className="map-toolbar-right">
                  <div className="layer-switch" aria-label="지도 배경 선택">
                    <button
                      type="button"
                      className={mapBase === 'satellite' ? 'active' : ''}
                      onClick={() => setMapBase('satellite')}
                    >
                      <Satellite /> 위성
                    </button>
                    <button
                      type="button"
                      className={mapBase === 'street' ? 'active' : ''}
                      onClick={() => setMapBase('street')}
                    >
                      <MapIcon /> 지도
                    </button>
                  </div>
                </div>
              </div>

              {mode !== 'analysis' && (
                <div className="map-edit-toolbar">
                  <button
                    type="button"
                    className={`map-tool ${interaction === 'add-base' ? 'active' : ''}`}
                    onClick={() =>
                      setInteraction(
                        interaction === 'add-base' ? 'inspect' : 'add-base',
                      )
                    }
                  >
                    <Crosshair /> 기지 지정
                  </button>
                  <button
                    type="button"
                    className="map-tool"
                    onClick={() => setDroneStatusOpen(true)}
                  >
                    <List /> 기체 현황
                  </button>
                  <button
                    type="button"
                    className={`map-tool ${interaction === 'add-waypoint' ? 'active' : ''}`}
                    onClick={() =>
                      setInteraction(
                        interaction === 'add-waypoint'
                          ? 'inspect'
                          : 'add-waypoint',
                      )
                    }
                  >
                    <MapPinPlus /> 정찰지점 추가
                  </button>
                  <button
                    type="button"
                    className={`map-tool ${interaction === 'add-drone' ? 'active' : ''}`}
                    onClick={() =>
                      setInteraction(
                        interaction === 'add-drone' ? 'inspect' : 'add-drone',
                      )
                    }
                  >
                    <Satellite /> 기체 추가
                  </button>
                  <button
                    type="button"
                    className="map-tool danger"
                    onClick={deleteSelectedMapEntity}
                    disabled={!canDeleteSelectedEntity}
                  >
                    <Trash2 /> {deleteTargetLabel} 삭제
                  </button>
                  <button
                    type="button"
                    className="map-tool"
                    onClick={() => replanNow()}
                  >
                    <BrainCircuit /> 경로 재계산
                  </button>
                  <button
                    type="button"
                    className={`map-tool ${interaction === 'move-drone' ? 'active' : ''}`}
                    onClick={() =>
                      setInteraction(
                        interaction === 'move-drone' ? 'inspect' : 'move-drone',
                      )
                    }
                    disabled={!selectedDrone}
                  >
                    <LocateFixed /> {selectedDroneId || '기체'} 위치 지정
                  </button>
                  {interaction !== 'inspect' && (
                    <span className="map-click-guide">
                      <Crosshair />
                      {interaction === 'move-drone'
                        ? `${selectedDroneId}를 이동할 위치를 클릭하세요`
                        : interaction === 'add-drone'
                          ? '기체를 배치할 위치를 클릭하세요'
                          : interaction === 'add-base'
                            ? '기지를 배치할 위치를 클릭하세요'
                            : '정찰지점을 등록할 위치를 클릭하세요'}
                    </span>
                  )}
                </div>
              )}
            </div>

            <OperationalMap
              mode={mode}
              mapBase={mapBase}
              mission={mission}
              logs={logs}
              selectedDroneId={selectedDroneId}
              selectedWaypointId={selectedWaypointId}
              interaction={interaction}
              onMapClick={handleMapClick}
              onMapContextMenu={handleMapContextMenu}
              onSelectDrone={(droneId) => {
                setSelectedDroneId(droneId);
                setSelectedMapEntity('drone');
              }}
              onSelectWaypoint={(waypointId) => {
                setSelectedWaypointId(waypointId);
                setSelectedMapEntity('waypoint');
              }}
            />

            {mapContextMenu && mode !== 'analysis' && (
              <div
                className="map-context-menu"
                role="menu"
                tabIndex={-1}
                aria-label="지도 편집 메뉴"
                onContextMenu={(event) => event.preventDefault()}
                style={
                  {
                    '--context-x': `${mapContextMenu.x}px`,
                    '--context-y': `${mapContextMenu.y}px`,
                  } as React.CSSProperties
                }
              >
                <div className="map-context-title">이 위치에 추가</div>
                <button
                  type="button"
                  onClick={() => setBaseAt(mapContextMenu.point)}
                >
                  <Crosshair /> 기지 지정
                </button>
                <button
                  type="button"
                  onClick={() => addWaypointAt(mapContextMenu.point)}
                >
                  <MapPinPlus /> 정찰 포인트 추가
                </button>
                <button
                  type="button"
                  onClick={() => requestDronePlacement(mapContextMenu.point)}
                >
                  <Satellite /> 드론 추가
                </button>
                {(mapContextMenu.droneId || mapContextMenu.waypointId) && (
                  <div className="map-context-divider" />
                )}
                {mapContextMenu.droneId && (
                  <>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedDroneId(mapContextMenu.droneId ?? '');
                        setSelectedMapEntity('drone');
                        closeMapContextMenu();
                      }}
                    >
                      <CheckCircle2 /> 기체 선택
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedDroneId(mapContextMenu.droneId ?? '');
                        setSelectedMapEntity('drone');
                        setInteraction('move-drone');
                        closeMapContextMenu();
                      }}
                    >
                      <LocateFixed /> 이 기체 위치 지정
                    </button>
                    <button
                      type="button"
                      className="danger"
                      onClick={() =>
                        deleteDroneById(mapContextMenu.droneId ?? '')
                      }
                    >
                      <Trash2 /> 기체 삭제
                    </button>
                  </>
                )}
                {mapContextMenu.waypointId && (
                  <button
                    type="button"
                    className="danger"
                    onClick={() =>
                      deleteWaypointById(mapContextMenu.waypointId ?? '')
                    }
                  >
                    <Trash2 /> 정찰 포인트 삭제
                  </button>
                )}
                <button
                  type="button"
                  className="subtle"
                  onClick={closeMapContextMenu}
                >
                  닫기
                </button>
              </div>
            )}

            <Dialog
              open={Boolean(pendingDronePoint)}
              onOpenChange={(open) => {
                if (open) return;
                setPendingDronePoint(null);
                setInteraction('inspect');
              }}
            >
              <DialogContent className="drone-profile-dialog">
                <DialogHeader>
                  <DialogTitle>투입 기체 선택</DialogTitle>
                  <DialogDescription>
                    DJI 공식 무풍 시험값 기준
                  </DialogDescription>
                </DialogHeader>
                <div
                  className="drone-profile-grid"
                  role="radiogroup"
                  aria-label="드론 기종"
                >
                  {DRONE_PROFILES.map((profile) => {
                    const selected = profile.id === pendingDroneProfileId;
                    return (
                      <label
                        key={profile.id}
                        className={`drone-profile-option ${selected ? 'active' : ''}`}
                      >
                        <input
                          type="radio"
                          name="drone-profile"
                          value={profile.id}
                          checked={selected}
                          onChange={() => setPendingDroneProfileId(profile.id)}
                        />
                        <span className="drone-profile-title">
                          <strong>{profile.model}</strong>
                          {profile.id === DEFAULT_DRONE_PROFILE_ID && (
                            <i>기본</i>
                          )}
                        </span>
                        <span className="drone-profile-specs">
                          <span>
                            <small>순항속도</small>
                            <b>{profile.cruiseSpeedMps} m/s</b>
                          </span>
                          <span>
                            <small>최대속도</small>
                            <b>{profile.maxSpeedMps} m/s</b>
                          </span>
                          <span>
                            <small>최대 비행</small>
                            <b>{profile.maxFlightTimeMin}분</b>
                          </span>
                          <span>
                            <small>배터리</small>
                            <b>
                              {profile.batteryCapacityMah.toLocaleString()} mAh
                            </b>
                          </span>
                          <span>
                            <small>에너지</small>
                            <b>{profile.batteryEnergyWh} Wh</b>
                          </span>
                          <span>
                            <small>이륙중량</small>
                            <b>{profile.takeoffWeightG.toLocaleString()} g</b>
                          </span>
                          <span>
                            <small>내풍 성능</small>
                            <b>{profile.maxWindResistanceMps} m/s</b>
                          </span>
                          <span>
                            <small>최대 전송</small>
                            <b>{profile.transmissionRangeKm} km</b>
                          </span>
                        </span>
                      </label>
                    );
                  })}
                </div>
                <DialogFooter className="drone-profile-actions">
                  <Button
                    variant="outline"
                    onClick={() => {
                      setPendingDronePoint(null);
                      setInteraction('inspect');
                    }}
                  >
                    취소
                  </Button>
                  <Button
                    onClick={() => {
                      if (!pendingDronePoint) return;
                      addDroneAt(pendingDronePoint, pendingDroneProfileId);
                    }}
                  >
                    <Satellite /> 선택 기체 배치
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

            <Dialog open={droneStatusOpen} onOpenChange={setDroneStatusOpen}>
              <DialogContent className="drone-status-dialog">
                <DialogHeader>
                  <DialogTitle>기체 현황</DialogTitle>
                  <DialogDescription>기체별 운용 상태</DialogDescription>
                </DialogHeader>
                <div className="drone-status-table" aria-label="기체 현황표">
                  <div className="drone-status-head">
                    <span>기체</span>
                    <span>기종</span>
                    <span>상태</span>
                    <span>배터리</span>
                    <span>속도</span>
                    <span>현재 목표</span>
                    <span>위치</span>
                  </div>
                  {mission.drones.map((drone) => {
                    const targetId = drone.route[drone.routeIndex];
                    const statusLabel = droneOperationalLabel(drone);
                    return (
                      <button
                        key={drone.id}
                        type="button"
                        className={`drone-status-row ${drone.id === selectedDroneId ? 'active' : ''}`}
                        onClick={() => {
                          setSelectedDroneId(drone.id);
                          setSelectedMapEntity('drone');
                          setDroneStatusOpen(false);
                        }}
                      >
                        <span>
                          <i style={{ background: drone.color }} />
                          {drone.id}
                        </span>
                        <span>{drone.model}</span>
                        <span>{statusLabel}</span>
                        <strong>{drone.battery.toFixed(0)}%</strong>
                        <span>
                          {drone.status === 'failed'
                            ? '—'
                            : `${drone.speedMps} m/s`}
                        </span>
                        <span>
                          {drone.status === 'returning'
                            ? mission.base.id
                            : drone.phase === 'dwell'
                              ? `${targetId} (${Math.ceil(drone.dwellRemainingSec)}초)`
                              : (targetId ?? '대기')}
                        </span>
                        <span className="drone-position">
                          {drone.lat.toFixed(5)}, {drone.lng.toFixed(5)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </DialogContent>
            </Dialog>

            <div className="map-footer">
              <div>
                <span>임무 시간</span>
                <strong>
                  {formatMissionTime(mission.time)}.
                  {Math.floor((mission.time % 1) * 10)}
                </strong>
              </div>
              <div>
                <span>활성 / 전체</span>
                <strong>
                  {metrics.activeDrones} / {mission.drones.length}
                </strong>
              </div>
              <div>
                <span>경로 산출</span>
                <strong>
                  {mission.inferenceMs
                    ? `${mission.inferenceMs.toFixed(1)} ms`
                    : '대기'}
                </strong>
              </div>
              <div>
                <span>위치 자료</span>
                <strong>
                  {mode === 'analysis'
                    ? 'DJI 로그'
                    : mode === 'field'
                      ? '운용자 입력'
                      : '모의'}
                </strong>
              </div>
            </div>
          </section>

          <aside className="right-rail">
            {mode === 'analysis' ? (
              <AnalysisRail
                metrics={selectedLogMetrics}
                logs={logs}
                selectedLog={selectedLog}
                handoverSeconds={handoverSeconds}
                flightSeries={flightSeries}
              />
            ) : (
              <>
                <div className="rail-section">
                  <div className="section-heading">
                    <span>임무 결과</span>
                    <span>
                      {mission.completed
                        ? '종료'
                        : mission.running
                          ? '진행 중'
                          : '대기'}
                    </span>
                  </div>
                  <dl className="result-table">
                    <ResultRow
                      label="이탈 후 평균 연속성"
                      value={
                        metrics.postFailureAverageContinuity === null
                          ? '—'
                          : `${metrics.postFailureAverageContinuity.toFixed(0)} %`
                      }
                    />
                    <ResultRow
                      label="재방문 기한 준수율"
                      value={
                        hasMissionResults
                          ? `${metrics.revisitCompliance.toFixed(0)} %`
                          : '—'
                      }
                    />
                    <ResultRow
                      label="이탈 후 최저 연속성"
                      value={
                        mission.failureTime === null
                          ? '—'
                          : `${metrics.minimumPostFailureContinuity.toFixed(0)} %`
                      }
                      tone={
                        mission.failureTime !== null &&
                        metrics.minimumPostFailureContinuity < 70
                          ? 'amber'
                          : 'neutral'
                      }
                    />
                    <ResultRow
                      label="담당구역 회복"
                      value={
                        metrics.recoverySeconds === null
                          ? '—'
                          : `${metrics.recoverySeconds.toFixed(0)} 초`
                      }
                    />
                    <ResultRow
                      label="누적 거리"
                      value={
                        hasMissionResults
                          ? `${metrics.totalDistanceKm.toFixed(2)} km`
                          : '—'
                      }
                    />
                    <ResultRow
                      label="기지 복귀"
                      value={
                        hasMissionResults ? `${metrics.returnCount} 회` : '—'
                      }
                    />
                    <ResultRow
                      label="최저 배터리"
                      value={
                        hasMissionResults && mission.drones.length
                          ? `${metrics.minimumBattery.toFixed(0)} %`
                          : '—'
                      }
                      tone={
                        !hasMissionResults || !mission.drones.length
                          ? 'neutral'
                          : metrics.minimumBattery < 20
                            ? 'red'
                            : metrics.minimumBattery < 30
                              ? 'amber'
                              : 'neutral'
                      }
                    />
                    <ResultRow
                      label="예비전력 위반"
                      value={
                        hasMissionResults
                          ? `${metrics.reserveViolations} 회`
                          : '—'
                      }
                      tone={
                        hasMissionResults && metrics.reserveViolations
                          ? 'red'
                          : 'neutral'
                      }
                    />
                  </dl>
                </div>

                <div className="rail-section current-model-card">
                  <div className="section-heading">
                    <span>경로계획 모델</span>
                    <span
                      className={`model-status ${config.planner === 'rl' ? 'active' : ''}`}
                    >
                      {config.planner === 'rl' ? '사용 중' : '대기'}
                    </span>
                  </div>
                  <div className="current-model-summary">
                    <div>
                      <strong>물리 제약 결합 DQN</strong>
                      <span>
                        {policyBundle.origin === 'pretrained'
                          ? '기본 사전학습 모델'
                          : '사용자 재학습 모델'}
                      </span>
                    </div>
                  </div>
                  <dl className="data-list current-model-data">
                    <div>
                      <dt>학습 규모</dt>
                      <dd>{policyStats.episodes.toLocaleString()}회</dd>
                    </div>
                    <div>
                      <dt>계획 방식</dt>
                      <dd>{PLANNER_LABEL[config.planner]}</dd>
                    </div>
                  </dl>
                </div>
                {mode === 'field' && (
                  <div className="rail-section flex-1">
                    <div className="section-heading">
                      <span>다음 정찰 경로</span>
                      <span>{selectedDrone?.id}</span>
                    </div>
                    <div className="route-order">
                      {(
                        selectedDrone?.route.slice(
                          selectedDrone.routeIndex,
                          selectedDrone.routeIndex + 6,
                        ) ?? []
                      ).map((id, index) => (
                        <div key={`${id}-${index}`}>
                          <span>{index + 1}</span>
                          <strong>{id}</strong>
                          <i>
                            {mission.waypoints.find(
                              (waypoint) => waypoint.id === id,
                            )?.priority ?? 0}{' '}
                            중요도
                          </i>
                        </div>
                      ))}
                      {!selectedDrone?.route.length && (
                        <div className="empty-inline">
                          경로 없음 · 재계산 필요
                        </div>
                      )}
                    </div>
                    <div className="section-heading mt-4">
                      <span>정찰 우선순위</span>
                      <span>상위 4개</span>
                    </div>
                    <div className="q-rank">
                      {candidateScores.map(({ waypoint, score }, index) => (
                        <div key={waypoint.id}>
                          <span>{index + 1}</span>
                          <strong>{waypoint.id}</strong>
                          <div>
                            <i
                              style={{
                                width: `${Math.max(12, Math.min(100, 50 + score * 4))}%`,
                              }}
                            />
                          </div>
                          <b>{score.toFixed(2)}</b>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="rail-section event-section">
                  <div className="section-heading">
                    <span>최근 기록</span>
                    <span>{mission.events.length}건</span>
                  </div>
                  <ol className="event-list">
                    {mission.events.slice(0, 5).map((event) => (
                      <li
                        key={event.id}
                        className={
                          event.kind === 'failure' || event.kind === 'warning'
                            ? 'warning'
                            : event.kind === 'replan'
                              ? 'success'
                              : ''
                        }
                      >
                        <time>{formatMissionTime(event.time)}</time>
                        <div>
                          <strong>{event.title}</strong>
                          <span>{event.detail}</span>
                        </div>
                      </li>
                    ))}
                  </ol>
                </div>
              </>
            )}
          </aside>
        </section>
      )}
    </main>
  );
}

function TrainingWorkspace({
  policyBundle,
  trainingEpisodes,
  trainingProgress,
  trainingStage,
  trainingNotice,
  currentSetupReady,
  onTrainingEpisodesChange,
  onTrain,
  onRestore,
  busy,
}: {
  policyBundle: PolicyBundle;
  trainingEpisodes: number;
  trainingProgress: number;
  trainingStage: string;
  trainingNotice: string;
  currentSetupReady: boolean;
  onTrainingEpisodesChange: (value: number) => void;
  onTrain: () => void;
  onRestore: () => void;
  busy: boolean;
}) {
  const stats = policyBundle.stats;
  return (
    <section className="workspace-page training-page">
      <div className="workspace-page-header">
        <div>
          <div className="workspace-kicker">
            <BrainCircuit /> 모델 관리
          </div>
          <h1>모델 학습</h1>
          <p>
            {currentSetupReady
              ? '현재 가상실험 구성 반영'
              : 'DJI 기체 프로파일 일반화 학습'}
          </p>
        </div>
        <Badge className={`workspace-status ${busy ? 'is-training' : ''}`}>
          <span className="status-pulse" />
          {busy ? '학습 중 ' + trainingProgress + '%' : '모델 준비됨'}
        </Badge>
      </div>

      <div className="workspace-grid training-grid">
        <div className="workspace-card training-main-card">
          <div className="workspace-card-header">
            <div>
              <span className="workspace-label">현재 정책</span>
              <h2>중앙집중형 물리 제약 결합 DQN</h2>
            </div>
            <Badge variant="outline">
              {policyBundle.origin === 'custom' ? '사용자 모델' : '기본 모델'}
            </Badge>
          </div>
          <div className="training-chart">
            <ResponsiveContainer
              width="100%"
              height="100%"
              minWidth={0}
              minHeight={0}
              initialDimension={{ width: 620, height: 250 }}
            >
              <AreaChart data={stats.rewardHistory}>
                <CartesianGrid stroke="#17303a" vertical={false} />
                <XAxis
                  dataKey="episode"
                  tick={{ fill: '#6f8994', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#6f8994', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <ChartTooltip
                  contentStyle={{
                    background: '#09151b',
                    border: '1px solid #24404b',
                    fontSize: 12,
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="reward"
                  name="평균 보상"
                  stroke="#67e8f9"
                  fill="#163742"
                  fillOpacity={0.55}
                  strokeWidth={2}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="workspace-chart-caption">
            <span>평균 보상 추이</span>
            <strong>{stats.episodes.toLocaleString()}회 학습 완료</strong>
          </div>
        </div>

        <div className="workspace-stack">
          <div className="workspace-card">
            <div className="workspace-card-header compact">
              <span className="workspace-label">모델 상태</span>
              <Bot className="text-cyan-300" size={20} />
            </div>
            <div className="workspace-metric-large">
              <strong>{stats.averageReward.toFixed(1)}</strong>
              <span>평균 보상</span>
            </div>
            <Progress
              value={Math.min(100, 100 - stats.epsilon * 100)}
              className="h-1.5 bg-white/10"
            />
            <div className="workspace-stat-list">
              <div>
                <span>TD 손실</span>
                <strong>{stats.finalLoss.toFixed(3)}</strong>
              </div>
              <div>
                <span>탐색률</span>
                <strong>{(stats.epsilon * 100).toFixed(1)}%</strong>
              </div>
              {stats.validation && (
                <>
                  <div>
                    <span>검증 점수</span>
                    <strong>{stats.validation.rlScore.toFixed(1)}</strong>
                  </div>
                  <div>
                    <span>기준 대비</span>
                    <strong>
                      {stats.validation.improvementVsBest >= 0 ? '+' : ''}
                      {stats.validation.improvementVsBest.toFixed(1)}
                    </strong>
                  </div>
                </>
              )}
            </div>
          </div>
          <div className="workspace-card">
            <span className="workspace-label">모델 구성</span>
            <div className="training-feature-grid">
              <div>
                <strong>16</strong>
                <span>상태 변수</span>
              </div>
              <div>
                <strong>1–6</strong>
                <span>이탈 후 잔여 기체</span>
              </div>
              <div>
                <strong>6–20</strong>
                <span>정찰지점</span>
              </div>
              <div>
                <strong>36</strong>
                <span>은닉 노드</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="workspace-card training-controls-card">
        <div>
          <span className="workspace-label">재학습</span>
          <h2>학습 횟수 선택</h2>
          <p>
            {currentSetupReady
              ? '현재 구성 변형 70% · 일반화 시나리오 30%'
              : 'DJI 5개 기체 프로파일 · 무작위 기지·정찰지점'}
          </p>
          <div className="training-progress-wrap">
            <div>
              <span>{trainingStage || '학습 진행률'}</span>
              <strong>{trainingProgress}%</strong>
            </div>
            <Progress value={trainingProgress} className="h-1.5 bg-white/10" />
          </div>
          {trainingNotice && (
            <p className="training-notice">{trainingNotice}</p>
          )}
        </div>
        <div className="training-action-row">
          <Select
            value={String(trainingEpisodes)}
            onValueChange={(value) => onTrainingEpisodesChange(Number(value))}
            disabled={busy}
          >
            <SelectTrigger className="training-episode-select">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1000">1,000회</SelectItem>
              <SelectItem value="2000">2,000회</SelectItem>
              <SelectItem value="5000">5,000회</SelectItem>
              <SelectItem value="10000">10,000회</SelectItem>
            </SelectContent>
          </Select>
          <Button
            className="primary-command training-action-button"
            onClick={onTrain}
            disabled={busy}
          >
            <Zap className={busy ? 'animate-pulse' : ''} />
            {busy
              ? '학습 중'
              : trainingEpisodes.toLocaleString() + '회 학습 시작'}
          </Button>
          <Button
            variant="outline"
            className="training-action-button"
            onClick={onRestore}
            disabled={busy || policyBundle.origin === 'pretrained'}
          >
            <RefreshCw /> 기본 모델 복원
          </Button>
        </div>
      </div>
    </section>
  );
}

function EvaluationWorkspace({
  results,
  onRun,
  busy,
  scenarioResults,
  onCompare,
  comparisonReady,
  comparisonBusy,
  missionTime,
}: {
  results: ReturnType<typeof runBatchEvaluation>;
  onRun: () => void;
  busy: boolean;
  scenarioResults: ScenarioComparisonResult[] | null;
  onCompare: () => void;
  comparisonReady: boolean;
  comparisonBusy: boolean;
  missionTime: number;
}) {
  return (
    <section className="workspace-page evaluation-page">
      <div className="workspace-page-header">
        <div>
          <div className="workspace-kicker">
            <FileChartColumn /> 비교 평가
          </div>
          <h1>성능 평가</h1>
          <p>동일 조건에서 세 경로 방식을 비교</p>
        </div>
        <Badge variant="outline" className="workspace-status">
          임무 시각 {formatMissionTime(missionTime)}
        </Badge>
      </div>

      <div className="evaluation-layout">
        <ComparisonRail
          results={results}
          onRun={onRun}
          busy={busy}
          scenarioResults={scenarioResults}
          onCompare={onCompare}
          comparisonReady={comparisonReady}
          comparisonBusy={comparisonBusy}
        />
        <div className="workspace-stack">
          <div className="workspace-card">
            <span className="workspace-label">비교 기준</span>
            <div className="evaluation-method-list">
              <div>
                <strong>01</strong>
                <span>동일 시나리오·이탈 조건</span>
              </div>
              <div>
                <strong>02</strong>
                <span>이탈 후 연속성·재방문·회복시간</span>
              </div>
              <div>
                <strong>03</strong>
                <span>실험 종료 후 동일 조건 비교</span>
              </div>
            </div>
          </div>
          <div className="workspace-card evaluation-note">
            <ShieldCheck className="text-cyan-300" />
            <div>
              <strong>논문용 비교</strong>
              <p>조건·반복 횟수·측정 지표 기록</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function DropoutControl({
  drones,
  droneId,
  reason,
  onDroneChange,
  onReasonChange,
  onExecute,
}: {
  drones: ReturnType<typeof createMission>['drones'];
  droneId: string;
  reason: string;
  onDroneChange: (value: string) => void;
  onReasonChange: (value: string) => void;
  onExecute: () => void;
}) {
  const activeDrones = drones.filter((drone) => drone.status !== 'failed');
  return (
    <div className="rail-section dropout-control">
      <div className="section-heading">
        <span>기체 이탈 입력</span>
      </div>
      <div className="dropout-fields">
        <label>
          <span>이탈 기체</span>
          <select
            value={droneId}
            onChange={(event) => onDroneChange(event.target.value)}
            disabled={!activeDrones.length}
          >
            {activeDrones.map((drone) => (
              <option key={drone.id} value={drone.id}>
                {drone.id} · {drone.model}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>이탈 원인</span>
          <select
            value={reason}
            onChange={(event) => onReasonChange(event.target.value)}
          >
            <option value="통신 두절">통신 두절</option>
            <option value="피격 추정">피격 추정</option>
            <option value="배터리 임계치 도달">배터리 임계치 도달</option>
            <option value="비행 제어 이상">비행 제어 이상</option>
          </select>
        </label>
      </div>
      <Button
        variant="destructive"
        className="mt-2 h-10 w-full justify-start"
        onClick={onExecute}
        disabled={!activeDrones.length}
      >
        <AlertTriangle /> 이탈 처리
      </Button>
    </div>
  );
}

function ParameterSlider({
  label,
  value,
  suffix,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  suffix: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="parameter-row">
      <div>
        <span className="parameter-label">{label}</span>
        <output>
          {value}
          {suffix}
        </output>
      </div>
      <Slider
        aria-label={label}
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
      />
    </div>
  );
}

function WaypointEditor({
  selected,
  onChange,
  onDelete,
  onVisit,
  onReplan,
}: {
  selected?: Waypoint;
  onChange: (patch: Partial<Waypoint>) => void;
  onDelete: () => void;
  onVisit?: () => void;
  onReplan: () => void;
}) {
  if (!selected) return null;
  return (
    <div className="rail-section">
      <div className="section-heading">
        <span>선택한 정찰지점</span>
        <span className="text-cyan-300">{selected.id}</span>
      </div>
      <ParameterSlider
        label="정찰 중요도"
        value={selected.priority}
        suffix=""
        min={1}
        max={5}
        step={1}
        onChange={(priority) => onChange({ priority })}
      />
      <ParameterSlider
        label="지점 정찰시간"
        value={selected.dwellSec}
        suffix="초"
        min={10}
        max={180}
        step={10}
        onChange={(dwellSec) => onChange({ dwellSec })}
      />
      <ParameterSlider
        label="재방문 한계"
        value={selected.revisitSec}
        suffix="초"
        min={60}
        max={300}
        step={10}
        onChange={(revisitSec) => onChange({ revisitSec })}
      />
      <dl className="data-list mt-3">
        <div>
          <dt>담당 기체</dt>
          <dd>{selected.assignedDrone ?? '미배정'}</dd>
        </div>
        <div>
          <dt>방문 횟수</dt>
          <dd>{selected.visitedCount}</dd>
        </div>
      </dl>
      <div className="waypoint-actions">
        {onVisit && (
          <Button className="h-9" onClick={onVisit}>
            <CheckCircle2 /> 정찰 완료 보고
          </Button>
        )}
        <Button variant="outline" className="h-9" onClick={onReplan}>
          <BrainCircuit /> 변경 반영
        </Button>
        <Button
          variant="outline"
          className="h-9 danger-outline"
          onClick={onDelete}
        >
          <Trash2 /> 지점 삭제
        </Button>
      </div>
    </div>
  );
}

function ComparisonRail({
  results,
  onRun,
  busy,
  scenarioResults,
  onCompare,
  comparisonReady,
  comparisonBusy,
}: {
  results: ReturnType<typeof runBatchEvaluation>;
  onRun: () => void;
  busy: boolean;
  scenarioResults: ScenarioComparisonResult[] | null;
  onCompare: () => void;
  comparisonReady: boolean;
  comparisonBusy: boolean;
}) {
  return (
    <div className="rail-section comparison-section">
      <div className="section-heading">
        <span>운용 방식 비교</span>
        <span>100회 반복</span>
      </div>
      <div className="comparison-chart">
        <ResponsiveContainer
          width="100%"
          height="100%"
          minWidth={0}
          minHeight={0}
          initialDimension={{ width: 240, height: 150 }}
        >
          <BarChart
            data={results}
            margin={{ top: 4, right: 0, left: -27, bottom: 0 }}
          >
            <CartesianGrid stroke="#17303a" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: '#6f8994', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: '#59727d', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <ChartTooltip
              contentStyle={{
                background: '#09151b',
                border: '1px solid #24404b',
                fontSize: 11,
              }}
            />
            <Bar
              dataKey="continuity"
              name="연속성 %"
              fill="#67e8f9"
              radius={[2, 2, 0, 0]}
            />
            <Bar
              dataKey="completion"
              name="경로 배정률 %"
              fill="#fbbf62"
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="comparison-table">
        {results.map((result) => (
          <div key={result.planner}>
            <span>{result.label}</span>
            <strong>{result.continuity.toFixed(1)}%</strong>
            <i>{result.distance.toFixed(2)} km</i>
          </div>
        ))}
      </div>
      <Button
        variant="outline"
        className="mt-3 h-8 w-full"
        onClick={onRun}
        disabled={busy}
      >
        <RefreshCw className={busy ? 'animate-spin' : ''} />
        {busy ? '100회 계산 중' : '무작위 100회 다시 평가'}
      </Button>
      <div className="scenario-comparison">
        <div className="section-heading mt-4">
          <span>현재 시나리오 비교</span>
          <span>{scenarioResults ? '동일 조건' : '실험 후'}</span>
        </div>
        {scenarioResults ? (
          <div className="scenario-comparison-table">
            <div className="scenario-comparison-head">
              <span>방식</span>
              <span>평균</span>
              <span>재방문</span>
              <span>최저</span>
              <span>회복</span>
              <span>거리</span>
            </div>
            {scenarioResults.map((result) => (
              <div className="scenario-comparison-row" key={result.planner}>
                <strong>{result.label}</strong>
                <span>{result.continuity.toFixed(1)}%</span>
                <span>{result.revisitCompliance.toFixed(0)}%</span>
                <span>{result.minimumPostFailureContinuity.toFixed(0)}%</span>
                <span>
                  {result.recoverySeconds === null
                    ? '—'
                    : `${result.recoverySeconds.toFixed(0)}초`}
                </span>
                <span>{result.distanceKm.toFixed(2)}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-inline">
            {comparisonReady
              ? '동일 조건·세 방식 비교'
              : '실험 종료 후 비교 가능'}
          </div>
        )}
        <Button
          variant="outline"
          className="mt-3 h-8 w-full"
          onClick={onCompare}
          disabled={!comparisonReady || comparisonBusy}
        >
          <Route className={comparisonBusy ? 'animate-pulse' : ''} />
          {comparisonBusy
            ? '현재 시나리오 계산 중'
            : '현재 시나리오 3방식 비교'}
        </Button>
      </div>
    </div>
  );
}

function AnalysisRail({
  metrics,
  logs,
  selectedLog,
  handoverSeconds,
  flightSeries,
}: {
  metrics: ReturnType<typeof calculateLogMetrics> | null;
  logs: FlightLog[];
  selectedLog?: FlightLog;
  handoverSeconds: number | null;
  flightSeries: {
    time: number;
    speed: number;
    battery: number;
    altitude: number;
  }[];
}) {
  return (
    <>
      <div className="rail-section">
        <div className="section-heading">
          <span>비행 요약</span>
          <span>{selectedLog ? '데이터 확인' : '데이터 없음'}</span>
        </div>
        {metrics ? (
          <>
            <div className="metric-grid">
              <MetricCard
                icon={<Clock3 />}
                label="비행시간"
                value={formatMissionTime(metrics.durationSec)}
              />
              <MetricCard
                icon={<Route />}
                label="비행거리"
                value={`${metrics.distanceKm.toFixed(2)} km`}
              />
              <MetricCard
                icon={<Gauge />}
                label="평균속도"
                value={`${metrics.avgSpeedMps.toFixed(1)} m/s`}
              />
              <MetricCard
                icon={<BatteryMedium />}
                label="배터리 사용"
                value={`${metrics.batteryUsed.toFixed(1)} %`}
                tone={metrics.batteryUsed > 50 ? 'amber' : 'cyan'}
              />
            </div>
            <dl className="analysis-data">
              <div>
                <dt>최대고도</dt>
                <dd>{metrics.maxAltitudeM.toFixed(1)} m</dd>
              </div>
              <div>
                <dt>표본주기</dt>
                <dd>{metrics.sampleRateHz.toFixed(1)} Hz</dd>
              </div>
              <div>
                <dt>경로 추종오차</dt>
                <dd>
                  {metrics.routeErrorM === null
                    ? '계획경로 없음'
                    : `${metrics.routeErrorM.toFixed(1)} m`}
                </dd>
              </div>
              <div>
                <dt>소모율</dt>
                <dd>{metrics.batteryPerKm.toFixed(1)} %/km</dd>
              </div>
            </dl>
          </>
        ) : (
          <div className="empty-inline">비행 로그를 불러오십시오.</div>
        )}
      </div>
      <div className="rail-section chart-section">
        <div className="section-heading">
          <span>비행 데이터</span>
          <span>{selectedLog?.fileName ?? '—'}</span>
        </div>
        <div className="telemetry-chart">
          <ResponsiveContainer
            width="100%"
            height="100%"
            minWidth={0}
            minHeight={0}
            initialDimension={{ width: 240, height: 160 }}
          >
            <LineChart
              data={flightSeries}
              margin={{ top: 6, right: 3, left: -29, bottom: 0 }}
            >
              <CartesianGrid stroke="#17303a" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fill: '#607985', fontSize: 9 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                yAxisId="left"
                tick={{ fill: '#607985', fontSize: 9 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0, 100]}
                hide
              />
              <ChartTooltip
                contentStyle={{
                  background: '#09151b',
                  border: '1px solid #24404b',
                  fontSize: 11,
                }}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="speed"
                name="속도 m/s"
                stroke="#67e8f9"
                dot={false}
                strokeWidth={1.7}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="battery"
                name="배터리 %"
                stroke="#fbbf62"
                dot={false}
                strokeWidth={1.7}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-legend">
          <span>
            <i className="cyan" />
            속도 m/s
          </span>
          <span>
            <i className="amber" />
            배터리 %
          </span>
        </div>
      </div>
      <div className="rail-section">
        <div className="section-heading">
          <span>임무 검증</span>
          <ShieldCheck size={14} />
        </div>
        <div className="validation-score">
          <span>
            {logs.length >= 2 ? '2-기체 로그 정합' : '추가 로그 필요'}
          </span>
          <strong>{logs.length >= 2 ? '검증 완료' : '대기'}</strong>
        </div>
        <dl className="analysis-data">
          <div>
            <dt>기체 이탈 추정</dt>
            <dd>
              {logs.length >= 2
                ? formatMissionTime(
                    Math.min(
                      ...logs.map((log) => log.points.at(-1)!.timeMs / 1000),
                    ),
                  )
                : '—'}
            </dd>
          </div>
          <div>
            <dt>공백 회복시간</dt>
            <dd>
              {handoverSeconds === null
                ? '미확인'
                : `${handoverSeconds.toFixed(1)} s`}
            </dd>
          </div>
          <div>
            <dt>시간축 정렬</dt>
            <dd>경과시간 기준</dd>
          </div>
        </dl>
      </div>
    </>
  );
}
