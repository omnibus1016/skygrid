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
  Activity,
  AlertTriangle,
  BatteryMedium,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Clock3,
  Crosshair,
  Database,
  Download,
  FileChartColumn,
  Gauge,
  LocateFixed,
  Map as MapIcon,
  MapPinPlus,
  Pause,
  Play,
  Radio,
  RefreshCw,
  Route,
  Satellite,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
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
  DialogClose,
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
  calculateLogMetrics,
  createDemoLogs,
  estimateHandoverSeconds,
  parseFlightCsv,
} from '@/lib/skygrid/log-parser';
import {
  assignRoutes,
  plannerScore,
  trainDqnPolicy,
} from '@/lib/skygrid/rl-policy';
import {
  advanceMission,
  createMission,
  missionMetrics,
  runBatchEvaluation,
  triggerFailure,
} from '@/lib/skygrid/simulation';
import type {
  AppMode,
  FlightLog,
  GeoPoint,
  PlannerKind,
  ScenarioConfig,
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
  droneCount: 4,
  waypointCount: 20,
  failureAt: 90,
  failureDroneId: 'UAV-02',
  planner: 'rl',
  simRate: 4,
  sensorRadiusM: 110,
  randomSeed: 20261125,
};

const PLANNER_LABEL: Record<PlannerKind, string> = {
  rl: '강화학습 DQN',
  nearest: '최근접 우선',
  priority: '중요도 우선',
};

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
  const training = useMemo(() => trainDqnPolicy(700, 2026), []);
  const [policyBundle, setPolicyBundle] = useState(training);
  const policyStats = policyBundle.stats;
  const [mode, setMode] = useState<AppMode>('simulation');
  const [config, setConfig] = useState<ScenarioConfig>(DEFAULT_CONFIG);
  const [mission, setMission] = useState(() =>
    createMission(DEFAULT_CONFIG, training.policy),
  );
  const [selectedDroneId, setSelectedDroneId] = useState('UAV-01');
  const [selectedWaypointId, setSelectedWaypointId] = useState('RP-01');
  const [interaction, setInteraction] = useState<
    'inspect' | 'add-waypoint' | 'move-drone'
  >('inspect');
  const [logs, setLogs] = useState<FlightLog[]>(() => createDemoLogs(mission));
  const [selectedLogId, setSelectedLogId] = useState('DEMO-UAV-01');
  const [uploadError, setUploadError] = useState('');
  const [batchResults, setBatchResults] = useState(() =>
    runBatchEvaluation(DEFAULT_CONFIG, training.policy, 48),
  );
  const [busyAction, setBusyAction] = useState<'train' | 'batch' | null>(null);
  const [dropoutOpen, setDropoutOpen] = useState(false);
  const [dropoutDroneId, setDropoutDroneId] = useState('UAV-02');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const metrics = useMemo(() => missionMetrics(mission), [mission]);
  const selectedDrone =
    mission.drones.find((drone) => drone.id === selectedDroneId) ??
    mission.drones[0];
  const selectedWaypoint = mission.waypoints.find(
    (waypoint) => waypoint.id === selectedWaypointId,
  );
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
        ),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);
  }, [mission.waypoints, mission.time, selectedDrone, policyBundle.policy]);

  useEffect(() => {
    if (!mission.running) return;
    const timer = window.setInterval(() => {
      setMission((current) =>
        advanceMission(
          current,
          0.25 * config.simRate,
          mode === 'field'
            ? { ...config, failureAt: Number.POSITIVE_INFINITY }
            : config,
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
      setInteraction('inspect');
    },
    [config, policyBundle.policy],
  );

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

  const confirmDropout = useCallback(() => {
    setMission((current) =>
      triggerFailure(
        current,
        dropoutDroneId,
        config.planner,
        policyBundle.policy,
      ),
    );
    setDropoutOpen(false);
  }, [dropoutDroneId, config.planner, policyBundle.policy]);

  const handleMapClick = useCallback(
    (point: GeoPoint) => {
      if (interaction === 'add-waypoint') {
        setMission((current) => {
          const waypoint: Waypoint = {
            id: `RP-${String(current.waypoints.length + 1).padStart(2, '0')}`,
            ...point,
            priority: 4,
            revisitSec: 150,
            dwellSec: 8,
            lastVisited: current.time,
            visitedCount: 0,
          };
          return { ...current, waypoints: [...current.waypoints, waypoint] };
        });
        setInteraction('inspect');
      } else if (interaction === 'move-drone' && selectedDroneId) {
        setMission((current) => ({
          ...current,
          drones: current.drones.map((drone) =>
            drone.id === selectedDroneId ? { ...drone, ...point } : drone,
          ),
        }));
        setInteraction('inspect');
      }
    },
    [interaction, selectedDroneId],
  );

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
    window.setTimeout(() => {
      const next = trainDqnPolicy(
        1_000,
        config.randomSeed + policyStats.episodes,
      );
      setPolicyBundle(next);
      setBusyAction(null);
    }, 40);
  }, [config.randomSeed, policyStats.episodes]);

  const runBatch = useCallback(() => {
    setBusyAction('batch');
    window.setTimeout(() => {
      setBatchResults(runBatchEvaluation(config, policyBundle.policy, 100));
      setBusyAction(null);
    }, 40);
  }, [config, policyBundle.policy]);

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
          description:
            '가상 실험의 기체 수, 정찰지점 수, 기체 이탈 시점을 설정하고 화면에 적용합니다.',
          inputSchema: {
            type: 'object',
            properties: {
              droneCount: { type: 'number', minimum: 2, maximum: 10 },
              waypointCount: { type: 'number', minimum: 6, maximum: 30 },
              failureAt: { type: 'number', minimum: 30, maximum: 480 },
            },
            required: ['droneCount', 'waypointCount', 'failureAt'],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          execute: async (input: unknown) => {
            const value = input as {
              droneCount: number;
              waypointCount: number;
              failureAt: number;
            };
            const next = {
              ...config,
              droneCount: value.droneCount,
              waypointCount: value.waypointCount,
              failureAt: value.failureAt,
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
          description: '현재 설정된 가상 정찰 임무 시뮬레이션을 시작합니다.',
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
          description:
            '지정한 무인기를 임무에서 이탈시키고 강화학습 정책으로 잔여 기체 임무를 재계획합니다.',
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
          <div className="brand-mark">
            <Crosshair size={19} />
          </div>
          <div>
            <div className="brand-name">SKYGRID</div>
            <div className="brand-sub">UAS MISSION CONTINUITY SYSTEM</div>
          </div>
        </div>
        <Tabs
          value={mode}
          onValueChange={(value) => changeMode(value as AppMode)}
          className="hidden lg:flex"
        >
          <TabsList variant="line" className="h-14 gap-1">
            <TabsTrigger value="field" className="mission-tab">
              <Radio /> 현장 운용
            </TabsTrigger>
            <TabsTrigger value="analysis" className="mission-tab">
              <Database /> 비행 검증
            </TabsTrigger>
            <TabsTrigger value="simulation" className="mission-tab">
              <Bot /> 가상 실험
            </TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="header-status">
          <Badge className="system-badge">
            <span className="status-pulse" />{' '}
            {mission.running ? 'MISSION ACTIVE' : 'SYSTEM READY'}
          </Badge>
          <button
            className="scenario-select"
            type="button"
            aria-label="기체 이탈 시나리오 열기"
            onClick={() => setDropoutOpen(true)}
          >
            SCN-{String(config.randomSeed).slice(-2)} / 기체 이탈{' '}
            <ChevronDown size={14} />
          </button>
        </div>
      </header>

      <section className="mission-shell">
        <aside className="left-rail">
          {mode === 'simulation' && (
            <>
              <div className="rail-section">
                <div className="section-heading">
                  <span>SIMULATION CONTROL</span>
                  <span>{formatMissionTime(mission.time)}</span>
                </div>
                <div className="button-pair">
                  <Button
                    className="primary-command"
                    onClick={() =>
                      setMission((current) => ({
                        ...current,
                        running: !current.running,
                      }))
                    }
                  >
                    {mission.running ? <Pause /> : <Play />}
                    {mission.running ? '일시 정지' : '실험 시작'}
                  </Button>
                  <Button
                    variant="outline"
                    className="icon-command"
                    size="icon"
                    onClick={() => applyScenario()}
                    aria-label="시뮬레이션 초기화"
                  >
                    <RefreshCw />
                  </Button>
                </div>
                <Button
                  variant="destructive"
                  className="mt-2 h-9 w-full justify-start"
                  onClick={() => setDropoutOpen(true)}
                  disabled={mission.failureTriggered}
                >
                  <AlertTriangle /> 지금 기체 이탈
                </Button>
              </div>

              <div className="rail-section parameter-panel">
                <div className="section-heading">
                  <span>EXPERIMENT SETUP</span>
                  <SlidersHorizontal size={14} />
                </div>
                <ParameterSlider
                  label="가상 기체"
                  value={config.droneCount}
                  suffix="대"
                  min={2}
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
                  label="정찰지점"
                  value={config.waypointCount}
                  suffix="개"
                  min={6}
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
                  max={360}
                  step={10}
                  onChange={(value) =>
                    setConfig((current) => ({ ...current, failureAt: value }))
                  }
                />
                <ParameterSlider
                  label="실험 배속"
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
                  <SelectTrigger id="planner-select" className="control-select">
                    <SelectValue>{PLANNER_LABEL[config.planner]}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rl">강화학습 DQN</SelectItem>
                    <SelectItem value="nearest">최근접 우선</SelectItem>
                    <SelectItem value="priority">중요도 우선</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  className="mt-3 h-9 w-full"
                  onClick={() => applyScenario()}
                >
                  <CheckCircle2 /> 설정 적용
                </Button>
              </div>

              <WaypointEditor
                selected={selectedWaypoint}
                onChange={updateWaypoint}
              />
            </>
          )}

          {mode === 'field' && (
            <>
              <div className="rail-section">
                <div className="section-heading">
                  <span>FIELD MISSION</span>
                  <span>{formatMissionTime(mission.time)}</span>
                </div>
                <div className="button-pair">
                  <Button
                    className="primary-command"
                    onClick={() =>
                      setMission((current) => ({
                        ...current,
                        running: !current.running,
                      }))
                    }
                  >
                    {mission.running ? <Pause /> : <Play />}
                    {mission.running ? '임무 정지' : '임무 개시'}
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    className="icon-command"
                    onClick={() => applyScenario()}
                    aria-label="현장 임무 초기화"
                  >
                    <RefreshCw />
                  </Button>
                </div>
                <Button
                  variant="destructive"
                  className="mt-2 h-10 w-full justify-start"
                  onClick={() => setDropoutOpen(true)}
                >
                  <AlertTriangle /> 기체 이탈 보고
                </Button>
              </div>

              <div className="rail-section">
                <div className="section-heading">
                  <span>MAP INPUT</span>
                  <LocateFixed size={14} />
                </div>
                <div className="button-stack">
                  <Button
                    variant={
                      interaction === 'add-waypoint' ? 'default' : 'outline'
                    }
                    className="justify-start"
                    onClick={() =>
                      setInteraction(
                        interaction === 'add-waypoint'
                          ? 'inspect'
                          : 'add-waypoint',
                      )
                    }
                  >
                    <MapPinPlus /> 지도에서 정찰지점 추가
                  </Button>
                  <Button
                    variant={
                      interaction === 'move-drone' ? 'default' : 'outline'
                    }
                    className="justify-start"
                    onClick={() =>
                      setInteraction(
                        interaction === 'move-drone' ? 'inspect' : 'move-drone',
                      )
                    }
                  >
                    <LocateFixed /> 선택 기체 위치 지정
                  </Button>
                </div>
                {interaction !== 'inspect' && (
                  <div className="inline-notice">
                    <CircleDot /> 지도에서 위치를 한 번 클릭하십시오.
                  </div>
                )}
              </div>

              {selectedDrone && (
                <div className="rail-section">
                  <div className="section-heading">
                    <span>SELECTED AIRCRAFT</span>
                    <span style={{ color: selectedDrone.color }}>
                      {selectedDrone.id}
                    </span>
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
                        {selectedDrone.status.toUpperCase()} ·{' '}
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
                  >
                    <BrainCircuit /> AI 경로 재계산
                  </Button>
                </div>
              )}

              <WaypointEditor
                selected={selectedWaypoint}
                onChange={updateWaypoint}
              />
            </>
          )}

          {mode === 'analysis' && (
            <>
              <div className="rail-section">
                <div className="section-heading">
                  <span>FLIGHT RECORDS</span>
                  <span>{logs.length} FILES</span>
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
                  <span>LOG INVENTORY</span>
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
          <div className="map-toolbar">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="map-badge">
                <MapIcon /> 청주 시험구역 / WGS84
              </Badge>
              {mode !== 'analysis' && (
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
                    : `이탈 예정 ${formatMissionTime(config.failureAt)}`}
                </Badge>
              )}
              {mode === 'analysis' && (
                <Badge variant="outline" className="event-badge">
                  <Database /> 비행기록 {logs.length}개 중첩
                </Badge>
              )}
            </div>
            <div className="map-mode-label">
              <span className="status-pulse" />
              {mode === 'field'
                ? 'FIELD ASSIST'
                : mode === 'analysis'
                  ? 'LOG REPLAY'
                  : 'MISSION SIM'}
            </div>
          </div>

          <OperationalMap
            mode={mode}
            mission={mission}
            logs={logs}
            selectedDroneId={selectedDroneId}
            interaction={interaction}
            onMapClick={handleMapClick}
            onSelectDrone={setSelectedDroneId}
            onSelectWaypoint={setSelectedWaypointId}
          />

          <div className="map-footer">
            <div>
              <span>MISSION TIME</span>
              <strong>
                {formatMissionTime(mission.time)}.
                {Math.floor((mission.time % 1) * 10)}
              </strong>
            </div>
            <div>
              <span>ACTIVE / TOTAL</span>
              <strong>
                {metrics.activeDrones} / {mission.drones.length}
              </strong>
            </div>
            <div>
              <span>AI INFERENCE</span>
              <strong>
                {mission.inferenceMs
                  ? `${mission.inferenceMs.toFixed(1)} ms`
                  : 'STANDBY'}
              </strong>
            </div>
            <div>
              <span>POSITION SOURCE</span>
              <strong>
                {mode === 'analysis'
                  ? 'DJI LOG'
                  : mode === 'field'
                    ? 'OPERATOR INPUT'
                    : 'SIMULATED'}
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
                  <span>MISSION EFFECT</span>
                  <span>{mission.running ? 'LIVE' : 'READY'}</span>
                </div>
                <div className="metric-hero">
                  <div
                    className="metric-ring"
                    style={
                      {
                        '--value': `${metrics.continuity * 3.6}deg`,
                      } as React.CSSProperties
                    }
                  >
                    <span>{metrics.continuity.toFixed(0)}</span>
                    <small>%</small>
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-slate-100">
                      정찰 임무 연속성
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      중요도 가중 현재 충족률
                    </div>
                  </div>
                </div>
                <div className="metric-grid">
                  <MetricCard
                    icon={<Clock3 />}
                    label="가중 공백"
                    value={`${metrics.weightedGapSeconds.toFixed(0)} p·s`}
                    tone={metrics.weightedGapSeconds > 100 ? 'amber' : 'cyan'}
                  />
                  <MetricCard
                    icon={<Gauge />}
                    label="누적 거리"
                    value={`${metrics.totalDistanceKm.toFixed(2)} km`}
                  />
                  <MetricCard
                    icon={<Activity />}
                    label="정찰 횟수"
                    value={`${metrics.completedVisits}`}
                  />
                  <MetricCard
                    icon={<BatteryMedium />}
                    label="평균 배터리"
                    value={`${metrics.averageBattery.toFixed(0)} %`}
                    tone={metrics.averageBattery < 35 ? 'red' : 'cyan'}
                  />
                </div>
              </div>

              <div className="rail-section">
                <div className="section-heading">
                  <span>AI POLICY</span>
                  <span className="text-cyan-300">
                    {busyAction === 'train' ? 'TRAINING' : 'ONLINE'}
                  </span>
                </div>
                <div className="policy-card">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="font-mono text-sm text-slate-100">
                        DQN-MC / CANDIDATE-Q
                      </div>
                      <div className="mt-1 text-xs text-slate-500">
                        8 → 14 RELU → Q(s,a)
                      </div>
                    </div>
                    <Bot className="text-cyan-300" size={20} />
                  </div>
                  <div className="policy-spark">
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                      minWidth={0}
                      minHeight={0}
                      initialDimension={{ width: 240, height: 76 }}
                    >
                      <AreaChart data={policyStats.rewardHistory}>
                        <defs>
                          <linearGradient
                            id="reward-fill"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="0%"
                              stopColor="#67e8f9"
                              stopOpacity=".26"
                            />
                            <stop
                              offset="100%"
                              stopColor="#67e8f9"
                              stopOpacity="0"
                            />
                          </linearGradient>
                        </defs>
                        <Area
                          type="monotone"
                          dataKey="reward"
                          stroke="#67e8f9"
                          fill="url(#reward-fill)"
                          strokeWidth={1.5}
                          dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                    <span>EPISODIC REWARD</span>
                  </div>
                  <div className="confidence">
                    <span>학습 에피소드</span>
                    <strong>{policyStats.episodes.toLocaleString()}</strong>
                  </div>
                  <Progress
                    value={Math.min(100, 100 - policyStats.epsilon * 100)}
                    className="h-1 bg-white/10"
                  />
                  <dl className="policy-stats">
                    <div>
                      <dt>평균 보상</dt>
                      <dd>{policyStats.averageReward.toFixed(2)}</dd>
                    </div>
                    <div>
                      <dt>TD 손실</dt>
                      <dd>{policyStats.finalLoss.toFixed(3)}</dd>
                    </div>
                  </dl>
                </div>
                <Button
                  variant="outline"
                  className="mt-3 h-8 w-full"
                  onClick={trainAgain}
                  disabled={busyAction !== null}
                >
                  <Zap
                    className={busyAction === 'train' ? 'animate-pulse' : ''}
                  />
                  {busyAction === 'train'
                    ? '정책 학습 중'
                    : '1,000 에피소드 정책 재학습'}
                </Button>
              </div>

              {mode === 'field' ? (
                <div className="rail-section flex-1">
                  <div className="section-heading">
                    <span>AI RECOMMENDATION</span>
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
                          PRI
                        </i>
                      </div>
                    ))}
                    {!selectedDrone?.route.length && (
                      <div className="empty-inline">
                        경로 재계산을 실행하십시오.
                      </div>
                    )}
                  </div>
                  <div className="section-heading mt-4">
                    <span>POLICY Q-RANK</span>
                    <span>TOP 4</span>
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
              ) : (
                <ComparisonRail
                  results={batchResults}
                  onRun={runBatch}
                  busy={busyAction === 'batch'}
                />
              )}

              <div className="rail-section event-section">
                <div className="section-heading">
                  <span>EVENT LOG</span>
                  <span>{mission.events.length} EVENTS</span>
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

      <Dialog open={dropoutOpen} onOpenChange={setDropoutOpen}>
        <DialogContent className="dropout-dialog">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="text-red-400" /> 기체 이탈 처리
            </DialogTitle>
            <DialogDescription>
              선택한 기체를 임무에서 제외하고 잔여 기체의 정찰경로를 즉시
              재계산합니다.
            </DialogDescription>
          </DialogHeader>
          <label className="control-label" htmlFor="dropout-drone-select">
            이탈 기체
          </label>
          <Select
            value={dropoutDroneId}
            onValueChange={(value) => {
              if (value) setDropoutDroneId(value);
            }}
          >
            <SelectTrigger id="dropout-drone-select" className="control-select">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {mission.drones
                .filter((drone) => drone.status === 'active')
                .map((drone) => (
                  <SelectItem key={drone.id} value={drone.id}>
                    {drone.id} · {drone.model}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          <div className="decision-preview">
            <BrainCircuit />
            <div>
              <strong>{PLANNER_LABEL[config.planner]}</strong>
              <span>
                현재 위치·배터리·정찰 중요도·미정찰 시간을 반영합니다.
              </span>
            </div>
          </div>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" />}>
              취소
            </DialogClose>
            <Button variant="destructive" onClick={confirmDropout}>
              <AlertTriangle /> 이탈 확정 및 재계획
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </main>
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
}: {
  selected?: Waypoint;
  onChange: (patch: Partial<Waypoint>) => void;
}) {
  if (!selected) return null;
  return (
    <div className="rail-section">
      <div className="section-heading">
        <span>RECON POINT</span>
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
    </div>
  );
}

function ComparisonRail({
  results,
  onRun,
  busy,
}: {
  results: ReturnType<typeof runBatchEvaluation>;
  onRun: () => void;
  busy: boolean;
}) {
  return (
    <div className="rail-section comparison-section">
      <div className="section-heading">
        <span>MONTE CARLO / 100 RUNS</span>
        <span>3 POLICIES</span>
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
              name="완료율 %"
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
          <span>FLIGHT SUMMARY</span>
          <span>{selectedLog ? 'VALID' : 'NO DATA'}</span>
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
          <span>TELEMETRY</span>
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
          <span>MISSION VALIDATION</span>
          <ShieldCheck size={14} />
        </div>
        <div className="validation-score">
          <span>
            {logs.length >= 2 ? '2-기체 로그 정합' : '추가 로그 필요'}
          </span>
          <strong>{logs.length >= 2 ? 'PASS' : 'WAIT'}</strong>
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
