import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";

import {
  Badge,
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  cn,
} from "@/lib/platform-ui-common";
import { ArrowLeft, Car, ChevronRight, ShieldAlert } from "lucide-react";

import {
  getDeviceScore,
  getEvents,
  getViolations,
  type DeviceScoreResponse,
  type EventRow,
} from "@/api/ingest";

type DeviceStats = {
  firstSeen?: string;
  lastSeen?: string;
  totalEvents: number;
};

function labelizeEventType(t: string) {
  return t.replace(/_/g, " ");
}

function errMsg(e: unknown, fallback: string) {
  return e instanceof Error ? e.message : fallback;
}

function formatDateTime(value?: string | null) {
  if (!value) return "None";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function riskBandClass(riskBand?: string) {
  switch (riskBand) {
    case "CRITICAL":
      return "border-red-200 bg-red-50 text-red-700";
    case "HIGH":
      return "border-orange-200 bg-orange-50 text-orange-700";
    case "MODERATE":
      return "border-yellow-200 bg-yellow-50 text-yellow-800";
    default:
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }
}

function riskAccent(score?: DeviceScoreResponse) {
  switch (score?.risk_band) {
    case "CRITICAL":
      return "#dc2626";
    case "HIGH":
      return "#ea7a2f";
    case "MODERATE":
      return "#ca8a04";
    default:
      return "#059669";
  }
}

function RiskGauge({ score }: { score?: DeviceScoreResponse }) {
  const value = score?.score ?? 0;
  const capped = Math.max(0, Math.min(value, 100));
  const dash = 245;
  const active = (capped / 100) * dash;
  const accent = riskAccent(score);

  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <div className="relative h-56 w-56">
        <svg viewBox="0 0 240 230" className="h-full w-full">
          <path
            d="M 42 150 A 78 78 0 0 1 198 150"
            fill="none"
            stroke="#e5e7eb"
            strokeLinecap="round"
            strokeWidth="16"
          />
          <path
            d="M 42 150 A 78 78 0 0 1 198 150"
            fill="none"
            stroke={accent}
            strokeDasharray={`${active} ${dash}`}
            strokeLinecap="round"
            strokeWidth="16"
          />
        </svg>
        <div className="absolute inset-x-0 top-28 text-center">
          <div className="text-5xl font-bold tabular-nums" style={{ color: accent }}>
            {value}
          </div>
          <div className="mt-1 text-xs font-semibold uppercase text-muted-foreground">
            Score
          </div>
          <div className="mt-2 text-sm font-semibold" style={{ color: accent }}>
            {score?.risk_band ?? "LOW"}
          </div>
        </div>
      </div>
      <div className="w-full rounded-md border bg-muted/30 px-3 py-2 text-center text-xs text-muted-foreground">
        Policy {score?.score_policy_version ?? "weighted-decay-v1"}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border bg-background p-4">
      <div className="text-xs font-semibold uppercase text-muted-foreground">{label}</div>
      <div className="mt-3 text-sm font-semibold leading-snug">{value}</div>
    </div>
  );
}

export function DeviceEventsPage() {
  const { deviceId } = useParams<{ deviceId: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const violation = useMemo(() => {
    const sp = new URLSearchParams(location.search);
    return sp.get("violation") ?? "";
  }, [location.search]);

  const [score, setScore] = useState<DeviceScoreResponse | null>(null);
  const [stats, setStats] = useState<DeviceStats>({ totalEvents: 0 });
  const [violations, setViolations] = useState<string[]>([]);
  const [rows, setRows] = useState<EventRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [eventsErr, setEventsErr] = useState<string | null>(null);

  useEffect(() => {
    if (!deviceId) return;

    let cancelled = false;

    (async () => {
      try {
        setLoading(true);
        setErr(null);

        const [scoreData, violationTypes, eventsData] = await Promise.all([
          getDeviceScore(deviceId),
          getViolations(deviceId),
          getEvents({ deviceId, limit: 500, offset: 0 }),
        ]);

        if (cancelled) return;

        const allRows = eventsData.rows ?? [];
        setScore(scoreData);
        setViolations(violationTypes);
        setStats({
          firstSeen: allRows[allRows.length - 1]?.ts,
          lastSeen: allRows[0]?.ts,
          totalEvents: eventsData.total ?? allRows.length,
        });
      } catch (e: unknown) {
        if (!cancelled) setErr(errMsg(e, "Failed to load device profile"));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [deviceId]);

  useEffect(() => {
    if (!deviceId || !violation) {
      setRows([]);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        setEventsLoading(true);
        setEventsErr(null);

        const res = await getEvents({
          deviceId,
          eventType: violation,
          limit: 200,
          offset: 0,
        });

        if (!cancelled) setRows(res.rows ?? []);
      } catch (e: unknown) {
        if (!cancelled) {
          setEventsErr(errMsg(e, "Failed to load events"));
          setRows([]);
        }
      } finally {
        if (!cancelled) setEventsLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [deviceId, violation]);

  function openEvent(eventId: string) {
    if (!deviceId) return;

    const search = violation ? `?violation=${encodeURIComponent(violation)}` : "";
    navigate(`/devices/${encodeURIComponent(deviceId)}/events/${encodeURIComponent(eventId)}${search}`);
  }

  if (!deviceId) {
    return (
      <div className="p-8 text-sm text-muted-foreground">
        Missing deviceId (route param). Open from Devices page.
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <Link
        to="/devices"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to devices
      </Link>

      {loading && (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            Loading device profile...
          </CardContent>
        </Card>
      )}

      {!loading && err && (
        <Card>
          <CardContent className="p-6 text-sm text-red-600">{err}</CardContent>
        </Card>
      )}

      {!loading && !err && (
        <>
          <div className="grid gap-5 lg:grid-cols-[1fr_280px]">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-md bg-gradient-to-br from-blue-500 to-indigo-500 text-white shadow-sm">
                    <Car className="h-6 w-6" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <h1 className="text-2xl font-semibold tracking-tight">{deviceId}</h1>
                      <Badge variant="secondary" className="rounded-full px-3">
                        ACTIVE
                      </Badge>
                      <Badge
                        variant="outline"
                        className={cn("rounded-full px-3", riskBandClass(score?.risk_band))}
                      >
                        {score?.risk_band ?? "LOW"}
                      </Badge>
                    </div>
                    <div className="mt-1 text-sm text-muted-foreground">{deviceId}</div>
                  </div>
                </div>

                <div className="mt-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <StatCard label="First seen" value={formatDateTime(stats.firstSeen)} />
                  <StatCard label="Last seen" value={formatDateTime(stats.lastSeen)} />
                  <StatCard label="Total events" value={stats.totalEvents} />
                  <StatCard
                    label="Last violation"
                    value={formatDateTime(score?.last_violation_at)}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Risk Score</CardTitle>
              </CardHeader>
              <CardContent>
                <RiskGauge score={score ?? undefined} />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Violation Types</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {violations.length === 0 && (
                <div className="text-sm text-muted-foreground">
                  No violations found for this device.
                </div>
              )}

              {violations.map((item) => {
                const selected = item === violation;
                return (
                  <Link
                    key={item}
                    to={`/devices/${encodeURIComponent(deviceId)}?violation=${encodeURIComponent(item)}`}
                    className={cn(
                      "flex items-center justify-between rounded-md border bg-background px-3 py-3 transition-colors",
                      selected
                        ? "border-blue-200 bg-blue-50 text-blue-900"
                        : "hover:bg-accent hover:text-accent-foreground"
                    )}
                  >
                    <div className="flex items-center gap-3">
                      <ShieldAlert
                        className={cn("h-4 w-4", selected ? "text-blue-600" : "text-blue-500")}
                      />
                      <span className="text-sm font-semibold">{labelizeEventType(item)}</span>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </Link>
                );
              })}
            </CardContent>
          </Card>

          {violation && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{labelizeEventType(violation)} Events</CardTitle>
              </CardHeader>
              <CardContent>
                {eventsLoading && (
                  <div className="text-sm text-muted-foreground">Loading events...</div>
                )}

                {!eventsLoading && eventsErr && (
                  <div className="text-sm text-red-600">{eventsErr}</div>
                )}

                {!eventsLoading && !eventsErr && (
                  <div className="w-full overflow-x-auto">
                    <Table className="min-w-max">
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[60px]">#</TableHead>
                          <TableHead>Timestamp</TableHead>
                          <TableHead>Event type</TableHead>
                          <TableHead>Severity</TableHead>
                        </TableRow>
                      </TableHeader>

                      <TableBody>
                        {rows.length === 0 ? (
                          <TableRow>
                            <TableCell colSpan={4} className="text-sm text-muted-foreground">
                              No events found for this device and violation type.
                            </TableCell>
                          </TableRow>
                        ) : (
                          rows.map((row) => (
                            <TableRow
                              key={row.event_id}
                              className="cursor-pointer transition-colors hover:bg-accent/70"
                              onClick={() => openEvent(row.event_id)}
                              onKeyDown={(event) => {
                                if (event.key === "Enter" || event.key === " ") {
                                  event.preventDefault();
                                  openEvent(row.event_id);
                                }
                              }}
                              role="link"
                              tabIndex={0}
                            >
                              <TableCell>{row.seq}</TableCell>
                              <TableCell className="whitespace-nowrap">
                                {formatDateTime(row.ts)}
                              </TableCell>
                              <TableCell className="whitespace-nowrap">
                                {labelizeEventType(row.event_type)}
                              </TableCell>
                              <TableCell className="whitespace-nowrap">{row.severity}</TableCell>
                            </TableRow>
                          ))
                        )}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
