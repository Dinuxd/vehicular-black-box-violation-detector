import { useEffect, useState } from "react";
import { useNavigate } from "react-router";

import { Badge, Button, Card, CardContent, cn } from "@/lib/platform-ui-common";
import { ChevronRight, Monitor } from "lucide-react";

import {
  getDeviceScore,
  getDevices,
  getEvents,
  type DeviceScoreResponse,
} from "@/api/ingest";

type DeviceSummary = {
  score?: DeviceScoreResponse;
  lastSeen?: string;
  totalEvents: number;
  error?: string;
};

function errMsg(e: unknown, fallback: string) {
  return e instanceof Error ? e.message : fallback;
}

function formatDateTime(value?: string) {
  if (!value) return "No events yet";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function scoreChipClass(riskBand?: string) {
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

async function loadSummary(deviceId: string): Promise<DeviceSummary> {
  const [scoreResult, eventsResult] = await Promise.allSettled([
    getDeviceScore(deviceId),
    getEvents({ deviceId, limit: 1, offset: 0 }),
  ]);

  const summary: DeviceSummary = {
    totalEvents: 0,
  };

  if (scoreResult.status === "fulfilled") {
    summary.score = scoreResult.value;
  } else {
    summary.error = errMsg(scoreResult.reason, "Score unavailable");
  }

  if (eventsResult.status === "fulfilled") {
    summary.totalEvents = eventsResult.value.total ?? 0;
    summary.lastSeen = eventsResult.value.rows?.[0]?.ts;
  }

  return summary;
}

export function DevicesPage() {
  const navigate = useNavigate();
  const [devices, setDevices] = useState<string[]>([]);
  const [summaries, setSummaries] = useState<Record<string, DeviceSummary>>({});
  const [devicesLoading, setDevicesLoading] = useState(true);
  const [devicesError, setDevicesError] = useState<string | null>(null);

  async function loadDevices() {
    try {
      setDevicesLoading(true);
      setDevicesError(null);
      setSummaries({});

      const list = await getDevices();
      setDevices(list);

      const results = await Promise.allSettled(
        list.map(async (deviceId) => ({
          deviceId,
          summary: await loadSummary(deviceId),
        }))
      );

      const next: Record<string, DeviceSummary> = {};
      results.forEach((result, index) => {
        const deviceId = list[index];
        if (result.status === "fulfilled") {
          next[result.value.deviceId] = result.value.summary;
        } else {
          next[deviceId] = {
            totalEvents: 0,
            error: errMsg(result.reason, "Summary unavailable"),
          };
        }
      });
      setSummaries(next);
    } catch (e: unknown) {
      setDevicesError(errMsg(e, "Failed to load devices"));
      setDevices([]);
    } finally {
      setDevicesLoading(false);
    }
  }

  useEffect(() => {
    void loadDevices();
  }, []);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Devices</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Device profiles, violation counts, and ingest history.
          </p>
        </div>

        <Button variant="outline" onClick={loadDevices}>
          Refresh
        </Button>
      </div>

      {devicesLoading && (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            Loading devices...
          </CardContent>
        </Card>
      )}

      {!devicesLoading && devicesError && (
        <Card>
          <CardContent className="p-6 text-sm text-red-600">{devicesError}</CardContent>
        </Card>
      )}

      {!devicesLoading && !devicesError && devices.length === 0 && (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            No devices found. (events table is empty)
          </CardContent>
        </Card>
      )}

      <div className="space-y-3">
        {devices.map((deviceId) => {
          const summary = summaries[deviceId];
          const score = summary?.score;
          const scoreLabel = score ? `Score ${score.score}` : "Score ...";
          const violationsLabel = score
            ? `${score.total_violations} violations`
            : "Loading violations";

          return (
            <button
              key={deviceId}
              type="button"
              onClick={() => navigate(`/devices/${encodeURIComponent(deviceId)}`)}
              className={cn(
                "flex w-full items-center justify-between gap-4 rounded-md border bg-card px-5 py-4 text-left shadow-sm transition-colors",
                "hover:border-slate-300 hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              )}
            >
              <div className="flex min-w-0 items-center gap-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-md bg-muted">
                  <Monitor className="h-5 w-5" />
                </div>
                <div className="min-w-0">
                  <div className="truncate text-lg font-semibold">{deviceId}</div>
                  <div className="text-sm text-muted-foreground">
                    Last seen {formatDateTime(summary?.lastSeen)}
                  </div>
                </div>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <Badge variant="secondary" className="rounded-full px-3">
                  ACTIVE
                </Badge>
                <Badge variant="outline" className="rounded-full px-3">
                  {scoreLabel}
                </Badge>
                <Badge
                  variant="outline"
                  className={cn("rounded-full px-3", scoreChipClass(score?.risk_band))}
                >
                  {violationsLabel}
                </Badge>
                <ChevronRight className="h-5 w-5 text-muted-foreground" />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
