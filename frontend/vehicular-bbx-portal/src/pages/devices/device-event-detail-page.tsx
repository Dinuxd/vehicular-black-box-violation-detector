import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useParams } from "react-router";

import { Badge, Card, CardContent, CardHeader, CardTitle } from "@/lib/platform-ui-common";
import { ArrowLeft, Crosshair, MapPin, Navigation } from "lucide-react";

import { getEvents, type EventRow } from "@/api/ingest";
import { EventLocationMap } from "@/components/maps/event-location-map";

function labelizeEventType(value: string) {
  return value.replace(/_/g, " ");
}

function errMsg(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function readViolation(search: string) {
  const params = new URLSearchParams(search);
  return params.get("violation") ?? "";
}

function formatCoordinate(value: number) {
  return value.toFixed(6);
}

export function DeviceEventDetailPage() {
  const { deviceId, eventId } = useParams<{ deviceId: string; eventId: string }>();
  const location = useLocation();

  const violation = useMemo(() => readViolation(location.search), [location.search]);

  const [eventRow, setEventRow] = useState<EventRow | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!deviceId || !eventId) {
      setEventRow(null);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        setLoading(true);
        setErr(null);

        const response = await getEvents({
          deviceId,
          eventType: violation || undefined,
          limit: 500,
          offset: 0,
        });

        const match = response.rows.find((row) => row.event_id === eventId) ?? null;

        if (!cancelled) {
          setEventRow(match);
          if (!match) {
            setErr("Event not found for this device.");
          }
        }
      } catch (error: unknown) {
        if (!cancelled) {
          setErr(errMsg(error, "Failed to load event details"));
          setEventRow(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [deviceId, eventId, violation]);

  const backToEventsHref = useMemo(() => {
    if (!deviceId) return "/devices";
    const search = violation ? `?violation=${encodeURIComponent(violation)}` : "";
    return `/devices/${encodeURIComponent(deviceId)}${search}`;
  }, [deviceId, violation]);

  const eventTitle = useMemo(() => {
    if (!eventRow) return eventId ?? "Event Detail";
    return `${labelizeEventType(eventRow.event_type)} - ${eventRow.event_id}`;
  }, [eventId, eventRow]);

  return (
    <div className="space-y-6 px-4 py-6 md:px-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link
          to={backToEventsHref}
          className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to event list
        </Link>
        {violation && <Badge variant="secondary">{labelizeEventType(violation)}</Badge>}
      </div>

      <Card>
        <CardHeader className="gap-4">
          <div className="space-y-1">
            <div className="text-sm text-muted-foreground">
              Device {deviceId ?? "Unknown device"}
            </div>
            <CardTitle>{eventTitle}</CardTitle>
          </div>
        </CardHeader>

        <CardContent>
          {loading && <div className="text-sm text-muted-foreground">Loading event details...</div>}

          {!loading && err && <div className="text-sm text-red-600">{err}</div>}

          {!loading && !err && eventRow && (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded-xl border bg-muted/20 p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Event ID
                </div>
                <div className="mt-2 break-all text-sm font-medium">{eventRow.event_id}</div>
              </div>

              <div className="rounded-xl border bg-muted/20 p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Event Time
                </div>
                <div className="mt-2 text-sm font-medium">
                  {new Date(eventRow.ts).toLocaleString()}
                </div>
              </div>

              <div className="rounded-xl border bg-muted/20 p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Event Type
                </div>
                <div className="mt-2 text-sm font-medium">
                  {labelizeEventType(eventRow.event_type)}
                </div>
              </div>

              <div className="rounded-xl border bg-muted/20 p-4">
                <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Severity
                </div>
                <div className="mt-2 text-sm font-medium">{eventRow.severity}</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {!loading && !err && eventRow && (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4 text-red-600" />
                <CardTitle>Violation Location</CardTitle>
              </div>
            </CardHeader>

            <CardContent>
              {eventRow.gps ? (
                <EventLocationMap
                  latitude={eventRow.gps.latitude}
                  longitude={eventRow.gps.longitude}
                  accuracyM={eventRow.gps.accuracy_m ?? null}
                  title={`${labelizeEventType(eventRow.event_type)} at ${new Date(eventRow.ts).toLocaleString()}`}
                />
              ) : (
                <div className="rounded-xl border border-dashed bg-muted/10 p-8 text-sm text-muted-foreground">
                  No GPS location was recorded for this event.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center gap-2">
                <Navigation className="h-4 w-4 text-orange-500" />
                <CardTitle>GPS Details</CardTitle>
              </div>
            </CardHeader>

            <CardContent>
              {eventRow.gps ? (
                <div className="space-y-4">
                  <div className="rounded-xl border bg-muted/20 p-4">
                    <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Latitude
                    </div>
                    <div className="mt-2 text-sm font-medium">
                      {formatCoordinate(eventRow.gps.latitude)}
                    </div>
                  </div>

                  <div className="rounded-xl border bg-muted/20 p-4">
                    <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Longitude
                    </div>
                    <div className="mt-2 text-sm font-medium">
                      {formatCoordinate(eventRow.gps.longitude)}
                    </div>
                  </div>

                  <div className="rounded-xl border bg-muted/20 p-4">
                    <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Captured At
                    </div>
                    <div className="mt-2 text-sm font-medium">
                      {new Date(eventRow.gps.captured_at).toLocaleString()}
                    </div>
                  </div>

                  <div className="rounded-xl border bg-muted/20 p-4">
                    <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      <Crosshair className="h-3.5 w-3.5" />
                      Accuracy
                    </div>
                    <div className="mt-2 text-sm font-medium">
                      {eventRow.gps.accuracy_m != null
                        ? `${eventRow.gps.accuracy_m.toFixed(1)} m`
                        : "Not recorded"}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-sm text-muted-foreground">
                  GPS metadata is unavailable for this event.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
