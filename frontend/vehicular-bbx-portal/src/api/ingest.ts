export type DevicesResponse = { devices: string[] };

export type ViolationsResponse = {
  device_id: string;
  violations: string[];
};

export type DeviceScoreResponse = {
  device_id: string;
  score: number;
  risk_band: "LOW" | "MODERATE" | "HIGH" | "CRITICAL" | string;
  total_violations: number;
  last_violation_at?: string | null;
  score_policy_version: string;
  half_life_days: number;
  updated_at: string;
};

export type EventGPS = {
  latitude: number;
  longitude: number;
  captured_at: string;
  accuracy_m?: number | null;
};

export type EventRow = {
  seq: number;
  event_id: string;
  ts: string; // RFC3339 / ISO string
  event_type: string;
  severity: string;
  gps?: EventGPS | null;
};

export type EventsResponse = {
  device_id: string;
  event_type: string;
  limit: number;
  offset: number;
  total: number;
  rows: EventRow[];
};

const INGEST_BASE = import.meta.env.VITE_INGEST_BASE_URL ?? "http://localhost:8080";

async function httpJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${INGEST_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    let text = "";
    try {
      text = await res.text();
    } catch {
      // ignore
    }
    throw new Error(`${res.status} ${res.statusText}${text ? ` - ${text}` : ""}`);
  }

  return (await res.json()) as T;
}

export async function getDevices(): Promise<string[]> {
  const data = await httpJSON<DevicesResponse>("/devices");
  return data.devices ?? [];
}

export async function getViolations(deviceId: string): Promise<string[]> {
  const data = await httpJSON<ViolationsResponse>(
    `/devices/${encodeURIComponent(deviceId)}/violations`
  );
  return data.violations ?? [];
}

export async function getDeviceScore(deviceId: string): Promise<DeviceScoreResponse> {
  return await httpJSON<DeviceScoreResponse>(
    `/devices/${encodeURIComponent(deviceId)}/score`
  );
}

export async function getEvents(params: {
  deviceId: string;
  eventType?: string;
  limit?: number;
  offset?: number;
}): Promise<EventsResponse> {
  const q = new URLSearchParams();
  if (params.eventType) q.set("event_type", params.eventType);
  if (typeof params.limit === "number") q.set("limit", String(params.limit));
  if (typeof params.offset === "number") q.set("offset", String(params.offset));

  const query = q.toString();

  return await httpJSON<EventsResponse>(
    `/devices/${encodeURIComponent(params.deviceId)}/events${query ? `?${query}` : ""}`
  );
}
