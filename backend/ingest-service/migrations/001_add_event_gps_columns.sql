ALTER TABLE events
  ADD COLUMN gps_latitude double precision,
  ADD COLUMN gps_longitude double precision,
  ADD COLUMN gps_captured_at timestamptz,
  ADD COLUMN gps_accuracy_m real;

ALTER TABLE events
  ADD CONSTRAINT events_gps_latitude_chk
    CHECK (gps_latitude IS NULL OR (gps_latitude >= -90 AND gps_latitude <= 90));

ALTER TABLE events
  ADD CONSTRAINT events_gps_longitude_chk
    CHECK (gps_longitude IS NULL OR (gps_longitude >= -180 AND gps_longitude <= 180));

ALTER TABLE events
  ADD CONSTRAINT events_gps_pair_chk
    CHECK (
      (gps_latitude IS NULL AND gps_longitude IS NULL) OR
      (gps_latitude IS NOT NULL AND gps_longitude IS NOT NULL)
    );

CREATE INDEX IF NOT EXISTS idx_events_device_ts
  ON events (device_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_events_device_type_ts
  ON events (device_id, event_type, ts DESC);
