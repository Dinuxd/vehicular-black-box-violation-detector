ALTER TABLE events
  ADD COLUMN IF NOT EXISTS score_points integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS score_policy_version text NOT NULL DEFAULT 'weighted-decay-v1';

CREATE TABLE IF NOT EXISTS device_scores (
  device_id text PRIMARY KEY,
  current_score integer NOT NULL DEFAULT 0,
  total_violations integer NOT NULL DEFAULT 0,
  last_violation_at timestamptz,
  score_policy_version text NOT NULL,
  updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_device_scores_updated_at
  ON device_scores (updated_at DESC);
