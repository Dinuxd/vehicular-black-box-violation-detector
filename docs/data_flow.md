# Data Flow

## Edge Event Flow

1. Raw sensor frame/window is captured.
2. Module-specific preprocessing converts it to the model input:
   - audio to mel/log-mel features
   - IMU rows to fixed windows and summary features
   - camera frames to crops or segmentation/detection inputs
3. The deploy artifact produces a score, class, or bounding box.
4. Rule logic applies thresholds, debouncing, cooldowns, and context.
5. A normalized event is emitted to the aggregator.
6. The aggregator writes local output and calls the scoring module periodically.
7. The upload client sends event JSON and optional media evidence to backend services.

## Backend Flow

1. Ingest service receives event JSON.
2. Event and GPS metadata are stored in PostgreSQL.
3. Media service prepares and verifies evidence uploads.
4. Dashboard requests devices, events, violation types, and score summaries.

## Driver Score Flow

`scoring/driver_violation_index/driving_index.py` maps normalized violation types to weighted risk contributions. Critical events such as crash and tamper can force higher minimum scores. The score is an advisory FYP metric, not a legal or insurance decision.
