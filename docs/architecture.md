# Architecture

The architecture has three layers:

1. Edge sensing and inference on Raspberry Pi 4B.
2. Backend ingestion/media APIs with PostgreSQL storage.
3. Dashboard views for devices, events, GPS, and scoring.

```mermaid
flowchart TB
  Audio["Microphone / audio input"] --> AudioModels["Horn, hello, shouting, crash audio"]
  IMU["BMI160 IMU"] --> IMUModels["Harsh braking, lane change, aggressive driving, crash IMU"]
  FrontCam["Front camera"] --> RoadModels["Road sign and road-line models"]
  DriverCam["Driver camera"] --> DriverModels["Drowsiness runtime"]
  GPS["GPS"] --> Rules["Violation logic"]
  Tamper["Tamper sensor"] --> Rules
  AudioModels --> Runtime["Pi event aggregator"]
  IMUModels --> Runtime
  RoadModels --> Runtime
  DriverModels --> Runtime
  Rules --> Runtime
  Runtime --> LocalStore["Local events JSON/JSONL"]
  Runtime --> Upload["GSM/LTE upload"]
  Upload --> Ingest["Go ingest service"]
  Upload --> Media["Go media service"]
  Ingest --> Postgres["PostgreSQL"]
  Media --> Postgres
  Dashboard["React dashboard"] --> Ingest
  Dashboard --> Media
```
