# System Design

The system is organized around an edge-first design. The Raspberry Pi performs live inference and rule evaluation inside the vehicle so that event detection does not depend on a continuous internet connection.

## Runtime Flow

1. Sensors stream raw inputs from audio, IMU, GPS, cameras, tamper hardware, and GSM/LTE status.
2. The Pi runtime runs module-specific detectors using lightweight deploy formats.
3. Detectors emit normalized violation events with timestamp, type, severity/confidence, and optional GPS/evidence metadata.
4. The local event aggregator writes JSON/JSONL output and periodically calls the driver violation index.
5. The upload layer sends events and evidence references to backend services when connectivity is available.
6. The dashboard reads backend APIs and visualizes devices, event history, GPS positions, and risk score.

## Runtime Design Choices

- TFLite, ONNX, NCNN, and joblib are preferred for live inference on Raspberry Pi.
- TensorFlow and PyTorch are documented as training/conversion dependencies, not preferred live runtime dependencies.
- The system tolerates missing optional model dependencies during checks and disables unavailable detectors rather than stopping the whole runtime.
- Standalone module scripts are kept for traceability, while `edge_device/raspberry_pi_deploy/` is the integrated runtime package.

## Main Data Contracts

The normalized event contract is:

```json
{
  "event_id": "unique-event-id",
  "device_id": "pi-001",
  "trip_id": "trip_live",
  "driver_id": "driver_live",
  "timestamp": "2026-06-25T10:00:00Z",
  "violation_type": "harsh_braking",
  "severity": "HIGH",
  "confidence": 0.91,
  "gps": {
    "latitude": 6.9158,
    "longitude": 79.9777,
    "accuracy_m": 5.0
  },
  "metadata": {}
}
```

## Failure Handling

- If GSM/LTE fails, events remain in local output until upload is retried by module/runtime logic.
- If GPS is unavailable, events can still be recorded without coordinates.
- If a model file is missing, the integrated check reports it before live deployment.
- If one detector fails, the remaining detectors should continue where possible.
